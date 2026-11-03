"""Prefix metadata of the waiting requests for the planner, kept on the CPU next to SGLang's HiCache radix tree.

For every waiting request it gives the prompt page path, the cached prefix in GPU and host memory and the cache nodes
the request would lock, without calling SGLang's match_prefix, touching device tensors or splitting a radix node. A
scheduled request is still matched and locked through SGLang before its prefill. Anchors are metadata references,
not substitutes for SGLang's last_node or prefix_indices.

As in SGLang's Req, origin_input_ids is immutable while a prefill is pending.
Replacement, length or namespace changes invalidate automatically. A caller
that edits that array in place must call invalidate_request before lookup.
Cache hits are determined by token and namespace equality, never by hashes.

A lookup keeps SGLang's LRU policy: each lookup, including a cached one, touches the root and every visited node in
traversal order. It does not split the last partial node of a request; SGLang's own match of a scheduled request does.
LRU touches do not invalidate the metadata. Cache mutations and locks advance separate epochs.

A changed epoch revalidates only the radix route of each request, so changes elsewhere in the tree do not repeat the
token matching. SGLang's insertion and splitting install new keys, so token containers are never edited in place; code
that edits them in place must call invalidate(topology=True), which forces complete rescans.
"""
from __future__ import annotations

from array import array
from collections import Counter, OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import wraps
import hashlib
import threading
import time
import weakref


TOPOLOGY_MUTATIONS = (
    'reset', 'insert', '_split_node', '_insert_helper_host', 'load_back',
    'init_load_back', 'evict', 'evict_host', '_detach_backuped',
    '_evict_backuped', '_evict_regular', '_drop_subtree_no_host', '_delete_leaf',
    'cache_finished_req', 'cache_unfinished_req', 'write_backup',
    '_finish_write_through_ack', '_record_store_event', '_record_remove_event',
    '_record_all_cleared_event',
)
LOCK_MUTATIONS = ('inc_lock_ref', 'dec_lock_ref')
REQUIRED_MUTATIONS = (
    'reset', 'insert', '_split_node', 'load_back', 'evict', 'evict_host',
    '_delete_leaf', 'cache_finished_req', 'cache_unfinished_req', 'write_backup',
    '_finish_write_through_ack', 'inc_lock_ref', 'dec_lock_ref',
)


@dataclass(frozen=True)
class Anchor:
    node: object = field(repr=False, compare=False)
    matched_tokens: int = 0
    partial: bool = False


@dataclass(frozen=True)
class Metadata:
    gpu_prefix: int
    host_prefix: int
    version: tuple[int, int]
    locks: tuple = field(repr=False)
    path: tuple = field(repr=False)
    last_device_anchor: Anchor = field(repr=False)
    last_host_anchor: Anchor = field(repr=False)
    best_anchor: Anchor = field(repr=False)
    held: bool = False


@dataclass(frozen=True)
class _Segment:
    node: object = field(repr=False)
    length: int
    end: int


@dataclass(frozen=True)
class _KeyDependency:
    # Strong CPU references prevent key/container ID reuse after replacements.
    # Identity is tested explicitly with ``is``, never container equality.
    key: object = field(repr=False, compare=False)
    tokens: object = field(repr=False, compare=False)
    scalars: tuple = field(repr=False)


@dataclass(frozen=True)
class _Edge:
    parent: object = field(repr=False)
    lookup_key: object = field(repr=False)
    child: object = field(repr=False)
    parent_key: _KeyDependency = field(repr=False)
    child_key: _KeyDependency | None = field(repr=False)
    query_offset: int
    matched_tokens: int


@dataclass
class _Entry:
    request: object = field(repr=False)
    input_container: object = field(repr=False)
    input_identity: tuple
    path: tuple = field(repr=False)
    topology_epoch: int = -1
    force_epoch: int = -1
    root: object = field(default=None, repr=False)
    dependencies: tuple = field(default=(), repr=False)
    segments: tuple = field(default=(), repr=False)
    locks_version: tuple = (-1, -1)
    locks_signature: tuple | None = field(default=None, repr=False)
    locks: tuple = field(default=(), repr=False)
    latest_metadata: object = field(default=None, repr=False)
    held_nodes: tuple = field(default=(), repr=False)
    held_prefix_reference: object = field(default=None, repr=False)


@dataclass(frozen=True)
class _TouchRequest:
    request: object = field(repr=False, compare=False)
    entry: object = field(repr=False, compare=False)
    input_container: object = field(repr=False, compare=False)
    scalars: tuple = field(repr=False)
    last_node: object = field(repr=False, compare=False)
    last_host_node: object = field(repr=False, compare=False)
    best_node: object = field(repr=False, compare=False)
    prefix_reference: object = field(repr=False, compare=False)
    segments: tuple | None = field(repr=False)


@dataclass(frozen=True)
class _TouchNode:
    node: object = field(repr=False, compare=False)
    parent: object = field(repr=False, compare=False)
    key: _KeyDependency = field(repr=False, compare=False)
    placement: tuple


@dataclass(frozen=True)
class TouchPlan:
    """The traversal of a complete lookup of the queue, replayed to reuse a decode-only certificate.

    It replays only the LRU touches of that lookup; the certificate itself is checked separately."""
    root: object = field(repr=False, compare=False)
    version: tuple
    requests: tuple = field(repr=False)
    nodes: tuple = field(repr=False)
    edges: tuple = field(repr=False)


class _TouchLease:
    def __init__(self, cache, plan):
        self.cache, self.plan, self.replayed = cache, plan, False

    def replay(self):
        if self.replayed:
            raise RuntimeError('native LRU touch plan replayed twice')
        # The caller retains the cache RLock across TP eligibility consensus.
        # A peer miss therefore performs no touch on either rank.
        for request in self.plan.requests:
            if request.segments is not None:
                self.cache._touch(request.segments)
        self.replayed = True
        self.cache.counters['early_touch_plan_replays'] += 1
        self.cache.counters['early_touch_requests'] += len(self.plan.requests)


def _namespace_key(tokens, offset, page, extra_key, cache_salt):
    plain = tokens[offset] if page == 1 else tuple(tokens[offset:offset+page])
    if cache_salt is not None:
        return ((extra_key, cache_salt), plain)
    return plain if extra_key is None else (extra_key, plain)


def _common_prefix(native_key, tokens, offset, maximum, page):
    """SGLang's RadixKey.match: galloping comparison of the tokens, rounded down to whole pages."""
    other = native_key.token_ids
    if type(other) is not type(tokens):
        raise ValueError('native radix token container type mismatch')
    n = min(len(native_key), maximum)
    lo = 0
    step = 1
    matched = n
    while lo < n:
        hi = min(n, lo+step)
        if other[lo:hi] != tokens[offset+lo:offset+hi]:
            while hi-lo > 1:
                mid = (lo+hi)//2
                if other[lo:mid] == tokens[offset+lo:offset+mid]:
                    lo = mid
                else:
                    hi = mid
            matched = lo
            break
        lo = hi
        step *= 2
    return matched//page*page


def _page_path(tokens, extra_key, cache_salt, page):
    parent = hashlib.sha256(repr((extra_key, cache_salt)).encode()).digest()
    result = []
    for start in range(0, len(tokens)//page*page, page):
        parent = hashlib.sha256(parent+array('q', tokens[start:start+page]).tobytes()).digest()
        result.append(parent)
    return tuple(result)


def _native_key_scalars(key):
    return (len(key.token_ids), len(key), getattr(key, 'limit', None),
            key.extra_key, getattr(key, 'cache_salt', None) or None,
            bool(getattr(key, 'is_bigram', False)))


def _native_key_dependency(node):
    """Identity of a node key plus every scalar that affects matching; a change of namespace, view or limit is
    detected even on the same key object."""
    key = node.key
    return _KeyDependency(key, key.token_ids, _native_key_scalars(key))


def _native_key_matches(node, dependency):
    key = node.key
    return (key is dependency.key and key.token_ids is dependency.tokens
            and _native_key_scalars(key) == dependency.scalars)


class MetaCache:
    def __init__(self, tree, page=128, *, clock=time.monotonic, strict=True,
                 max_entries=1024):
        if (page < 1 or int(page) != page or tree.page_size != page
                or getattr(tree, 'disable', False)
                or getattr(tree, 'is_eagle', False)
                or getattr(tree, 'enable_storage', False)
                or (callable(getattr(tree, 'supports_mamba', None)) and tree.supports_mamba())
                or (callable(getattr(tree, 'swa_reprefill_tail_tokens', None))
                    and tree.swa_reprefill_tail_tokens())):
            raise ValueError('unsupported CPU metadata cache configuration')
        if max_entries < 1:
            raise ValueError('metadata entry limit must be positive')
        if getattr(tree, '_b300_page_metadata_owner', None) is not None:
            raise ValueError('tree already has a metadata cache')
        if strict and any(not callable(getattr(tree, name, None)) for name in REQUIRED_MUTATIONS):
            raise ValueError('SGLang radix cache mutation API is incomplete')
        self.tree = tree
        self.page = int(page)
        self.clock = clock
        self.max_entries = int(max_entries)
        self.topology_epoch = self.lock_epoch = 0
        self._force_epoch = 0
        self._entries = OrderedDict()
        self._mutex = threading.RLock()
        self._depth = 0
        self._pending = set()
        self._originals = {}
        self.counters = Counter()
        tree._b300_page_metadata_owner = self
        for name in TOPOLOGY_MUTATIONS:
            self._install(name, 'topology')
        for name in LOCK_MUTATIONS:
            self._install(name, 'locks')

    @property
    def version(self):
        return self.topology_epoch, self.lock_epoch

    def _install(self, name, kind):
        original = getattr(self.tree, name, None)
        if not callable(original):
            return
        self._originals[name] = original

        @wraps(original)
        def wrapped(*args, **kwargs):
            with self._mutex:
                self._depth += 1
                succeeded = False
                try:
                    result = original(*args, **kwargs)
                    succeeded = True
                    return result
                finally:
                    # A failed native mutation may still have changed state.
                    self.counters['mutation_'+name] += 1
                    self._pending.add(kind)
                    if not succeeded:
                        self._pending.add('force')
                    self._depth -= 1
                    if not self._depth:
                        if 'topology' in self._pending:
                            self.topology_epoch += 1
                        if 'locks' in self._pending:
                            self.lock_epoch += 1
                        if 'force' in self._pending:
                            self._force_epoch += 1
                        self._pending.clear()
        setattr(self.tree, name, wrapped)

    def invalidate(self, *, topology=True, locks=False):
        """Invalidate all metadata after a cache mutation that the wrapped methods do not cover."""
        with self._mutex:
            if topology:
                self.topology_epoch += 1
                self._force_epoch += 1
            if locks:
                self.lock_epoch += 1

    def invalidate_request(self, request):
        with self._mutex:
            self._entries.pop(id(request), None)

    def prune(self, requests):
        with self._mutex:
            live = {id(req) for req in requests}
            for identity in tuple(self._entries):
                if identity not in live:
                    del self._entries[identity]

    def _touch_request_scalars(self, req):
        tokens = req.origin_input_ids
        held = getattr(req, 'req_pool_idx', None) is not None
        finished = getattr(req, 'finished', None)
        if (getattr(req, 'output_ids', ()) or (callable(finished) and finished())
                or getattr(req, 'return_logprob', False)
                or any(getattr(req, name, None) is not None for name in
                       ('multimodal_inputs', 'input_embeds', 'positional_embed_overrides'))):
            raise ValueError('touch plan only supports pending dense prefills')
        cap = getattr(getattr(req, 'sampling_params', None), 'max_new_tokens', None)
        return (len(tokens), getattr(req, 'extra_key', None),
                getattr(req, 'cache_salt', None) or None,
                getattr(req, 'req_pool_idx', None),
                len(req.prefix_indices) if held else 0,
                max(0, int(cap)) if cap is not None else None,
                getattr(req, '_b300_page_inflight', 0),
                getattr(req, 'inflight_middle_chunks', 0))

    @staticmethod
    def _touch_node_placement(node):
        # These are CPU properties/lengths only. Do not retain value tensors.
        value=getattr(node,'value',None)
        host=getattr(node,'host_value',None)
        return (bool(node.evicted), bool(node.backuped), int(node.lock_ref),
                len(value) if value is not None else None,
                len(host) if host is not None else None)

    @staticmethod
    def _prefix_reference(req, held):
        if not held:
            return None
        container=req.prefix_indices
        try:
            # torch.Tensor supports weak references: never hold its device
            # storage merely to validate a resource-only metadata lease.
            return weakref.ref(container)
        except TypeError:
            if isinstance(container,(array,list,tuple)):
                return lambda:container  # plain CPU containers
            raise ValueError('held prefix container has no safe identity guard')

    def capture_touch_plan(self, requests, *, expected=None, host_threshold=0):
        """Capture the traversal of the queue right after a complete lookup that proved the decode-only certificate.

        Any change of a cache epoch rejects it, and checks of placement and held slots catch the other changes."""
        with self._mutex:
            if expected is not None and len(expected)!=len(requests):
                raise ValueError('touch plan expected frontier length changed')
            captured = []
            nodes = {}
            edges = {}
            def remember_node(node):
                if node is not None and id(node) not in nodes:
                    nodes[id(node)] = _TouchNode(node, node.parent,
                        _native_key_dependency(node), self._touch_node_placement(node))
            remember_node(self.tree.root_node)
            for index,req in enumerate(requests):
                entry = self._entries.get(id(req))
                if entry is None or entry.request is not req:
                    raise ValueError('touch plan has no complete metadata lookup')
                scalars = self._touch_request_scalars(req)
                if entry.input_container is not req.origin_input_ids:
                    raise ValueError('touch plan input changed since lookup')
                held = scalars[3] is not None
                meta=entry.latest_metadata
                if (meta is None or meta.version!=self.version or meta.held!=held
                        or entry.input_identity[1:]!=scalars[:3]):
                    raise ValueError('touch plan metadata witness changed since lookup')
                if expected is not None:
                    actual=(meta.gpu_prefix,meta.host_prefix if meta.host_prefix>=host_threshold else 0,
                            scalars[5],held,meta.locks)
                    if actual!=expected[index]:
                        raise ValueError('touch plan no longer matches resource proof rows')
                touch = None
                if held:
                    if (entry.held_prefix_reference is None
                            or entry.held_prefix_reference() is not req.prefix_indices
                            or len(req.prefix_indices)!=meta.gpu_prefix
                            or req.last_node is not meta.last_device_anchor.node):
                        raise ValueError('held touch witness changed since lookup')
                    for saved in entry.held_nodes:
                        if (saved.node.parent is not saved.parent
                                or not _native_key_matches(saved.node,saved.key)
                                or self._touch_node_placement(saved.node)!=saved.placement):
                            raise ValueError('held placement witness changed since lookup')
                    node = req.last_node
                    while node is not None:
                        remember_node(node)
                        node = node.parent
                else:
                    if (entry.root is not self.tree.root_node
                            or entry.topology_epoch != self.topology_epoch
                            or entry.force_epoch != self._force_epoch
                            or entry.locks_version != self.version
                            or not self._dependencies_valid(entry)):
                        raise ValueError('touch plan lookup version is incomplete')
                    gpu=host=0;resident=[];saw_evicted=False
                    for segment in entry.segments:
                        node=segment.node
                        placement=self._touch_node_placement(node)
                        if node.lock_ref<0:
                            raise ValueError('touch plan native lock count is negative')
                        # Supported native resident/backup tensors cover the
                        # node key exactly; malformed truncation is a miss.
                        if ((not placement[0] and placement[3]!=len(node.key))
                                or (placement[1] and placement[4]!=len(node.key))):
                            raise ValueError('touch plan native placement length changed')
                        if node.evicted:
                            if not node.backuped:
                                raise ValueError('touch plan evicted prefix has no host backup')
                            saw_evicted=True
                            host+=segment.length
                        else:
                            if saw_evicted:
                                raise ValueError('touch plan GPU prefix has an evicted ancestor')
                            gpu+=segment.length;resident.append(segment)
                    signature=tuple((id(s.node),s.length,s.end,s.node.lock_ref) for s in resident)
                    if (gpu!=meta.gpu_prefix or host!=meta.host_prefix
                            or signature!=entry.locks_signature):
                        raise ValueError('touch plan placement witness changed since lookup')
                    if len(req.origin_input_ids)-1 >= self.page:
                        touch = entry.segments
                    for edge in entry.dependencies:
                        remember_node(edge.parent)
                        remember_node(edge.child)
                        edges.setdefault((id(edge.parent), edge.lookup_key), edge)
                    for segment in entry.segments:
                        remember_node(segment.node)
                captured.append(_TouchRequest(req, entry, req.origin_input_ids,
                    scalars, getattr(req, 'last_node', None),
                    getattr(req, 'last_host_node', None),
                    getattr(req, 'best_match_node', None),
                    self._prefix_reference(req,held), touch))
            self.counters['early_touch_plan_captures'] += 1
            return TouchPlan(self.tree.root_node,
                (*self.version, self._force_epoch), tuple(captured),
                tuple(nodes.values()), tuple(edges.values()))

    def _touch_plan_valid(self, plan, requests):
        if (plan is None or plan.root is not self.tree.root_node
                or plan.version != (*self.version, self._force_epoch)
                or len(requests) != len(plan.requests)):
            return False
        for req, old in zip(requests, plan.requests):
            if (req is not old.request
                    or self._entries.get(id(req)) is not old.entry
                    or req.origin_input_ids is not old.input_container
                    or self._touch_request_scalars(req) != old.scalars
                    or getattr(req, 'last_node', None) is not old.last_node
                    or getattr(req, 'last_host_node', None) is not old.last_host_node
                    or getattr(req, 'best_match_node', None) is not old.best_node
                    or (old.prefix_reference is not None
                        and old.prefix_reference() is not req.prefix_indices)):
                return False
        for saved in plan.nodes:
            node = saved.node
            if (node.parent is not saved.parent
                    or not _native_key_matches(node, saved.key)
                    or self._touch_node_placement(node) != saved.placement):
                return False
        for edge in plan.edges:
            if edge.parent.children.get(edge.lookup_key) is not edge.child:
                return False
        return True

    @contextmanager
    def early_touch_lease(self, plan, requests):
        """Hold the cache lock while both ranks check a captured plan, then replay its LRU touches.

        The check itself touches nothing. On a miss on either rank, the caller releases the lease and does a complete
        lookup."""
        with self._mutex:
            self.counters['early_touch_plan_checks'] += 1
            try:
                valid = self._touch_plan_valid(plan, requests)
            except Exception:
                valid = False
                self.counters['early_touch_plan_validation_errors'] += 1
            if not valid:
                self.counters['early_touch_plan_misses'] += 1
            yield _TouchLease(self, plan) if valid else None

    def _entry(self, req, path):
        tokens = req.origin_input_ids
        if not isinstance(tokens, (array, list, tuple)) or len(tokens) < 1:
            raise ValueError('invalid pending input token container')
        extra = getattr(req, 'extra_key', None)
        salt = getattr(req, 'cache_salt', None) or None
        identity = (id(tokens), len(tokens), extra, salt)
        entry = self._entries.get(id(req))
        if (entry is None or entry.request is not req
                or entry.input_container is not tokens or entry.input_identity != identity):
            if path is None:
                path = _page_path(tokens, extra, salt, self.page)
            elif (not isinstance(path, tuple) or len(path) != len(tokens)//self.page
                  or any(not isinstance(key, bytes) or len(key) != 32 for key in path)):
                raise ValueError('supplied canonical page path has invalid geometry')
            entry = _Entry(req, tokens, identity, path)
            self._entries[id(req)] = entry
            self.counters['new_inputs'] += 1
        elif path is not None and path is not entry.path and path != entry.path:
            raise ValueError('canonical page path changed for an immutable input')
        self._entries.move_to_end(id(req))
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
        return entry

    def _dependencies_valid(self, entry):
        self.counters['dependency_validations'] += 1
        try:
            if entry.root is not self.tree.root_node:
                return False
            for edge in entry.dependencies:
                if (not _native_key_matches(edge.parent, edge.parent_key)
                        or edge.parent.children.get(edge.lookup_key) is not edge.child):
                    return False
                if edge.child is not None:
                    if (edge.child.parent is not edge.parent
                            or not _native_key_matches(edge.child, edge.child_key)):
                        return False
            return True
        except Exception:
            # Unsupported geometry never authorizes a stale metadata hit.
            self.counters['dependency_validation_fallbacks'] += 1
            return False

    def _scan(self, req, entry):
        tokens = req.origin_input_ids
        extra = getattr(req, 'extra_key', None)
        salt = getattr(req, 'cache_salt', None) or None
        maximum = max(0, len(tokens)-1)//self.page*self.page
        node = self.tree.root_node
        offset = 0
        segments = []
        dependencies = []
        while offset < maximum:
            key = _namespace_key(tokens, offset, self.page, extra, salt)
            child = node.children.get(key)
            if child is None:
                dependencies.append(_Edge(node, key, None,
                    _native_key_dependency(node), None, offset, 0))
                break
            if child.parent is not node:
                raise ValueError('native radix route has an inconsistent parent')
            native_key = child.key
            if (getattr(native_key, 'is_bigram', False)
                    or native_key.extra_key != extra
                    or (getattr(native_key, 'cache_salt', None) or None) != salt):
                raise ValueError('native radix namespace or layout mismatch')
            matched = _common_prefix(native_key, tokens, offset, maximum-offset, self.page)
            if matched <= 0 or len(native_key) % self.page:
                raise ValueError('native radix child geometry is not page aligned')
            dependencies.append(_Edge(node, key, child,
                _native_key_dependency(node), _native_key_dependency(child), offset, matched))
            offset += matched
            segments.append(_Segment(child, matched, offset))
            if matched < len(native_key):
                break
            node = child
        self.counters['radix_scans'] += 1
        entry.root = self.tree.root_node
        entry.dependencies = tuple(dependencies)
        return tuple(segments)

    def _touch(self, segments):
        self.tree.root_node.last_access_time = self.clock()
        for segment in segments:
            segment.node.last_access_time = self.clock()

    def _locks(self, segments, path):
        result = []
        for segment in segments:
            if segment.node.lock_ref == 0:
                if segment.end % self.page or segment.end//self.page > len(path):
                    raise ValueError('prefix lock exceeds canonical page path')
                # Page identities give the exact protected GPU union even when
                # requests end inside the same physical radix node, or TP ranks
                # temporarily have different physical split boundaries.
                start = (segment.end-segment.length)//self.page
                result.extend((path[i], self.page) for i in range(start, segment.end//self.page))
        return tuple(result)

    def _held(self, req, entry):
        # len(tensor) is scalar metadata; never copy/store/inspect its values.
        gpu = len(req.prefix_indices)
        if gpu < 0 or gpu >= len(req.origin_input_ids):
            raise ValueError('held request violates positive prefill convention')
        nodes = []
        node = req.last_node
        while node is not None and node.parent is not None:
            nodes.append(node)
            node = node.parent
        end = 0
        segments = []
        for node in reversed(nodes):
            end += len(node.key)
            if node.evicted or end > gpu:
                raise ValueError('held prefix metadata no longer has resident ancestors')
            segments.append(_Segment(node, len(node.key), end))
        anchor = Anchor(req.last_node if req.last_node is not None else self.tree.root_node,
                        len(req.last_node.key) if req.last_node is not None else 0)
        host_node = getattr(req, 'last_host_node', None) or self.tree.root_node
        host = Anchor(host_node, len(host_node.key))
        self.counters['held_lookups'] += 1
        return Metadata(gpu, 0, self.version, self._locks(segments, entry.path), entry.path,
                        anchor, host, host, True)

    def lookup(self, req, held=False, *, path=None):
        """Prefix lengths and locks of one request. A scheduled request is still matched again through SGLang.

        ``path`` may pass a page path computed before, to avoid hashing the prompt twice. Token comparison, not the
        path, determines the cache hits."""
        with self._mutex:
            if (getattr(req, 'output_ids', ()) or getattr(req, 'return_logprob', False)
                    or any(getattr(req, name, None) is not None for name in
                           ('multimodal_inputs', 'input_embeds', 'positional_embed_overrides'))):
                raise ValueError('metadata lookup only supports pending dense prefills')
            entry = self._entry(req, path)
            if held:
                # Live partials change without a radix mutation; do not memoize.
                meta=self._held(req, entry)
                nodes=[];node=req.last_node
                while node is not None:
                    nodes.append(_TouchNode(node,node.parent,_native_key_dependency(node),
                        self._touch_node_placement(node)))
                    node=node.parent
                entry.held_nodes=tuple(nodes)
                entry.held_prefix_reference=self._prefix_reference(req,True)
                entry.latest_metadata=meta
                return meta
            if (entry.force_epoch != self._force_epoch
                    or (entry.topology_epoch != self.topology_epoch
                        and not self._dependencies_valid(entry))):
                entry.segments = self._scan(req, entry)
                entry.topology_epoch = self.topology_epoch
                entry.force_epoch = self._force_epoch
                self.counters['misses'] += 1
            else:
                if entry.topology_epoch != self.topology_epoch:
                    self.counters['unrelated_epoch_reuses'] += 1
                    entry.topology_epoch = self.topology_epoch
                self.counters['hits'] += 1
            segments = entry.segments
            if len(req.origin_input_ids)-1 >= self.page:
                self._touch(segments)
            root = Anchor(self.tree.root_node)
            gpu = host = 0
            resident = []
            last_device = root
            last_host = root
            saw_evicted = False
            for segment in segments:
                node = segment.node
                if node.lock_ref < 0:
                    raise ValueError('native radix node has negative lock count')
                anchor = Anchor(node, segment.length, segment.length < len(node.key))
                if node.backuped:
                    last_host = anchor
                if node.evicted:
                    if not node.backuped:
                        raise ValueError('evicted matched prefix has no host backup')
                    saw_evicted = True
                    host += segment.length
                else:
                    if saw_evicted:
                        raise ValueError('GPU prefix has an evicted ancestor')
                    gpu += segment.length
                    resident.append(segment)
                    last_device = anchor
            if gpu+host >= len(req.origin_input_ids):
                raise ValueError('metadata violates native positive prefill convention')
            # GPU/host placement and lock counts are read fresh above, even
            # with an unchanged epoch. Only the actual resident path's lock
            # dependency can authorize reuse of its exact canonical page union.
            lock_signature = tuple((id(segment.node), segment.length,
                segment.end, segment.node.lock_ref) for segment in resident)
            if entry.locks_signature != lock_signature:
                entry.locks = self._locks(resident, entry.path)
                entry.locks_signature = lock_signature
                self.counters['lock_rebuilds'] += 1
            entry.locks_version = self.version
            meta=Metadata(gpu, host, self.version, entry.locks,
                          entry.path, last_device, last_host, last_host)
            entry.latest_metadata=meta
            return meta
