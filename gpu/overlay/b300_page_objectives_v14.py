"""The planner of Caesura: its objective, resource model and step-time model, and its search.

For one step, the planner chooses how many prompt tokens of each waiting request to prefill. It scores a schedule by
the objective at the end of the step: the requests that can still meet their deadline minus those that complete their
prefill on time (K), plus those that miss it (M), plus a quarter of the remaining prefill work (Phi), in which each
request counts its uncomputed share of the prompt, weighted by its urgency. Remaining work accounts for the prompt pages
that the scheduled requests publish to the prefix cache, which other waiting requests then reuse.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import re
import threading
import time

PAGE = 128


def paged(n):
    return ((int(n) + PAGE - 1) // PAGE) * PAGE


@dataclass(frozen=True)
class Row:
    index: int
    age_s: float
    total: int
    gpu_prefix: int
    host_prefix: int = 0
    output_limit: int = 0
    held_slot: bool = False
    locks: tuple = ()             # (cache node, tokens) of matched prefix nodes that the request would lock
    path: tuple = ()              # ids of the prompt pages
    future_output_guard: int | None = None  # KV tokens reserved for the output on completion, with one page

    @property
    def remaining(self):
        return self.total - self.gpu_prefix

    @property
    def query(self):
        return self.total - self.gpu_prefix - self.host_prefix


@dataclass
class Limits:
    token_budget: int
    kv_budget: int
    slots: int
    batch_slots: int
    decode_count: int = 0
    decode_context: int = 0
    time_cap_s: float = .75


@dataclass
class CostModel:
    # Initial coefficients from the B300 calibration. observe() adapts the scales and the decode time online.
    intercept_s: float = .24945264601867195
    query_token_s: float = 2.6016077165803982e-6
    query_context_billion_s: float = .14116817468998816
    decode_s: float = .025
    transfer_token_s: float = 1e-6
    mixed_scale: float = 1.
    pure_scale: float = 1.
    launch_s: float = .004
    observations: int = 0

    def features(self, rows, action, limits):
        q = attention = host = 0
        for row in rows:
            n = action.get(row.index, 0)
            if n:
                q += n
                c = row.gpu_prefix + row.host_prefix
                attention += n * (c + (n + 1) / 2)
                host += row.host_prefix
        return q, attention, host

    def predict(self, rows, action, limits):
        q, attention, host = self.features(rows, action, limits)
        if not q:
            return self.decode_s + self.launch_s if limits.decode_count else 0.
        base = self.intercept_s + self.query_token_s*q + self.query_context_billion_s*attention/1e9
        scale = self.mixed_scale if limits.decode_count else self.pure_scale
        return max(base * scale, self.decode_s if limits.decode_count else 0.) + host*self.transfer_token_s + self.launch_s

    def observe(self, features, decode_count, cuda_s):
        q, attention, _ = features
        if not math.isfinite(cuda_s) or not 0 < cuda_s < 30:
            raise ValueError('invalid real CUDA timing observation')
        if q:
            base = self.intercept_s + self.query_token_s*q + self.query_context_billion_s*attention/1e9
            attr = 'mixed_scale' if decode_count else 'pure_scale'
            ratio = min(8., max(.1, cuda_s/base))
            setattr(self, attr, .9*getattr(self, attr)+.1*ratio)
        else:
            self.decode_s = .9*self.decode_s + .1*cuda_s
        self.observations += 1


def remaining_after(rows, action, check=None):
    """Remaining prefill tokens of every row after ACTION, and the rows ACTION completes.

    A row gains the prefix cache hits of the pages that the scheduled rows publish. A completed prompt still leaves
    one token to compute, as in SGLang."""
    published = {}
    for position, row in enumerate(rows):
        if check is not None and position % 32 == 0:
            check()
        q = action.get(row.index, 0)
        if not q:
            continue
        prefix = row.gpu_prefix + row.host_prefix + q
        for at, key in enumerate(row.path[:prefix//PAGE]):
            if check is not None and at % 64 == 0:
                check()
            published[key] = True
    result = {}
    completed = set()
    for position, row in enumerate(rows):
        if check is not None and position % 32 == 0:
            check()
        q = action.get(row.index, 0)
        prefix = row.gpu_prefix
        if q:
            prefix += row.host_prefix + q
            if q == row.query:
                completed.add(row.index)
        if row.path:
            lo, hi = 0, len(row.path)
            while lo < hi:
                mid = (lo+hi+1)//2
                if row.path[mid-1] in published:
                    lo = mid
                else:
                    hi = mid-1
            match = lo*PAGE
            prefix = max(prefix, min(match, ((row.total-1)//PAGE)*PAGE))
        result[row.index] = 0 if row.index in completed else max(1, row.total-prefix)
    return result, completed


@dataclass(frozen=True)
class _CompiledPath:
    path: tuple
    nodes: tuple
    runs: tuple


class _RegistryFull(Exception):
    pass


class _RegistryNoncanonical(Exception):
    pass


@dataclass(frozen=True)
class _ActiveGroup:
    token: object
    compiled: _CompiledPath
    cap: int
    runs: tuple


@dataclass(frozen=True)
class _ActiveOwnerSnapshot:
    namespace: object
    signature: frozenset
    groups: tuple


class StaticOwners:
    """Owner groups of compiled paths: for each (path, page cap), the contiguous node-id runs of its pages.

    It holds no residency, request, age or score, so it is shared across planning rounds."""
    def __init__(self, max_entries=1200000):
        self.max_entries = max_entries
        self.mutex = threading.RLock()
        self.groups, self.pending, self.owners = {}, {}, {}
        self.entries = 0
        self.reserved = 0
        self.masks = OrderedDict()
        self._active_namespace = object()
        self.active_vectors = OrderedDict()
        self.historical_run_entries = 0
        self.active_stats = dict(active_match_calls=0, active_match_hits=0,
            active_group_visits=0, historical_groups_available=0,
            active_run_visits=0, active_binary_probes=0)

    def register(self, compiled, cap, check):
        cap = min(cap, len(compiled.nodes))
        key = id(compiled), cap
        with self.mutex:
            if key in self.groups:
                return self.groups[key][1], True, 0
            if self.reserved + cap > self.max_entries and key not in self.pending:
                raise _RegistryFull
            entry = self.pending.get(key)
            if entry is None:
                entry = [compiled, object(), 0, [], 0]
                self.pending[key] = entry
                self.reserved += cap
            before = entry[4]
            while entry[2] < len(compiled.runs):
                if check is not None and entry[2] % 32 == 0:
                    check()
                depth, length, first = compiled.runs[entry[2]]
                if depth >= cap:
                    break
                length = min(length, cap-depth)
                entry[3].append((depth, length, first, (1 << length)-1))
                self.entries += length
                entry[4] += length
                entry[2] += 1
            token = entry[1]
            self.owners[token] = tuple(entry[3])
            self.historical_run_entries += len(entry[3])
            self.groups[key] = compiled, token
            del self.pending[key]
            return token, False, cap-before

    def prefix(self, compiled, count, check):
        key = id(compiled), count
        with self.mutex:
            found = self.masks.get(key)
            if found is not None and found[0] is compiled:
                self.masks.move_to_end(key)
                return found[1], True
            mask = 0
            for at, (depth, length, first) in enumerate(compiled.runs):
                if check is not None and at % 32 == 0:
                    check()
                if depth >= count:
                    break
                mask |= ((1 << min(length, count-depth))-1) << first
            self.masks[key] = compiled, mask
            if len(self.masks) > 512:
                self.masks.popitem(last=False)
            return mask, False

    def active_snapshot(self, compiled_paths, caps, check=None):
        """The owner groups of the current rows, as a key for cached prefix matches."""
        if len(compiled_paths) != len(caps):
            raise ValueError('active group geometry length mismatch')
        with self.mutex:
            groups = {}
            for position, (compiled, cap) in enumerate(zip(compiled_paths, caps)):
                if check is not None and position % 16 == 0:
                    check()
                cap = min(cap, len(compiled.nodes))
                if cap < 0:
                    raise ValueError('negative APC cap')
                found = self.groups.get((id(compiled), cap))
                if found is None or found[0] is not compiled:
                    raise ValueError('unregistered active group')
                token = found[1]
                groups.setdefault(token, _ActiveGroup(token, compiled, cap, self.owners[token]))
            return _ActiveOwnerSnapshot(self._active_namespace,
                frozenset(groups), tuple(groups.values()))

    def matches_active(self, snapshot, published, check):
        """Prefix-match depth of every owner group after the pages in PUBLISHED become cached.

        Published pages are unions of whole prefixes, so the match depth of a group is the deepest published page on its
        path: short run lists take the highest set bit, and long ones a binary search along the path."""
        if snapshot.namespace is not self._active_namespace:
            raise ValueError('foreign active owner namespace')
        if not isinstance(published, int) or published < 0:
            raise ValueError('invalid published mask')
        key = snapshot.signature, published
        with self.mutex:
            self.active_stats['active_match_calls'] += 1
            self.active_stats['historical_groups_available'] = len(self.owners)
            cached = self.active_vectors.get(key)
            if cached is not None:
                self.active_vectors.move_to_end(key)
                self.active_stats['active_match_hits'] += 1
                return cached[1], True, 0, 0, 0
            values, visits, probes, operations = {}, 0, 0, 0
            encoded = None
            for position, group in enumerate(snapshot.groups):
                if check is not None and position % 16 == 0:
                    check()
                visits += 1
                depth = 0
                if len(group.runs) <= max(1, group.cap.bit_length()):
                    for at, _, first, low_mask in group.runs:
                        if check is not None and operations % 16 == 0:
                            check()
                        operations += 1
                        bits = (published >> first) & low_mask
                        if bits:
                            depth = max(depth, (at + bits.bit_length()) * PAGE)
                    self.active_stats['active_run_visits'] += len(group.runs)
                else:
                    if encoded is None:
                        encoded = published.to_bytes((published.bit_length()+7)//8, 'little')
                    lo, hi = 0, group.cap
                    while lo < hi:
                        if check is not None and operations % 16 == 0:
                            check()
                        operations += 1
                        probes += 1
                        mid = (lo + hi + 1) // 2
                        node = group.compiled.nodes[mid - 1]
                        byte, bit = divmod(node, 8)
                        if byte < len(encoded) and encoded[byte] & (1 << bit):
                            lo = mid
                        else:
                            hi = mid - 1
                    depth = lo * PAGE
                if depth:
                    values[group.token] = depth
            self.active_vectors[key] = snapshot, values
            if len(self.active_vectors) > 12:
                self.active_vectors.popitem(last=False)
            self.active_stats['active_group_visits'] += visits
            self.active_stats['active_binary_probes'] += probes
            return values, False, published.bit_count(), visits, probes


class PathRegistry:
    """Compiled page paths: each page id gets a node id, and each path its node ids as contiguous runs.

    The registry is bounded and resets when full; a planning round never mixes node ids from before and after a
    reset."""
    def __init__(self, max_nodes=131072, max_paths=1024):
        self.max_nodes, self.max_paths = max_nodes, max_paths
        self._mutex = threading.RLock()
        self.stats = dict(prefix_registry_alias_hits=0, prefix_registry_content_hits=0,
            prefix_registry_path_compiles=0, prefix_registry_capacity_resets=0,
            prefix_registry_noncanonical_roots=0)
        self.reset()

    def reset(self):
        with self._mutex:
            self._ids, self._parents, self._paths, self._aliases = {}, {}, {}, OrderedDict()
            self._pending_paths = {}
            self._node_prefixes = {}
            self.layout_token = object()
            self.static_owners = StaticOwners()
            self._retired_owners = []

    def _one(self, path, check=None):
        alias = self._aliases.get(id(path))
        if alias is not None and alias[0] is path:
            self._aliases.move_to_end(id(path))
            self.stats['prefix_registry_alias_hits'] += 1
            return alias[1]
        endpoint = len(path), path[-1] if path else None
        compiled = self._paths.get(endpoint)
        if compiled is not None:
            if compiled.path != path:
                raise _RegistryNoncanonical
            self.stats['prefix_registry_content_hits'] += 1
        else:
            if len(self._paths) >= self.max_paths:
                raise _RegistryFull
            pending = self._pending_paths.get(endpoint)
            if pending is not None and pending[0] != path:
                raise _RegistryNoncanonical
            if pending is None:
                nodes, runs, parent = [], [], None
                # Reuse the compiled prefix of a nearby ancestor page, so that a path compiles only its new
                # suffix.
                probe = range(len(path)-1, max(-1, len(path)-65), -1) if self._node_prefixes else ()
                for at in probe:
                    if check is not None and (len(path)-1-at) % 64 == 0:
                        check()
                    prior = self._node_prefixes.get(path[at])
                    if prior is None:
                        continue
                    old, depth = prior
                    if depth != at+1 or path[:depth] != old.path[:depth]:
                        raise _RegistryNoncanonical
                    nodes = list(old.nodes[:depth])
                    for start, length, first in old.runs:
                        if start >= depth:
                            break
                        runs.append((start, min(length, depth-start), first))
                    parent = nodes[-1]
                    break
                pending = [path, nodes, runs, parent, [], 0]
                self._pending_paths[endpoint] = pending
            nodes, runs, parent = pending[1:4]
            for depth in range(len(nodes), len(path)):
                if check is not None and depth % 64 == 0:
                    check()
                key = path[depth]
                node = self._ids.get(key)
                if node is None:
                    if len(self._ids) >= self.max_nodes:
                        raise _RegistryFull
                    node = len(self._ids)
                    self._ids[key] = node
                ancestry = parent, depth
                if node in self._parents and self._parents[node] != ancestry:
                    raise _RegistryNoncanonical
                self._parents[node] = ancestry
                parent = node
                nodes.append(node)
                pending[3] = parent
                if key not in self._node_prefixes:
                    pending[4].append((key, depth+1))
                if runs and node == runs[-1][2]+runs[-1][1]:
                    start, length, first = runs[-1]
                    runs[-1] = start, length+1, first
                else:
                    runs.append((depth, 1, node))
            compiled = _CompiledPath(path, tuple(nodes), tuple(runs))
            while pending[5] < len(pending[4]):
                if check is not None and pending[5] % 64 == 0:
                    check()
                key, depth = pending[4][pending[5]]
                self._node_prefixes.setdefault(key, (compiled, depth))
                pending[5] += 1
            self._paths[endpoint] = compiled
            del self._pending_paths[endpoint]
            self.stats['prefix_registry_path_compiles'] += 1
        self._aliases[id(path)] = path, compiled
        if len(self._aliases) > self.max_paths:
            self._aliases.popitem(last=False)
        return compiled

    def compile(self, paths, check=None):
        with self._mutex:
            before = dict(self.stats)
            compiled = None
            for attempt in range(2):
                try:
                    result = []
                    for path in paths:
                        if check is not None:
                            check()
                        result.append(self._one(path, check))
                    compiled = tuple(result)
                    break
                except _RegistryFull:
                    self.stats['prefix_registry_capacity_resets'] += 1
                    self.reset()
                except _RegistryNoncanonical:
                    self.stats['prefix_registry_noncanonical_roots'] += 1
                    self.reset()
                    break
            return compiled, {key: value-before[key] for key,value in self.stats.items()}

    def compile_layout(self, paths, check=None):
        with self._mutex:
            compiled, stats = self.compile(paths, check)
            stats['prefix_registry_layout_nodes'] = len(self._ids)
            return compiled, stats, self.layout_token, self.static_owners

    def reset_owners(self, prior):
        with self._mutex:
            if self.static_owners is prior:
                self._retired_owners[:] = [prior]
                self.static_owners = StaticOwners(prior.max_entries)


_PATH_REGISTRY = PathRegistry()


_NONZERO_BYTES = re.compile(rb'[^\x00]')


def _delta_nodes(delta):
    """The set bits of DELTA, low to high."""
    if delta.bit_length() <= 4096:
        while delta:
            bit = delta & -delta
            yield bit.bit_length()-1
            delta ^= bit
        return
    offset = 0
    for _ in range(8):
        if not delta:
            return
        skip = (delta & -delta).bit_length()-1
        shifted = delta >> skip
        length = (shifted ^ (shifted+1)).bit_length()-1
        yield from range(offset+skip, offset+skip+length)
        offset += skip+length
        delta = shifted >> length
    if not delta:
        return
    encoded = delta.to_bytes((delta.bit_length()+7)//8, 'little')
    for match in _NONZERO_BYTES.finditer(encoded):
        byte, position = match.group()[0], offset+match.start()*8
        while byte:
            bit = byte & -byte
            yield position+bit.bit_length()-1
            byte ^= bit


class PrefixMatchVectors:
    """Prefix-cache hits of every row of one planning round after a candidate schedule publishes its pages.

    Page ids form a tree, and a schedule publishes whole prefixes, so only the rows that own a newly published page can
    gain a hit. When the paths do not fit the bounded registry, it falls back to remaining_after(), with the same
    result."""
    def __init__(self, rows, check=None, clock=time.perf_counter):
        self.rows = tuple(rows)
        self.check, self.clock = check, clock
        self.snapshot_version = object()
        self._positions = {r.index: i for i, r in enumerate(self.rows)}
        self._base_hits = tuple(r.gpu_prefix for r in self.rows)
        self._vectors = OrderedDict({(self.snapshot_version, 0): self._base_hits})
        self._prefix_masks = OrderedDict()
        self._runs = self._owners = None
        self._canonical = True
        self._legacy_backend = False
        self._static = None
        self.stats = dict(prefix_vector_calls=0, prefix_vector_cache_hits=0,
            prefix_vector_cache_misses=0, prefix_vector_delta_pages=0,
            prefix_vector_owner_visits=0, prefix_vector_owner_updates=0,
            prefix_vector_legacy_fallbacks=0, prefix_mask_cache_hits=0,
            prefix_mask_cache_misses=0, prefix_index_nodes=0,
            prefix_index_row_pages=0, prefix_index_build_s=0.,
            prefix_backend_legacy_roots=0, prefix_backend_legacy_calls=0,
            prefix_backend_cold_paths=0, prefix_backend_cold_pages_upper=0,
            prefix_backend_probe_s=0., static_owner_hits=0, static_owner_misses=0,
            static_owner_entries_added=0, cross_root_match_hits=0,
            cross_root_match_misses=0, index_budget_aborts=0,
            index_setup_s=0.)
        self.stats['prefix_binary_probes'] = 0
        self.stats['static_owner_capacity_resets'] = 0
        self.stats['static_owner_root_capacity_fallbacks'] = 0
        self.stats['active_match_active_groups'] = 0
        self.stats['active_match_historical_groups'] = 0
        self.stats['active_match_historical_runs'] = 0
        self.stats['prefix_backend_disjoint_nodes_lower'] = 0
        self.stats['prefix_backend_capacity_roots'] = 0

    def _build(self):
        began = self.clock()
        try:
            paths = tuple(r.path for r in self.rows)
            # Paths with distinct first pages are disjoint trees; if even their longest paths exceed the registry,
            # use remaining_after().
            lengths = {}
            for position, path in enumerate(paths):
                if self.check is not None and position % 32 == 0:
                    self.check()
                if path:
                    lengths[path[0]] = max(lengths.get(path[0], 0), len(path))
            lower = sum(lengths.values())
            self.stats['prefix_backend_disjoint_nodes_lower'] = lower
            if lower > _PATH_REGISTRY.max_nodes:
                self._legacy_backend = True
                self._runs, self._owners, self._owner_groups = (), {}, {}
                self.stats['prefix_backend_legacy_roots'] = 1
                self.stats['prefix_backend_capacity_roots'] = 1
                return
            compiled, registry_stats, layout, static = _PATH_REGISTRY.compile_layout(paths, self.check)
            self.stats.update(registry_stats)
            if compiled is None:
                self._canonical = False
                self._runs, self._owners, self._owner_groups = (), {}, {}
                return
            need = sum(cap for (_, cap) in {
                (id(path), min((row.total-1)//PAGE, len(path.nodes)))
                for path, row in zip(compiled, self.rows)})
            if need > static.max_entries:
                self._legacy_backend = True
                self._runs, self._owners, self._owner_groups = (), {}, {}
                self.stats['prefix_backend_legacy_roots'] = 1
                self.stats['static_owner_root_capacity_fallbacks'] = 1
                return
            groups, row_groups = {}, []
            for position, row in enumerate(self.rows):
                if self.check is not None:
                    self.check()
                group, hit, added = static.register(compiled[position], (row.total-1)//PAGE, self.check)
                groups.setdefault(group, []).append(position)
                row_groups.append(group)
                self.stats['static_owner_hits'] += hit
                self.stats['static_owner_misses'] += not hit
                self.stats['static_owner_entries_added'] += added
            self._compiled, self._layout, self._static = compiled, layout, static
            self._active_snapshot = static.active_snapshot(compiled,
                tuple((row.total - 1) // PAGE for row in self.rows), self.check)
            self._row_groups = tuple(row_groups)
            self._runs = tuple(path.runs for path in compiled)
            self._owners, self._owner_groups = static.owners, groups
            self.stats['prefix_index_row_pages'] = sum(len(row.path) for row in self.rows)
            self.stats['prefix_index_nodes'] = registry_stats['prefix_registry_layout_nodes']
            self.stats['prefix_index_owner_groups'] = len(groups)
            self.stats['active_match_active_groups'] = len(self._active_snapshot.groups)
            self.stats['active_match_historical_groups'] = len(static.owners)
            self.stats['active_match_historical_runs'] = static.historical_run_entries
        except _RegistryFull:
            self._legacy_backend = True
            self._runs, self._owners, self._owner_groups = (), {}, {}
            self.stats['prefix_backend_legacy_roots'] = 1
            _PATH_REGISTRY.reset_owners(static)
            self.stats['static_owner_capacity_resets'] = 1
        except TimeoutError:
            self.stats['index_budget_aborts'] += 1
            raise
        finally:
            duration = self.clock()-began
            self.stats['prefix_index_build_s'] += duration
            self.stats['index_setup_s'] += duration

    def _prefix(self, position, count):
        mask, hit = self._static.prefix(self._compiled[position], count, self.check)
        self.stats['prefix_mask_cache_hits' if hit else 'prefix_mask_cache_misses'] += 1
        return mask

    def _matches(self, published):
        key = self.snapshot_version, published
        found = self._vectors.get(key)
        if found is not None:
            self._vectors.move_to_end(key)
            self.stats['prefix_vector_cache_hits'] += 1
            return found
        self.stats['prefix_vector_cache_misses'] += 1
        raw, hit, pages, visits, probes = self._static.matches_active(self._active_snapshot, published, self.check)
        self.stats['cross_root_match_hits' if hit else 'cross_root_match_misses'] += 1
        self.stats['prefix_vector_delta_pages'] += pages
        self.stats['prefix_vector_owner_visits'] += visits
        self.stats['prefix_binary_probes'] += probes
        values = []
        for position, group in enumerate(self._row_groups):
            if self.check is not None and position % 32 == 0:
                self.check()
            depth = max(self._base_hits[position], raw.get(group, 0))
            self.stats['prefix_vector_owner_updates'] += depth > self._base_hits[position]
            values.append(depth)
        found = tuple(values)
        self._vectors[key] = found
        if len(self._vectors) > 12:
            self._vectors.popitem(last=False)
        return found

    def remaining(self, action, legacy=False):
        self.stats['prefix_vector_calls'] += 1
        if legacy:
            return (remaining_after(self.rows, action) if action else
                    ({r.index: r.remaining for r in self.rows}, set()))
        selected = [(self._positions[index], q) for index, q in action.items()
                    if index in self._positions and q]
        if any(q < 0 for _, q in selected):
            self.stats['prefix_vector_legacy_fallbacks'] += 1
            return remaining_after(self.rows, action, self.check)
        if selected and self._runs is None:
            self._build()
        if self._legacy_backend:
            self.stats['prefix_backend_legacy_calls'] += 1
            return remaining_after(self.rows, action, self.check)
        if not self._canonical:
            self.stats['prefix_vector_legacy_fallbacks'] += 1
            return remaining_after(self.rows, action, self.check)
        published = 0
        for position, q in selected:
            row = self.rows[position]
            published |= self._prefix(position, (row.gpu_prefix+row.host_prefix+q)//PAGE)
        hits = self._matches(published)
        result, completed = {}, set()
        for position, row in enumerate(self.rows):
            if self.check is not None and position % 32 == 0:
                self.check()
            q = action.get(row.index, 0)
            prefix = hits[position]
            if q:
                prefix = max(prefix, row.gpu_prefix+row.host_prefix+q)
                if q == row.query:
                    completed.add(row.index)
            result[row.index] = 0 if row.index in completed else max(1, row.total-prefix)
        return result, completed


class RootObjective:
    """The objective of one planning round. The rows that can still meet the deadline, the urgency weights and the
    normalizers of the remaining work are fixed when the round starts."""
    def __init__(self, rows, check=None, clock=time.perf_counter):
        self.rows = tuple(rows)
        self.eligible = tuple(r for r in self.rows if r.age_s <= 3.)
        # Urgency: the inverse of the time left to the deadline, at most 1 / 0.05 s; 1 / 3 s once it has passed.
        self.weights = tuple(1./max(3.-r.age_s, .05) if r.age_s <= 3. else 1./3. for r in self.rows)
        self.denominator = math.fsum(self.weights)
        self.normalizers = tuple(max(1, r.remaining) for r in self.rows)
        self.check = check
        self.prefix_vectors = PrefixMatchVectors(self.rows, check, clock)

    def marginal_weights(self):
        """Objective weight of one remaining prefill token of each row, which orders the urgency fill."""
        return {r.index: .25*w/(n*self.denominator)
                for r, w, n in zip(self.rows, self.weights, self.normalizers)} if self.denominator else {}

    def potential(self, action, elapsed_s, legacy=False):
        """Objective of ACTION when the step ends ELAPSED_S from now: |E| - K + M + Phi / 4."""
        after, completed = self.prefix_vectors.remaining(action, legacy=legacy)
        def checked(values):
            for position, value in enumerate(values):
                if not legacy and self.check is not None and position % 32 == 0:
                    self.check()
                yield value
        ontime = {r.index for r in self.eligible
                  if r.index in completed and r.age_s+elapsed_s <= 3.}
        misses = sum(r.age_s+elapsed_s > 3. and r.index not in ontime for r in self.eligible)
        fraction = math.fsum(checked(w*after[r.index]/n for r, w, n in
                             zip(self.rows, self.weights, self.normalizers)))/self.denominator if self.denominator else 0.
        return len(self.eligible)-len(ontime)+misses+.25*fraction


def completion_guard(row):
    """KV tokens reserved for the output of a request that completes its prefill in this step.

    SGLang derives it from the output cap and its own clip, never from the unknown output length; it has one extra page.
    Without it, the reserve is the full output limit plus one page."""
    guard = row.future_output_guard
    if guard is None:
        return paged(row.output_limit) + PAGE
    if type(guard) is not int or guard < PAGE or guard % PAGE:
        raise ValueError('invalid root-frozen completion guard')
    return guard


def usage(rows, action):
    tokens = kv = new_slots = count = 0
    locked = set()
    for row in rows:
        q = action.get(row.index, 0)
        if not q:
            continue
        if not 0 < q <= row.query or (q != row.query and q % PAGE):
            return None
        count += 1
        tokens += paged(q)
        kv += paged(q) + row.host_prefix + PAGE
        if q == row.query:
            kv += completion_guard(row)
        new_slots += not row.held_slot
        for node, cost in row.locks:
            if node not in locked:
                locked.add(node)
                kv += cost
    return tokens, kv, new_slots, count


def feasible(rows, action, limits, model):
    if not action:
        return limits.decode_count > 0
    consumed = usage(rows, action)
    if consumed is None:
        return False
    tok, kv, slots, count = consumed
    return (tok <= limits.token_budget and kv < limits.kv_budget
            and slots <= limits.slots and count <= limits.batch_slots
            and model.predict(rows, action, limits) <= limits.time_cap_s)


def resource_feasible(rows, action, limits):
    """Resource feasibility only, without the step-time cap."""
    if not action:
        return limits.decode_count > 0
    consumed = usage(rows, action)
    if consumed is None:
        return False
    tok, kv, slots, count = consumed
    return (tok <= limits.token_budget and kv < limits.kv_budget
            and slots <= limits.slots and count <= limits.batch_slots)


def _validate_resource_rows(rows):
    # Each lock node has one size; negative or inconsistent sizes would break the decode-only certificate.
    costs = {}
    for row in rows:
        if row.query < 1 or row.gpu_prefix < 0 or row.host_prefix < 0 or row.output_limit < 0:
            raise ValueError('invalid resource frontier')
        completion_guard(row)
        for node, cost in row.locks:
            if cost < 0 or (node in costs and costs[node] != cost):
                raise ValueError('inconsistent root lock cost')
            costs[node] = cost


def forced_current_action(rows, limits, check=None):
    """The decode-only certificate: proves that no prefill fits in this step, without the step-time model.

    It checks every waiting request alone with its smallest chunk (one page, or a shorter final chunk) against the token,
    KV and slot limits. Adding requests or tokens never lowers these costs, so if no single request fits, only the
    decode-only schedule is feasible. Returns dict(forced, action, reason, checked_singletons)."""
    rows = tuple(rows)
    _validate_resource_rows(rows)
    if not limits.decode_count:
        return dict(forced=False, action=None, reason='no_active_decoder', checked_singletons=0)
    checked = 0
    for row in rows:
        if check is not None and checked % 32 == 0:
            check()
        checked += 1
        if resource_feasible((row,), {row.index: min(PAGE, row.query)}, limits):
            return dict(forced=False, action=None, reason='positive_prefill_not_ruled_out',
                        checked_singletons=checked)
    return dict(forced=True, action={}, reason='no_positive_prefill_by_resources',
                checked_singletons=checked)


def merge_locks(rows,rank_locks):
    """Merge the lock costs that the two tensor-parallel ranks report, taking the larger cost of each node."""
    from dataclasses import replace
    merged=[]
    for index,row in enumerate(rows):
        union={}
        for locks in rank_locks:
            for node,cost in locks[index]:union[node]=max(cost,union.get(node,0))
        merged.append(replace(row,locks=tuple(sorted(union.items()))))
    return tuple(merged)


def _usage_checked(rows, action, check):
    """usage() with a deadline check every 16 rows."""
    tokens = kv = new_slots = count = 0
    locked = set()
    for position, row in enumerate(rows):
        if position % 16 == 0:
            check()
        q = action.get(row.index, 0)
        if not q:
            continue
        if not 0 < q <= row.query or (q != row.query and q % PAGE):
            return None
        count += 1
        tokens += paged(q)
        kv += paged(q)+row.host_prefix+PAGE
        if q == row.query:
            kv += completion_guard(row)
        new_slots += not row.held_slot
        for position, (node, cost) in enumerate(row.locks):
            if position % 16 == 0:
                check()
            if node not in locked:
                locked.add(node)
                kv += cost
    return tokens, kv, new_slots, count


def _predict_checked(model, rows, action, limits, check):
    """CostModel.predict() with a deadline check every 16 rows."""
    q = attention = host = 0
    for position, row in enumerate(rows):
        if position % 16 == 0:
            check()
        n = action.get(row.index, 0)
        if n:
            q += n
            c = row.gpu_prefix+row.host_prefix
            attention += n*(c+(n+1)/2)
            host += row.host_prefix
    if not q:
        return model.decode_s+model.launch_s if limits.decode_count else 0.
    base = model.intercept_s+model.query_token_s*q+model.query_context_billion_s*attention/1e9
    scale = model.mixed_scale if limits.decode_count else model.pure_scale
    return max(base*scale,model.decode_s if limits.decode_count else 0.)+host*model.transfer_token_s+model.launch_s


def solve(rows, limits, model, *, budget_s=.010, predicted_wait_s=0., planning_allowance_s=None,
          clock=time.perf_counter, rotation_round):
    """Plan one step and return dict(action={row index: prefill tokens}, score, status, predicted_execution_s, ...).

    If the decode-only certificate holds, the schedule is decode only. Otherwise the planner first scores the mandatory
    schedule: decode only when requests decode, else the first single request that fits with its smallest chunk. Then
    it fills candidate batches greedily in four orders of the requests that fit alone (urgency, completion and size,
    age, prefix sharing), each in two chunk modes: the largest chunk that fits, or completion or one page. Finally it
    tries up to fourteen moves around the best batch with a short list of 16 requests. Optional work stops at budget_s
    on CLOCK, after 24 scored or 128 probed schedules. Every returned schedule is feasible and scored; ties go to the
    schedule whose requests get their first tokens earlier. The step is predicted to end after predicted_wait_s, the
    planning time planning_allowance_s and its own predicted execution time.
    """
    started = clock()
    deadline = started+budget_s
    rows = tuple(rows)
    if (not math.isfinite(budget_s) or budget_s <= 0
            or len({r.index for r in rows}) != len(rows)
            or any(r.remaining < 1 or r.query < 1 for r in rows)):
        raise ValueError('invalid shadow frontier or budget')
    if planning_allowance_s is not None and (not math.isfinite(planning_allowance_s)
                                            or planning_allowance_s < 0):
        raise ValueError('invalid planning allowance')
    stats = dict(origin=None, full_cohort_count=len(rows),
        resource_proof_s=0., setup_s=0., strata_s=0., portfolio_s=0., moves_s=0.,
        scoring_s=0., mandatory_fallback_s=0., positive_fallback_atomic_s=0.,
        finite_candidates_exhausted=False, deadline_reached=False,
        deadline_overshoot_s=0., work_budget_reached=False,
        resource_singletons_checked=0, forced_resource_decode=False,
        resource_proof_budget_aborts=0, fallback_singletons_checked=0,
        mandatory_probe_calls=0, optional_probe_calls=0, optional_unique_probes=0,
        probe_cache_hits=0, optional_score_attempts=0, optional_fully_scored_actions=0,
        mandatory_fully_scored_actions=0, fully_scored_actions=0,
        objective_cache_hits=0, score_budget_aborts=0,
        objective_constructed=False, selected_action_scored=False,
        fallback_scored=False, positive_fallback_scored=False,
        strata_resource_singletons_checked=0, shortlist_size=0,
        strata_short_count=0, strata_oldest_count=0, strata_urgent_count=0,
        strata_share_count=0, strata_rotating_count=0, strata_fill_count=0,
        rotation_round=0, portfolios_started=0, portfolios_completed=0,
        portfolios_unique=0, portfolios_scored=0, moves_proposed=0,
        moves_feasible=0, moves_scored=0, portfolio_rows_scanned=0,
        portfolio_full_cohort_orders=0, portfolios_probe_sealed=0,
        portfolios_deadline_sealed=0, portfolios_resource_sealed=0,
        portfolios_full_scan_completed=0, portfolio_score_reserve_s=.001,
        portfolio_probe_cap_per_family=16, portfolio_usage_cache_hits=0,
        portfolio_usage_cache_misses=0, first_complete_fill_scored=False,
        first_complete_fill_atomic_s=0., first_complete_fill_deadline_exemptions=0,
        usage_s=0., prediction_s=0., max_atomic_usage_s=0., max_atomic_prediction_s=0.,
        checked_prediction_calls=0,
        optional_score_budget_guard_skips=0, first_complete_fill_selected=False,
        first_complete_fill_score=0., initial_fallback_score=0.,
        selected_incumbent_origin_id=0, selected_prefill_count=0, selected_query_tokens=0,
        optional_probe_cap_hit=False, optional_score_cap_hit=False,
        partial_swap_proposals=0, partial_swap_geometry_s=0.)
    diagnostics = dict(schema_version=1, root_count=len(rows),
        age_le1_count=0, age_le1_remaining_tokens=0,
        age_1to2_count=0, age_1to2_remaining_tokens=0,
        age_2to295_count=0, age_2to295_remaining_tokens=0,
        age_ge295_count=0, age_ge295_remaining_tokens=0,
        age_nonfinite_count=0, age_min_s=0., age_max_s=0.,
        root_remaining_tokens=0, resource_rejected_unique_probes=0,
        timing_rejected_unique_probes=0, accepted_unique_probes=0,
        mandatory_resource_rejected_unique_probes=0,
        mandatory_timing_rejected_unique_probes=0,
        optional_resource_rejected_unique_probes=0,
        optional_timing_rejected_unique_probes=0,
        partial_swap_proposals=0, partial_swap_unique_proposals=0,
        partial_swap_unique_offers=0, partial_swap_scored=0,
        partial_swap_new_scores=0, partial_swap_cached_scores=0,
        partial_swap_strict_raw_improvements=0,
        partial_swap_delta_present=0, partial_swap_min_raw_delta=0.,
        selected_matches_partial_swap=0)
    age_seen = False
    for row in rows:
        diagnostics['root_remaining_tokens'] += row.remaining
        if not math.isfinite(row.age_s):
            diagnostics['age_nonfinite_count'] += 1
            continue
        prefix = ('age_le1' if row.age_s <= 1. else
                  'age_1to2' if row.age_s <= 2. else
                  'age_2to295' if row.age_s < 2.95 else 'age_ge295')
        diagnostics[prefix+'_count'] += 1
        diagnostics[prefix+'_remaining_tokens'] += row.remaining
        diagnostics['age_min_s'] = min(diagnostics['age_min_s'], row.age_s) if age_seen else row.age_s
        diagnostics['age_max_s'] = max(diagnostics['age_max_s'], row.age_s) if age_seen else row.age_s
        age_seen = True
    def finite_numeric(value):
        return type(value) in (int, float) and math.isfinite(value)
    for name in ('mixed_scale', 'pure_scale', 'decode_s', 'observations'):
        value = getattr(model, name, None)
        diagnostics['model_'+name+'_present'] = int(finite_numeric(value))
        diagnostics['model_'+name] = value if finite_numeric(value) else 0.
    diagnostics['predicted_wait_s'] = predicted_wait_s if finite_numeric(predicted_wait_s) else 0.
    allowance = budget_s if planning_allowance_s is None else planning_allowance_s
    diagnostics['planning_allowance_s'] = allowance
    diagnostics['planning_allowance_explicit'] = int(planning_allowance_s is not None)
    diagnostics['budget_s'] = budget_s
    best_positive_key = None
    partial_swap_keys, partial_swap_offered = set(), set()

    def checkpoint():
        if clock() >= deadline:
            stats['deadline_reached'] = True
            raise TimeoutError

    proof_started = clock()
    try:
        proof = forced_current_action(rows, limits, checkpoint)
    except TimeoutError:
        stats['resource_proof_budget_aborts'] = 1
        proof = dict(forced=False, checked_singletons=0)
    stats['resource_proof_s'] = clock()-proof_started
    stats['resource_singletons_checked'] = proof['checked_singletons']
    stats['forced_resource_decode'] = proof['forced']
    objective = RootObjective(rows, checkpoint, clock)
    stats['objective_constructed'] = True
    elapsed_base = predicted_wait_s+(budget_s if planning_allowance_s is None else planning_allowance_s)
    rowmap = {r.index: r for r in rows}
    best, best_key = None, None
    first_complete_fill_key = None
    probes, values, scores = {}, {}, {}
    resource_usages = {(): (0, 0, 0, 0)}
    stats['setup_s'] = clock()-started

    def key_for(action):
        return tuple(sorted((index, q) for index, q in action.items() if q))

    def probe(key, mandatory=False, complete_incumbent=False):
        if mandatory:
            stats['mandatory_probe_calls'] += 1
        else:
            if not (complete_incumbent and key in probes):
                checkpoint()
            stats['optional_probe_calls'] += 1
        if key in probes:
            stats['probe_cache_hits'] += 1
            return probes[key]
        if not mandatory:
            if stats['optional_unique_probes'] >= 128:
                stats['work_budget_reached'] = True
                stats['optional_probe_cap_hit'] = True
                raise TimeoutError
            stats['optional_unique_probes'] += 1
        probe_started = clock()
        try:
            used = usage(rows, dict(key)) if mandatory else _usage_checked(rows, dict(key), checkpoint)
        finally:
            spent = clock()-probe_started
            stats['usage_s'] += spent
            stats['max_atomic_usage_s'] = max(stats['max_atomic_usage_s'], spent)
        resource_usages[key] = used
        valid = (bool(key) and used is not None and used[0] <= limits.token_budget
            and used[1] < limits.kv_budget and used[2] <= limits.slots
            and used[3] <= limits.batch_slots) or (not key and limits.decode_count > 0)
        predicted = None
        if not valid:
            diagnostics['resource_rejected_unique_probes'] += 1
            diagnostics[('mandatory' if mandatory else 'optional')+'_resource_rejected_unique_probes'] += 1
        if valid:
            probe_started = clock()
            try:
                if mandatory:
                    predicted = model.predict(rows, dict(key), limits)
                else:
                    stats['checked_prediction_calls'] += 1
                    predicted = _predict_checked(model, rows, dict(key), limits, checkpoint)
            finally:
                spent = clock()-probe_started
                stats['prediction_s'] += spent
                stats['max_atomic_prediction_s'] = max(stats['max_atomic_prediction_s'], spent)
            valid = not key or predicted <= limits.time_cap_s
            if not valid:
                diagnostics['timing_rejected_unique_probes'] += 1
                diagnostics[('mandatory' if mandatory else 'optional')+'_timing_rejected_unique_probes'] += 1
        diagnostics['accepted_unique_probes'] += int(valid)
        answer = (used, predicted) if valid else None
        probes[key] = answer
        return answer

    def offer(action, origin, mandatory=False, atomic=False):
        nonlocal best, best_key, best_positive_key
        key = key_for(action)
        if key in values:
            stats['objective_cache_hits'] += 1
            value = values[key]
        else:
            checked = probe(key, mandatory, complete_incumbent=atomic)
            if checked is None:
                values[key] = None
                return False
            if not mandatory:
                if stats['optional_score_attempts'] >= 24:
                    stats['work_budget_reached'] = True
                    stats['optional_score_cap_hit'] = True
                    raise TimeoutError
                stats['optional_score_attempts'] += 1
                if not atomic:
                    checkpoint()
                    # Keep 1 ms to finish an optional score before the deadline.
                    if clock() >= deadline-.001:
                        stats['optional_score_budget_guard_skips'] += 1
                        raise TimeoutError
                elif clock() >= deadline:
                    stats['first_complete_fill_deadline_exemptions'] += 1
            score_started = clock()
            try:
                # Mandatory work and the first complete fill are scored without the prefix registry, and always
                # complete; later candidates build the registry incrementally.
                cost = objective.potential(dict(key), elapsed_base+checked[1], legacy=mandatory or atomic)
            except TimeoutError:
                stats['score_budget_aborts'] += 1
                raise
            finally:
                spent = clock()-score_started
                stats['scoring_s'] += spent
                if atomic:
                    stats['positive_fallback_atomic_s'] += spent
                    stats['first_complete_fill_atomic_s'] += spent
            scores[key] = cost
            # Compare the objective first, then the first-token times of the completed requests, all at the end of
            # the step; among equal schedules the first one found stays.
            completions = sum(q == rowmap[index].query for index, q in key)
            first_tokens = (elapsed_base+checked[1],)*completions+(math.inf,)
            value = (cost, first_tokens)
            values[key] = value
            stats['fully_scored_actions'] += 1
            stats['mandatory_fully_scored_actions' if mandatory else 'optional_fully_scored_actions'] += 1
        if value is None:
            return False
        if key and (best_positive_key is None or value < values[best_positive_key]):
            best_positive_key = key
        if mandatory:
            stats['fallback_scored'] = True
        if key and (mandatory or atomic):
            stats['positive_fallback_scored'] = True
        if best_key is None or value < best_key:
            best, best_key = dict(key), value
            stats['origin'] = origin
        return True

    def finish():
        selected = best or {}
        diagnostics['elapsed_base_s'] = elapsed_base if finite_numeric(elapsed_base) else 0.
        # Summaries of schedules that were already scored; the release-aware rollout reads the decode-only one.
        for prefix, candidate_key in (('decode', () if () in scores else None),
                ('best_positive', best_positive_key),
                ('first_fill', first_complete_fill_key),
                ('selected', key_for(selected) if best_key is not None else None)):
            complete = candidate_key is not None and candidate_key in scores
            diagnostics[prefix+'_scored'] = int(complete)
            diagnostics[prefix+'_raw_score'] = 0.
            diagnostics[prefix+'_raw_score_finite'] = 0
            diagnostics[prefix+'_predicted_execution_s'] = 0.
            diagnostics[prefix+'_query_tokens'] = 0
            diagnostics[prefix+'_completion_count'] = 0
            if complete:
                raw = scores[candidate_key]
                diagnostics[prefix+'_raw_score_finite'] = int(finite_numeric(raw))
                diagnostics[prefix+'_raw_score'] = raw if finite_numeric(raw) else 0.
                predicted = probes[candidate_key][1]
                diagnostics[prefix+'_predicted_execution_s'] = predicted if finite_numeric(predicted) else 0.
                diagnostics[prefix+'_query_tokens'] = sum(q for _, q in candidate_key)
                diagnostics[prefix+'_completion_count'] = len(values[candidate_key][1])-1
        for prefix in ('best_positive', 'first_fill', 'selected'):
            present = bool(diagnostics[prefix+'_raw_score_finite'] and diagnostics['decode_raw_score_finite'])
            diagnostics[prefix+'_minus_decode_present'] = int(present)
            diagnostics[prefix+'_minus_decode_raw_score'] = (
                diagnostics[prefix+'_raw_score']-diagnostics['decode_raw_score'] if present else 0.)
        diagnostics['partial_swap_proposals'] = stats['partial_swap_proposals']
        diagnostics['partial_swap_unique_proposals'] = len(partial_swap_keys)
        diagnostics['selected_matches_partial_swap'] = int(key_for(selected) in partial_swap_keys)
        stats['selected_action_scored'] = best_key is not None
        stats['first_complete_fill_selected'] = first_complete_fill_key is not None and key_for(selected) == first_complete_fill_key
        stats['selected_incumbent_origin_id'] = {'forced_resource_decode':1, 'mandatory_decode':2,
            'minimal_prefill_fallback':3, 'complete_fill_portfolio':4, 'fixed_complete_action_move':5}.get(stats['origin'],0)
        stats['selected_prefill_count'] = len(selected)
        stats['selected_query_tokens'] = sum(selected.values())
        stats['evaluated'] = stats['fully_scored_actions']
        stats['seed_s'] = stats['strata_s']+stats['portfolio_s']+stats['moves_s']
        stats['dfs_s'] = 0.
        stats['fallback_atomic_s'] = stats['mandatory_fallback_s']+stats['first_complete_fill_atomic_s']
        stats.update(objective.prefix_vectors.stats)
        key = key_for(selected)
        checked = probes.get(key)
        execution = checked[1] if checked is not None else model.predict(rows, selected, limits)
        wall = clock()-started
        stats['deadline_overshoot_s'] = max(0., wall-budget_s)
        stats['deadline_reached'] |= wall >= budget_s
        if stats['origin'] is None:
            stats['origin'] = 'no_feasible_prefill'
        return dict(action=selected, score=best_key[0] if best_key else None,
            wall_s=wall, predicted_execution_s=execution, status='feasible_unproven', stats=stats,
            diagnostics=diagnostics)

    fallback_started = clock()
    if proof['forced']:
        elapsed_base = predicted_wait_s+clock()-started
        offer({}, 'forced_resource_decode', mandatory=True)
        stats['mandatory_fallback_s'] = clock()-fallback_started
        result = finish()
        result['status'] = 'search_exhausted'
        return result
    if limits.decode_count:
        offer({}, 'mandatory_decode', mandatory=True)
    else:
        # Without decodes, scan for the first request that fits alone. The scan is not cut by the budget, so that
        # an unfit request at the head cannot block a later one.
        for row in rows:
            stats['fallback_singletons_checked'] += 1
            q = min(PAGE, row.query)
            if offer({row.index: q}, 'minimal_prefill_fallback', mandatory=True):
                break
    stats['mandatory_fallback_s'] = clock()-fallback_started
    stats['initial_fallback_score'] = best_key[0] if best_key else 0.

    def resource_usage_for(action):
        key = key_for(action)
        if key in resource_usages:
            stats['portfolio_usage_cache_hits'] += 1
            return resource_usages[key]
        stats['portfolio_usage_cache_misses'] += 1
        resource_usages[key] = usage(rows, action)
        return resource_usages[key]

    def options(row, action, used):
        tok, kv, slots, count = used
        if count >= limits.batch_slots or slots+(not row.held_slot) > limits.slots:
            return ()
        locked = {node for index in action for node, _ in rowmap[index].locks}
        extra = 0
        for node, cost in row.locks:
            if node not in locked:
                locked.add(node)
                extra += cost
        base = kv+row.host_prefix+PAGE+extra
        pages = min((limits.token_budget-tok)//PAGE,
                    (limits.kv_budget-1-base)//PAGE, (row.query-1)//PAGE)
        maximum = max(0, pages)*PAGE
        full = row.query if (paged(row.query) <= limits.token_budget-tok and
            base+paged(row.query)+completion_guard(row) < limits.kv_budget) else 0
        # At most four chunk sizes: completion, the largest page multiple that fits, half of it and one page.
        minimum = min(PAGE, row.query) if full or maximum >= PAGE else 0
        return tuple(dict.fromkeys(q for q in (full, maximum, (maximum//(2*PAGE))*PAGE,
                                                minimum) if q > 0))

    phase_started = clock()
    phase = 'strata_s'
    def change_phase(name):
        nonlocal phase, phase_started
        stats[phase] += clock()-phase_started
        phase, phase_started = name, clock()
    seen_portfolios = set()
    def fill_family(order, mode):
        nonlocal first_complete_fill_key
        checkpoint()
        stats['portfolios_started'] += 1
        action = {}
        stop_after_seal = False
        probe_start = stats['optional_unique_probes']
        scanned = 0
        expired = False
        try:
            for row in order:
                checkpoint()
                # A fill may stop early; its batch so far is feasible, so it is sealed and scored.
                if action and clock() >= deadline-stats['portfolio_score_reserve_s']:
                    stats['portfolios_deadline_sealed'] += 1
                    stop_after_seal = True
                    break
                used = resource_usage_for(action)
                if (used[0]+PAGE > limits.token_budget or
                        used[1]+2*PAGE >= limits.kv_budget or
                        used[3] >= limits.batch_slots):
                    stats['portfolios_resource_sealed'] += 1
                    break
                if (stats['optional_unique_probes']-probe_start >= 16 or
                        stats['optional_unique_probes'] >= 128):
                    stats['portfolios_probe_sealed'] += 1
                    stop_after_seal = stats['optional_unique_probes'] >= 128
                    break
                scanned += 1
                stats['portfolio_rows_scanned'] += 1
                qs = options(row, action, used)
                if mode == 'largest':
                    qs = sorted(qs, reverse=True)
                else:
                    qs = tuple(dict.fromkeys(q for q in (row.query, min(PAGE,row.query)) if q in qs))
                for q in qs:
                    checkpoint()
                    candidate = dict(action)
                    candidate[row.index] = q
                    key = key_for(candidate)
                    if key not in probes and (stats['optional_unique_probes']-probe_start >= 16 or
                                               stats['optional_unique_probes'] >= 128):
                        break
                    if probe(key) is not None:
                        action = candidate
                        break
        except TimeoutError:
            if not action or stats['first_complete_fill_scored']:
                raise
            expired = True
            stats['portfolios_deadline_sealed'] += 1
        if scanned == len(order):
            stats['portfolios_full_scan_completed'] += 1
        stats['portfolios_completed'] += 1
        key = key_for(action)
        if key and key not in seen_portfolios:
            seen_portfolios.add(key)
            stats['portfolios_unique'] += 1
            atomic = not stats['first_complete_fill_scored']
            if offer(action, 'complete_fill_portfolio', atomic=atomic):
                stats['portfolios_scored'] += 1
                if not stats['first_complete_fill_scored']:
                    first_complete_fill_key = key
                    stats['first_complete_fill_score'] = values[key][0]
                stats['first_complete_fill_scored'] = True
        if expired or stop_after_seal:
            if stats['optional_unique_probes'] >= 128:
                stats['work_budget_reached'] = True
                stats['optional_probe_cap_hit'] = True
            raise TimeoutError
    try:
        checkpoint()
        eligible = []
        for row in rows:
            checkpoint()
            stats['strata_resource_singletons_checked'] += 1
            if resource_feasible((row,), {row.index:min(PAGE,row.query)}, limits):
                eligible.append(row)
        urgency = objective.marginal_weights()
        def ordered(source, key):
            checkpoint()
            result = sorted(source, key=key)
            checkpoint()
            return result
        first_order = ordered(eligible,lambda r:(-urgency[r.index],-r.age_s,r.index))
        stats['portfolio_full_cohort_orders'] = 1
        change_phase('portfolio_s')
        # The urgency fill is completed and scored before the other orders and the short list are prepared.
        fill_family(first_order, 'largest')
        change_phase('strata_s')
        completable = {}
        for row in eligible:
            checkpoint()
            completable[row.index] = resource_feasible((row,), {row.index:row.query}, limits)
        short = ordered(eligible, lambda r: (not completable[r.index], r.query, -r.age_s, r.index))
        old = ordered(eligible, lambda r: (-r.age_s, r.index))
        near = ordered((r for r in eligible if r.age_s <= 3.),
                       lambda r: (3.-r.age_s, r.query, r.index))
        fanout = {}
        next_nodes = {}
        for position, row in enumerate(eligible):
            checkpoint()
            cap = min((row.total-1)//PAGE, len(row.path))
            depth = min(cap, (row.gpu_prefix+row.host_prefix+min(PAGE,row.query))//PAGE)
            node = row.path[depth-1] if depth else None
            next_nodes[row.index] = node
            if node is not None:
                fanout[node] = fanout.get(node, 0)+1
        shared = ordered(eligible, lambda r: (-fanout.get(next_nodes[r.index],0), r.query, -r.age_s, r.index))
        stats['rotation_round'] = int(rotation_round)
        rotation = int(rotation_round)*2%len(eligible) if eligible else 0
        rotating = eligible[rotation:]+eligible[:rotation]
        selected, picked = [], set()
        def choose(order, count, stat):
            added = 0
            for row in order:
                checkpoint()
                if added >= count or len(selected) >= 16:
                    break
                if row.index not in picked:
                    selected.append(row)
                    picked.add(row.index)
                    added += 1
            stats[stat] += added
        choose(short, 4, 'strata_short_count')
        held = ordered((r for r in eligible if r.held_slot), lambda r: (-r.age_s,r.index))
        choose(held, 2, 'strata_oldest_count')
        choose(old, 4-stats['strata_oldest_count'], 'strata_oldest_count')
        choose(near, 4, 'strata_urgent_count')
        choose(shared, 2, 'strata_share_count')
        choose(rotating, 2, 'strata_rotating_count')
        choose(rotating, 16-len(selected), 'strata_fill_count')
        stats['shortlist_size'] = len(selected)
        checkpoint()
        # The eight fills scan every request that fits alone; the short list restricts only the moves.
        orders = (first_order, short, old, shared)
        stats['portfolio_full_cohort_orders'] = len(orders)
        change_phase('portfolio_s')
        for order_number, order in enumerate(orders):
            for mode in ('largest', 'completion_or_page'):
                if order_number == 0 and mode == 'largest':
                    continue
                fill_family(order, mode)
        change_phase('moves_s')
        # Moves around the best batch: add one page of a short-listed request, swap one in for the largest chunk
        # cut to one page or removed, or double a scheduled chunk.
        base = dict(best or {})
        outside = [r for r in selected if r.index not in base]
        inside = sorted((rowmap[i] for i in base), key=lambda r:(-base[r.index],r.index))
        proposals = []
        for row in outside[:4]:
            checkpoint()
            candidate = dict(base)
            candidate[row.index] = min(PAGE,row.query)
            proposals.append(candidate)
        if inside:
            victim = inside[0]
            for row in outside[:4]:
                checkpoint()
                candidate = dict(base)
                candidate[victim.index] = min(PAGE,victim.query)
                candidate[row.index] = row.query
                proposals.append(candidate)
            for row in outside[:4]:
                checkpoint()
                candidate = dict(base)
                candidate.pop(victim.index)
                geometry_started = clock()
                # If completion does not fit, take the largest chunk that does.
                qs = options(row, candidate, resource_usage_for(candidate))
                q = max(qs, default=row.query)
                stats['partial_swap_geometry_s'] += clock()-geometry_started
                stats['partial_swap_proposals'] += int(q != row.query)
                candidate[row.index] = q
                if q != row.query:
                    partial_swap_keys.add(key_for(candidate))
                proposals.append(candidate)
        for row in inside[:4]:
            checkpoint()
            candidate = dict(base)
            candidate[row.index] = (row.query if row.query <= 2*base[row.index] else
                                    min(row.query-1, 2*base[row.index])//PAGE*PAGE)
            proposals.append(candidate)
        for candidate in proposals[:14]:
            checkpoint()
            stats['moves_proposed'] += 1
            candidate_key = key_for(candidate)
            track_partial = candidate_key in partial_swap_keys and candidate_key not in partial_swap_offered
            if track_partial:
                partial_swap_offered.add(candidate_key)
                diagnostics['partial_swap_unique_offers'] += 1
            before_raw = best_key[0] if best_key is not None else None
            cached_score = candidate_key in scores
            if probe(candidate_key) is not None:
                stats['moves_feasible'] += 1
                if offer(candidate,'fixed_complete_action_move'):
                    stats['moves_scored'] += 1
                    if track_partial:
                        diagnostics['partial_swap_scored'] += 1
                        diagnostics['partial_swap_cached_scores' if cached_score else 'partial_swap_new_scores'] += 1
                        raw = scores[candidate_key]
                        if finite_numeric(before_raw) and finite_numeric(raw):
                            delta = raw-before_raw
                            diagnostics['partial_swap_strict_raw_improvements'] += int(raw < before_raw)
                            if not diagnostics['partial_swap_delta_present'] or delta < diagnostics['partial_swap_min_raw_delta']:
                                diagnostics['partial_swap_min_raw_delta'] = delta
                            diagnostics['partial_swap_delta_present'] = 1
        stats['finite_candidates_exhausted'] = True
    except TimeoutError:
        pass
    finally:
        stats[phase] += clock()-phase_started
    return finish()
