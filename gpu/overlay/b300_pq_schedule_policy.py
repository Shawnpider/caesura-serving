"""PQ+aging, the priority-queue baseline: SGLang's admission with a priority order of the waiting queue.

When a request enters the queue, its prefill time is estimated from its uncached prompt; an estimate up to 0.2 s puts
it on the top level, up to 0.4 s on the middle level, and a longer one, or one outside the calibrated range, on the
bottom level. Each level is served in arrival order, and a request that has waited 5 s moves to the end of the top
level. Only the order changes: SGLang's admission, chunking, allocation, load-back and retraction stay in charge. Rank 0
of the two tensor-parallel ranks owns the queue state and broadcasts the order.
"""
from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import time


LOG = logging.getLogger(__name__)
# Prefill time of one chunk on two H200 GPUs serving Qwen3-30B-A3B-Instruct-2507, fitted to 18 measured single-request
# prefills (CUDA events): intercept + per query token + per billion query x context, valid up to 8192 query and
# 122880 context tokens.
COEFFICIENTS = (0.061864795456745024, 1.6400297490029218e-06, 0.52809255111424)
DOMAIN = dict(min_query_tokens=1, max_query_tokens=8192, max_context_tokens=122880)


def finite(value, name, *, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    value = float(value)
    if not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError(f"invalid {name}")
    return value


class PrefillCostModel:
    """Prefill time of a prompt: the sum over its 8192-token chunks of a + b q + c q (context + (q + 1) / 2) / 1e9.

    It counts only the prefill itself, without queueing, load-back or decodes. estimate() also returns the number of
    chunks outside the calibrated range."""

    def __init__(self, coefficients=COEFFICIENTS, domain=DOMAIN):
        self.coefficients = tuple(finite(c, "coefficient") for c in coefficients)
        self.domain = dict(domain)
        self.page_tokens, self.chunk_tokens = 128, 8192

    def estimate(self, prompt_tokens, matched_tokens):
        if (type(prompt_tokens) is not int or prompt_tokens < 1
                or type(matched_tokens) is not int or not 0 <= matched_tokens <= prompt_tokens
                or matched_tokens % self.page_tokens):
            raise ValueError("invalid prompt/prefix geometry")
        remaining, context = prompt_tokens - matched_tokens, matched_tokens
        if not remaining:
            return 0., 0
        total, extrapolated = 0., 0
        a, b, c = self.coefficients
        while remaining:
            query = min(remaining, self.chunk_tokens)
            total += a + b * query + c * query * (context + (query + 1) / 2) / 1e9
            extrapolated += int(not (
                self.domain["min_query_tokens"] <= query <= self.domain["max_query_tokens"]
                and context <= self.domain["max_context_tokens"]))
            context += query
            remaining -= query
        return finite(total, "estimated prefill cost", positive=True), extrapolated


@dataclass
class QueueEntry:
    key: str
    received_s: float
    queue_entry_s: float
    prompt_tokens: int
    matched_tokens: int
    estimated_s: float | None = None
    level: int | None = None
    initial_level: int | None = None
    fifo_sequence: int | None = None
    waiting_since: float | None = None
    observed_queue_entry_s: float | None = None
    promotions: int = 0
    extrapolated_chunks: int = 0


class PQAgingQueue:
    """Three levels in arrival order, a level fixed at queue entry, and the top level after 5 s of waiting.

    Entries live on SGLang's request objects. A retracted request keeps its estimate and level, and its waiting time
    restarts when it re-enters the queue."""

    def __init__(self, model):
        self.model = model
        self.sequence = 0
        self.last_now = None
        self.calls = 0
        self.initial_levels = [0, 0, 0]
        self.promotions = 0
        self.extrapolated_requests = 0
        self.out_of_domain_requests = 0

    def _sequence(self):
        value = self.sequence
        self.sequence += 1
        return value

    def order(self, entries, now):
        now = finite(now, "queue clock")
        if self.last_now is not None and now < self.last_now:
            raise ValueError("PQ clock moved backwards")
        if len({e.key for e in entries}) != len(entries):
            raise ValueError("duplicate PQ request key")
        for entry in entries:
            if not isinstance(entry.key, str) or not entry.key:
                raise ValueError("invalid PQ request key")
            for name in ("received_s", "queue_entry_s"):
                if finite(getattr(entry, name), name) > now:
                    raise ValueError("PQ request timestamp is in the future")
        # A stable sort keeps the queue order of exact ties.
        for entry in sorted(entries, key=lambda e: e.received_s):
            if entry.estimated_s is None:
                entry.estimated_s, entry.extrapolated_chunks = self.model.estimate(
                    entry.prompt_tokens, entry.matched_tokens)
                entry.level = (2 if entry.extrapolated_chunks else
                               (0 if entry.estimated_s <= .2 else (1 if entry.estimated_s <= .4 else 2)))
                entry.initial_level = entry.level
                entry.fifo_sequence = self._sequence()
                entry.waiting_since = entry.received_s
                entry.observed_queue_entry_s = entry.queue_entry_s
                self.initial_levels[entry.level] += 1
                self.extrapolated_requests += bool(entry.extrapolated_chunks)
                self.out_of_domain_requests += bool(entry.extrapolated_chunks)
            elif entry.queue_entry_s > entry.observed_queue_entry_s:
                # A retracted request re-enters the queue: keep its estimate and position, restart its waiting.
                entry.waiting_since = entry.queue_entry_s
                entry.observed_queue_entry_s = entry.queue_entry_s
        promoted = [e for e in entries if e.level > 0 and now - e.waiting_since >= 5.]
        for entry in sorted(promoted, key=lambda e: (e.waiting_since, e.fifo_sequence)):
            entry.level = 0
            entry.fifo_sequence = self._sequence()
            entry.waiting_since = now
            entry.promotions += 1
            self.promotions += 1
        self.last_now = now
        self.calls += 1
        return sorted(entries, key=lambda e: (e.level, e.fifo_sequence))


class NativePQAging:
    """The order of SGLang's waiting queue under PQ+aging, called from SGLang's calc_priority.

    Each rank matches every waiting request in its radix cache, as SGLang does. Rank 0 orders the queue and broadcasts
    the order, and both ranks check it before applying it: one broadcast and one reduction per call."""

    def __init__(self, tree_cache, match_prefix, torch_module):
        self.tree_cache = tree_cache
        self.match_prefix = match_prefix
        self.torch = torch_module
        self.dist = torch_module.distributed
        self.group = tree_cache.tp_group
        if (not self.dist.is_initialized() or tree_cache.pp_size != 1
                or self.dist.get_world_size(self.group) != 2
                or str(self.dist.get_backend(self.group)).lower() != "gloo"):
            raise ValueError("PQ requires the verified TP2/PP1 Gloo cache group")
        self.rank = self.dist.get_rank(self.group)
        self.source_rank = self.dist.get_global_rank(self.group, 0)
        self.queue = PQAgingQueue(PrefillCostModel()) if self.rank == 0 else None
        self.applied_calls = 0
        self.elapsed_total_s = 0.
        self.elapsed_max_s = 0.
        self.waiters_total = 0
        LOG.info("B300_PQ_AGING_INIT rank=%d tp=2 pp=1 thresholds_s=0.2,0.4 aging_s=5", self.rank)

    def _entry(self, req):
        existing = getattr(req, "_b300_pq_aging_entry", None)
        if existing is None:
            prompt = len(req.origin_input_ids)
            # Cached tokens count GPU and host hits; a fully cached prompt still prefills at least one token.
            maximum = (prompt - 1) // 128 * 128
            matched = min(len(req.prefix_indices) + int(req.host_hit_length), maximum)
            existing = QueueEntry(
                key=req.rid, received_s=req.time_stats.scheduler_recv_time,
                queue_entry_s=req.time_stats.wait_queue_entry_time,
                prompt_tokens=prompt, matched_tokens=matched,
            )
            req._b300_pq_aging_entry = existing
        existing.queue_entry_s = req.time_stats.wait_queue_entry_time
        return existing

    def order(self, waiting_queue):
        started = time.perf_counter()
        error, payload, by_key = None, None, {}
        try:
            for req in waiting_queue:
                if not isinstance(req.rid, str) or not req.rid or req.rid in by_key:
                    raise ValueError("invalid or duplicate native request key")
                by_key[req.rid] = req
                self.match_prefix(self.tree_cache, req, include_req=True)
            if self.rank == 0:
                entries = [self._entry(req) for req in waiting_queue]
                ordered = self.queue.order(entries, time.perf_counter())
                payload = {"order": [e.key for e in ordered], "error": None}
        except Exception as exc:
            error = type(exc).__name__ + ": " + str(exc)
            if self.rank == 0:
                payload = {"order": [], "error": error}
        shared = [payload]
        self.dist.broadcast_object_list(shared, src=self.source_rank, group=self.group)
        payload = shared[0]
        order = payload.get("order") if isinstance(payload, dict) else None
        valid = (error is None and isinstance(payload, dict) and payload.get("error") is None
                 and isinstance(order, list) and all(isinstance(x, str) for x in order)
                 and len(order) == len(by_key) and len(set(order)) == len(order)
                 and set(order) == set(by_key))
        agreed = self.torch.tensor(int(valid), dtype=self.torch.int, device="cpu")
        self.dist.all_reduce(agreed, op=self.dist.ReduceOp.MIN, group=self.group)
        if not agreed.item():
            # The same failure on both ranks, before the queue changes.
            raise RuntimeError("PQ TP decision validation failed")
        waiting_queue[:] = [by_key[key] for key in order]
        self.applied_calls += 1
        elapsed = time.perf_counter() - started
        self.elapsed_total_s += elapsed
        self.elapsed_max_s = max(elapsed, self.elapsed_max_s)
        self.waiters_total += len(order)
        if self.applied_calls == 1 or self.applied_calls % 128 == 0:
            LOG.info("B300_PQ_AGING_STATS rank=%d calls=%d waiters_total=%d cpu_sync_total_s=%.9f cpu_sync_max_s=%.9f initial_levels=%s promotions=%s out_of_domain_requests=%s",
                self.rank, self.applied_calls, self.waiters_total,
                self.elapsed_total_s, self.elapsed_max_s,
                self.queue.initial_levels if self.queue else None,
                self.queue.promotions if self.queue else None,
                self.queue.out_of_domain_requests if self.queue else None)
