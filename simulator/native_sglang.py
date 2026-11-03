"""Discrete-event simulator of one SGLang serving instance.

The simulator replays a request trace on one serving instance and returns the time to first token (TTFT) of every
request. One iteration of the main loop is one GPU step. It reproduces SGLang's whole-prompt admission with chunked
prefill, the FCFS, LPM and PQ+aging orders, the GPU and host prefix cache of HiCache with eviction and load-back, and
the one-step lag of overlap scheduling. Step times and scheduler CPU costs come from cost models calibrated against GPU
runs. For the planner and Caesura, it calls the planner code that runs in SGLang (gpu/overlay) once per step, on a
virtual clock, so that the results do not depend on the speed of the machine that runs the simulation.

A KV page holds 128 tokens. The path of a request lists the ids of its prompt pages, and requests with a common prefix
share the ids of its pages, so the parent of a page is the page before it on every path. Times are in seconds.
"""
from __future__ import annotations

import bisect
import heapq
import importlib.util
import math
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

DEADLINE_S = 3.   # TTFT SLO


@dataclass
class Cfg:
    # Scheduler: 'fcfs', 'lpm' and 'pq-aging' are SGLang's admission with that queue order, 'b01' is the planner.
    policy: str = 'fcfs'
    # PQ+aging: the prefill time of a request is estimated per chunk of q tokens on a context of c tokens as
    # a + b q + e q (c + (q + 1) / 2) / 1e9. The estimate sets its level, with the thresholds below, and a request
    # outside the calibrated context range takes the lowest level. Requests move to the top level after pq_aging_s.
    pq_cost: tuple = (0.24945264601876208, 2.6016077165691653e-06, 0.1411681746899885)
    pq_domain_max_ctx: int = 426880
    pq_thresholds: tuple = (0.2, 0.4)
    pq_aging_s: float = 5.0

    # Engine, as in the SGLang arguments of the GPU runs.
    page: int = 128
    gpu_tokens: int = 568960             # KV cache in GPU memory
    host_tokens: int = 1707008           # KV cache in host memory
    chunk: int = 8192                    # chunked prefill size
    max_prefill: int = 16384             # prefill tokens per step
    max_running: int = 128               # running requests
    clip: int = 4096                     # cap on the output tokens SGLang reserves per request
    ratio_init: float = 0.7              # share of the remaining output that SGLang reserves, decaying to
    ratio_min_factor: float = 0.14       #   ratio_init * ratio_min_factor over ratio_decay_steps decode steps
    ratio_decay_steps: int = 600
    in_batch_threshold: int = 32         # LPM moves uncached requests with the first page of an earlier one to the back

    # GPU step time. A step with prefill takes max(A, B + C q) + e attention + d context + n q d_n / 8192, and at least
    # the time to load its host pages back: q prefill tokens, attention = sum q (cached + q / 2) / 1e9, context = the
    # decode KV tokens / 1e6, n decodes. A decode step takes the time of its CUDA-graph batch size plus
    # decode_ctx_s_per_mtok per million context tokens.
    roofline: tuple = (0.2340, 0.1300, 20.0e-6, 0.0588, 0.0061)    # A, B, C, e, d
    mixed_nd_q_s: float = 0.                                        # d_n
    decode_buckets: tuple = (1, 2, 4, 8, 12)
    decode_bucket_base: tuple = (6.896e-3, 7.768e-3, 9.642e-3, 9.702e-3, 10.141e-3)
    decode_ctx_s_per_mtok: float = 5.718e-3
    loadback_s_per_token: float = 2.2e-6

    # Scheduler CPU of the SGLang policies, overlapped with the GPU time of the previous decode step. Every step costs
    # cpu_base. An admission pass adds cpu_admit, cpu_admit_per_kseg per 1000 radix segments of the waiting requests,
    # cpu_admit_per_kseg_exam per 1000 segments of the requests it examines and, for LPM, cpu_lpm_cold_per_mtok per
    # million tokens of the uncached waiting prompts. Each eviction and load-back call adds its cost. A decode step
    # exposes decode_exposed_per_kseg per 1000 segments of the waiting requests.
    cpu_base: float = 3.5e-3
    cpu_admit: float = 14.19e-3
    cpu_admit_per_kseg: float = 0.0
    cpu_admit_per_kseg_exam: float = 0.0
    cpu_lpm_cold_per_mtok: float = 0.0
    cpu_evict_chunk: float = 51.0e-3     # eviction call while allocating a prefill chunk
    cpu_evict_loadback: float = 16.3e-3  # eviction call while allocating pages for a load-back
    cpu_evict_decode: float = 0.7e-3     # eviction call while allocating decode pages
    cpu_loadback_call: float = 8.97e-3
    cpu_loadback_per_token: float = 0.055e-6
    decode_exposed_per_kseg: float = 0.

    # Request path. A request reaches the frontend client_upload after its arrival and the scheduler d_in later, plus
    # d_in_per_token per prompt token on one ingress server shared in arrival order. Its first token reaches the
    # frontend d_out after the scheduler emits it. TTFT runs from the frontend receive to the first token.
    client_upload: float = 0.0
    d_in: float = 5e-3
    d_in_per_token: float = 0.32e-6
    d_out: float = 4e-3

    # Planner: the solving budget on the virtual clock, the virtual time of one clock reading, the round preparation
    # before the solve, the request slots and the step-time cap.
    b01_real_budget_s: float = 0.1
    b01_real_clock_tick_s: float = 1e-5
    b01_real_prep_s: float = 0.005
    b01_real_slots: int = 128
    b01_time_cap_s: float = 0.75

    # Exposed planning CPU of the planner, fitted to the GPU runs. Before a step, the GPU idles for
    # max(floor, R + X + p - s). R is the CPU time of the planning round, linear in its features per round type: a
    # round that schedules prefill (slk_Rp), a decode-only round proven by the certificate (slk_cert) and any other
    # decode-only round (slk_Rd). X is the CPU time of the extra solves of the rollout, p the preparation before the
    # kernels and s the slack of the step in flight, namely its GPU time minus its CPU dispatch time. Residuals are
    # drawn from measured quantiles. Prefill steps get a mean-preserving lognormal noise of log sd slk_gpu_noise,
    # the step-time prediction error of the planner on the GPU.
    slk_disp: tuple = (0.2199, 0.0170, 0.0503)     # prefill dispatch: 1, decodes / 100, prefill tokens / 8192
    slk_Rp: tuple = (0.03156, 0.003, 0.01898, 0.02983, 0.00588)   # 1, waiting / 100, scheduled / 10,
    slk_Rp_off: float = 0.                                         #   host tokens / 1e5, decodes / 100
    slk_pp: float = 0.010
    slk_Rd: tuple = (0.02226, 0.02134, -0.00807)   # 1, waiting / 100, decodes / 100
    slk_pd: float = 0.007
    slk_cert: tuple = ()     # (reused certificate: 1, waiting / 100, decodes / 100), (new certificate: 1,
    slk_pf: float = 0.0055   #   waiting / 100, waiting prompt tokens / 1e6), reused residuals, new residuals
    slk_nr_after_p: float = 0.0083                 # idle of a step without a planning round after a prefill step
    slk_nr: float = 0.00012                        # ... after a decode step
    slk_i0: tuple = (0.0030, 0.0050)               # floor before a prefill step: 1, decodes / 100
    slk_i0_d: float = 0.0005                       # floor before a decode step
    slk_disp_res: tuple = ()
    slk_Rp_res: tuple = ()
    slk_Rd_res: tuple = ()
    slk_Rf_res: tuple = ()   # a certificate round also draws once from the main residual stream, of this length
    slk_gpu_noise: float = 0.
    slk_search_s: float = 0.0044                   # CPU time of one solve
    slk_seed: int = 0

    # Caesura on top of the planner: the release-aware rollout (rc_*), the thin-step veto (veto_*) and the lower
    # planning CPU of its implementation, in seconds less R per round type (slk_R_cut_*).
    rc_steps: tuple = ()                 # numbers of decode steps to look ahead, () = the planner alone
    rc_margin: int = 1                   # wait only if it puts this many more requests on time
    rc_min_decoders: int = 0             # look ahead only with at least this many decodes
    rc_len_q: float = 0.5                # quantile of the remaining-output predictor
    rc_len_min_n: int = 3                # completed requests a class needs before it is used
    rc_cost_scale: float = 0.            # CPU time of one extra solve, in units of slk_search_s
    veto_tokens: int = 0                 # 0 = no veto
    veto_min_decoders: int = 16
    veto_slack_s: float = 0.3
    veto_max_steps: int = 6
    veto_req_budget_s: float = 1.0
    slk_R_cut_P: float = 0.
    slk_R_cut_Fh: float = 0.
    slk_R_cut_Fm: float = 0.
    slk_R_cut_D: float = 0.


class NativeSim:
    def __init__(self, jobs, cfg: Cfg):
        self.cfg = cfg
        self.jobs = jobs
        P = cfg.page
        # Radix segments of each request: its path splits where the trie of all paths branches or another prompt
        # ends. Prefix matching walks one tree node per segment, so the admission CPU grows with them.
        children, ends = {}, {}
        for j in jobs:
            prev = 0
            for pg in j['path']:
                children.setdefault(prev, set()).add(pg)
                prev = pg
            ends[prev] = ends.get(prev, 0) + 1
        self.segs = []
        for j in jobs:
            n_seg = 1
            for pg in j['path'][:-1]:
                if len(children.get(pg, ())) > 1 or ends.get(pg, 0):
                    n_seg += 1
            self.segs.append(n_seg)
        self.cap = cfg.gpu_tokens // P
        self.hcap = cfg.host_tokens // P
        n = len(jobs)
        self.parent, self.depth = {}, {}
        for j in jobs:
            p = j['path']
            for k, pid in enumerate(p):
                self.parent[pid] = p[k - 1] if k else -1
                self.depth[pid] = k
        self.next_pid = max(self.parent, default=0) + 1
        # Prefix cache: the pages in GPU and host memory, their last access, radix node (pages inserted together),
        # lock count and children, the unlocked GPU pages and the LRU heaps of evictable leaves.
        self.dev, self.host = set(), set()
        self.last = {}
        self.seg = {}
        self.next_seg = 1
        self.lock = defaultdict(int)
        self.dev_children = defaultdict(int)
        self.tree_children = defaultdict(int)
        self.evict_cnt = 0
        self.dheap, self.hheap = [], []
        self.n_evicted = self.n_loaded = self.n_host_evicted = 0
        # Requests.
        self.st = [0] * n                # 0 future, 1 waiting, 2 prefilling, 3 decoding, 4 done
        self.kv = [0] * n                # KV tokens
        self.gen = [0] * n               # output tokens
        self.locked = [0] * n            # path pages locked in the tree
        self.priv = [0] * n              # pages outside the tree
        self.priv_total = 0
        self.first_tok = [None] * n      # first token at the frontend
        self.finish = [None] * n
        self.admit_t = [None] * n
        self.dev_hit = [0] * n
        self.host_hit = [0] * n
        self.queue_at = [0.] * n
        free = 0.
        for i in sorted(range(n), key=lambda i: (jobs[i]['arrival'], i)):
            start = max(free, jobs[i]['arrival'] + cfg.client_upload + cfg.d_in)
            free = start + cfg.d_in_per_token * jobs[i]['prompt']
            self.queue_at[i] = free
        self.order = sorted(range(n), key=lambda i: (self.queue_at[i], i))
        self.next_arr = 0
        self.recv_t = [None] * n         # entry into the waiting queue
        self.waiting = []
        self.running = []
        self.chunked = None              # SGLang's request in chunked prefill
        self.partials = []               # planner: started prefills, which hold their slot and KV
        self.batch_is_full = False
        self.pq = {}                     # PQ+aging: request -> [estimate, level, order, waiting since]
        self.pq_seq = 0
        self.ratio_min = cfg.ratio_init * cfg.ratio_min_factor
        self.ratio_decay = (cfg.ratio_init - self.ratio_min) / cfg.ratio_decay_steps
        self.ratio = cfg.ratio_init
        self.t = 0.
        self.prev_gpu_overlap = 0.
        self.pending_first = []
        # Planner state.
        self._ov = None
        self._round = None               # None, 'solve' or 'forced' (decode-only proven by the certificate)
        self._extra_cpu = 0.
        self._cert_key = None
        self._cert_hit = False
        self._slack = 0.
        self._prev_prefill = False
        self._dispatch = 0.
        self._rng = random.Random(1000003 * (cfg.slk_seed + 1) + 17)
        self._noise_rng = random.Random(7919 * (cfg.slk_seed + 1) + 3)
        self._cert_rng = random.Random(1000033 * (cfg.slk_seed + 1) + 29)
        self._done_by_class, self._done_all = {}, []   # output lengths of completed requests
        self._veto_used, self._veto_run = {}, 0

    # ------------------------------------------------------------------ prefix cache
    @property
    def free_pages(self):
        return self.cap - len(self.dev) - self.priv_total

    def _touch(self, pid, t):
        self.last[pid] = t
        if pid in self.dev and self.lock[pid] == 0 and self.dev_children[pid] == 0:
            heapq.heappush(self.dheap, (t, -self.depth[pid], pid))
        if pid in self.host and pid not in self.dev and self.tree_children[pid] == 0:
            heapq.heappush(self.hheap, (t, -self.depth[pid], pid))

    def _in_tree(self, pid):
        return pid in self.dev or pid in self.host

    def _add_dev(self, pid, t):
        """The page becomes resident in GPU memory, by insertion or load-back."""
        if pid in self.dev:
            return
        was = self._in_tree(pid)
        self.dev.add(pid)
        par = self.parent[pid]
        if par >= 0:
            self.dev_children[par] += 1
            if not was:
                self.tree_children[par] += 1
        if self.lock[pid] == 0:
            self.evict_cnt += 1
        self.last[pid] = max(self.last.get(pid, t), t)
        self._touch(pid, self.last[pid])

    def _lock_range(self, path, a, b, delta):
        for pid in path[a:b]:
            old = self.lock[pid]
            self.lock[pid] = old + delta
            if pid in self.dev:
                if old == 0 and delta > 0:
                    self.evict_cnt -= 1
                elif old + delta == 0:
                    self.evict_cnt += 1
                    self._touch(pid, self.last.get(pid, self.t))

    def _drop_from_tree(self, pid):
        par = self.parent[pid]
        if par >= 0:
            self.tree_children[par] -= 1
            if self.tree_children[par] == 0:
                self._touch(par, self.last.get(par, self.t))

    def evict_device(self, need):
        """Evict unlocked GPU leaves in LRU order, one radix node at a time. Pages backed up to host memory stay there,
        the others leave the cache. Returns the pages freed."""
        freed = 0
        while freed < need and self.dheap:
            t0, nd, pid = heapq.heappop(self.dheap)
            if (pid not in self.dev or self.lock[pid] or self.dev_children[pid] or self.last.get(pid) != t0):
                continue
            chain = [pid]
            cur = pid
            while True:
                par = self.parent[cur]
                if (par < 0 or par not in self.dev or self.lock[par] or self.dev_children[par] != 1
                        or self.last.get(par) != t0 or self.tree_children[par] != 1
                        or self.seg.get(par) != self.seg.get(cur)):
                    break
                chain.append(par)
                cur = par
            for p in chain:
                self.dev.discard(p)
                self.evict_cnt -= 1
                par = self.parent[p]
                if par >= 0:
                    self.dev_children[par] -= 1
                if p in self.host:
                    self._touch(p, self.last[p])
                else:
                    self._drop_from_tree(p)
                freed += 1
            top = self.parent[chain[-1]]
            if top >= 0 and top in self.dev and self.dev_children[top] == 0:
                self._touch(top, self.last.get(top, t0))
        self.n_evicted += freed
        return freed

    def evict_host(self, need):
        """Evict host leaves that are not in GPU memory, in LRU order, one radix node at a time."""
        freed = 0
        while freed < need and self.hheap:
            t0, nd, pid = heapq.heappop(self.hheap)
            if pid not in self.host or pid in self.dev or self.tree_children[pid] or self.last.get(pid) != t0:
                continue
            cur = pid
            while True:
                par = self.parent[cur]
                self.host.discard(cur)
                self._drop_from_tree(cur)
                freed += 1
                if (par < 0 or par not in self.host or par in self.dev or self.tree_children[par] != 0
                        or self.last.get(par) != t0 or self.seg.get(par) != self.seg.get(cur)):
                    break
                cur = par
        self.n_host_evicted += freed
        return freed

    def backup(self, pages):
        """Write-through: copy new GPU pages to host memory, evicting host pages if it is full."""
        todo = [p for p in pages if p not in self.host]
        if not todo:
            return
        room = self.hcap - len(self.host)
        if room < len(todo):
            self.evict_host(len(todo) - room)
            room = self.hcap - len(self.host)
        for p in todo[:max(0, room)]:
            self.host.add(p)

    def alloc(self, pages, path='chunk'):
        """Allocate pages outside the tree, evicting if needed. Returns (ok, eviction CPU seconds)."""
        cost = 0.
        c = self.cfg
        if pages > self.free_pages:
            self.evict_device(pages - self.free_pages)
            cost = {'chunk': c.cpu_evict_chunk, 'loadback': c.cpu_evict_loadback, 'decode': c.cpu_evict_decode}[path]
        if pages > self.free_pages:
            return False, cost
        return True, cost

    def match(self, i):
        """(prefix pages in GPU memory, the following pages in host memory), at most (prompt - 1) // page pages."""
        j = self.jobs[i]
        path = j['path']
        cap = min(len(path), (j['prompt'] - 1) // self.cfg.page)
        d = 0
        while d < cap and path[d] in self.dev:
            d += 1
        h = 0
        while d + h < cap and path[d + h] in self.host and path[d + h] not in self.dev:
            h += 1
        return d, h

    def touch_path(self, i, n, t):
        path = self.jobs[i]['path']
        for pid in path[:n]:
            if self._in_tree(pid):
                self._touch(pid, t)

    def evictable_tokens(self):
        return self.evict_cnt * self.cfg.page

    def avail_tokens(self):
        return self.free_pages * self.cfg.page + self.evictable_tokens()

    # ------------------------------------------------------------------ SGLang admission
    def prefill_pass(self, t):
        """SGLang's prefill admission (PrefillAdder): walk the ordered waiting queue and admit each request whose whole
        uncached prompt and reserved output fit in the free and evictable KV; stop at the first that does not.
        Returns ([(request, prefill tokens, cached tokens, loaded pages)], CPU seconds)."""
        c = self.cfg
        P = c.page
        cpu = 0.
        if (self.batch_is_full or not self.waiting) and self.chunked is None:
            return [], cpu
        running_bs = len(self.running)
        if c.max_running - running_bs <= 0 and self.chunked is None:
            self.batch_is_full = True
            return [], cpu
        wait_kseg = sum(self.segs[i] for i in self.waiting) / 1e3 if c.cpu_admit_per_kseg else 0.
        if c.policy == 'lpm' and len(self.waiting) > 128:
            wait_kseg = 0.     # above 128 waiting requests LPM keeps the queue order without prefix matching
        cpu += c.cpu_admit + c.cpu_admit_per_kseg * wait_kseg
        m = {}
        touch_all = c.policy != 'fcfs'   # LPM and PQ+aging match every waiting request, FCFS only those it examines
        for i in self.waiting:
            d, h = self.match(i)
            m[i] = (d, h)
            if touch_all:
                self.touch_path(i, d + h, t)
        if c.policy == 'pq-aging':
            self._pq_order(t, m)
        if c.policy == 'lpm' and len(self.waiting) <= 128:
            if c.cpu_lpm_cold_per_mtok:
                cpu += c.cpu_lpm_cold_per_mtok * sum(self.jobs[i]['prompt'] for i in self.waiting
                                                     if (m[i][0] + m[i][1]) * P <= c.in_batch_threshold) / 1e6
            depri = set()
            seen_first = set()
            for i in self.waiting:
                d, h = m[i]
                if d * P <= c.in_batch_threshold:
                    path = self.jobs[i]['path']
                    key = path[0] if path else None
                    if key is not None and key in seen_first:
                        depri.add(i)
                    elif key is not None:
                        seen_first.add(key)
            maxp = {i: min((m[i][0] + m[i][1]) * P, self.jobs[i]['prompt'] - 1) for i in self.waiting}
            self.waiting.sort(key=lambda i: (math.inf if i in depri else -maxp[i]))
        mixed_tokens = running_bs        # each running decode adds one token to a mixed step
        rem_input = c.max_prefill - mixed_tokens
        rem_chunk = c.chunk - mixed_tokens
        off = mixed_tokens + sum(min(self.jobs[r]['output'] - self.gen[r], c.clip) * self.ratio for r in self.running)
        cur_off = mixed_tokens
        out = []

        def rem_total():
            return self.avail_tokens() - off

        def cur_rem():
            return self.avail_tokens() - cur_off

        def ceilp(x):
            return -(-x // P) * P

        if self.chunked is not None:
            i = self.chunked
            j = self.jobs[i]
            rem = min(rem_chunk, int(rem_total()))
            if rem <= 0:
                rem = rem_chunk
            ext = j['prompt'] - self.kv[i]
            qk = min(ext, rem)
            trunc = ext > rem
            off += ceilp(qk) + (0 if trunc else min(j['output'], c.clip)) + P
            cur_off += ceilp(qk) + P
            rem_input -= ceilp(qk)
            rem_chunk -= ceilp(qk)
            out.append((i, qk, self.kv[i], 0))
            if not trunc:
                self.chunked = None
        new_chunked = None
        no_token = False
        admitted = []
        exam_seg = 0
        for i in list(self.waiting):
            exam_seg += self.segs[i]
            if len(out) >= c.max_running - running_bs:
                self.batch_is_full = True
                break
            j = self.jobs[i]
            d, h = self.match(i)
            if not touch_all:
                self.touch_path(i, d + h, t)
            max_new = min(j['output'], c.clip)
            total = j['prompt'] - d * P + max_new + P
            if total >= rem_total():
                no_token = True
                break
            self._lock_range(j['path'], 0, d, +1)
            if total >= rem_total():
                self._lock_range(j['path'], 0, d, -1)
                no_token = True
                break
            loaded = 0
            if h:
                ok, ecost = self.alloc(h, 'loadback')
                if not ok:
                    self._lock_range(j['path'], 0, d, -1)
                    break
                for pid in j['path'][d:d + h]:
                    self._add_dev(pid, t)
                self._lock_range(j['path'], d, d + h, +1)
                loaded = h
                cpu += ecost + c.cpu_loadback_call + c.cpu_loadback_per_token * h * P
                self.n_loaded += h
            prefix = (d + h) * P
            inp = ceilp(j['prompt'] - prefix)
            if inp <= rem_chunk:
                qk = j['prompt'] - prefix
                off += inp + max_new + P
                cur_off += inp + P
                rem_input -= inp
                rem_chunk -= inp
                chunk_req = False
            else:
                trunc = rem_chunk // P * P
                trunc = (trunc + prefix) // P * P - prefix
                if trunc <= 0:
                    self._lock_range(j['path'], 0, d + h, -1)
                    break
                qk = trunc
                off += trunc + P
                cur_off += trunc + P
                rem_input -= trunc
                rem_chunk -= trunc
                chunk_req = True
            self.locked[i] = d + h
            self.kv[i] = prefix
            self.dev_hit[i] = d * P
            self.host_hit[i] = h * P
            self.admit_t[i] = t
            self.st[i] = 2
            admitted.append(i)
            out.append((i, qk, prefix, loaded))
            if chunk_req:
                new_chunked = i
                break
            if rem_total() <= 0 or cur_rem() <= 0:
                no_token = True
                break
            if rem_input <= 0 or rem_chunk <= 0:
                break
        if no_token:
            # SGLang marks the batch full and skips admission until a request finishes.
            self.batch_is_full = bool(admitted) or bool(self.running) or self.chunked is not None
        for i in admitted:
            self.waiting.remove(i)
        if new_chunked is not None:
            self.chunked = new_chunked
        cpu += c.cpu_admit_per_kseg_exam * exam_seg / 1e3
        return out, cpu

    def _pq_order(self, t, m):
        """PQ+aging: a level fixed at queue entry from the estimated prefill time, arrival order within a level, and
        the top level after pq_aging_s of waiting."""
        c = self.cfg
        P = c.page
        a, b, cc = c.pq_cost
        for i in sorted(self.waiting, key=lambda i: self.queue_at[i]):
            if i in self.pq:
                continue
            j = self.jobs[i]
            d, h = m[i]
            matched = min((d + h) * P, (j['prompt'] - 1) // P * P)
            rem, ctx, est, extra = j['prompt'] - matched, matched, 0., 0
            while rem:
                qk = min(rem, c.chunk)
                est += a + b * qk + cc * qk * (ctx + (qk + 1) / 2) / 1e9
                extra += int(not (1 <= qk <= 8192 and ctx <= c.pq_domain_max_ctx))
                ctx += qk
                rem -= qk
            lvl = 2 if extra else (0 if est <= c.pq_thresholds[0] else (1 if est <= c.pq_thresholds[1] else 2))
            self.pq[i] = [est, lvl, self.pq_seq, self.queue_at[i]]
            self.pq_seq += 1
        prom = [i for i in self.waiting if self.pq[i][1] > 0 and t - self.pq[i][3] >= c.pq_aging_s]
        for i in sorted(prom, key=lambda i: (self.pq[i][3], self.pq[i][2])):
            e = self.pq[i]
            e[1] = 0
            e[2] = self.pq_seq
            self.pq_seq += 1
            e[3] = t
        self.waiting.sort(key=lambda i: (self.pq[i][1], self.pq[i][2]))

    # ------------------------------------------------------------------ planner and Caesura
    def _planner_init(self):
        """Load the planner modules of gpu/overlay, fresh for every simulation."""
        if self._ov is not None:
            return
        d = Path(__file__).resolve().parents[1] / 'gpu' / 'overlay'
        mods = {}
        for name in ('b300_page_resources_v9', 'b300_page_objectives_v14'):
            spec = importlib.util.spec_from_file_location('sim_' + name, d / (name + '.py'))
            m = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = m
            spec.loader.exec_module(m)
            mods[name] = m
        self._res = mods['b300_page_resources_v9']
        self._ov = mods['b300_page_objectives_v14']
        self._ov_model = self._ov.CostModel()   # step-time model, updated from every executed step as on the GPU
        self._ov_rounds = 0
        self._ov_allowance = 0.008              # planning-time estimate added to the predicted step start
        self._ov_last = None

    def _planner_clock(self):
        """Virtual clock of one solve: every reading advances it by b01_real_clock_tick_s."""
        tick = self.cfg.b01_real_clock_tick_s
        state = [0]

        def clock():
            state[0] += 1
            return state[0] * tick
        return clock

    def _planner_solve(self, rows, lim, wait):
        c = self.cfg
        return self._ov.solve(rows, lim, self._ov_model, budget_s=c.b01_real_budget_s,
                              planning_allowance_s=self._ov_allowance, predicted_wait_s=wait,
                              clock=self._planner_clock(), rotation_round=self._ov_rounds)

    def _planner_rows(self, t):
        """One planner row per waiting request and paused partial prefill, in queue order."""
        c, P = self.cfg, self.cfg.page
        ov = self._ov
        order = sorted(list(self.partials) + list(self.waiting), key=lambda i: (self.queue_at[i], i))
        rows, ids, extra = [], [], []
        for k, i in enumerate(order):
            j = self.jobs[i]
            guard = self._res.future_output_guard(int(j['output']), int(c.clip))
            path = tuple(j['path'])
            if i in self.partials:
                gpu, host, locks, d, h = self.kv[i], 0, (), None, None
            else:
                d, h = self.match(i)
                self.touch_path(i, d + h, t)
                gpu, host = d * P, h * P
                if host < 10:   # HiCache does not load back fewer than 10 tokens
                    host, h = 0, 0
                if gpu + host >= j['prompt']:   # at least one prompt token is computed
                    if h:
                        h -= 1; host = h * P
                    else:
                        d -= 1; gpu = d * P
                locks = tuple((pid, P) for pid in j['path'][:d] if self.lock[pid] == 0)
            age = max(0., t - self.recv_t[i])
            rows.append(ov.Row(k, age, int(j['prompt']), int(gpu), int(host),
                               max(0, int(j['output'])), i in self.partials, locks, path, guard))
            ids.append(i); extra.append((d, h))
        return rows, ids, extra

    def _planner_limits(self):
        """The resources of the next step: token budget, KV left after the decode reserve, slots and decodes."""
        c, P = self.cfg, self.cfg.page

        class _SP:
            __slots__ = ('max_new_tokens',)

        class _Req:
            __slots__ = ('sampling_params', 'output_ids')
        decs = []
        for r in self.running:
            q = _Req(); q.sampling_params = _SP(); q.sampling_params.max_new_tokens = int(self.jobs[r]['output'])
            q.output_ids = range(int(self.gen[r]))
            decs.append(q)
        nd = len(self.running)
        available = max(0, int(self.free_pages * P + self.evictable_tokens()))
        required = sum(P for r in self.running if self.kv[r] % P == 0)
        budget = self._res.resource_budget(available, int(required), decs, float(self.ratio), int(c.clip), True)
        return self._ov.Limits(min(8192 - nd, 16384 - nd), max(0, int(available - budget['reserved_tokens'])),
                               max(0, c.b01_real_slots - nd - len(self.partials)), max(0, 128 - nd), nd,
                               int(sum(self.kv[r] for r in self.running)), c.b01_time_cap_s)

    def planner_pass(self, t):
        """One planning round: the planner's solve, then Caesura's release-aware rollout and veto, then the schedule
        is applied. Returns [(request, prefill tokens, cached tokens, loaded pages)]."""
        c = self.cfg
        self._planner_init()
        self._round = None
        self._extra_cpu = 0.
        if not self.waiting and not self.partials:
            return []
        self._round = 'solve'
        rows, ids, extra = self._planner_rows(t)
        lim = self._planner_limits()
        wait = self.prev_gpu_overlap + c.b01_real_prep_s
        decision = self._planner_solve(rows, lim, wait)
        # The decode-only certificate: no waiting request fits, so the round runs decode only without a search.
        forced = decision.get('status') == 'forced_resource_decode' or (
            decision.get('status') == 'search_exhausted' and not decision.get('action'))
        if forced:
            self._round = 'forced'
        # A certificate is reused while the queue, the partials, the decodes and the cache state stay the same.
        key = (tuple(self.running), tuple(self.waiting), tuple(self.partials), self.n_evicted, self.n_loaded,
               self.n_host_evicted)
        self._cert_hit = forced and self._cert_key == key
        self._cert_key = key if forced else None
        if not forced:
            self._ov_allowance = .9 * self._ov_allowance + .1 * c.slk_search_s
        self._ov_rounds += 1
        action = decision['action']
        if c.rc_steps and action and not forced and self.running:
            action = self._release_check(rows, lim, wait, dict(action))
        if c.veto_tokens and action and not forced:
            action = self._veto(rows, ids, lim, wait, dict(action))
        return self._apply(t, rows, ids, extra, lim, action)

    def _remaining_output(self, r):
        """Predicted remaining output tokens of decode r. Among the completed requests with the same first prompt page,
        or all completed requests if fewer than rc_len_min_n of those are longer than r's output so far, it takes the
        rc_len_q quantile of the outputs longer than r's output so far, minus that output."""
        c = self.cfg
        g = int(self.gen[r])
        for v in (self._done_by_class.get(tuple(self.jobs[r]['path'][:1]), ()), self._done_all):
            k = bisect.bisect_right(v, g)
            n = len(v) - k
            if n >= c.rc_len_min_n or (v is self._done_all and n > 0):
                return v[k + min(n - 1, int(c.rc_len_q * n))] - g
        return 1 << 30

    def _release_check(self, rows, lim, wait, action):
        """Release-aware rollout: for each look-ahead of j decode steps, release the KV and slots of the decodes
        predicted to finish within j steps and solve again on that state. Wait with a decode-only step instead of the
        planner's prefill if this puts at least rc_margin more requests on time."""
        c, O, model = self.cfg, self._ov, self._ov_model
        elig = [r for r in rows if r.age_s <= DEADLINE_S]
        if not elig or (c.rc_min_decoders and len(self.running) < c.rc_min_decoders):
            return action

        def n_ontime(rws, act, lm, extra_s):
            _, done = O.remaining_after(rws, act)
            el = wait + extra_s + model.predict(rws, act, lm)
            return sum(1 for r in elig if r.index in done and r.age_s + el <= DEADLINE_S)
        now = n_ontime(rows, action, lim, 0.)
        step = model.decode_s + model.launch_s
        for j in c.rc_steps:
            gone = [r for r in self.running if self._remaining_output(r) <= j]
            if not gone:
                continue
            dt = j * step
            lim2 = replace(lim, slots=lim.slots + len(gone), batch_slots=lim.batch_slots + len(gone),
                           decode_count=max(0, lim.decode_count - len(gone)),
                           kv_budget=lim.kv_budget + sum(self.kv[r] for r in gone),
                           decode_context=max(0, lim.decode_context - sum(self.kv[r] for r in gone)))
            rows2 = tuple(replace(r, age_s=r.age_s + dt) for r in rows)
            dec2 = self._planner_solve(rows2, lim2, wait)
            if c.rc_cost_scale:
                self._extra_cpu += c.rc_cost_scale * c.slk_search_s
            if n_ontime(rows2, dict(dec2['action']), lim2, dt) >= now + c.rc_margin:
                return {}
        return action

    def _veto(self, rows, ids, lim, wait, action):
        """Thin-step veto: run decode only instead of a small prefill step under KV pressure that completes no request
        on time, if every request that can still be on time keeps veto_slack_s plus one step of slack."""
        c, O, model = self.cfg, self._ov, self._ov_model
        if (len(self.running) < c.veto_min_decoders or sum(action.values()) >= c.veto_tokens
                or lim.kv_budget >= c.veto_tokens or self._veto_run >= c.veto_max_steps):
            self._veto_run = 0
            return action
        elig = [r for r in rows if r.age_s <= DEADLINE_S]
        _, done = O.remaining_after(rows, action)
        el = wait + model.predict(rows, action, lim)
        step = model.decode_s + model.launch_s
        if (any(r.index in done and r.age_s + el <= DEADLINE_S for r in elig)
                or any(DEADLINE_S - r.age_s - wait < c.veto_slack_s + step for r in elig)):
            self._veto_run = 0
            return action
        if any(self._veto_used.get(ids[k], 0.) + step > c.veto_req_budget_s for k in action):
            self._veto_run = 0
            return action
        for k in action:
            self._veto_used[ids[k]] = self._veto_used.get(ids[k], 0.) + step
        self._veto_run += 1
        return {}

    def _apply(self, t, rows, ids, extra, lim, action):
        """Apply a schedule {row: prefill tokens}. A paused partial prefill that the schedule does not continue
        releases its KV: its computed prefix enters the cache, backed up to host memory, and it rejoins the queue."""
        P = self.cfg.page
        self._ov_last = (rows, dict(action), lim)
        sel = {ids[k] for k in action}
        for i in [i for i in self.partials if i not in sel]:
            j = self.jobs[i]
            self.backup(j['path'][:self.locked[i]])
            self._lock_range(j['path'], 0, self.locked[i], -1)
            self.locked[i] = 0
            self.priv_total -= self.priv[i]; self.priv[i] = 0; self.kv[i] = 0
            self.st[i] = 1; self.partials.remove(i); self.waiting.append(i)
        out = []
        for k, q in sorted(action.items()):
            i = ids[k]; row = rows[k]; j = self.jobs[i]
            if row.held_slot:
                out.append((i, q, self.kv[i], 0))
                continue
            d, h = extra[k]
            self._lock_range(j['path'], 0, d, +1)
            loaded = 0
            if h:
                ok, _ = self.alloc(h, 'loadback')
                if not ok:
                    self._lock_range(j['path'], 0, d, -1)
                    continue
                for pid in j['path'][d:d + h]:
                    self._add_dev(pid, t)
                self._lock_range(j['path'], d, d + h, +1)
                loaded = h
                self.n_loaded += h
            self.locked[i] = d + h
            self.kv[i] = (d + h) * P
            if self.admit_t[i] is None:
                self.dev_hit[i] = d * P; self.host_hit[i] = h * P; self.admit_t[i] = t
            self.st[i] = 2
            self.waiting.remove(i)
            self.partials.append(i)
            out.append((i, q, self.kv[i], loaded))
        return out

    def _planner_observe(self, gpu):
        """Update the planner's step-time model with the GPU time of the executed step."""
        if self._ov_last is None:
            return
        rows, action, lim = self._ov_last
        self._ov_last = None
        self._ov_model.observe(self._ov_model.features(rows, action, lim), len(self.running), float(gpu))

    # ------------------------------------------------------------------ step time
    def decode_time(self, ctxs):
        c = self.cfg
        R = len(ctxs)
        tot = (sum(ctxs) + R) / 1e6
        for b, size in enumerate(c.decode_buckets):
            if R <= size:
                return c.decode_bucket_base[b] + c.decode_ctx_s_per_mtok * tot
        raise ValueError('more decodes than the largest CUDA-graph batch size')

    def mixed_time(self, prefills, ctxs):
        qs = sum(q for _, q, _, _ in prefills)
        attn = sum(q * (cc + q / 2.) for _, q, cc, _ in prefills) / 1e9
        dctx = (sum(ctxs) + len(ctxs)) / 1e6
        Aa, B, C, e, d = self.cfg.roofline
        comp = max(Aa, B + C * qs) + e * attn + d * dctx + self.cfg.mixed_nd_q_s * len(ctxs) * qs / 8192.
        load = sum(l for *_, l in prefills) * self.cfg.page * self.cfg.loadback_s_per_token
        return max(comp, load)

    # ------------------------------------------------------------------ main loop
    def _arrivals(self, t):
        while self.next_arr < len(self.order) and self.queue_at[self.order[self.next_arr]] <= t:
            i = self.order[self.next_arr]
            self.st[i] = 1
            self.waiting.append(i)
            self.recv_t[i] = t
            self.next_arr += 1

    def _finish(self, i, t):
        c = self.cfg
        j = self.jobs[i]
        o = int(j['output'])
        bisect.insort(self._done_all, o)
        bisect.insort(self._done_by_class.setdefault(tuple(j['path'][:1]), []), o)
        path = j['path']
        self._lock_range(path, 0, self.locked[i], -1)
        # The finished request's prompt and output enter the cache; its pages beyond the prompt are new leaves.
        kv_final = j['prompt'] + j['output'] - 1
        ntail = kv_final // c.page - len(path)
        newp = []
        par = path[-1] if path else -1
        sg = self.next_seg
        self.next_seg += 1
        for _ in range(max(0, ntail)):
            pid = self.next_pid
            self.next_pid += 1
            self.parent[pid] = par
            self.depth[pid] = (self.depth[par] + 1) if par >= 0 else 0
            self.seg[pid] = sg
            self._add_dev(pid, t)
            newp.append(pid)
            par = pid
        self.backup(newp)
        self.priv_total -= self.priv[i]
        self.priv[i] = 0
        self.st[i] = 4
        self.finish[i] = t
        self.locked[i] = 0

    def _insert_prompt_pages(self, i, t, final):
        """After a chunk, the prompt pages it completed enter the tree, locked by the request."""
        j = self.jobs[i]
        path = j['path']
        full = min(self.kv[i] // self.cfg.page, len(path))
        a = self.locked[i]
        if full > a:
            sg = self.next_seg
            self.next_seg += 1
            for pid in path[a:full]:
                if pid not in self.dev:     # a duplicate of an already cached prefix is freed
                    if not self._in_tree(pid):
                        self.seg[pid] = sg
                    self._add_dev(pid, t)
            self._lock_range(path, a, full, +1)
            self.locked[i] = full
        if final:
            self.backup(path[:full])
        newpriv = -(-self.kv[i] // self.cfg.page) - self.locked[i]
        self.priv_total += newpriv - self.priv[i]
        self.priv[i] = newpriv

    def run(self, t_end=None):
        c = self.cfg
        n = len(self.jobs)
        done = 0
        while done < n:
            t = self.t
            if t_end is not None and t >= t_end:
                return
            self._arrivals(t)
            if not self.waiting and not self.running and self.chunked is None and not self.partials:
                # Idle: emit the pending first tokens and jump to the next arrival.
                for i in self.pending_first:
                    self._emit_first(i, t)
                self.pending_first = []
                self.ratio = c.ratio_init
                if self.next_arr >= len(self.order):
                    break
                self.t = max(t, self.queue_at[self.order[self.next_arr]])
                self.prev_gpu_overlap = 0.
                self._slack = 0.
                self._prev_prefill = False
                continue
            idle = 0.
            if c.policy == 'b01':
                nwait = len(self.waiting) + len(self.partials)
                prefills = self.planner_pass(t)
                cpu = 0.
            else:
                cpu = c.cpu_base
                prefills, pcpu = self.prefill_pass(t)
                cpu += pcpu
            ctxs = [self.kv[r] for r in self.running]
            kind = 'decode'
            ecpu = 0.
            if prefills:
                kind = 'mixed' if self.running else 'extend'
                # Pages for the chunks and one decode token per running request.
                need = 0
                for i, q, pre, loaded in prefills:
                    need += -(-(self.kv[i] + q) // c.page) - (-(-self.kv[i] // c.page))
                need += sum(1 for r in self.running if self.kv[r] % c.page == 0)
                ok, ecost = self.alloc(need, 'chunk')
                ecpu += ecost
                gpu = self.mixed_time(prefills, ctxs)
                if c.policy == 'b01':
                    if c.slk_gpu_noise > 0:
                        gpu *= math.exp(c.slk_gpu_noise * self._noise_rng.gauss(0., 1.) - 0.5 * c.slk_gpu_noise ** 2)
                    # Prefill steps run without CUDA graphs and cannot end before their CPU dispatch does.
                    dres = c.slk_disp_res[self._rng.randrange(len(c.slk_disp_res))] if c.slk_disp_res else 0.
                    self._dispatch = max(0.02, c.slk_disp[0] + c.slk_disp[1] * len(self.running) / 100.
                                         + c.slk_disp[2] * sum(x[1] for x in prefills) / 8192. + dres)
                    gpu = max(gpu, self._dispatch - 0.001)
            elif self.running:
                need = sum(1 for r in self.running if self.kv[r] % c.page == 0)
                ok, ecost = self.alloc(need, 'decode')
                ecpu += ecost
                if not ok:
                    self._retract(t)
                    ctxs = [self.kv[r] for r in self.running]
                gpu = self.decode_time(ctxs)
                self.ratio = max(self.ratio - self.ratio_decay, self.ratio_min)
            else:
                # Requests wait, but none can start and nothing runs: try again shortly.
                self.t = t + max(cpu, 1e-3)
                if self.next_arr < len(self.order):
                    self.t = max(self.t, min(self.t + 0.01, self.queue_at[self.order[self.next_arr]]))
                continue
            cpu += ecpu
            if c.policy != 'b01':
                if kind == 'decode' and c.decode_exposed_per_kseg and self.waiting:
                    idle += c.decode_exposed_per_kseg * sum(self.segs[i] for i in self.waiting) / 1e3
            else:
                idle = self._planning_idle(prefills, nwait)
                cpu = 0.
                self.prev_gpu_overlap = 0.
            dur = gpu + max(0., cpu - self.prev_gpu_overlap) + idle
            launch_done = t + max(0., cpu - self.prev_gpu_overlap) + idle + (gpu if kind != 'decode' else 0.)
            # One-step lag of overlap scheduling: the first tokens of the previous step are processed after this
            # step is launched.
            for i in self.pending_first:
                self._emit_first(i, launch_done)
            self.pending_first = []
            te = t + dur
            if c.policy == 'b01' and gpu > 0:
                self._planner_observe(gpu)
            self.prev_gpu_overlap = gpu if kind == 'decode' else 0.
            if c.policy == 'b01':
                self._prev_prefill = kind != 'decode'
                self._slack = gpu - self._dispatch if kind != 'decode' else gpu
            # Effects at the end of the step.
            dec = list(self.running)
            for i, q, pre, loaded in prefills:
                self.kv[i] += q
                final = self.kv[i] >= self.jobs[i]['prompt']
                self._insert_prompt_pages(i, te, final)
                if final and i in self.partials:
                    self.partials.remove(i)
                if final:
                    self.gen[i] = 1
                    self.pending_first.append(i)
                    if self.jobs[i]['output'] <= 1:
                        self._finish(i, te)
                        done += 1
                    else:
                        self.st[i] = 3
                        self.running.append(i)
            finished_any = False
            for r in dec:
                self.kv[r] += 1
                self.gen[r] += 1
                newpriv = -(-self.kv[r] // c.page) - self.locked[r]
                self.priv_total += newpriv - self.priv[r]
                self.priv[r] = newpriv
                if self.gen[r] >= self.jobs[r]['output']:
                    self._finish(r, te)
                    finished_any = True
                    done += 1
            if finished_any:
                self.running = [r for r in self.running if self.st[r] == 3]
                self.batch_is_full = False
            if prefills and (self.chunked is not None or any(self.jobs[i]['output'] <= 1 for i, *_ in prefills)):
                self.batch_is_full = False
            self.t = te
        for i in self.pending_first:
            self._emit_first(i, self.t)
        self.pending_first = []

    def _planning_idle(self, prefills, nwait):
        """GPU idle before a step of the planner: max(floor, R + X + p - s)."""
        c = self.cfg
        nd0 = len(self.running)
        pick = lambda res: res[self._rng.randrange(len(res))] if res else 0.
        R = None
        if self._round is not None and prefills:
            co = c.slk_Rp
            hk = sum(x[3] for x in prefills) * c.page
            R = (co[0] + co[1] * nwait / 100. + co[2] * len(prefills) / 10. + co[3] * hk / 1e5 + co[4] * nd0 / 100.
                 + c.slk_Rp_off + pick(c.slk_Rp_res))
            R -= c.slk_R_cut_P
            pp = c.slk_pp
            floor = c.slk_i0[0] + c.slk_i0[1] * nd0 / 100.
        elif self._round == 'forced':
            hit = self._cert_hit
            co = c.slk_cert[0 if hit else 1]
            x3 = nd0 / 100. if hit else sum(self.jobs[i]['prompt'] for i in list(self.waiting) + list(self.partials)) / 1e6
            res = c.slk_cert[2 if hit else 3]
            R = co[0] + co[1] * nwait / 100. + co[2] * x3 + (res[self._cert_rng.randrange(len(res))] if res else 0.)
            if c.slk_Rf_res:
                self._rng.randrange(len(c.slk_Rf_res))
            R -= c.slk_R_cut_Fh if hit else c.slk_R_cut_Fm
            pp = c.slk_pf
            floor = c.slk_i0_d
        elif self._round is not None:
            co = c.slk_Rd
            R = co[0] + co[1] * nwait / 100. + co[2] * nd0 / 100. + pick(c.slk_Rd_res)
            R -= c.slk_R_cut_D
            pp = c.slk_pd
            floor = c.slk_i0_d
        if R is None:
            idle = c.slk_nr_after_p if self._prev_prefill else c.slk_nr
        else:
            R = max(0.002, R)
            idle = max(floor, R + self._extra_cpu + pp - self._slack)
        self._round = None
        self._extra_cpu = 0.
        return idle

    def _emit_first(self, i, t):
        if self.first_tok[i] is None:
            self.first_tok[i] = t + self.cfg.d_out

    def _retract(self, t):
        """SGLang's decode retraction when the decode pages run out (it did not occur in the GPU runs)."""
        c = self.cfg
        while self.running:
            need = sum(1 for r in self.running if self.kv[r] % c.page == 0)
            if need <= self.free_pages + self.evict_cnt or len(self.running) == 1:
                break
            r = sorted(self.running, key=lambda r: (self.gen[r], -self.jobs[r]['prompt']))[0]
            self.running.remove(r)
            j = self.jobs[r]
            self._lock_range(j['path'], 0, self.locked[r], -1)
            self.priv_total -= self.priv[r]
            self.priv[r] = 0
            self.locked[r] = 0
            self.kv[r] = 0
            self.gen[r] = 0
            self.st[r] = 1
            self.waiting.insert(0, r)
            total_dec = sum(self.gen[x] for x in self.running)
            total_max = sum(self.jobs[x]['output'] for x in self.running)
            self.ratio = min(1., (total_dec + 20 * len(self.running)) / (total_max + 1))
        self.alloc(sum(1 for r in self.running if self.kv[r] % c.page == 0))

    def results(self):
        """Per request: arrival, prompt and output tokens, first token at the frontend, TTFT, completion time and the
        prompt tokens found in the GPU and host prefix cache at admission."""
        out = []
        for i, j in enumerate(self.jobs):
            ft = self.first_tok[i]
            out.append(dict(id=j['id'], arrival_s=j['arrival'], prompt=j['prompt'], output=j['output'],
                            first_token_s=ft, ttft_s=(ft - j['arrival'] - self.cfg.client_upload) if ft is not None else None,
                            completion_s=self.finish[i], gpu_hit=self.dev_hit[i], host_hit=self.host_hit[i]))
        return out
