"""The release-aware rollout of Caesura and its thin-step veto, applied on rank 0 after the planner's solve.

Release-aware rollout (release_check): for each look-ahead of j decode steps, the decodes that the output-length
predictor expects to finish within j steps release their slots and KV, the waiting requests age by j steps, and the
planner solves once more on that state. If waiting for that release puts at least rc_margin more requests on time than
the planner's schedule now, the step runs decode only.

Output-length predictor (remaining): among the completed requests with the same first prompt page, or all completed
requests if fewer than rc_len_min_n of those are longer than the decode's output so far, the rc_len_q quantile of the
outputs longer than that output, minus it. It learns only from completed requests (note_finished).

Thin-step veto (veto): run decode only instead of a small prefill step under KV pressure that completes no request on
time, as long as every request that can still be on time keeps veto_slack_s plus one step of slack.

The simulator (simulator/native_sglang.py) implements the same rules.
"""
from __future__ import annotations

from array import array
import bisect
from dataclasses import replace
import hashlib
import math

from sglang.srt.managers.b300_page_objectives_v14 import PAGE, remaining_after, solve

ELIGIBLE_AGE_S = 3.              # TTFT SLO
SHORT_PROMPT_CLASS = ()          # prompts shorter than one page share one class
VETO_USED_MAX_ENTRIES = 4096
_CLASS_ATTR = '_b300_b01ext_class'
_RECORDED_ATTR = '_b300_b01ext_recorded'
FIELDS = ('rc_steps', 'rc_margin', 'rc_min_decoders', 'rc_len_min_n', 'rc_len_q', 'veto_tokens',
          'veto_min_decoders', 'veto_slack_s', 'veto_max_steps', 'veto_req_budget_s', 'tp_wire')


def class_key(req):
    """The class of a request for the predictor: a hash of its first 128-token page (with its cache namespace), or the
    shared class of prompts shorter than a page. Cached on the request."""
    key = getattr(req, _CLASS_ATTR, None)
    if key is None:
        ids = req.origin_input_ids
        if len(ids) >= PAGE:
            seed = hashlib.sha256(repr((getattr(req, 'extra_key', None),
                                        getattr(req, 'cache_salt', None) or None)).encode()).digest()
            key = (hashlib.sha256(seed + array('q', ids[:PAGE]).tobytes()).digest(),)
        else:
            key = SHORT_PROMPT_CLASS
        setattr(req, _CLASS_ATTR, key)
    return key


def _aborted(req):
    reason = getattr(req, 'finished_reason', None)
    return reason is not None and any(c.__name__ == 'FINISH_ABORT' for c in type(reason).__mro__)


def _finite(value):
    return type(value) in (int, float) and math.isfinite(value)


class ReleaseVeto:
    """The state of the rollout and the veto of one page scheduler: predictor samples and veto bookkeeping."""

    def __init__(self, ext):
        if type(ext) is not dict or set(ext) != set(FIELDS):
            raise ValueError('b01_ext must have exactly the fields ' + ', '.join(FIELDS))
        self.ext = dict(ext, rc_steps=list(ext['rc_steps']))
        self.rc_steps = tuple(int(j) for j in ext['rc_steps'])
        self.rc_margin = int(ext['rc_margin'])
        self.rc_min_decoders = int(ext['rc_min_decoders'])
        self.rc_len_min_n = int(ext['rc_len_min_n'])
        self.rc_len_q = float(ext['rc_len_q'])
        self.veto_tokens = int(ext['veto_tokens'])
        self.veto_min_decoders = int(ext['veto_min_decoders'])
        self.veto_slack_s = float(ext['veto_slack_s'])
        self.veto_max_steps = int(ext['veto_max_steps'])
        self.veto_req_budget_s = float(ext['veto_req_budget_s'])
        self.class_lengths = {}
        self.all_lengths = []
        self.veto_used = {}
        self.veto_run = 0

    # ---- output-length predictor -------------------------------------------------------------------------------
    def note_finished(self, reqs):
        """Record the output length of every request that finished normally, once. Never raises; an error skips only
        that request."""
        try:
            reqs = list(reqs)
        except Exception:
            return
        for req in reqs:
            try:
                if getattr(req, _RECORDED_ATTR, False) or not req.finished():
                    continue
                setattr(req, _RECORDED_ATTR, True)
                if _aborted(req) or getattr(req, 'is_retracted', False):
                    continue
                length = len(req.output_ids)
                key = class_key(req)          # before any insert, so that a failing request changes neither list
                bisect.insort(self.all_lengths, length)
                bisect.insort(self.class_lengths.setdefault(key, []), length)
            except Exception:
                pass

    def remaining(self, req):
        """Predicted remaining output tokens of a running decode."""
        g = len(req.output_ids)
        for v in (self.class_lengths.get(class_key(req), ()), self.all_lengths):
            k = bisect.bisect_right(v, g)
            n = len(v) - k
            if n >= self.rc_len_min_n or (v is self.all_lengths and n > 0):
                return v[k + min(n - 1, int(self.rc_len_q * n))] - g
        return 1 << 30

    # ---- release-aware rollout ---------------------------------------------------------------------------------
    def release_check(self, rows, limits, model, wait, action, decoders, solve_kwargs):
        elig = [r for r in rows if r.age_s <= ELIGIBLE_AGE_S]
        if not elig or (self.rc_min_decoders and limits.decode_count < self.rc_min_decoders):
            return action

        def n_ontime(rws, act, lm, extra_s):
            _, done = remaining_after(rws, act)
            el = wait + extra_s + model.predict(rws, act, lm)
            return sum(1 for r in elig if r.index in done and r.age_s + el <= ELIGIBLE_AGE_S)

        now = n_ontime(rows, action, limits, 0.)
        step = model.decode_s + model.launch_s
        credit = [(self.remaining(r), int(r.kv_committed_len)) for r in decoders]
        for j in self.rc_steps:
            gone = [kv for left, kv in credit if left <= j]
            if not gone:
                continue
            dt = j * step
            n, kv = len(gone), sum(gone)
            lim2 = replace(limits, slots=limits.slots + n, batch_slots=limits.batch_slots + n,
                           decode_count=max(0, limits.decode_count - n), kv_budget=limits.kv_budget + kv,
                           decode_context=max(0, limits.decode_context - kv))
            rows2 = tuple(replace(r, age_s=r.age_s + dt) for r in rows)
            dec2 = solve(rows2, lim2, model, **solve_kwargs)
            if n_ontime(rows2, dict(dec2['action']), lim2, dt) >= now + self.rc_margin:
                return {}
        return action

    # ---- thin-step veto ----------------------------------------------------------------------------------------
    def veto(self, rows, waiting_queue, limits, model, wait, action):
        if (limits.decode_count < self.veto_min_decoders or sum(action.values()) >= self.veto_tokens
                or limits.kv_budget >= self.veto_tokens or self.veto_run >= self.veto_max_steps):
            self.veto_run = 0
            return action
        elig = [r for r in rows if r.age_s <= ELIGIBLE_AGE_S]
        _, done = remaining_after(rows, action)
        el = wait + model.predict(rows, action, limits)
        step = model.decode_s + model.launch_s
        if (any(r.index in done and r.age_s + el <= ELIGIBLE_AGE_S for r in elig)
                or any(ELIGIBLE_AGE_S - r.age_s - wait < self.veto_slack_s + step for r in elig)):
            self.veto_run = 0
            return action
        rids = [waiting_queue[k].rid for k in action]
        if any(self.veto_used.get(rid, 0.) + step > self.veto_req_budget_s for rid in rids):
            self.veto_run = 0
            return action
        for rid in rids:
            self.veto_used[rid] = self.veto_used.get(rid, 0.) + step
        self.veto_run += 1
        if len(self.veto_used) > VETO_USED_MAX_ENTRIES:
            live = {req.rid for req in waiting_queue}
            self.veto_used = {rid: used for rid, used in self.veto_used.items() if rid in live}
        return {}


def solve_kwargs(st, wait):
    """The arguments of the planner's own solve, with its updated planning allowance and the next round number."""
    return dict(budget_s=st.search_budget_s, planning_allowance_s=st.planning_allowance_s, predicted_wait_s=wait,
                rotation_round=st.rounds + 1)


def decode_only(decision, rows, limits, model):
    """The planner's decision with the decode-only schedule; its score and predicted step time come from the planner's
    scored decode-only candidate when there is one."""
    new = dict(decision)
    new['action'] = {}
    diagnostics = decision.get('diagnostics') or {}
    raw = diagnostics.get('decode_raw_score')
    predicted = diagnostics.get('decode_predicted_execution_s')
    if (diagnostics.get('decode_scored') == 1 and diagnostics.get('decode_raw_score_finite') == 1
            and _finite(raw) and _finite(predicted)):
        new['score'] = raw
        new['predicted_execution_s'] = predicted
    else:
        new['predicted_execution_s'] = model.predict(rows, {}, limits)
    return new


def apply_round(rules, st, scheduler, running, rows, limits, decision, predicted_wait_s):
    """Rank 0, after the planner's solve on a round without the decode-only certificate: the release check first, then
    the veto if the schedule still prefills. Never raises; any error keeps the planner's decision."""
    try:
        action = dict(decision['action'])
        if not action:
            return decision
        model = st.model
        wait = float(predicted_wait_s)
        result = action
        if rules.rc_steps and limits.decode_count > 0:
            decoders = [r for r in running.reqs if not r.finished()]
            result = rules.release_check(rows, limits, model, wait, result, decoders, solve_kwargs(st, wait))
        if rules.veto_tokens and result:
            result = rules.veto(rows, scheduler.waiting_queue, limits, model, wait, result)
        if not result and limits.decode_count > 0:
            return decode_only(decision, rows, limits, model)
        return decision
    except Exception:
        return decision
