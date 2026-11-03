"""Dispatch between SGLang's own scheduling and the scheduler of Caesura.

gpu/serve.sh selects the method at startup. B300_PERSISTENT_METHOD is fcfs, lpm or pq-aging for SGLang's admission
with that queue order (mode native), or b01 for the planner (mode page); B300_PAGE_B01_EXT_JSON adds the release-aware
rollout and the veto of Caesura to the planner. The hooks in SGLang's scheduler call the functions below.
"""
from __future__ import annotations

import importlib
import json
import math
import os

CONFIG = None
METHODS = ('fcfs', 'lpm', 'pq-aging', 'b01')
B01_EXT_INT_FIELDS = ('rc_margin', 'rc_min_decoders', 'rc_len_min_n', 'veto_tokens', 'veto_min_decoders',
                      'veto_max_steps', 'tp_wire')
B01_EXT_FLOAT_FIELDS = ('rc_len_q', 'veto_slack_s', 'veto_req_budget_s')
B01_EXT_FIELDS = frozenset(('rc_steps', *B01_EXT_INT_FIELDS, *B01_EXT_FLOAT_FIELDS))


def _require(condition, reason):
    if not condition:
        raise ValueError(reason)


def validate_b01_ext(value):
    """Check the settings of Caesura on top of the planner and return them with sorted keys.

    rc_steps: increasing look-ahead step counts (empty: no rollout); rc_margin: requests the wait must put on time in
    addition; rc_min_decoders: decodes needed to look ahead; rc_len_q, rc_len_min_n: quantile and minimum class size of
    the output-length predictor; veto_*: the thin-step veto (veto_tokens 0: off); tp_wire: numeric exchange between the
    tensor-parallel ranks.
    """
    _require(type(value) is dict and set(value) == B01_EXT_FIELDS, 'b01_ext fields')
    steps = value['rc_steps']
    _require(type(steps) is list and len(steps) <= 16
             and all(type(j) is int and 1 <= j <= 4096 for j in steps)
             and all(a < b for a, b in zip(steps, steps[1:])), 'b01_ext rc_steps')
    for name in B01_EXT_INT_FIELDS:
        _require(type(value[name]) is int, 'b01_ext integer ' + name)
    for name in B01_EXT_FLOAT_FIELDS:
        _require(type(value[name]) is float and math.isfinite(value[name]), 'b01_ext float ' + name)
    _require(value['rc_margin'] >= 0 and value['rc_min_decoders'] >= 0 and value['rc_len_min_n'] >= 1
             and value['veto_tokens'] >= 0 and value['veto_min_decoders'] >= 1 and value['veto_max_steps'] >= 0
             and value['tp_wire'] in (0, 1), 'b01_ext integer range')
    _require(0. <= value['rc_len_q'] <= 1. and value['veto_slack_s'] >= 0. and value['veto_req_budget_s'] >= 0.,
             'b01_ext float range')
    return {name: (list(steps) if name == 'rc_steps' else value[name]) for name in sorted(B01_EXT_FIELDS)}


def validate_config(config):
    _require(type(config) is dict and config.get('method') in METHODS, 'unsupported method')
    method = config['method']
    _require(set(config) == {'mode', 'method', 'search_budget_s'} | ({'b01_ext'} if 'b01_ext' in config else set()),
             'configuration fields')
    _require(method == 'b01' or 'b01_ext' not in config, 'b01_ext requires method b01')
    mode = 'page' if method == 'b01' else 'native'
    _require(config['mode'] == mode, 'mode does not match method')
    budget = config['search_budget_s']
    _require(type(budget) in (int, float) and math.isfinite(budget) and .001 <= budget <= .3, 'search budget')
    result = dict(mode=mode, method=method, search_budget_s=float(budget))
    if 'b01_ext' in config:
        result['b01_ext'] = validate_b01_ext(config['b01_ext'])
    return result


def initial_config():
    method = os.environ.get('B300_PERSISTENT_METHOD', 'b01')
    value = dict(mode=os.environ.get('B300_PERSISTENT_MODE', 'page'), method=method,
                 search_budget_s=float(os.environ.get('B300_PAGE_SEARCH_WALL_S', '.1')))
    ext = json.loads(os.environ.get('B300_PAGE_B01_EXT_JSON', 'null'))
    if method == 'b01' and ext is not None:
        value['b01_ext'] = ext
    return validate_config(value)


def is_page(scheduler=None):
    config = getattr(scheduler, '_b300_policy_config', None) if scheduler is not None else None
    return (config or CONFIG or initial_config())['mode'] == 'page'


def _page():
    return importlib.import_module('sglang.srt.managers.b300_page_scheduler_rollout_v12')


def configure(scheduler):
    """Called once before the scheduler loop starts, on both ranks."""
    global CONFIG
    config = validate_config(getattr(scheduler, '_b300_policy_config', None) or initial_config())
    expected_native = 'fcfs' if config['mode'] == 'page' else config['method']
    if scheduler.schedule_policy != expected_native:
        raise ValueError('startup native policy differs from dispatch configuration')
    CONFIG = config
    scheduler._b300_policy_config = dict(config)
    # The page scheduler reads these process-local variables when it is constructed.
    os.environ['B300_PAGE_METHOD'] = 'b01'
    os.environ['B300_PAGE_SEARCH_WALL_S'] = str(config['search_budget_s'])
    os.environ['B300_PAGE_B01_EXT_JSON'] = json.dumps(config.get('b01_ext'), sort_keys=True, separators=(',', ':'),
                                                      allow_nan=False)


def prefill(scheduler, running_batch, *args, **kwargs):
    if is_page(scheduler):
        return _page().prefill(scheduler, running_batch, *args, **kwargs)
    return scheduler._b300_v9_native_recovery_prefill(
        prefill_delayer_single_pass=kwargs['prefill_delayer_single_pass'],
        running_batch=running_batch)


def prepare_running(scheduler, running_batch):
    return _page().prepare_running(scheduler, running_batch) if is_page(scheduler) else running_batch


def prepare_decode(scheduler, running_batch):
    return (_page().prepare_decode(scheduler, running_batch) if is_page(scheduler)
            else scheduler.update_running_batch(running_batch))


def stash_round(scheduler, excluded):
    if is_page(scheduler):
        return _page().stash_round(scheduler, excluded)
    # SGLang's own continuation of its single chunked request.
    if scheduler.chunked_req is not None:
        excluded.add(scheduler.chunked_req)
        if scheduler.chunked_req.extend_range.end > len(scheduler.chunked_req.prefix_indices):
            scheduler.stash_chunked_request(scheduler.chunked_req)


def note_launch(scheduler, batch):
    if is_page(scheduler):
        _page().note_launch(scheduler, batch)


def note_result(scheduler, batch, result):
    if is_page(scheduler):
        _page().note_result(scheduler, batch, result)


def synchronize_decode_memory(batch, native_ok, selected_indices=None):
    if not is_page():
        return native_ok
    return _page().synchronize_decode_memory(batch, native_ok, selected_indices=selected_indices)
