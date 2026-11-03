"""KV reserve for the running decodes, as SGLang computes it, and the KV left to the planner.

The reserve is the larger of SGLang's estimate of the future output (its new-token ratio times the clipped remaining
output of each decode, plus one token per decode) and the pages of the next decode step plus two pages per decode for
overlap scheduling. It is an estimate, not a guarantee: SGLang's decode memory check and retraction still protect the
actual allocation.
"""
from __future__ import annotations

import math

PAGE = 128


def _integer(value, name, *, positive=False):
    if type(value) is not int or value < (1 if positive else 0):
        raise ValueError('invalid '+name)
    return value


def _policy(ratio, clip):
    if type(ratio) not in (int, float) or not math.isfinite(ratio) or not 0 < ratio <= 1:
        raise ValueError('invalid native ratio')
    return float(ratio), _integer(clip, 'native clip', positive=True)


def paged(tokens):
    tokens = _integer(tokens, 'tokens')
    return ((tokens+PAGE-1)//PAGE)*PAGE


def future_output_guard(public_cap, clip):
    """KV reserved for the output of a request that completes its prefill: its clipped output cap plus one page."""
    cap = _integer(public_cap, 'public cap')
    clip = _integer(clip, 'native clip', positive=True)
    return paged(min(cap, clip))+PAGE


def native_future_reserve(decoders, ratio, clip, mixed_decode_tokens=None):
    """SGLang's reserve for the remaining output of the running decodes, as in its PrefillAdder, rounded up."""
    ratio, clip = _policy(ratio, clip)
    decoders = tuple(decoders)
    mixed = len(decoders) if mixed_decode_tokens is None else _integer(mixed_decode_tokens, 'mixed decode tokens')
    values = []
    for req in decoders:
        cap = _integer(req.sampling_params.max_new_tokens, 'public cap')
        completed = len(req.output_ids)
        values.append(min(max(0, cap-completed), clip)*ratio)
    return math.ceil(math.fsum(values))+mixed


def decode_headroom(next_decode_tokens, decode_count, overlap=True):
    """The pages of the next decode step, plus two pages per decode because overlap scheduling keeps two steps in
    flight."""
    required = _integer(next_decode_tokens, 'next decode tokens')
    count = _integer(decode_count, 'decode count')
    if type(overlap) is not bool or required % PAGE:
        raise ValueError('invalid decode headroom geometry')
    return required+(2*PAGE*count if overlap else 0)


def resource_budget(available, next_decode_tokens, decoders, ratio, clip, overlap=True):
    available = _integer(available, 'available tokens')
    decoders = tuple(decoders)
    future = native_future_reserve(decoders, ratio, clip)
    current = decode_headroom(next_decode_tokens, len(decoders), overlap)
    reserved = max(future, current)
    return dict(future_estimate_tokens=future,
                current_decode_headroom_tokens=current,
                reserved_tokens=reserved,
                kv_budget_tokens=max(0, available-reserved))
