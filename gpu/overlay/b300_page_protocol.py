"""The root packet of a planning round: both tensor-parallel ranks agree on the resources and the queue geometry."""
from __future__ import annotations

from dataclasses import replace


def pack_root(limits, digest, waiting, forced, valid=True):
    """Packet of the resource limits and the geometry digest: the geometry must match exactly, while the resources may
    differ by rank and the smaller one counts."""
    if not valid:
        return [0] * 18
    if len(digest) != 64:
        raise ValueError('invalid root geometry fingerprint')
    return [
        1, int(bool(forced)), limits.token_budget, limits.kv_budget,
        limits.slots, limits.batch_slots, limits.decode_count,
        limits.decode_context, int(round(limits.time_cap_s * 1e9)), waiting,
        *[int(digest[i:i+8], 16) for i in range(0, 64, 8)],
    ]


def unpack_root(low, high, limits):
    """The limits both ranks share, from the element-wise minimum and maximum of their packets, and whether both proved
    the decode-only certificate."""
    if len(low) != 18 or len(high) != 18 or low[0] != 1:
        raise RuntimeError('TP root preparation failed')
    # The decodes and the queue geometry must be identical on both ranks.
    if low[6:] != high[6:]:
        raise RuntimeError('queue or cache geometry differs between TP ranks')
    return replace(limits, token_budget=low[2], kv_budget=low[3],
                   slots=low[4], batch_slots=low[5]), low[1] == 1


def synchronize_root(torch, dist, group, limits, digest, waiting, forced,
                     valid=True):
    values = pack_root(limits, digest, waiting, forced, valid)
    low = torch.tensor(values, dtype=torch.int64, device='cpu')
    high = low.clone()
    dist.all_reduce(low, op=dist.ReduceOp.MIN, group=group)
    dist.all_reduce(high, op=dist.ReduceOp.MAX, group=group)
    return unpack_root(low.tolist(), high.tolist(), limits)
