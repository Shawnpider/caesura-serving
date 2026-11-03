"""The numeric exchange of Caesura: fixed-size integer collectives in place of pickled object exchanges.

With tp_wire, the page scheduler exchanges state between its two tensor-parallel ranks as int64 vectors over Gloo,
with the same outcome as the exchanges they replace; both ranks branch only on gathered, hence identical, data.

* synchronize_root: one all_gather in place of the MIN and MAX all_reduce pair of the root packet.
* early_root: the validity flag of a reused decode-only certificate travels with the root packet in one all_gather.
* exchange_locks: the ranks compare sha256 words of their lock lists; if equal, each uses its own list twice, and
  otherwise both fall back to all_gather_object.
* agree_geometry: the ranks compare sha256 words of their prefix geometry before and after load-back, and both raise
  unless they agree.
"""
from __future__ import annotations

import hashlib

from sglang.srt.managers.b300_page_protocol import pack_root, unpack_root

ROOT_WIDTH = 18
EARLY_WIDTH = 1 + ROOT_WIDTH
HASH_WIDTH = 9
CONSENSUS_ERROR = 'TP consensus on the planned step failed'


def words(text):
    """Eight 32-bit words of sha256(text)."""
    digest = hashlib.sha256(text.encode()).hexdigest()
    return [int(digest[i:i + 8], 16) for i in range(0, 64, 8)]


def gather(torch, dist, group, values):
    """One fixed-size int64 all_gather over the two ranks; returns the rows of both ranks in rank order."""
    packet = torch.tensor(list(values), dtype=torch.int64, device='cpu')
    parts = [packet.clone(), packet.clone()]
    dist.all_gather(parts, packet, group=group)
    rows = [part.tolist() for part in parts]
    if len(rows) != 2 or any(len(row) != len(values) for row in rows):
        raise RuntimeError('numeric exchange packet shape changed')
    return rows


def synchronize_root(torch, dist, group, limits, digest, waiting, forced, valid=True):
    """Same arguments and outcome as b300_page_protocol.synchronize_root."""
    values = pack_root(limits, digest, waiting, forced, valid)
    left, right = gather(torch, dist, group, values)
    low = [min(a, b) for a, b in zip(left, right)]
    high = [max(a, b) for a, b in zip(left, right)]
    return unpack_root(low, high, limits)


def early_root(torch, dist, group, limits, digest, waiting, valid):
    """None when either rank cannot reuse the certificate, else the (low, high) root vectors for unpack_root. A local
    error is a miss on both ranks."""
    values = [0] * EARLY_WIDTH
    if valid:
        try:
            packed = pack_root(limits, digest, waiting, True, True)
            if len(packed) == ROOT_WIDTH:
                values = [1, *packed]
        except Exception:
            values = [0] * EARLY_WIDTH
    left, right = gather(torch, dist, group, values)
    if min(left[0], right[0]) == 0:
        return None
    low = [min(a, b) for a, b in zip(left[1:], right[1:])]
    high = [max(a, b) for a, b in zip(left[1:], right[1:])]
    return low, high


def exchange_locks(st, locks):
    """The lock lists of both ranks, in rank order, for merge_locks. Equal lists merge to the same result."""
    try:
        packet = [1, *words(repr(locks))]
    except Exception:
        packet = [0] * HASH_WIDTH
    left, right = gather(st.torch, st.dist, st.group, packet)
    if left[0] == right[0] == 1 and left[1:] == right[1:]:
        return [locks, locks]
    gathered = [None, None]
    st.dist.all_gather_object(gathered, locks, group=st.group)
    return gathered


def agree_geometry(st, ok, geometry):
    """Raises on both ranks unless both are ok with equal geometry."""
    packet = [0] * HASH_WIDTH
    if ok is True:
        try:
            packet = [1, *words(repr(geometry))]
        except Exception:
            packet = [0] * HASH_WIDTH
    left, right = gather(st.torch, st.dist, st.group, packet)
    if not (left[0] == right[0] == 1 and left[1:] == right[1:]):
        raise RuntimeError(CONSENSUS_ERROR)
