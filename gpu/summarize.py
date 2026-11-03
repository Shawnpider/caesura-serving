"""TTFT statistics of the Qwen3-30B-A3B experiment, from our measurements or from your own replay.

usage: python3 gpu/summarize.py                               # our runs, from gpu/results/
       python3 gpu/summarize.py WORKLOAD METHOD=DIR [...]     # replay.py output directories, for example
       python3 gpu/summarize.py leval90 fcfs=runs/leval90-fcfs caesura=runs/leval90-caesura

Our runs took the TTFT at the HTTP frontend in front of SGLang, and replay.py takes it at the client.
"""
import csv
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path

GPU = Path(__file__).resolve().parent
sys.path.insert(0, str(GPU.parent))

from metrics import LABELS, METHODS, WINDOWS, in_window, print_table, ttft_stats


def phases(workload):
    """The phase of every request: the warm-up, or the window that its arrival falls in."""
    m = json.loads((GPU / 'workloads' / workload / 'WORKLOAD.json').read_text())
    windows = [(k, a, b) for k, (a, b) in m['windows'].items() if k != 'warmup']
    return {e['request']: 'warmup' if e['warmup'] else next(k for k, a, b in windows if a <= e['offset_s'] < b)
            for e in m['events']}


def stats(ttft, phase):
    return {w: ttft_stats([t for i, t in ttft.items() if in_window(phase[i], w)]) for w in WINDOWS}


def ours(workload):
    ttft = defaultdict(dict)
    with gzip.open(GPU / 'results' / f'{workload}.csv.gz', 'rt') as f:
        for r in csv.DictReader(f):
            ttft[r['method']][int(r['id'])] = float(r['ttft_s'])
    phase = phases(workload)
    return {m: stats(t, phase) for m, t in ttft.items()}


def replayed(workload, runs):
    phase, table = phases(workload), {}
    for run in runs:
        method, directory = run.split('=', 1)
        if method not in METHODS:
            sys.exit(f'unknown method {method}, expected one of {METHODS}')
        rows = [json.loads(line) for line in (Path(directory) / 'client_records.jsonl').read_text().splitlines()]
        if len(rows) != len(phase) or not all(r['ok'] for r in rows):
            sys.exit(f'{directory}: the replay is incomplete or has failed requests')
        table[method] = stats({r['request']: r['client_ttft_s'] for r in rows}, phase)
    return table


if __name__ == '__main__':
    if len(sys.argv) == 1:
        print_table({w: ours(w) for w in ('leval90', 'mooncake')}, ['leval90', 'mooncake'], METHODS, LABELS)
    else:
        table = replayed(sys.argv[1], sys.argv[2:])
        print_table({sys.argv[1]: table}, [sys.argv[1]], [m for m in METHODS if m in table], LABELS)
