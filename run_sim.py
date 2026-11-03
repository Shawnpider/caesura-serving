"""Run the CPU simulator on the L-Eval and Mooncake cells of Table 2 and print the TTFT statistics, pooled over
timing members.

usage: python3 run_sim.py [--workloads leval mooncake] [--methods fcfs lpm pq-aging planner caesura]
                          [--members all|base] [--procs N] [--out DIR]

Each cell is one workload and one method. A timing member scales the calibrated GPU step times: f scales every step
and d the decode steps on top, e.g. f101 = x1.01. configs/<workload>/<method>.json lists the members of each cell.
"""
import argparse
import gzip
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'simulator'))

from metrics import LABELS, METHODS, WINDOWS, in_window, print_table, ttft_stats


def load_jobs(workload):
    return json.loads(gzip.decompress((ROOT / 'workloads' / f'{workload}.jobs.json.gz').read_bytes()))


def simulate(task):
    """Run one member of one cell and return {request id: TTFT in s}."""
    from native_sglang import Cfg, NativeSim
    workload, method, member = task
    conf = json.loads((ROOT / 'configs' / workload / f'{method}.json').read_text())
    sim = NativeSim(load_jobs(workload), Cfg(**dict(conf['base'], **conf['members'][member])))
    sim.run()
    return {r['id']: r['ttft_s'] for r in sim.results()}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--workloads', nargs='+', default=['leval', 'mooncake'], choices=['leval', 'mooncake'])
    ap.add_argument('--methods', nargs='+', default=METHODS, choices=METHODS)
    ap.add_argument('--members', default='all', choices=['all', 'base'])
    ap.add_argument('--procs', type=int, default=4)
    ap.add_argument('--out', type=Path, default=ROOT / 'results/sim')
    a = ap.parse_args()

    tasks = []
    for w in a.workloads:
        for m in a.methods:
            members = json.loads((ROOT / 'configs' / w / f'{m}.json').read_text())['members']
            tasks += [(w, m, member) for member in (members if a.members == 'all' else ['base'])]
    print(f'{len(tasks)} simulator runs on {a.procs} processes', flush=True)

    table = {}
    with ProcessPoolExecutor(a.procs) as pool:
        for (w, m, member), ttft in zip(tasks, pool.map(simulate, tasks)):
            out = a.out / w / m / f'{member}.json.gz'
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(gzip.compress(json.dumps(ttft).encode(), mtime=0))
            print(f'  done {w} {m} {member}', flush=True)
            table.setdefault(w, {}).setdefault(m, []).append(ttft)

    stats = {}
    for w, by_method in table.items():
        phase = {j['id']: j['phase'] for j in load_jobs(w)}
        for m, runs in by_method.items():
            stats.setdefault(w, {})[m] = {
                window: ttft_stats([t for ttft in runs for i, t in ttft.items() if in_window(phase[i], window)])
                for window in WINDOWS}
    print_table(stats, a.workloads, a.methods, LABELS)


if __name__ == '__main__':
    main()
