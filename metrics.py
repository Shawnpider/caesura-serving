"""Methods, windows and TTFT statistics shared by run_sim.py and gpu/summarize.py.

The measurement window holds every request after the warm-up, and the long burst the requests that arrive during it.
Attainment is the share of requests with TTFT <= 3 s. P50 is the median, and P90 and P95 use the nearest rank.
"""
import math
import statistics

METHODS = ['fcfs', 'lpm', 'pq-aging', 'planner', 'caesura']
LABELS = {'fcfs': 'SGLang FCFS', 'lpm': 'SGLang LPM', 'pq-aging': 'PQ+aging',
          'planner': 'Caesura planner', 'caesura': 'Caesura'}
SLO_S = 3.0
WINDOWS = ('measurement', 'long_burst')


def in_window(phase, window):
    return phase not in (None, 'warmup') if window == 'measurement' else phase == 'long_burst'


def ttft_stats(values):
    v = sorted(values)
    rank = lambda f: v[max(0, math.ceil(f * len(v)) - 1)]
    return dict(n=len(v), attainment_pct=100 * sum(x <= SLO_S for x in v) / len(v),
                p50_s=statistics.median(v), p90_s=rank(.9), p95_s=rank(.95))


def fmt(x):
    return f'{x:.2f}' if x < 10 else f'{x:.1f}'


def print_table(table, workloads, methods, labels):
    """table[workload][method][window] -> ttft_stats; prints one block per window, like Table 2 of the paper."""
    head = ''.join(f'{w:>30s}' for w in workloads)
    for window in WINDOWS:
        print(f'\n{window:<18s}{head}')
        print(f'{"":18s}' + f'{"Att.":>8s}{"P50":>8s}{"P90":>7s}{"P95":>7s}' * len(workloads))
        for m in methods:
            cells = []
            for w in workloads:
                s = table.get(w, {}).get(m, {}).get(window)
                cells.append(f'{s["attainment_pct"]:8.1f}{fmt(s["p50_s"]):>8s}{fmt(s["p90_s"]):>7s}{fmt(s["p95_s"]):>7s}'
                             if s else f'{"-":>8s}{"":22s}')
            print(f'{labels[m]:<18s}' + ''.join(cells))
