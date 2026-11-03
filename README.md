# Caesura

Official implementation of the paper **Caesura: SLO-Aware Prefill Scheduling with Release-Aware Rollout for Bursty
Traffic**.


| path | content |
|---|---|
| `gpu/` | Caesura for SGLang 0.5.18, and the Qwen3-30B-A3B experiment of Figure 9 |
| `run_sim.py`, `simulator/`, `configs/`, `workloads/` | the CPU simulator of the Table 2 GPU experiments |
| `metrics.py` | SLO attainment (TTFT ≤ 3 s) and TTFT percentiles |

## GPU: Qwen3-30B-A3B on two H200 GPUs

Our results, the data of Figure 9:

```bash
python3 gpu/summarize.py
```

```
measurement                              leval90                      mooncake
                      Att.     P50    P90    P95    Att.     P50    P90    P95
SGLang FCFS            0.0    9.76   12.8   12.9    68.8    0.17   26.0   26.5
SGLang LPM             5.5    8.21   11.3   11.4    88.5    0.13   6.00   25.8
PQ+aging               0.0    16.4   18.0   18.1    73.2    0.17   23.5   24.7
Caesura planner       86.2    0.97   6.10   11.2    94.9    0.18   0.95   3.07
Caesura               96.0    0.49   0.84   1.52    97.0    0.13   0.68   1.31

long_burst                               leval90                      mooncake
                      Att.     P50    P90    P95    Att.     P50    P90    P95
SGLang FCFS            0.0    9.97   10.4   10.4    22.9    15.0   26.7   27.0
SGLang LPM             0.0    8.38   8.77   8.80    72.6    0.56   29.7   33.8
PQ+aging               0.0    16.4   16.5   16.6    35.4    11.2   25.1   26.0
Caesura planner       77.9    1.74   6.16   7.06    86.8    0.33   12.5   54.8
Caesura               98.5    0.62   1.12   1.55    92.2    0.25   2.02   16.5
```

### Rerun on your GPUs

You need two GPUs (our runs used two H200), the `lmsysorg/sglang:v0.5.18-cu130-runtime` image, which has SGLang 0.5.18,
and Python 3.11 or later for the client. The server downloads `Qwen/Qwen3-30B-A3B-Instruct-2507` from Hugging Face on
its first start.

**1. Start a container** of the image with the repository mounted, for example:

```bash
docker run --gpus '"device=0,1"' --ipc=host --shm-size 32g -p 30000:30000 \
    -v ~/.cache/huggingface:/root/.cache/huggingface -v "$PWD":/caesura -w /caesura \
    -it lmsysorg/sglang:v0.5.18-cu130-runtime bash
```

**2. Install Caesura into SGLang**, once per container:

```bash
bash gpu/install.sh
```

It checks that SGLang is 0.5.18, applies `gpu/sglang_hooks.patch` to the installed SGLang and copies
`gpu/overlay/*.py` next to its scheduler. This changes the SGLang installation in place, so run it in the container.

**3. Start the server with one method:**

```bash
CUDA_VISIBLE_DEVICES=0,1 gpu/serve.sh caesura
```

| method | scheduling |
|---|---|
| `fcfs` | SGLang FCFS |
| `lpm` | SGLang LPM |
| `pq-aging` | PQ+aging: SGLang's admission, with the queue ordered by estimated prefill time and aging |
| `planner` | the planner of Caesura alone |
| `caesura` | Caesura: the planner with the release-aware rollout |

`serve.sh` starts SGLang with the settings of our runs: tensor parallelism 2, 128-token pages, an FP8 KV cache of
568,960 tokens, chunked prefill of 8192 tokens and HiCache. Further arguments go to SGLang, for example `--port 30001`.
The server is ready when this answers:

```bash
curl -s http://127.0.0.1:30000/v1/models
```

**4. Replay a workload** from a second shell, for example `docker exec -it <container> bash` and `cd /caesura`:

```bash
python3 gpu/replay.py gpu/workloads/leval90 --verify                    # optional: check the workload, no server
python3 gpu/replay.py gpu/workloads/leval90 --run-http --smoke --url http://127.0.0.1:30000/v1/completions \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --output-dir runs/smoke    # optional: 4 short requests
python3 gpu/replay.py gpu/workloads/leval90 --run-http --url http://127.0.0.1:30000/v1/completions \
    --model Qwen/Qwen3-30B-A3B-Instruct-2507 --output-dir runs/leval90-caesura
```

The client sends every request at its arrival time, without waiting for earlier ones, and measures the TTFT. It writes
`client_records.jsonl` (one line per request) and `RUN.json` (a summary) into the output directory, which must not
exist yet. `gpu/workloads/leval90` sends 804 requests within 34 s, and `gpu/workloads/mooncake` 2,119 requests within
930 s.

**5. Repeat steps 3 and 4 for every method:** stop the server with Ctrl-C, start it with the next method, and replay into
a new directory, for example `runs/leval90-fcfs`.

**6. Summarize** the replays of one workload, one `method=directory` per run:

```bash
python3 gpu/summarize.py leval90 fcfs=runs/leval90-fcfs lpm=runs/leval90-lpm pq-aging=runs/leval90-pq-aging \
    planner=runs/leval90-planner caesura=runs/leval90-caesura
```

It prints the table above for these runs. Att. is the share of requests with a TTFT of at most 3 s, in percent, and
P50, P90 and P95 are TTFT percentiles in seconds. The measurement window holds every request after the warm-up (the
first 20 s of L-Eval and 300 s of Mooncake), and the long burst the requests that arrive during it (26-27.3 s and
570-630 s). A run with failed requests is rejected.

## Simulator of the Table 2 GPU experiments

The simulator replays the L-Eval and Mooncake workloads of Table 2 on one simulated SGLang instance with the B300 step
times, for the same five methods. It needs Python 3.10 or later, no packages and no GPU.

```bash
python3 run_sim.py --members base                                        # 10 runs, about 1 CPU-minute
python3 run_sim.py --procs 8                                             # all 120 runs, about 12 CPU-minutes
python3 run_sim.py --workloads leval --methods planner caesura --members base   # a subset
```

| option | default | |
|---|---|---|
| `--workloads` | `leval mooncake` | workloads to simulate |
| `--methods` | `fcfs lpm pq-aging planner caesura` | methods to simulate |
| `--members` | `all` | `base`: the calibrated step times only; `all`: also the scaled ones |
| `--procs` | `4` | parallel processes |
| `--out` | `results/sim` | directory for the per-request results |

It prints the same table as `gpu/summarize.py`, with the TTFTs of all members of a cell pooled, and writes the TTFT of
every request to `results/sim/<workload>/<method>/<member>.json.gz`.

Each cell is one workload and one method. `workloads/<workload>.jobs.json.gz` lists its requests: arrival time, prompt
and output tokens, prompt pages and window. `configs/<workload>/<method>.json` holds the settings of the cell under
`base`, calibrated to our B300 runs, and the timing members under `members`: `f` scales the time of every GPU step and
`d` that of the decode steps on top, so `f101` runs 1% slower GPU steps. The class `Cfg` in `simulator/native_sglang.py`
describes every setting. For the planner and Caesura, the simulator calls the planner in `gpu/overlay/`, the same code
that runs in SGLang.

## Code

| file | content |
|---|---|
| `gpu/overlay/b300_page_objectives_v14.py` | the planner: objective and search |
| `gpu/overlay/b300_page_scheduler_rollout_v12.py` | runs the planner in every step |
| `gpu/overlay/b300_b01_release_veto_v1.py` | the release-aware rollout |
| `gpu/overlay/b300_page_cache.py` | prefix-cache matches of the waiting requests |
| `gpu/overlay/b300_page_resources_v9.py` | KV reserved for the running decodes |
| `gpu/overlay/b300_page_protocol.py`, `b300_page_tpwire_v1.py` | agreement of the two tensor-parallel ranks |
| `gpu/overlay/b300_persistent_dispatch_rollout_v12.py` | SGLang's own scheduling or Caesura, chosen at startup |
| `gpu/overlay/b300_pq_schedule_policy.py` | the PQ+aging baseline |
| `gpu/sglang_hooks.patch` | the hooks in SGLang's scheduler |
| `simulator/native_sglang.py` | the simulator |

The planner is method `b01`. Caesura adds `b01_ext`.

## Workloads and licenses

The workloads replay L-Eval questions (clock 36× in Table 2, 90× on Qwen3) and a 30% sample of the Mooncake
tool-agent trace. The token IDs in `gpu/workloads/` encode L-Eval text and keep its GPL-3.0 license
(`gpu/workloads/LICENSE.LEval`). The Mooncake trace and our code are under Apache-2.0.

## Citation

```bibtex
@misc{zhou2026caesura,
  title  = {Caesura: SLO-Aware Prefill Scheduling with Release-Aware Rollout for Bursty Traffic},
  author = {Zhou, Zhixiang and Sha, Long and Chen, Ming and Liu, Yong},
  year   = {2026}
}
```
