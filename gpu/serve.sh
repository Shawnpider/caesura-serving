#!/usr/bin/env bash
# Serve Qwen3-30B-A3B on two GPUs with one scheduling method and the settings of our Qwen3 runs.
# usage: gpu/serve.sh fcfs|lpm|pq-aging|planner|caesura [more SGLang arguments, e.g. --port 30001]
set -euo pipefail
method=${1:?usage: gpu/serve.sh fcfs|lpm|pq-aging|planner|caesura [SGLang arguments]}
shift
case $method in
  fcfs|lpm|pq-aging)
    policy=$method
    export B300_PERSISTENT_MODE=native B300_PERSISTENT_METHOD=$method ;;
  planner)
    policy=fcfs
    export B300_PERSISTENT_MODE=page B300_PERSISTENT_METHOD=b01 B300_PAGE_B01_EXT_JSON=null ;;
  caesura)
    policy=fcfs
    export B300_PERSISTENT_MODE=page B300_PERSISTENT_METHOD=b01
    export B300_PAGE_B01_EXT_JSON='{"rc_steps": [1, 2, 4], "rc_margin": 1, "rc_min_decoders": 32, "rc_len_q": 0.5,
      "rc_len_min_n": 3, "veto_tokens": 4096, "veto_min_decoders": 32, "veto_max_steps": 6, "veto_slack_s": 0.3,
      "veto_req_budget_s": 1.0, "tp_wire": 1}' ;;
  *)
    echo "unknown method $method" >&2
    exit 1 ;;
esac
export B300_PAGE_SEARCH_WALL_S=0.1   # planner search budget

set -x
exec python3 -m sglang.launch_server \
  --model-path Qwen/Qwen3-30B-A3B-Instruct-2507 --revision 0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe \
  --tensor-parallel-size 2 --dtype bfloat16 --bf16-gemm-backend torch --mem-fraction-static 0.8 \
  --context-length 131072 --page-size 128 --kv-cache-dtype fp8_e4m3 \
  --chunked-prefill-size 8192 --max-prefill-tokens 16384 --enable-mixed-chunk --disable-prefill-cuda-graph \
  --max-total-tokens 568960 --max-running-requests 128 --max-queued-requests 1024 \
  --enable-hierarchical-cache --hicache-ratio 3 --hicache-write-policy write_through \
  --hicache-io-backend kernel --hicache-mem-layout page_first \
  --tokenizer-worker-num 1 --detokenizer-worker-num 1 --enable-metrics \
  --schedule-policy "$policy" "$@"
