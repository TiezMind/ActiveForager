#!/usr/bin/env bash
set -euo pipefail
MODEL_FAMILY="${1:-qwen25vl}"
SAMPLES_PER_SUBTASK="${2:-20}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

for p in $(pgrep -f eval_streamingbench.py || true); do
    kill -9 "$p" 2>/dev/null || true
done
sleep 1

export SCRATCH_DIR="/tmp/forager_sb_${MODEL_FAMILY}"
rm -rf "$SCRATCH_DIR"
mkdir -p "$SCRATCH_DIR/sb_shards" "$ROOT/outputs" "$ROOT/outputs/logs"

if [[ -f "$ROOT/scripts/driver_env.sh" && -e /proc/driver/nvidia/version ]]; then
    set +e
    export DA_SCRATCH="$SCRATCH_DIR"
    source "$ROOT/scripts/driver_env.sh"
    set -e
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

if [[ "$MODEL_FAMILY" == "qwen3vl" ]]; then
    MODEL_PATH="/tmp/da_models/Qwen3-VL-8B-SFT"
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        MODEL_PATH="/mnt/hdfs/lipengyu/DA/checkpoints/forager/rl/qwen3vl_ig_grpo_v2"
    fi
    OUT_JSON="$ROOT/outputs/streamingbench_qwen3vl_ig_grpo_v2.json"
else
    MODEL_PATH="/tmp/da_models/Qwen2.5-VL-7B-Forager-Eval"
    if [[ ! -f "$MODEL_PATH/config.json" ]]; then
        MODEL_PATH="/mnt/hdfs/lipengyu/DA/checkpoints/forager/rl/qwen25vl_ig_grpo_v2"
    fi
    OUT_JSON="$ROOT/outputs/streamingbench_qwen25vl_ig_grpo_v2.json"
fi

pids=()
shards=()
for gpu_idx in $(seq 0 7); do
    sf="$SCRATCH_DIR/sb_shards/shard_${gpu_idx}.json"
    shards+=("$sf")
    CUDA_VISIBLE_DEVICES="$gpu_idx" python3 scripts/eval_streamingbench.py \
        --backend hf \
        --model-path "$MODEL_PATH" \
        --max-samples-per-subtask "$SAMPLES_PER_SUBTASK" \
        --shard "${gpu_idx}/8" \
        --output-json "$sf" > "$SCRATCH_DIR/sb_shards/shard_${gpu_idx}.log" 2>&1 &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

python3 scripts/eval_streamingbench.py \
    --merge-shards "${shards[@]}" \
    --output-json "$OUT_JSON"

rm -rf "$SCRATCH_DIR"
echo "=== [StreamingBench Completed: $OUT_JSON] ==="
