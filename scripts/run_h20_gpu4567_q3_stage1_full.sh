#!/usr/bin/env bash
# ==============================================================================
# Strict H20 GPU 4,5,6,7 Full 1,640-Sample Qwen3-VL-8B Stage 1 Evaluation
# SAFETY GUARDRAIL:
#   - Strictly uses ONLY CUDA_VISIBLE_DEVICES=4, 5, 6, 7
#   - NEVER touches GPU 0, 1, 2, 3
# ==============================================================================
set -eo pipefail

ROOT_DIR="/mlx/users/lipengyu.seed/playground/agentic_video_understanding"
cd "$ROOT_DIR"
export DA_SCRATCH="${DA_SCRATCH:-/tmp/forager_scratch}"
mkdir -p "$DA_SCRATCH" /tmp/da_models "$ROOT_DIR/outputs/logs"
source scripts/driver_env.sh
set -u

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OVOBENCH_JSON="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json"
OVOBENCH_VIDEOS="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos"
QWEN3_HDFS="/mnt/hdfs/lipengyu/DA/models/Qwen3-VL-8B-Instruct"
QWEN3_LOCAL="/tmp/da_models/Qwen3-VL-8B-Instruct"

if [ ! -f "$QWEN3_LOCAL/config.json" ]; then
    cp -r "$QWEN3_HDFS" "$QWEN3_LOCAL.tmp.$$" && mv "$QWEN3_LOCAL.tmp.$$" "$QWEN3_LOCAL"
fi

OUT_DIR="$ROOT_DIR/outputs/clean_zero_leakage"
SHARD_DIR="/tmp/forager_clean_h20_gpu4567/q3_s1_full1640"
rm -rf "$SHARD_DIR"
mkdir -p "$OUT_DIR" "$SHARD_DIR"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Launching Full 1,640-Sample Qwen3-VL-8B Stage 1 on H20 GPUs 4,5,6,7 ==="
pids=()
for idx in 0 1 2 3; do
    gpu_id=$((idx + 4))
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
        --backend hf \
        --model-path "$QWEN3_LOCAL" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --shard "${idx}/4" \
        --output-json "$SHARD_DIR/shard_${idx}.json" \
        > "$SHARD_DIR/shard_${idx}.log" 2>&1 &
    pids+=("$!")
done

for pid in "${pids[@]}"; do
    wait "$pid"
done

python3 scripts/eval_ovobench.py \
    --merge-shards "$SHARD_DIR"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen3vl_stage1_zero_leakage.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Completed Full 1,640-Sample Qwen3-VL-8B Stage 1 on H20 GPUs 4,5,6,7! ==="
