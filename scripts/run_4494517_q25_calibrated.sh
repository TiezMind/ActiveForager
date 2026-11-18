#!/usr/bin/env bash
# ==============================================================================
# Isolated 8xH800 Evaluation on Worker 4494517:
# Entropy-Gated / Calibrated Turn-1 Routing on Qwen2.5-VL-7B (3D-IG-GRPO)
# ==============================================================================
set -eo pipefail

ROOT_DIR="/mlx/users/lipengyu.seed/playground/agentic_video_understanding"
cd "$ROOT_DIR"

export DA_SCRATCH="/tmp/forager_4494517"
SHARD_DIR="/tmp/forager_4494517/ovobench_q25_calibrated_shards"
LOCAL_MODEL="/tmp/forager_4494517/models/qwen25vl_7b_v4"
OUT_DIR="$ROOT_DIR/outputs/clean_zero_leakage/calibrated_4494517"

mkdir -p "$DA_SCRATCH" "$SHARD_DIR" "/tmp/forager_4494517/models" "$OUT_DIR"
source scripts/driver_env.sh
set -u

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

HDFS_CKPT="/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen25vl_7b_v4"
if [[ ! -f "$LOCAL_MODEL/config.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Staging $HDFS_CKPT to $LOCAL_MODEL on local NVMe..."
    mkdir -p "$LOCAL_MODEL"
    cp -r "$HDFS_CKPT"/* "$LOCAL_MODEL"/
fi

OVOBENCH_JSON="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json"
OVOBENCH_VIDEOS="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos"

# First run a fast 8-GPU N=360 (30 samples/task) calibrated evaluation to verify routing & accuracy gains
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1: 8-GPU Calibrated Routing Evaluation (N=360 Stratified, 30/task) on Worker 4494517 ==="
SHARD_360="$SHARD_DIR/n360"
rm -rf "$SHARD_360"
mkdir -p "$SHARD_360"

pids_360=()
for idx in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$idx" python3 scripts/eval_qwen25vl_calibrated_4494517.py \
        --model-path "$LOCAL_MODEL" \
        --conf-threshold 0.65 \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --max-samples-per-task 30 \
        --shard "${idx}/8" \
        --output-json "$SHARD_360/shard_${idx}.json" \
        > "$SHARD_360/shard_${idx}.log" 2>&1 &
    pids_360+=("$!")
done

for pid in "${pids_360[@]}"; do
    wait "$pid"
done

python3 scripts/eval_qwen25vl_calibrated_4494517.py \
    --merge-shards "$SHARD_360"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen25vl_ig_grpo_calibrated_n360.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Step 1 (N=360 Calibrated Evaluation)!"

# Launch full N=1,640 8-GPU evaluation in background if needed
if [[ "${RUN_FULL_1640:-0}" == "1" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 2: 8-GPU Full N=1,640 Calibrated Routing Evaluation on Worker 4494517 ==="
    SHARD_1640="$SHARD_DIR/n1640"
    rm -rf "$SHARD_1640"
    mkdir -p "$SHARD_1640"
    pids_1640=()
    for idx in 0 1 2 3 4 5 6 7; do
        CUDA_VISIBLE_DEVICES="$idx" python3 scripts/eval_qwen25vl_calibrated_4494517.py \
            --model-path "$LOCAL_MODEL" \
            --conf-threshold 0.65 \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --shard "${idx}/8" \
            --output-json "$SHARD_1640/shard_${idx}.json" \
            > "$SHARD_1640/shard_${idx}.log" 2>&1 &
        pids_1640+=("$!")
    done
    for pid in "${pids_1640[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_qwen25vl_calibrated_4494517.py \
        --merge-shards "$SHARD_1640"/shard_*.json \
        --output-json "$OUT_DIR/ovobench_qwen25vl_ig_grpo_calibrated_n1640.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Step 2 (Full N=1,640 Calibrated Evaluation)!"
fi
