#!/usr/bin/env bash
# ==============================================================================
# Isolated 8xH800 Full Official OVO-Bench FAR Evaluation on Worker 4494517
# Evaluates all 172 FAR records (1,567 causal probes across REC, SSR, CRR)
# under 100% blind zero-leakage isolation for both:
#   1. Qwen3-VL-8B (3D-IG-GRPO, qwen3vl_8b_v4)
#   2. Qwen2.5-VL-7B (3D-IG-GRPO, qwen25vl_7b_v4)
# and merges with existing full zero-leakage RT (837) + BT (631) results into
# `outputs/clean_zero_leakage/official_far_4494517/`.
# ==============================================================================
set -eo pipefail

ROOT_DIR="/mlx/users/lipengyu.seed/playground/agentic_video_understanding"
cd "$ROOT_DIR"

export DA_SCRATCH="/tmp/forager_4494517"
SHARD_ROOT="/tmp/forager_4494517/official_far_shards"
OUT_DIR="$ROOT_DIR/outputs/clean_zero_leakage/official_far_4494517"
mkdir -p "$DA_SCRATCH" "$SHARD_ROOT" "$OUT_DIR"

source scripts/driver_env.sh
set -u

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OVOBENCH_JSON="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json"
OVOBENCH_VIDEOS="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos"

Q3_MODEL="/tmp/forager_4494517/models/qwen3vl_8b_v4"
Q25_MODEL="/tmp/forager_4494517/models/qwen25vl_7b_v4"

# Ensure both checkpoints are staged on local NVMe
if [[ ! -f "$Q3_MODEL/config.json" ]]; then
    mkdir -p "$Q3_MODEL"
    cp -r /mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen3vl_8b_v4/* "$Q3_MODEL"/
fi
if [[ ! -f "$Q25_MODEL/config.json" ]]; then
    mkdir -p "$Q25_MODEL"
    cp -r /mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen25vl_7b_v4/* "$Q25_MODEL"/
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1: 8-GPU Full Official FAR Evaluation on Qwen3-VL-8B (172 records, 1,567 probes) ==="
Q3_SHARDS="$SHARD_ROOT/qwen3vl"
rm -rf "$Q3_SHARDS"
mkdir -p "$Q3_SHARDS"

pids_q3=()
for idx in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$idx" python3 scripts/eval_ovobench_official_far_4494517.py \
        --model-path "$Q3_MODEL" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --shard "${idx}/8" \
        --output-json "$Q3_SHARDS/shard_${idx}.json" \
        > "$Q3_SHARDS/shard_${idx}.log" 2>&1 &
    pids_q3+=("$!")
done

for pid in "${pids_q3[@]}"; do
    wait "$pid"
done

python3 scripts/eval_ovobench_official_far_4494517.py \
    --merge-shards "$Q3_SHARDS"/shard_*.json \
    --merge-with-full-json "$ROOT_DIR/outputs/clean_zero_leakage/ovobench_qwen3vl_final_ig_grpo_v2_zero_leakage.json" \
    --output-json "$OUT_DIR/ovobench_qwen3vl_official_far_n1640.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Step 1 (Qwen3-VL-8B Full Official FAR)!"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 2: 8-GPU Full Official FAR Evaluation on Qwen2.5-VL-7B (172 records, 1,567 probes) ==="
Q25_SHARDS="$SHARD_ROOT/qwen25vl"
rm -rf "$Q25_SHARDS"
mkdir -p "$Q25_SHARDS"

pids_q25=()
for idx in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$idx" python3 scripts/eval_ovobench_official_far_4494517.py \
        --model-path "$Q25_MODEL" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --shard "${idx}/8" \
        --output-json "$Q25_SHARDS/shard_${idx}.json" \
        > "$Q25_SHARDS/shard_${idx}.log" 2>&1 &
    pids_q25+=("$!")
done

for pid in "${pids_q25[@]}"; do
    wait "$pid"
done

python3 scripts/eval_ovobench_official_far_4494517.py \
    --merge-shards "$Q25_SHARDS"/shard_*.json \
    --merge-with-full-json "$ROOT_DIR/outputs/clean_zero_leakage/ovobench_qwen25vl_final_ig_grpo_v2_zero_leakage.json" \
    --output-json "$OUT_DIR/ovobench_qwen25vl_official_far_n1640.json"

python3 scripts/eval_ovobench_official_far_4494517.py \
    --merge-shards "$Q25_SHARDS"/shard_*.json \
    --merge-with-full-json "$ROOT_DIR/outputs/clean_zero_leakage/calibrated_4494517/ovobench_qwen25vl_ig_grpo_calibrated_n1640.json" \
    --output-json "$OUT_DIR/ovobench_qwen25vl_calibrated_official_far_n1640.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === All 8-GPU Official FAR Evaluations Completed! ==="
