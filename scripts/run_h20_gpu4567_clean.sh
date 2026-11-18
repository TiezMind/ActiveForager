#!/usr/bin/env bash
# ==============================================================================
# Strict H20 GPU 4,5,6,7 Zero-Leakage Evaluation & Training Pipeline
# SAFETY GUARDRAIL:
#   - NEVER touches GPU 0, 1, 2, 3 on H20 4485507 or H800 4480127
#   - Every Python invocation explicitly binds CUDA_VISIBLE_DEVICES to 4, 5, 6, or 7
#   - NEVER runs pkill or killall
# ==============================================================================
set -eo pipefail

ROOT_DIR="/mlx/users/lipengyu.seed/playground/agentic_video_understanding"
cd "$ROOT_DIR"
source scripts/driver_env.sh
set -u
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OVOBENCH_JSON="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json"
OVOBENCH_VIDEOS="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos"
QWEN25_PATH="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct"
QWEN3_PATH="/mnt/hdfs/lipengyu/DA/models/Qwen3-VL-8B-Instruct"

OUT_DIR="$ROOT_DIR/outputs/clean_zero_leakage"
SHARD_ROOT="/tmp/forager_clean_h20_gpu4567"
mkdir -p "$OUT_DIR" "$ROOT_DIR/outputs/logs" "$SHARD_ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Starting Zero-Leakage H20 GPU 4-7 Pipeline ==="

# ------------------------------------------------------------------------------
# Phase 1A: Fast 60-Sample (5 per task x 12 tasks) Zero-Leakage Calibration on GPUs 4,5,6,7
# ------------------------------------------------------------------------------
CAL_SHARD_DIR="$SHARD_ROOT/cal60_qwen25vl"
mkdir -p "$CAL_SHARD_DIR"
if [[ ! -f "$OUT_DIR/ovobench_qwen25vl_zero_leakage_cal60.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running 60-sample Zero-Leakage OVO-Bench calibration (Qwen2.5-VL-7B) on GPUs 4,5,6,7..."
    pids=()
    for idx in 0 1 2 3; do
        gpu_id=$((idx + 4))
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
            --backend hf \
            --model-path "$QWEN25_PATH" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --max-samples-per-task 5 \
            --shard "${idx}/4" \
            --output-json "$CAL_SHARD_DIR/shard_${idx}.json" \
            > "$CAL_SHARD_DIR/shard_${idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_ovobench.py \
        --merge-shards "$CAL_SHARD_DIR"/shard_*.json \
        --output-json "$OUT_DIR/ovobench_qwen25vl_zero_leakage_cal60.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed 60-sample Qwen2.5-VL-7B Zero-Leakage Calibration!"
fi

# ------------------------------------------------------------------------------
# Phase 1B: Fast 60-Sample (5 per task x 12 tasks) Zero-Leakage Calibration (Qwen3-VL-8B) on GPUs 4,5,6,7
# ------------------------------------------------------------------------------
CAL_Q3_DIR="$SHARD_ROOT/cal60_qwen3vl"
mkdir -p "$CAL_Q3_DIR"
if [[ ! -f "$OUT_DIR/ovobench_qwen3vl_zero_leakage_cal60.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running 60-sample Zero-Leakage OVO-Bench calibration (Qwen3-VL-8B) on GPUs 4,5,6,7..."
    pids=()
    for idx in 0 1 2 3; do
        gpu_id=$((idx + 4))
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
            --backend hf \
            --model-path "$QWEN3_PATH" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --max-samples-per-task 5 \
            --shard "${idx}/4" \
            --output-json "$CAL_Q3_DIR/shard_${idx}.json" \
            > "$CAL_Q3_DIR/shard_${idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_ovobench.py \
        --merge-shards "$CAL_Q3_DIR"/shard_*.json \
        --output-json "$OUT_DIR/ovobench_qwen3vl_zero_leakage_cal60.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed 60-sample Qwen3-VL-8B Zero-Leakage Calibration!"
fi

# ------------------------------------------------------------------------------
# Phase 2: Full 1,640-Sample Zero-Leakage OVO-Bench Stage 1 Evaluation (Qwen2.5-VL-7B) on GPUs 4,5,6,7
# ------------------------------------------------------------------------------
FULL_Q25_DIR="$SHARD_ROOT/full_qwen25vl_stage1"
mkdir -p "$FULL_Q25_DIR"
if [[ ! -f "$OUT_DIR/ovobench_qwen25vl_stage1_zero_leakage.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Full 1640-sample Zero-Leakage OVO-Bench Stage 1 (Qwen2.5-VL-7B) on GPUs 4,5,6,7..."
    pids=()
    for idx in 0 1 2 3; do
        gpu_id=$((idx + 4))
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
            --backend hf \
            --model-path "$QWEN25_PATH" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --shard "${idx}/4" \
            --output-json "$FULL_Q25_DIR/shard_${idx}.json" \
            > "$FULL_Q25_DIR/shard_${idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_ovobench.py \
        --merge-shards "$FULL_Q25_DIR"/shard_*.json \
        --output-json "$OUT_DIR/ovobench_qwen25vl_stage1_zero_leakage.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Full 1640-sample Qwen2.5-VL-7B Stage 1!"
fi

# ------------------------------------------------------------------------------
# Phase 3: Multimodal SDAM-SFT + 3D-IG-GRPO Training on GPUs 4,5,6,7
# ------------------------------------------------------------------------------
SFT_CKPT_DIR="/tmp/forager_ckpts/clean_sft_qwen25vl"
RL_CKPT_DIR="/tmp/forager_ckpts/clean_rl_qwen25vl"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Multimodal SDAM-SFT on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python3 scripts/train_sft.py \
    --backend hf \
    --model-path "$QWEN25_PATH" \
    --train-jsonl "$ROOT_DIR/data/sft_coldstart_6k.jsonl" \
    --local-ckpt-dir "$SFT_CKPT_DIR" \
    --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_sft/qwen25vl_7b" \
    --max-steps 50 \
    --batch-size 2 \
    --lr 1e-5 \
    > "$OUT_DIR/train_sft_qwen25vl.log" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running 3D-IG-GRPO RL Training on GPU 4..."
CUDA_VISIBLE_DEVICES=4 python3 scripts/train_ig_grpo.py \
    --mode hf \
    --model-path "$SFT_CKPT_DIR/global_step_50" \
    --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
    --local-ckpt-dir "$RL_CKPT_DIR" \
    --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen25vl_7b" \
    --max-steps 100 \
    --save-freq 50 \
    --group-size 4 \
    > "$OUT_DIR/train_ig_grpo_qwen25vl.log" 2>&1

# ------------------------------------------------------------------------------
# Phase 4: Full Post-RL Zero-Leakage OVO-Bench Evaluation on GPUs 4,5,6,7
# ------------------------------------------------------------------------------
FULL_RL_DIR="$SHARD_ROOT/full_qwen25vl_rl"
mkdir -p "$FULL_RL_DIR"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Full 1640-sample Post-RL Zero-Leakage OVO-Bench on GPUs 4,5,6,7..."
pids=()
for idx in 0 1 2 3; do
    gpu_id=$((idx + 4))
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
        --backend hf \
        --model-path "$RL_CKPT_DIR/global_step_100" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --shard "${idx}/4" \
        --output-json "$FULL_RL_DIR/shard_${idx}.json" \
        > "$FULL_RL_DIR/shard_${idx}.log" 2>&1 &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
python3 scripts/eval_ovobench.py \
    --merge-shards "$FULL_RL_DIR"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_zero_leakage.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === All Zero-Leakage H20 GPU 4-7 Stages Complete! ==="
