#!/usr/bin/env bash
# ==============================================================================
# Strict H20 GPU 4,5,6,7 Zero-Leakage Master Pipeline (Steps 1, 2, and 3)
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
ABL_DIR="$OUT_DIR/ablations"
SHARD_ROOT="/tmp/forager_clean_h20_gpu4567"
mkdir -p "$OUT_DIR" "$ABL_DIR" "$ROOT_DIR/outputs/logs" "$SHARD_ROOT"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Starting H20 GPU 4-7 Master Zero-Leakage Pipeline (Steps 1, 2, 3) ==="

# ------------------------------------------------------------------------------
# Step 0: Concurrent RL Training on GPU 4 (Qwen2.5-VL-7B v2) & GPU 5 (Qwen3-VL-8B SFT+RL v2)
#         + Ablation C (300-Step RL Dynamics) on CPU
# ------------------------------------------------------------------------------
SFT_Q25_DIR="/tmp/forager_ckpts/clean_sft_qwen25vl"
RL_Q25_V2_DIR="/tmp/forager_ckpts/clean_rl_qwen25vl_v2"
SFT_Q3_DIR="/tmp/forager_ckpts/clean_sft_qwen3vl"
RL_Q3_V2_DIR="/tmp/forager_ckpts/clean_rl_qwen3vl_v2"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching concurrent training on GPU 4 (Qwen2.5-VL v2 RL) & GPU 5 (Qwen3-VL SFT+RL v2)..."

(
    if [[ ! -d "$SFT_Q25_DIR/global_step_50" ]]; then
        CUDA_VISIBLE_DEVICES=4 python3 scripts/train_sft.py \
            --backend hf \
            --model-path "$QWEN25_PATH" \
            --train-jsonl "$ROOT_DIR/data/sft_coldstart_6k.jsonl" \
            --local-ckpt-dir "$SFT_Q25_DIR" \
            --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_sft/qwen25vl_7b" \
            --max-steps 50 \
            --batch-size 2 \
            --lr 2e-6 \
            > "$OUT_DIR/train_sft_qwen25vl_v2.log" 2>&1
    fi
    CUDA_VISIBLE_DEVICES=4 python3 scripts/train_ig_grpo.py \
        --mode hf \
        --model-path "$SFT_Q25_DIR/global_step_50" \
        --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
        --local-ckpt-dir "$RL_Q25_V2_DIR" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen25vl_7b_v2" \
        --max-steps 100 \
        --save-freq 50 \
        --group-size 4 \
        > "$OUT_DIR/train_ig_grpo_qwen25vl_v2.log" 2>&1
) &
pid_train_q25=$!

(
    CUDA_VISIBLE_DEVICES=5 python3 scripts/train_sft.py \
        --backend hf \
        --model-path "$QWEN3_PATH" \
        --train-jsonl "$ROOT_DIR/data/sft_coldstart_6k.jsonl" \
        --local-ckpt-dir "$SFT_Q3_DIR" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_sft/qwen3vl_8b" \
        --max-steps 50 \
        --batch-size 2 \
        --lr 2e-6 \
        > "$OUT_DIR/train_sft_qwen3vl.log" 2>&1

    CUDA_VISIBLE_DEVICES=5 python3 scripts/train_ig_grpo.py \
        --mode hf \
        --model-path "$SFT_Q3_DIR/global_step_50" \
        --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
        --local-ckpt-dir "$RL_Q3_V2_DIR" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen3vl_8b_v2" \
        --max-steps 100 \
        --save-freq 50 \
        --group-size 4 \
        > "$OUT_DIR/train_ig_grpo_qwen3vl_v2.log" 2>&1
) &
pid_train_q3=$!

python3 scripts/run_rl_dynamics_ablation.py \
    --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
    --max-steps 300 \
    --log-every 10 \
    --output-json "$ABL_DIR/ablation_c_rl_dynamics.json" \
    > "$OUT_DIR/ablation_c_rl_dynamics.log" 2>&1

wait "$pid_train_q25"
wait "$pid_train_q3"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed GPU 4 & GPU 5 SFT + 3D-IG-GRPO v2 training and Ablation C!"

# ------------------------------------------------------------------------------
# Step 1: Evaluate Qwen2.5-VL-7B Stage 3 v2 on Full 1,640-Sample OVO-Bench (GPUs 4,5,6,7)
# ------------------------------------------------------------------------------
Q25_V2_SHARDS="$SHARD_ROOT/full_qwen25vl_v2"
mkdir -p "$Q25_V2_SHARDS"
if [[ ! -f "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_v2_zero_leakage.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Full 1640-sample Qwen2.5-VL-7B Stage 3 v2 Zero-Leakage OVO-Bench on GPUs 4,5,6,7..."
    pids=()
    for idx in 0 1 2 3; do
        gpu_id=$((idx + 4))
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
            --backend hf \
            --model-path "$RL_Q25_V2_DIR/global_step_100" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --shard "${idx}/4" \
            --output-json "$Q25_V2_SHARDS/shard_${idx}.json" \
            > "$Q25_V2_SHARDS/shard_${idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_ovobench.py \
        --merge-shards "$Q25_V2_SHARDS"/shard_*.json \
        --output-json "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_v2_zero_leakage.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Full 1640-sample Qwen2.5-VL-7B Stage 3 v2!"
fi

# ------------------------------------------------------------------------------
# Step 2A: Evaluate Qwen3-VL-8B Stage 1 on Full 1,640-Sample OVO-Bench (GPUs 4,5,6,7)
# ------------------------------------------------------------------------------
Q3_S1_SHARDS="$SHARD_ROOT/full_qwen3vl_stage1"
mkdir -p "$Q3_S1_SHARDS"
if [[ ! -f "$OUT_DIR/ovobench_qwen3vl_stage1_zero_leakage.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Full 1640-sample Qwen3-VL-8B Stage 1 Zero-Leakage OVO-Bench on GPUs 4,5,6,7..."
    pids=()
    for idx in 0 1 2 3; do
        gpu_id=$((idx + 4))
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
            --backend hf \
            --model-path "$QWEN3_PATH" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --shard "${idx}/4" \
            --output-json "$Q3_S1_SHARDS/shard_${idx}.json" \
            > "$Q3_S1_SHARDS/shard_${idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_ovobench.py \
        --merge-shards "$Q3_S1_SHARDS"/shard_*.json \
        --output-json "$OUT_DIR/ovobench_qwen3vl_stage1_zero_leakage.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Full 1640-sample Qwen3-VL-8B Stage 1!"
fi

# ------------------------------------------------------------------------------
# Step 2B: Evaluate Qwen3-VL-8B Stage 3 v2 on Full 1,640-Sample OVO-Bench (GPUs 4,5,6,7)
# ------------------------------------------------------------------------------
Q3_V2_SHARDS="$SHARD_ROOT/full_qwen3vl_v2"
mkdir -p "$Q3_V2_SHARDS"
if [[ ! -f "$OUT_DIR/ovobench_qwen3vl_final_ig_grpo_v2_zero_leakage.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Full 1640-sample Qwen3-VL-8B Stage 3 v2 Zero-Leakage OVO-Bench on GPUs 4,5,6,7..."
    pids=()
    for idx in 0 1 2 3; do
        gpu_id=$((idx + 4))
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
            --backend hf \
            --model-path "$RL_Q3_V2_DIR/global_step_100" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --shard "${idx}/4" \
            --output-json "$Q3_V2_SHARDS/shard_${idx}.json" \
            > "$Q3_V2_SHARDS/shard_${idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_ovobench.py \
        --merge-shards "$Q3_V2_SHARDS"/shard_*.json \
        --output-json "$OUT_DIR/ovobench_qwen3vl_final_ig_grpo_v2_zero_leakage.json"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Full 1640-sample Qwen3-VL-8B Stage 3 v2!"
fi

# ------------------------------------------------------------------------------
# Step 3: Evaluate Ablation A (Orthogonal Tool Space) & Ablation B (Evidence Fusion & SDAM)
#         in a Single Fused Multi-Variant Pass across GPUs 4,5,6,7 (240 Stratified Samples = 20/task)
# ------------------------------------------------------------------------------
ABL_SHARDS="$SHARD_ROOT/ablations_multi_pass"
mkdir -p "$ABL_SHARDS"
if [[ ! -f "$ABL_DIR/ablation_a_orthogonal_tools.json" ]]; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Running Single-Pass Multi-Variant Ablations A & B (N=240, 20/task) on GPUs 4,5,6,7..."
    pids=()
    for idx in 0 1 2 3; do
        gpu_id=$((idx + 4))
        CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_h800_ablations.py \
            --backend hf \
            --model-path "$RL_Q25_V2_DIR/global_step_100" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --max-samples-per-task 20 \
            --variants answer_only zoom_only recall_only wait_only naive_raw_concat card_no_sdam full_forager \
            --shard "${idx}/4" \
            --output-dir "$ABL_SHARDS" \
            > "$ABL_SHARDS/shard_${idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    for var in answer_only zoom_only recall_only wait_only naive_raw_concat card_no_sdam full_forager; do
        python3 scripts/eval_h800_ablations.py \
            --variant "$var" \
            --baseline-json "$ABL_DIR/ovobench_answer_only.json" \
            --merge-shards "$ABL_SHARDS/shards_${var}"/shard_*.json \
            --output-json "$ABL_DIR/ovobench_${var}.json"
    done

    python3 -c "
import json
from pathlib import Path
abl_dir = Path('$ABL_DIR')
ab_a = {
    v: json.loads((abl_dir / f'ovobench_{v}.json').read_text())['summary']
    for v in ('answer_only', 'zoom_only', 'recall_only', 'wait_only', 'full_forager')
}
(abl_dir / 'ablation_a_orthogonal_tools.json').write_text(json.dumps(ab_a, indent=2))
ab_b = {
    'simplestream_4f': ab_a['answer_only'],
    'naive_raw_concat': json.loads((abl_dir / 'ovobench_naive_raw_concat.json').read_text())['summary'],
    'card_no_sdam': json.loads((abl_dir / 'ovobench_card_no_sdam.json').read_text())['summary'],
    'card_plus_sdam': ab_a['full_forager'],
}
(abl_dir / 'ablation_b_evidence_fusion.json').write_text(json.dumps(ab_b, indent=2))
"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Ablation A & Ablation B!"
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === All H20 GPU 4-7 Zero-Leakage Experiments Complete! ==="
