#!/usr/bin/env bash
# ==============================================================================
# Strict H20 GPU 4,5,6,7 Zero-Leakage Fast Single-Decode Master Pipeline
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

RL_Q25_V2_DIR="/tmp/forager_ckpts/clean_rl_qwen25vl_v2"
RL_Q3_V2_DIR="/tmp/forager_ckpts/clean_rl_qwen3vl_v2"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Starting Wave-Parallel Zero-Leakage Evaluation on GPUs 4,5,6,7 ==="

# ------------------------------------------------------------------------------
# Wave 1 (Simultaneous across GPUs 4,5,6,7):
#   - GPUs 4,5: Ablations A & B (7 variants in a single pass, N=180 = 15/task)
#   - GPU 6: Video-Holmes Real-Video Zero-Shot Generalization (Qwen2.5-VL-7B v2, N=140 = 20/subtask)
#   - GPU 7: Video-Holmes Real-Video Zero-Shot Generalization (Qwen3-VL-8B v2, N=140 = 20/subtask)
# ------------------------------------------------------------------------------
ABL_SHARDS="$SHARD_ROOT/ablations_multi_pass_fast"
mkdir -p "$ABL_SHARDS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Wave 1 (GPUs 4,5: Ablations A & B | GPU 6: Video-Holmes Qwen2.5-VL | GPU 7: Video-Holmes Qwen3-VL)..."
pids_w1=()
for idx in 0 1; do
    gpu_id=$((idx + 4))
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_h800_ablations.py \
        --backend hf \
        --model-path "$RL_Q25_V2_DIR/global_step_100" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --max-samples-per-task 15 \
        --variants answer_only zoom_only recall_only wait_only naive_raw_concat card_no_sdam full_forager \
        --shard "${idx}/2" \
        --output-dir "$ABL_SHARDS" \
        > "$ABL_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w1+=("$!")
done

CUDA_VISIBLE_DEVICES=6 python3 scripts/eval_videoholmes.py \
    --backend hf \
    --model-path "$RL_Q25_V2_DIR/global_step_100" \
    --max-samples-per-subtask 20 \
    --shard "0/1" \
    --output-json "$OUT_DIR/videoholmes_qwen25vl_ig_grpo_v2_zero_leakage.json" \
    > "$OUT_DIR/videoholmes_qwen25vl.log" 2>&1 &
pids_w1+=("$!")

CUDA_VISIBLE_DEVICES=7 python3 scripts/eval_videoholmes.py \
    --backend hf \
    --model-path "$RL_Q3_V2_DIR/global_step_100" \
    --max-samples-per-subtask 20 \
    --shard "0/1" \
    --output-json "$OUT_DIR/videoholmes_qwen3vl_ig_grpo_v2_zero_leakage.json" \
    > "$OUT_DIR/videoholmes_qwen3vl.log" 2>&1 &
pids_w1+=("$!")

for pid in "${pids_w1[@]}"; do
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
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Wave 1 (Ablations A & B + Video-Holmes Qwen2.5-VL & Qwen3-VL)!"

# ------------------------------------------------------------------------------
# Wave 2 (Simultaneous across GPUs 4,5,6,7 with Single-Decode Joint Evaluator):
#   - GPUs 4,5: Qwen2.5-VL-7B Stage 3 v2 Zero-Leakage OVO-Bench
#   - GPUs 6,7: Qwen3-VL-8B Stage 1 Zero-Leakage OVO-Bench
# ------------------------------------------------------------------------------
Q25_V2_SHARDS="$SHARD_ROOT/fast_qwen25vl_v2"
Q3_S1_SHARDS="$SHARD_ROOT/fast_qwen3vl_stage1"
Q3_V2_SHARDS="$SHARD_ROOT/fast_qwen3vl_v2"
mkdir -p "$Q25_V2_SHARDS" "$Q3_S1_SHARDS" "$Q3_V2_SHARDS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Wave 2A: Qwen2.5-VL-7B Stage 3 v2 (GPUs 4,5) + Qwen3-VL-8B Stage 1 (GPUs 6,7)..."
pids_w2=()
for idx in 0 1; do
    gpu_q25=$((idx + 4))
    CUDA_VISIBLE_DEVICES="$gpu_q25" python3 scripts/eval_ovobench.py \
        --backend hf \
        --model-path "$RL_Q25_V2_DIR/global_step_100" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --max-samples-per-task 30 \
        --shard "${idx}/2" \
        --output-json "$Q25_V2_SHARDS/shard_${idx}.json" \
        > "$Q25_V2_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w2+=("$!")

    gpu_q3=$((idx + 6))
    CUDA_VISIBLE_DEVICES="$gpu_q3" python3 scripts/eval_ovobench.py \
        --backend hf \
        --model-path "$QWEN3_PATH" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --max-samples-per-task 30 \
        --shard "${idx}/2" \
        --output-json "$Q3_S1_SHARDS/shard_${idx}.json" \
        > "$Q3_S1_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w2+=("$!")
done

for pid in "${pids_w2[@]}"; do
    wait "$pid"
done

python3 scripts/eval_ovobench.py \
    --merge-shards "$Q25_V2_SHARDS"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_v2_zero_leakage.json"

python3 scripts/eval_ovobench.py \
    --merge-shards "$Q3_S1_SHARDS"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen3vl_stage1_zero_leakage.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching Wave 2B: Qwen3-VL-8B Stage 3 v2 on all 4 GPUs (4,5,6,7)..."
pids_w2b=()
for idx in 0 1 2 3; do
    gpu_id=$((idx + 4))
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_ovobench.py \
        --backend hf \
        --model-path "$RL_Q3_V2_DIR/global_step_100" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --max-samples-per-task 30 \
        --shard "${idx}/4" \
        --output-json "$Q3_V2_SHARDS/shard_${idx}.json" \
        > "$Q3_V2_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w2b+=("$!")
done

for pid in "${pids_w2b[@]}"; do
    wait "$pid"
done

python3 scripts/eval_ovobench.py \
    --merge-shards "$Q3_V2_SHARDS"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen3vl_final_ig_grpo_v2_zero_leakage.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === All Wave-Parallel Zero-Leakage Evaluations Complete! ==="
