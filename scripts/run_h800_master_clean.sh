#!/usr/bin/env bash
# ==============================================================================
# Master 8xH800 Zero-Leakage Training & Full 1,640-Sample Evaluation Pipeline
# (Worker 4480127: GPUs 0,1,2,3,4,5,6,7)
# ==============================================================================
set -eo pipefail

ROOT_DIR="/mlx/users/lipengyu.seed/playground/agentic_video_understanding"
cd "$ROOT_DIR"
export DA_SCRATCH="${DA_SCRATCH:-/tmp/forager_scratch}"
mkdir -p "$DA_SCRATCH" /tmp/da_models /tmp/forager_ckpts "$ROOT_DIR/outputs/logs"
source scripts/driver_env.sh
set -u

export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

OVOBENCH_JSON="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json"
OVOBENCH_VIDEOS="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos"
QWEN25_BASE="/tmp/da_models/Qwen2.5-VL-7B-Instruct"
QWEN3_BASE="/tmp/da_models/Qwen3-VL-8B-Instruct"

OUT_DIR="$ROOT_DIR/outputs/clean_zero_leakage"
ABL_DIR="$OUT_DIR/ablations"
SHARD_ROOT="/tmp/forager_clean_h800"
mkdir -p "$OUT_DIR" "$ABL_DIR" "$SHARD_ROOT"

SFT_Q25="/tmp/forager_ckpts/clean_sft_qwen25vl_v4"
RL_Q25="/tmp/forager_ckpts/clean_rl_qwen25vl_v4"
SFT_Q3="/tmp/forager_ckpts/clean_sft_qwen3vl_v4"
RL_Q3="/tmp/forager_ckpts/clean_rl_qwen3vl_v4"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1: Training Balanced SDAM-SFT + 3D-IG-GRPO (GPUs 0,1: Qwen2.5-VL-7B | GPUs 2,3: Qwen3-VL-8B) & Ablation C ==="

(
    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_sft.py \
        --backend hf \
        --model-path "$QWEN25_BASE" \
        --train-jsonl "$ROOT_DIR/data/sft_coldstart_6k.jsonl" \
        --local-ckpt-dir "$SFT_Q25" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_sft/qwen25vl_7b_v4" \
        --max-steps 25 \
        --batch-size 2 \
        --lr 2e-7 \
        > "$OUT_DIR/train_sft_qwen25vl_v3.log" 2>&1

    CUDA_VISIBLE_DEVICES=0,1 python3 scripts/train_ig_grpo.py \
        --mode hf \
        --model-path "$SFT_Q25/global_step_25" \
        --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
        --local-ckpt-dir "$RL_Q25" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen25vl_7b_v4" \
        --max-steps 50 \
        --save-freq 25 \
        --group-size 4 \
        > "$OUT_DIR/train_ig_grpo_qwen25vl_v3.log" 2>&1
) &
p_q25=$!

(
    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/train_sft.py \
        --backend hf \
        --model-path "$QWEN3_BASE" \
        --train-jsonl "$ROOT_DIR/data/sft_coldstart_6k.jsonl" \
        --local-ckpt-dir "$SFT_Q3" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_sft/qwen3vl_8b_v4" \
        --max-steps 25 \
        --batch-size 2 \
        --lr 2e-7 \
        > "$OUT_DIR/train_sft_qwen3vl_v3.log" 2>&1

    CUDA_VISIBLE_DEVICES=2,3 python3 scripts/train_ig_grpo.py \
        --mode hf \
        --model-path "$SFT_Q3/global_step_25" \
        --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
        --local-ckpt-dir "$RL_Q3" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen3vl_8b_v4" \
        --max-steps 50 \
        --save-freq 25 \
        --group-size 4 \
        > "$OUT_DIR/train_ig_grpo_qwen3vl_v3.log" 2>&1
) &
p_q3=$!

python3 scripts/run_rl_dynamics_ablation.py \
    --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
    --max-steps 300 \
    --log-every 10 \
    --output-json "$ABL_DIR/ablation_c_rl_dynamics.json" \
    > "$OUT_DIR/ablation_c_rl_dynamics.log" 2>&1 &
p_ab_c=$!

wait "$p_q25"
wait "$p_q3"
wait "$p_ab_c"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Step 1 (SFT + 3D-IG-GRPO v4 on Qwen2.5-VL & Qwen3-VL + Ablation C)!"

# ==============================================================================
# Step 2: 8-GPU Concurrent Wave 1
# - GPUs 0,1,2,3 (4 shards): Ablations A & B (N=360, 30 samples/task across all 7 variants)
# - GPUs 4,5 (2 shards): Video-Holmes Qwen2.5-VL-7B (N=140, 20 samples/subtask)
# - GPUs 6,7 (2 shards): Video-Holmes Qwen3-VL-8B (N=140, 20 samples/subtask)
# ==============================================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 2: Launching 8-GPU Wave 1 (Ablations A&B on GPUs 0-3 | Video-Holmes on GPUs 4-7) ==="
ABL_SHARDS="$SHARD_ROOT/ablations"
VH_Q25_SHARDS="$SHARD_ROOT/vh_q25"
VH_Q3_SHARDS="$SHARD_ROOT/vh_q3"
rm -rf "$ABL_SHARDS" "$VH_Q25_SHARDS" "$VH_Q3_SHARDS"
mkdir -p "$ABL_SHARDS" "$VH_Q25_SHARDS" "$VH_Q3_SHARDS"

pids_w1=()
for idx in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$idx" python3 scripts/eval_h800_ablations.py \
        --backend hf \
        --model-path "$RL_Q25/global_step_50" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --max-samples-per-task 30 \
        --variants answer_only zoom_only recall_only wait_only naive_raw_concat card_no_sdam full_forager \
        --shard "${idx}/4" \
        --output-dir "$ABL_SHARDS" \
        > "$ABL_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w1+=("$!")
done

for idx in 0 1; do
    gpu_id=$((idx + 4))
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_videoholmes.py \
        --backend hf \
        --model-path "$RL_Q25/global_step_50" \
        --max-samples-per-subtask 20 \
        --shard "${idx}/2" \
        --output-json "$VH_Q25_SHARDS/shard_${idx}.json" \
        > "$VH_Q25_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w1+=("$!")
done

for idx in 0 1; do
    gpu_id=$((idx + 6))
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_videoholmes.py \
        --backend hf \
        --model-path "$RL_Q3/global_step_50" \
        --max-samples-per-subtask 20 \
        --shard "${idx}/2" \
        --output-json "$VH_Q3_SHARDS/shard_${idx}.json" \
        > "$VH_Q3_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w1+=("$!")
done

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

python3 scripts/eval_videoholmes.py \
    --merge-shards "$VH_Q25_SHARDS"/shard_*.json \
    --output-json "$OUT_DIR/videoholmes_qwen25vl_ig_grpo_v2_zero_leakage.json"

python3 scripts/eval_videoholmes.py \
    --merge-shards "$VH_Q3_SHARDS"/shard_*.json \
    --output-json "$OUT_DIR/videoholmes_qwen3vl_ig_grpo_v2_zero_leakage.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Step 2 (Ablations A & B N=360 + Video-Holmes N=140)!"

# ==============================================================================
# Step 3: Full 1,640-Sample OVO-Bench Stage 3 Evaluation on Qwen2.5-VL-7B across all 8 H800 GPUs
# ==============================================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 3: Running Full 1,640-Sample Qwen2.5-VL-7B Stage 3 (3D-IG-GRPO) across all 8 H800 GPUs ==="
Q25_FULL_SHARDS="$SHARD_ROOT/q25_s3_full"
rm -rf "$Q25_FULL_SHARDS"
mkdir -p "$Q25_FULL_SHARDS"

pids_w2=()
for idx in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$idx" python3 scripts/eval_ovobench.py \
        --backend hf \
        --model-path "$RL_Q25/global_step_50" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --shard "${idx}/8" \
        --output-json "$Q25_FULL_SHARDS/shard_${idx}.json" \
        > "$Q25_FULL_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w2+=("$!")
done

for pid in "${pids_w2[@]}"; do
    wait "$pid"
done

python3 scripts/eval_ovobench.py \
    --merge-shards "$Q25_FULL_SHARDS"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_v2_zero_leakage.json"

cp "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_v2_zero_leakage.json" "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_zero_leakage.json"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Step 3 (Full N=1,640 Qwen2.5-VL-7B 3D-IG-GRPO)!"

# ==============================================================================
# Step 4: Full 1,640-Sample OVO-Bench Stage 3 Evaluation on Qwen3-VL-8B across all 8 H800 GPUs
# ==============================================================================
echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 4: Running Full 1,640-Sample Qwen3-VL-8B Stage 3 (3D-IG-GRPO) across all 8 H800 GPUs ==="
Q3_FULL_SHARDS="$SHARD_ROOT/q3_s3_full"
rm -rf "$Q3_FULL_SHARDS"
mkdir -p "$Q3_FULL_SHARDS"

pids_w3=()
for idx in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES="$idx" python3 scripts/eval_ovobench.py \
        --backend hf \
        --model-path "$RL_Q3/global_step_50" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --shard "${idx}/8" \
        --output-json "$Q3_FULL_SHARDS/shard_${idx}.json" \
        > "$Q3_FULL_SHARDS/shard_${idx}.log" 2>&1 &
    pids_w3+=("$!")
done

for pid in "${pids_w3[@]}"; do
    wait "$pid"
done

python3 scripts/eval_ovobench.py \
    --merge-shards "$Q3_FULL_SHARDS"/shard_*.json \
    --output-json "$OUT_DIR/ovobench_qwen3vl_final_ig_grpo_v2_zero_leakage.json"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === All 8xH800 Zero-Leakage Evaluations Completed Successfully! ==="
