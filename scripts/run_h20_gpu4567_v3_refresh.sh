#!/usr/bin/env bash
# ==============================================================================
# Strict H20 GPU 4,5,6,7 Zero-Leakage SFT+RL Retrain (with Turn-1 Supervision Fix)
# & Fast Evaluation Refresh
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

SFT_Q25_V3="/tmp/forager_ckpts/clean_sft_qwen25vl_v3"
RL_Q25_V3="/tmp/forager_ckpts/clean_rl_qwen25vl_v3"
SFT_Q3_V3="/tmp/forager_ckpts/clean_sft_qwen3vl_v3"
RL_Q3_V3="/tmp/forager_ckpts/clean_rl_qwen3vl_v3"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Step 1: Retraining Balanced Turn-1+Turn-2 SFT & 3D-IG-GRPO on GPU 4 (Qwen2.5-VL) and GPU 5 (Qwen3-VL) ==="

(
    CUDA_VISIBLE_DEVICES=4 python3 scripts/train_sft.py \
        --backend hf \
        --model-path "$QWEN25_PATH" \
        --train-jsonl "$ROOT_DIR/data/sft_coldstart_6k.jsonl" \
        --local-ckpt-dir "$SFT_Q25_V3" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_sft/qwen25vl_7b_v3" \
        --max-steps 40 \
        --batch-size 2 \
        --lr 2e-6 \
        > "$OUT_DIR/train_sft_qwen25vl_v3.log" 2>&1

    CUDA_VISIBLE_DEVICES=4 python3 scripts/train_ig_grpo.py \
        --mode hf \
        --model-path "$SFT_Q25_V3/global_step_40" \
        --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
        --local-ckpt-dir "$RL_Q25_V3" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen25vl_7b_v3" \
        --max-steps 80 \
        --save-freq 40 \
        --group-size 4 \
        > "$OUT_DIR/train_ig_grpo_qwen25vl_v3.log" 2>&1
) &
p_q25=$!

(
    CUDA_VISIBLE_DEVICES=5 python3 scripts/train_sft.py \
        --backend hf \
        --model-path "$QWEN3_PATH" \
        --train-jsonl "$ROOT_DIR/data/sft_coldstart_6k.jsonl" \
        --local-ckpt-dir "$SFT_Q3_V3" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_sft/qwen3vl_8b_v3" \
        --max-steps 40 \
        --batch-size 2 \
        --lr 2e-6 \
        > "$OUT_DIR/train_sft_qwen3vl_v3.log" 2>&1

    CUDA_VISIBLE_DEVICES=5 python3 scripts/train_ig_grpo.py \
        --mode hf \
        --model-path "$SFT_Q3_V3/global_step_40" \
        --rl-jsonl "$ROOT_DIR/data/rl_ig_grpo_8k.jsonl" \
        --local-ckpt-dir "$RL_Q3_V3" \
        --hdfs-export-dir "/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen3vl_8b_v3" \
        --max-steps 80 \
        --save-freq 40 \
        --group-size 4 \
        > "$OUT_DIR/train_ig_grpo_qwen3vl_v3.log" 2>&1
) &
p_q3=$!

wait "$p_q25"
wait "$p_q3"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Completed Turn-1+Turn-2 SFT + 3D-IG-GRPO v3 training!"

# ------------------------------------------------------------------------------
# Step 2: Evaluate Ablations A & B (GPUs 4,5) + Qwen2.5-VL v3 & Qwen3-VL v3 (GPUs 6,7)
# ------------------------------------------------------------------------------
ABL_SHARDS="$SHARD_ROOT/ablations_v3"
Q25_V3_SHARDS="$SHARD_ROOT/qwen25vl_v3"
Q3_V3_SHARDS="$SHARD_ROOT/qwen3vl_v3"
rm -rf "$ABL_SHARDS" "$Q25_V3_SHARDS" "$Q3_V3_SHARDS"
mkdir -p "$ABL_SHARDS" "$Q25_V3_SHARDS" "$Q3_V3_SHARDS"

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Launching 4-GPU concurrent evaluation (GPUs 4,5: Ablations A&B | GPU 6: Qwen2.5-VL v3 | GPU 7: Qwen3-VL v3)..."
pids_eval=()
for idx in 0 1; do
    gpu_id=$((idx + 4))
    CUDA_VISIBLE_DEVICES="$gpu_id" python3 scripts/eval_h800_ablations.py \
        --backend hf \
        --model-path "$RL_Q25_V3/global_step_80" \
        --data-json "$OVOBENCH_JSON" \
        --video-root "$OVOBENCH_VIDEOS" \
        --max-samples-per-task 15 \
        --variants answer_only zoom_only recall_only wait_only naive_raw_concat card_no_sdam full_forager \
        --shard "${idx}/2" \
        --output-dir "$ABL_SHARDS" \
        > "$ABL_SHARDS/shard_${idx}.log" 2>&1 &
    pids_eval+=("$!")
done

CUDA_VISIBLE_DEVICES=6 python3 scripts/eval_ovobench.py \
    --backend hf \
    --model-path "$RL_Q25_V3/global_step_80" \
    --data-json "$OVOBENCH_JSON" \
    --video-root "$OVOBENCH_VIDEOS" \
    --max-samples-per-task 20 \
    --shard "0/1" \
    --output-json "$OUT_DIR/ovobench_qwen25vl_final_ig_grpo_v2_zero_leakage.json" \
    > "$Q25_V3_SHARDS/eval.log" 2>&1 &
pids_eval+=("$!")

CUDA_VISIBLE_DEVICES=7 python3 scripts/eval_ovobench.py \
    --backend hf \
    --model-path "$RL_Q3_V3/global_step_80" \
    --data-json "$OVOBENCH_JSON" \
    --video-root "$OVOBENCH_VIDEOS" \
    --max-samples-per-task 20 \
    --shard "0/1" \
    --output-json "$OUT_DIR/ovobench_qwen3vl_final_ig_grpo_v2_zero_leakage.json" \
    > "$Q3_V3_SHARDS/eval.log" 2>&1 &
pids_eval+=("$!")

for pid in "${pids_eval[@]}"; do
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

echo "[$(date '+%Y-%m-%d %H:%M:%S')] === Completed v3 Retrain & Evaluation Refresh! ==="
