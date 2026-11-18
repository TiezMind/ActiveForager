#!/usr/bin/env bash
# ==============================================================================
# Isolated H800 Ablation & ZoomIn Enhancement Runner (`scripts/run_h800_ablations.sh`)
#
# Target Worker: 4480127 (8x NVIDIA H800 80GB)
# Isolation Guarantee:
#   - Local Scratch : /tmp/forager_h800_ablations
#   - Local Outputs : outputs/h800_ablations/
#   - Local Logs    : outputs/logs/h800_ablations_*.log
#   - HDFS Export   : /mnt/hdfs/lipengyu/DA/checkpoints/forager_h800_ablations/
#   - NEVER touches /mnt/hdfs/lipengyu/DA/checkpoints/forager/{sft,rl} used by H20!
# ==============================================================================
set -euo pipefail

MODE="${1:-all}"
DRY_RUN="${2:-0}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export H800_SCRATCH="/tmp/forager_h800_ablations"
export H800_OUT_DIR="$ROOT/outputs/h800_ablations"
export H800_LOG_DIR="$ROOT/outputs/logs"
export H800_HDFS_CKPT="/mnt/hdfs/lipengyu/DA/checkpoints/forager_h800_ablations"

mkdir -p "$H800_SCRATCH" "$H800_OUT_DIR" "$H800_LOG_DIR" "$H800_HDFS_CKPT" /tmp/da_models

if [[ -f "$ROOT/scripts/driver_env.sh" && -e /proc/driver/nvidia/version ]]; then
    set +e
    export DA_SCRATCH="$H800_SCRATCH"
    # shellcheck disable=SC1091
    source "$ROOT/scripts/driver_env.sh"
    set -e
fi

export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NGPUS=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

OVOBENCH_JSON="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json"
OVOBENCH_VIDEOS="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos"
BASELINE_JSON="$ROOT/outputs/ovobench_qwen25vl_stage1.json"

# Prefer the trained IG-GRPO / SFT checkpoint from HDFS when staging to 4480127 local NVMe
HDFS_RL_CKPT="/mnt/hdfs/lipengyu/DA/checkpoints/forager/rl/qwen25vl_ig_grpo"
HDFS_SFT_CKPT="/mnt/hdfs/lipengyu/DA/checkpoints/forager/sft/qwen25vl_sft"
HDFS_BASE_MODEL="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct"
LOCAL_MODEL_DIR="/tmp/da_models/Qwen2.5-VL-7B-Forager-Eval"

stage_eval_model() {
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        echo "[DRY-RUN] Skipping model staging"
        return 0
    fi
    local src_model="$HDFS_BASE_MODEL"
    if [[ -f "$HDFS_RL_CKPT/config.json" && -f "$HDFS_RL_CKPT/.export_complete" ]]; then
        src_model="$HDFS_RL_CKPT"
    elif [[ -f "$HDFS_SFT_CKPT/config.json" && -f "$HDFS_SFT_CKPT/.export_complete" ]]; then
        src_model="$HDFS_SFT_CKPT"
    fi
    if [[ ! -f "$LOCAL_MODEL_DIR/config.json" ]]; then
        echo "[H800 Staging] Copying $src_model -> $LOCAL_MODEL_DIR on local NVMe..."
        mkdir -p "$LOCAL_MODEL_DIR.tmp.$$"
        cp -r "$src_model/"* "$LOCAL_MODEL_DIR.tmp.$$/"
        mv -T "$LOCAL_MODEL_DIR.tmp.$$" "$LOCAL_MODEL_DIR" || rm -rf "$LOCAL_MODEL_DIR.tmp.$$"
    else
        echo "[H800 Staging] Model ready at $LOCAL_MODEL_DIR"
    fi
}

run_8gpu_variant() {
    local variant="$1"
    local out_file="$H800_OUT_DIR/ovobench_${variant}.json"
    if [[ -f "$out_file" && "$DRY_RUN" != "--dry-run" ]]; then
        echo "[Skip] $out_file already exists."
        return 0
    fi
    echo ">>> [H800 Ablation] Evaluating variant=$variant across $NGPUS x H800..."
    if [[ "$DRY_RUN" == "--dry-run" ]]; then
        python3 scripts/eval_h800_ablations.py \
            --backend mock \
            --variant "$variant" \
            --data-json "$OVOBENCH_JSON" \
            --baseline-json "$BASELINE_JSON" \
            --max-samples-per-task 2 \
            --output-json "$H800_OUT_DIR/dryrun_${variant}.json"
        return 0
    fi

    local shard_dir="$H800_SCRATCH/shards_${variant}"
    mkdir -p "$shard_dir"
    local pids=()
    local shard_files=()
    for gpu_idx in $(seq 0 $((NGPUS - 1))); do
        local sf="$shard_dir/shard_${gpu_idx}.json"
        shard_files+=("$sf")
        CUDA_VISIBLE_DEVICES="$gpu_idx" python3 scripts/eval_h800_ablations.py \
            --backend hf \
            --model-path "$LOCAL_MODEL_DIR" \
            --variant "$variant" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --baseline-json "$BASELINE_JSON" \
            --shard "${gpu_idx}/${NGPUS}" \
            --output-json "$sf" > "$shard_dir/shard_${gpu_idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done
    python3 scripts/eval_h800_ablations.py \
        --variant "$variant" \
        --baseline-json "$BASELINE_JSON" \
        --merge-shards "${shard_files[@]}" \
        --output-json "$out_file"
}

# 1. Run Ablation C (300-Step RL Training Dynamics: Vanilla GRPO vs StreamScout vs PARA-GRPO vs IG-GRPO)
echo ">>> [1/3] Running Ablation C (300-Step RL Training Dynamics & Tool Collapse Comparison)..."
python3 scripts/run_rl_dynamics_ablation.py \
    --rl-jsonl "$ROOT/data/rl_ig_grpo_8k.jsonl" \
    --max-steps 300 \
    --log-every 10 \
    --output-json "$H800_OUT_DIR/ablation_c_rl_dynamics_300steps.json"

# 2. Stage model & Run All Ablation Variants in a Single Fused 8-GPU Pass
stage_eval_model

if [[ "$DRY_RUN" == "--dry-run" ]]; then
    echo ">>> [2/3] Running Enhanced Full Forager (True 896x896 ZoomIn + SDAM + Sub-Agent Card)..."
    run_8gpu_variant "full_forager"
    if [[ "$MODE" == "all" ]]; then
        echo ">>> [3/3] Running Orthogonal Tool Decomposition (Ablation A) & Fusion Variants (Ablation B)..."
        for var in "zoom_only" "recall_only" "wait_only" "naive_raw_concat" "card_no_sdam"; do
            run_8gpu_variant "$var"
        done
    fi
else
    ALL_VARS=("full_forager")
    if [[ "$MODE" == "all" ]]; then
        ALL_VARS=("full_forager" "zoom_only" "recall_only" "wait_only" "naive_raw_concat" "card_no_sdam")
    fi
    echo ">>> [2/2] Running Fused Single-Pass 8-GPU Evaluation for variants: ${ALL_VARS[*]}..."
    mkdir -p "$H800_SCRATCH/multi_pass_logs"
    pids=()
    for gpu_idx in $(seq 0 $((NGPUS - 1))); do
        CUDA_VISIBLE_DEVICES="$gpu_idx" python3 scripts/eval_h800_ablations.py \
            --backend hf \
            --model-path "$LOCAL_MODEL_DIR" \
            --variants "${ALL_VARS[@]}" \
            --output-dir "$H800_SCRATCH" \
            --data-json "$OVOBENCH_JSON" \
            --video-root "$OVOBENCH_VIDEOS" \
            --baseline-json "$BASELINE_JSON" \
            --shard "${gpu_idx}/${NGPUS}" > "$H800_SCRATCH/multi_pass_logs/shard_${gpu_idx}.log" 2>&1 &
        pids+=("$!")
    done
    for pid in "${pids[@]}"; do
        wait "$pid"
    done

    for var in "${ALL_VARS[@]}"; do
        shard_files=()
        for gpu_idx in $(seq 0 $((NGPUS - 1))); do
            shard_files+=("$H800_SCRATCH/shards_${var}/shard_${gpu_idx}.json")
        done
        python3 scripts/eval_h800_ablations.py \
            --variant "$var" \
            --baseline-json "$BASELINE_JSON" \
            --merge-shards "${shard_files[@]}" \
            --output-json "$H800_OUT_DIR/ovobench_${var}.json"
    done
fi

echo "=== [H800 Ablation Suite Completed] Outputs in $H800_OUT_DIR ==="
