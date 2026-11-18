#!/usr/bin/env bash
# ==============================================================================
# Turnkey H800 Launch Script for Spatio-Temporal Foraging Agent
# (`scripts/launch_h800_all.sh`)
#
# Supports:
#   bash scripts/launch_h800_all.sh [--stage all|stage1|stage2|stage3] [--backbone qwen25vl|qwen3vl] [--dry-run]
#
# Strictly enforces INFRA_LESSONS.md operational guardrails:
#   1. Sources driver_env.sh (NVML & CUDA forward-compat protection)
#   2. Never kills gpu_keeper; sets vLLM gpu_memory_utilization=0.80 (<= 0.95)
#   3. Stages models to worker local NVMe /tmp/da_models (copy-only, no overwrite)
#   4. Saves intermediate SFT/RL checkpoints strictly to /tmp/forager_ckpts (max_keep=2)
#   5. Exports final HF weights sequentially to /mnt/hdfs/lipengyu/DA/checkpoints/forager/
# ==============================================================================
set -euo pipefail

STAGE="all"
BACKBONE="qwen25vl"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --stage)
            STAGE="$2"
            shift 2
            ;;
        --backbone)
            BACKBONE="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift 1
            ;;
        *)
            echo "Usage: bash scripts/launch_h800_all.sh [--stage all|stage1|stage2|stage3] [--backbone qwen25vl|qwen3vl] [--dry-run]" >&2
            exit 2
            ;;
    esac
done

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export DA_SCRATCH="${DA_SCRATCH:-/tmp/forager_scratch}"
mkdir -p "$DA_SCRATCH" /tmp/da_models /tmp/forager_ckpts "$ROOT/outputs/logs"

# 1. Source driver_env.sh (local scripts/ or sibling DA-train-video) to protect libnvidia-ml.so.1 & libcuda.so.1
DRIVER_ENV="$ROOT/scripts/driver_env.sh"
[[ -f "$DRIVER_ENV" ]] || DRIVER_ENV="/mlx/users/lipengyu.seed/playground/DA-train-video/scripts/driver_env.sh"
if [[ -f "$DRIVER_ENV" && -e /proc/driver/nvidia/version ]]; then
    set +e
    # shellcheck disable=SC1090
    source "$DRIVER_ENV"
    set -e
fi

# 2. Environment & network loopback guardrails
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NO_PROXY="127.0.0.1,localhost,${NO_PROXY:-}"
export no_proxy="127.0.0.1,localhost,${no_proxy:-}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export NCCL_SOCKET_FAMILY="${NCCL_SOCKET_FAMILY:-AF_INET}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"

# Auto-detect cluster topology (single-node 8xH800 vs multi-node 8x8 H800)
NNODES="${ARNOLD_WORKER_NUM:-${NNODES:-1}}"
NGPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr ',' '\n' | wc -l)

TRAIN_VENV_PY="/mlx/users/lipengyu.seed/playground/DA-train/envs/train/.venv/bin/python"
if [[ -x "$TRAIN_VENV_PY" ]]; then
    PYTHON_BIN="${PYTHON_BIN:-python3}"
    export DA_TRAIN_PYTHON="$TRAIN_VENV_PY"
else
    PYTHON_BIN="python3"
    export DA_TRAIN_PYTHON="python3"
fi

# 3. Resolve backbone paths (HDFS canonical -> worker /tmp/da_models local NVMe staging)
if [[ "$BACKBONE" == "qwen25vl" ]]; then
    MODEL_TAG="Qwen2.5-VL-7B-Instruct"
    HDFS_MODEL_SRC="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct"
elif [[ "$BACKBONE" == "qwen3vl" ]]; then
    MODEL_TAG="Qwen3-VL-8B-Instruct"
    HDFS_MODEL_SRC="/mnt/hdfs/lipengyu/DA/models/Qwen3-VL-8B-Instruct"
else
    echo "Unsupported --backbone $BACKBONE (use qwen25vl or qwen3vl)" >&2
    exit 2
fi

LOCAL_MODEL_DIR="/tmp/da_models/$MODEL_TAG"
OVOBENCH_JSON="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json"
OVOBENCH_VIDEOS="/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos"
SFT_LOCAL_CKPT="/tmp/forager_ckpts/sft/${BACKBONE}_sft"
SFT_HDFS_EXPORT="/mnt/hdfs/lipengyu/DA/checkpoints/forager/sft/${BACKBONE}_sft"
RL_LOCAL_CKPT="/tmp/forager_ckpts/rl/${BACKBONE}_ig_grpo"
RL_HDFS_EXPORT="/mnt/hdfs/lipengyu/DA/checkpoints/forager/rl/${BACKBONE}_ig_grpo"

echo "=================================================================="
echo "[Forager H800 Launcher]"
echo "  Stage          : $STAGE (dry_run=$DRY_RUN)"
echo "  Backbone       : $MODEL_TAG"
echo "  Topology       : NNODES=$NNODES x NGPUS_PER_NODE=$NGPUS_PER_NODE (CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES)"
echo "  OVO-Bench JSON : $OVOBENCH_JSON"
echo "  OVO-Bench Vids : $OVOBENCH_VIDEOS"
echo "  Local SFT Ckpt : $SFT_LOCAL_CKPT -> HDFS: $SFT_HDFS_EXPORT"
echo "  Local RL Ckpt  : $RL_LOCAL_CKPT  -> HDFS: $RL_HDFS_EXPORT"
echo "=================================================================="

stage_model_to_nvme() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY-RUN] Would stage $HDFS_MODEL_SRC -> $LOCAL_MODEL_DIR"
        return 0
    fi
    if [[ ! -f "$LOCAL_MODEL_DIR/config.json" ]]; then
        echo "[Staging] Copying $HDFS_MODEL_SRC -> $LOCAL_MODEL_DIR on local NVMe..."
        mkdir -p "$LOCAL_MODEL_DIR.tmp.$$"
        cp -r "$HDFS_MODEL_SRC/"* "$LOCAL_MODEL_DIR.tmp.$$/"
        mv -T "$LOCAL_MODEL_DIR.tmp.$$" "$LOCAL_MODEL_DIR" || rm -rf "$LOCAL_MODEL_DIR.tmp.$$"
    else
        echo "[Staging] Model already staged at $LOCAL_MODEL_DIR"
    fi
}

# ==============================================================================
# Stage 1: OVO-Bench Full 12-Task Evaluation (SimpleStream-4f + Zero-Shot Forager)
# ==============================================================================
run_stage1() {
    echo ">>> [Stage 1] Running OVO-Bench 12-Task Streaming Evaluation..."
    stage_model_to_nvme
    if [[ "$DRY_RUN" -eq 1 ]]; then
        "$PYTHON_BIN" scripts/eval_ovobench.py \
            --backend mock \
            --data-json "$OVOBENCH_JSON" \
            --max-samples-per-task 1 \
            --output-json "$ROOT/outputs/ovobench_stage1_dryrun.json"
    else
        mkdir -p "$DA_SCRATCH/ovobench_s1_${BACKBONE}"
        pids=()
        shard_files=()
        for gpu_idx in $(seq 0 $((NGPUS_PER_NODE - 1))); do
            sf="$DA_SCRATCH/ovobench_s1_${BACKBONE}/shard_${gpu_idx}.json"
            shard_files+=("$sf")
            CUDA_VISIBLE_DEVICES="$gpu_idx" "$PYTHON_BIN" scripts/eval_ovobench.py \
                --backend hf \
                --model-path "$LOCAL_MODEL_DIR" \
                --data-json "$OVOBENCH_JSON" \
                --video-root "$OVOBENCH_VIDEOS" \
                --shard "${gpu_idx}/${NGPUS_PER_NODE}" \
                --output-json "$sf" > "$DA_SCRATCH/ovobench_s1_${BACKBONE}/shard_${gpu_idx}.log" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        "$PYTHON_BIN" scripts/eval_ovobench.py \
            --merge-shards "${shard_files[@]}" \
            --output-json "$ROOT/outputs/ovobench_${BACKBONE}_stage1.json"
    fi
}

# ==============================================================================
# Stage 2: 8-GPU Counterfactual Probing + SDAM-SFT Cold-Start Training (6K)
# ==============================================================================
run_stage2() {
    echo ">>> [Stage 2] Running Counterfactual Log-Likelihood Probing & SDAM-SFT Training..."
    stage_model_to_nvme
    if [[ ! -f "$ROOT/data/sft_coldstart_6k.jsonl" ]]; then
        "$PYTHON_BIN" scripts/prepare_candidate_manifests.py
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        "$PYTHON_BIN" scripts/run_gpu_probing.py \
            --backend mock \
            --input-jsonl "$ROOT/data/sft_coldstart_6k.jsonl" \
            --output-jsonl "$DA_SCRATCH/sft_probed_dryrun.jsonl" \
            --limit 16
        "$PYTHON_BIN" scripts/train_sft.py \
            --backend mock \
            --train-jsonl "$DA_SCRATCH/sft_probed_dryrun.jsonl" \
            --local-ckpt-dir "$SFT_LOCAL_CKPT" \
            --hdfs-export-dir "$SFT_HDFS_EXPORT" \
            --max-steps 3 \
            --batch-size 2
    else
        # Launch 8 data-parallel probing shards across the 8 GPUs on the node
        mkdir -p "$DA_SCRATCH/probed_shards"
        pids=()
        for gpu_idx in $(seq 0 $((NGPUS_PER_NODE - 1))); do
            CUDA_VISIBLE_DEVICES="$gpu_idx" "$PYTHON_BIN" scripts/run_gpu_probing.py \
                --backend hf \
                --model-path "$LOCAL_MODEL_DIR" \
                --input-jsonl "$ROOT/data/sft_coldstart_6k.jsonl" \
                --output-jsonl "$DA_SCRATCH/probed_shards/sft_shard_${gpu_idx}.jsonl" \
                --shard "${gpu_idx}/${NGPUS_PER_NODE}" &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        cat "$DA_SCRATCH/probed_shards/"sft_shard_*.jsonl > "$DA_SCRATCH/sft_probed_6k.jsonl"

        "$PYTHON_BIN" scripts/train_sft.py \
            --backend hf \
            --model-path "$LOCAL_MODEL_DIR" \
            --train-jsonl "$DA_SCRATCH/sft_probed_6k.jsonl" \
            --local-ckpt-dir "$SFT_LOCAL_CKPT" \
            --hdfs-export-dir "$SFT_HDFS_EXPORT" \
            --max-steps 375 \
            --batch-size 4 \
            --max-ckpts-to-keep 2
    fi
}

# ==============================================================================
# Stage 3: Counterfactual IG-GRPO Training (8K, 300 Steps) + Final OVO-Bench Eval
# ==============================================================================
run_stage3() {
    echo ">>> [Stage 3] Running IG-GRPO Reinforcement Learning & Final Evaluation..."
    stage_model_to_nvme
    if [[ ! -f "$ROOT/data/rl_ig_grpo_8k.jsonl" ]]; then
        "$PYTHON_BIN" scripts/prepare_candidate_manifests.py
    fi

    if [[ "$DRY_RUN" -eq 1 ]]; then
        "$PYTHON_BIN" scripts/train_ig_grpo.py \
            --mode mock \
            --rl-jsonl "$ROOT/data/rl_ig_grpo_8k.jsonl" \
            --local-ckpt-dir "$RL_LOCAL_CKPT" \
            --hdfs-export-dir "$RL_HDFS_EXPORT" \
            --max-steps 3
        "$PYTHON_BIN" scripts/train_ig_grpo.py \
            --mode verl-dry-run \
            --rl-jsonl "$ROOT/data/rl_ig_grpo_8k.jsonl" \
            --model-path "$LOCAL_MODEL_DIR" \
            --run-name "forager_${BACKBONE}_ig_grpo" \
            --local-ckpt-dir "$RL_LOCAL_CKPT" \
            --nnodes "$NNODES" \
            --gpus-per-node "$NGPUS_PER_NODE" \
            --max-steps 300
    else
        INIT_MODEL="$LOCAL_MODEL_DIR"
        if [[ -f "$SFT_HDFS_EXPORT/config.json" ]]; then
            INIT_MODEL="$SFT_HDFS_EXPORT"
        fi
        "$PYTHON_BIN" scripts/train_ig_grpo.py \
            --mode verl \
            --rl-jsonl "$ROOT/data/rl_ig_grpo_8k.jsonl" \
            --model-path "$INIT_MODEL" \
            --run-name "forager_${BACKBONE}_ig_grpo" \
            --local-ckpt-dir "$RL_LOCAL_CKPT" \
            --hdfs-export-dir "$RL_HDFS_EXPORT" \
            --nnodes "$NNODES" \
            --gpus-per-node "$NGPUS_PER_NODE" \
            --max-steps 300 \
            --save-freq 25 \
            --group-size 8 \
            --prompts-per-step 32

        # Final OVO-Bench evaluation on RL checkpoint across all 8 GPUs
        mkdir -p "$DA_SCRATCH/ovobench_s3_${BACKBONE}"
        pids=()
        shard_files=()
        for gpu_idx in $(seq 0 $((NGPUS_PER_NODE - 1))); do
            sf="$DA_SCRATCH/ovobench_s3_${BACKBONE}/shard_${gpu_idx}.json"
            shard_files+=("$sf")
            CUDA_VISIBLE_DEVICES="$gpu_idx" "$PYTHON_BIN" scripts/eval_ovobench.py \
                --backend hf \
                --model-path "$RL_HDFS_EXPORT" \
                --data-json "$OVOBENCH_JSON" \
                --video-root "$OVOBENCH_VIDEOS" \
                --shard "${gpu_idx}/${NGPUS_PER_NODE}" \
                --output-json "$sf" > "$DA_SCRATCH/ovobench_s3_${BACKBONE}/shard_${gpu_idx}.log" 2>&1 &
            pids+=("$!")
        done
        for pid in "${pids[@]}"; do
            wait "$pid"
        done
        "$PYTHON_BIN" scripts/eval_ovobench.py \
            --merge-shards "${shard_files[@]}" \
            --output-json "$ROOT/outputs/ovobench_${BACKBONE}_final_ig_grpo.json"
    fi
}

case "$STAGE" in
    stage1)
        run_stage1
        ;;
    stage2)
        run_stage2
        ;;
    stage3)
        run_stage3
        ;;
    all)
        run_stage1
        run_stage2
        run_stage3
        ;;
    *)
        echo "Unknown --stage $STAGE (expected all, stage1, stage2, or stage3)" >&2
        exit 2
        ;;
esac

echo "[Forager H800 Launcher] Completed stage=$STAGE successfully."
