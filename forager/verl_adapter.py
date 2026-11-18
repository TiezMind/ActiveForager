"""
veRL & IG-GRPO Training Adapter (`forager/verl_adapter.py`).

Provides:
1. `sample_2d_sdg_mode`: Group-level 2D Stochastic Degradation Gating (`clean`: 0.4, `spatial`: 0.3, `temporal`: 0.3).
2. `verl_compute_score`: Compatible with `veRL`'s `custom_reward_function` signature
   (`data_source, solution_str, ground_truth, extra_info=None`). Computes the composite
   `IG-GRPO` reward (R_acc + lambda_ig * R_IG + lambda_fmt * R_anchor - lambda_cost * C).
3. `verl_compute_std_free_advantage`: Group-indexed Std-Free Group-Mean advantage estimator
   with degenerate group skipping (`max(R) - min(R) < 1e-3`), eliminating `sigma_R -> 0` explosion.
4. `export_rl_jsonl_to_verl_parquet` & `build_verl_ig_grpo_command`: Seamless integration
   with `verl.trainer.main_ppo` on 8-16 x H800 GPUs obeying all `INFRA_LESSONS.md` guardrails.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import torch

from forager.ig_grpo import (
    IGGRPOConfig,
    RolloutSample,
    compute_composite_reward,
    compute_std_free_group_advantages,
)
from forager.protocol import ActionType, AnswerAction, parse_model_response


SDGMode = Literal["clean", "spatial", "temporal", "premature"]


def sample_2d_sdg_mode(
    rng: Optional[random.Random] = None,
    probs: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> SDGMode:
    """
    Sample 2D Stochastic Degradation Gating (2D-SDG) mode identically for a prompt group:
    - `clean` (p=0.4): W_t^rec has w=4 frames at 224x224.
    - `spatial` (p=0.3): W_t^rec frames downsampled to 112x112 (making <ZoomIn> strictly necessary).
    - `temporal` (p=0.3): W_t^rec truncated to w=1 frame (making <Recall> strictly necessary).
    """
    p_clean, p_spat, p_temp = probs
    total = p_clean + p_spat + p_temp
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"2D-SDG probabilities must sum to 1.0, got {probs}")
    r = (rng or random).random()
    if r < p_clean:
        return "clean"
    if r < p_clean + p_spat:
        return "spatial"
    return "temporal"


def sample_3d_sdg_mode(
    rng: Optional[random.Random] = None,
    probs: tuple[float, float, float, float] = (0.28, 0.24, 0.24, 0.24),
) -> SDGMode:
    """
    Sample Symmetric 3D Spatio-Temporal Causal Degradation Gating (`3D-SDG`) mode identically for a prompt group:
    - `clean` (p=0.28): W_t^rec has w=4 frames at 224x224 at t_q (direct <Answer> suffices).
    - `spatial` (p=0.24): Present frames downsampled to 112x112 (making <ZoomIn> strictly necessary).
    - `temporal` (p=0.24): Recent window truncated to w=1 frame (making <Recall> strictly necessary).
    - `premature` (p=0.24): Query evaluated at pre-clue window t_q < t_clue (making <Wait> strictly necessary
      and yielding positive Temporal Lookahead Information Gain Delta I_wait > 0).
    """
    p_clean, p_spat, p_temp, p_prem = probs
    total = p_clean + p_spat + p_temp + p_prem
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"3D-SDG probabilities must sum to 1.0, got {probs}")
    r = (rng or random).random()
    if r < p_clean:
        return "clean"
    if r < p_clean + p_spat:
        return "spatial"
    if r < p_clean + p_spat + p_temp:
        return "temporal"
    return "premature"


_TURN_SPLIT_RE = re.compile(
    r"(<think>.*?</think>\s*<(?:Answer|ZoomIn|Recall|Wait)>.*?</(?:Answer|ZoomIn|Recall|Wait)>)",
    re.DOTALL,
)


def parse_multi_turn_solution_str(solution_str: str) -> tuple[str, Optional[str], str]:
    """
    Split a 1-turn or 2-turn rollout transcript into `(turn1_response, turn2_response, final_answer)`.
    """
    matches = _TURN_SPLIT_RE.findall(solution_str)
    if not matches:
        parsed = parse_model_response(solution_str)
        final_ans = parsed.action.text if isinstance(parsed.action, AnswerAction) else solution_str.strip()
        return solution_str, None, final_ans

    t1 = matches[0].strip()
    t2 = matches[1].strip() if len(matches) >= 2 else None
    last_turn = t2 if t2 is not None else t1
    parsed_last = parse_model_response(last_turn)
    if isinstance(parsed_last.action, AnswerAction):
        final_ans = parsed_last.action.text
    else:
        m_ans = re.search(r"<Answer>(.*?)</Answer>", solution_str, re.DOTALL)
        final_ans = m_ans.group(1).strip() if m_ans else ""
    return t1, t2, final_ans


def verl_compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: Optional[dict[str, Any]] = None,
    config: Optional[IGGRPOConfig] = None,
) -> dict[str, Any]:
    """
    veRL-compatible `custom_reward_function` (`compute_score`).

    Returns a dictionary containing `score` (total composite reward) and all diagnostic
    sub-rewards (`r_acc`, `r_ig`, `r_anchor`, `cost`, `is_correct`, `ccv_valid`).
    """
    info = extra_info or {}
    cfg = config or IGGRPOConfig()
    t1_resp, t2_resp, final_ans = parse_multi_turn_solution_str(solution_str)

    parsed_t1 = parse_model_response(t1_resp)
    if "foraged_visual_tokens" in info:
        foraged_tokens = int(info["foraged_visual_tokens"])
    elif parsed_t1.action_type == ActionType.ZOOM_IN:
        foraged_tokens = 256
    elif parsed_t1.action_type == ActionType.RECALL:
        foraged_tokens = 0 if info.get("recall_status") == "NONE" else 64
    else:
        foraged_tokens = 0

    task_type: Literal["mcq", "open", "proactive"] = info.get("task_type", "mcq")
    if parsed_t1.action_type == ActionType.WAIT:
        lp_with = float(
            info.get(
                "logprob_after_wait",
                info.get("logprob_with_evidence", info.get("probe_logprob_with_evidence", 0.0)),
            )
        )
        lp_without = float(
            info.get(
                "logprob_before_wait",
                info.get("logprob_without_evidence", info.get("probe_logprob_without_evidence", 0.0)),
            )
        )
    else:
        lp_with = float(info.get("logprob_with_evidence", info.get("probe_logprob_with_evidence", 0.0)))
        lp_without = float(info.get("logprob_without_evidence", info.get("probe_logprob_without_evidence", 0.0)))
    ans_len = max(1, int(info.get("answer_token_len", len(str(ground_truth).split()))))

    sample = RolloutSample(
        turn1_response=t1_resp,
        turn2_response=t2_resp,
        final_answer=final_ans,
        gt_answer=str(ground_truth),
        task_type=task_type,
        foraged_visual_tokens=foraged_tokens,
        logprob_with_evidence=lp_with,
        logprob_without_evidence=lp_without,
        answer_token_len=ans_len,
        pred_time=info.get("pred_time"),
        gt_time=info.get("target_reveal_time"),
    )
    breakdown = compute_composite_reward(sample, config=cfg)
    return {
        "score": float(breakdown.total_reward),
        "r_acc": float(breakdown.r_acc),
        "r_ig": float(breakdown.r_ig),
        "r_anchor": float(breakdown.r_anchor),
        "cost": float(breakdown.cost),
        "is_correct": float(breakdown.is_correct),
        "ccv_valid": float(breakdown.ccv_valid),
        "used_foraging_tool": float(breakdown.used_foraging_tool),
    }


def verl_compute_std_free_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: Sequence[Any],
    degenerate_eps: float = 1e-3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    veRL-compatible Std-Free Group-Mean Advantage Estimator:
        A_i = R_i - mean_{j in group(i)}(R_j)
    Broadcasts scalar rollout advantage across valid response tokens (`response_mask`).
    Skips degenerate groups (`max(R) - min(R) < degenerate_eps`) by setting their advantage to 0.
    """
    scores = token_level_rewards.sum(dim=-1)  # (B_total,)
    device = scores.device
    dtype = scores.dtype

    groups: dict[Any, list[int]] = defaultdict(list)
    for idx_pos, group_key in enumerate(index):
        key = int(group_key.item()) if isinstance(group_key, torch.Tensor) else group_key
        groups[key].append(idx_pos)

    scalar_advantages = torch.zeros_like(scores)
    for _, member_indices in groups.items():
        idx_tensor = torch.tensor(member_indices, dtype=torch.long, device=device)
        group_r = scores.index_select(0, idx_tensor)
        group_adv, _ = compute_std_free_group_advantages(group_r, degenerate_eps=degenerate_eps)
        scalar_advantages.index_copy_(0, idx_tensor, group_adv.to(device=device, dtype=dtype))

    token_advantages = scalar_advantages.unsqueeze(-1) * response_mask.to(dtype=dtype)
    return token_advantages, token_advantages


def export_rl_jsonl_to_verl_parquet(
    jsonl_path: Path | str,
    parquet_path: Path | str,
    max_samples: Optional[int] = None,
) -> dict[str, Any]:
    """
    Convert `data/rl_ig_grpo_8k.jsonl` into `veRL`'s standard Parquet format
    (`data_source`, `prompt`, `reward_model`, `extra_info`).
    """
    import pandas as pd

    src = Path(jsonl_path)
    dst = Path(parquet_path)
    records = [
        json.loads(line)
        for line in src.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if max_samples is not None and max_samples > 0:
        records = records[:max_samples]

    verl_rows: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        verl_rows.append(
            {
                "data_source": rec.get("source_dataset", "forager_rl"),
                "prompt": [{"role": "user", "content": str(rec["prompt"])}],
                "reward_model": {
                    "style": "rule",
                    "ground_truth": str(rec["gt_answer"]),
                },
                "extra_info": {
                    "index": idx,
                    "prompt_id": rec.get("prompt_id", f"rl_{idx:05d}"),
                    "rl_bucket": rec.get("rl_bucket", "realtime_perception"),
                    "video_id": rec.get("video_id", ""),
                    "video_path": rec.get("video_path", ""),
                    "query_timestamp": float(rec.get("query_timestamp", 24.0)),
                    "target_reveal_time": rec.get("target_reveal_time"),
                    "task_type": rec.get("task_type", "mcq"),
                    "probe_logprob_without_evidence": float(rec.get("probe_logprob_without_evidence", -1.386)),
                    "probe_logprob_with_evidence": float(rec.get("probe_logprob_with_evidence", -0.083)),
                },
            }
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(verl_rows).to_parquet(dst)
    sha = hashlib.sha256(dst.read_bytes()).hexdigest()
    return {"rows": len(verl_rows), "parquet_path": str(dst), "sha256": sha}


def build_verl_ig_grpo_command(
    train_parquet: Path | str,
    model_path: str,
    run_name: str = "forager_qwen25vl_7b_ig_grpo",
    local_ckpt_dir: Optional[Path | str] = None,
    nnodes: int = 1,
    n_gpus_per_node: int = 8,
    total_steps: int = 300,
    save_freq: int = 25,
    group_size: int = 8,
    prompts_per_step: int = 32,
    gpu_memory_utilization: float = 0.80,
    python_bin: Optional[str] = None,
) -> list[str]:
    """
    Construct the production `veRL` (`verl.trainer.main_ppo`) launch command
    enforcing all `INFRA_LESSONS.md` guardrails:
    - `trainer.max_actor_ckpt_to_keep=2`
    - `trainer.default_local_dir=/tmp/forager_ckpts/rl/<run_name>` (local NVMe, never HDFS FUSE)
    - `actor_rollout_ref.rollout.gpu_memory_utilization=0.80` (<= 0.95 to coexist with gpu_keeper)
    - `reward.custom_reward_function.path=pkg://forager.verl_adapter`
    """
    if not (0.1 <= gpu_memory_utilization <= 0.95):
        raise ValueError(f"gpu_memory_utilization must be <= 0.95 for gpu_keeper coexistence, got {gpu_memory_utilization}")

    ckpt_dir = Path(local_ckpt_dir) if local_ckpt_dir else Path(f"/tmp/forager_ckpts/rl/{run_name}")
    ray_short_hash = hashlib.sha1(run_name.encode("utf-8")).hexdigest()[:10]
    ray_tmp_dir = f"/tmp/forager_ray/{ray_short_hash}"
    py = python_bin or sys.executable

    return [
        py,
        "-m",
        "verl.trainer.main_ppo",
        "trainer.use_v1=False",
        "algorithm.adv_estimator=grpo",
        "algorithm.norm_adv_by_std_in_grpo=False",
        "algorithm.use_kl_in_reward=False",
        f"data.train_files={train_parquet}",
        f"data.val_files={train_parquet}",
        "data.val_max_samples=16",
        f"data.train_batch_size={prompts_per_step}",
        "data.max_prompt_length=2048",
        "data.max_response_length=384",
        f"actor_rollout_ref.model.path={model_path}",
        "actor_rollout_ref.model.enable_gradient_checkpointing=True",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.actor.optim.lr=1e-6",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={max(1, prompts_per_step // 2)}",
        "actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean",
        "actor_rollout_ref.actor.use_kl_loss=True",
        "actor_rollout_ref.actor.kl_loss_coef=0.01",
        "actor_rollout_ref.actor.checkpoint.save_contents=[model,optimizer,extra,hf_model]",
        "actor_rollout_ref.rollout.name=vllm",
        f"actor_rollout_ref.rollout.gpu_memory_utilization={gpu_memory_utilization}",
        f"actor_rollout_ref.rollout.n={group_size}",
        "actor_rollout_ref.rollout.temperature=0.8",
        "actor_rollout_ref.rollout.calculate_log_probs=True",
        "reward.custom_reward_function.path=pkg://forager.verl_adapter",
        "reward.custom_reward_function.name=verl_compute_score",
        "reward.reward_manager.name=naive",
        "trainer.project_name=forager-video",
        f"trainer.experiment_name={run_name}",
        "trainer.logger=[console,file]",
        f"trainer.nnodes={nnodes}",
        f"trainer.n_gpus_per_node={n_gpus_per_node}",
        f"trainer.total_training_steps={total_steps}",
        f"trainer.save_freq={save_freq}",
        "trainer.max_actor_ckpt_to_keep=2",
        "trainer.val_before_train=False",
        f"trainer.default_local_dir={ckpt_dir}",
        "+ray_kwargs.ray_init.include_dashboard=False",
        f"+ray_kwargs.ray_init._temp_dir={ray_tmp_dir}",
    ]
