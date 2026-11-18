"""
Counterfactual Information-Gain GRPO (`forager/ig_grpo.py`):
Implements:
1. Task Accuracy Reward (`R_acc`) for MCQ, Open-Ended QA, and Proactive Timing (FAR).
2. Counterfactual Visual Information Gain (`R_IG`):
   Rewards `<ZoomIn>` / `<Recall>` iff the foraged evidence E_t genuinely lifts the
   per-token log-probability of ground-truth answer y^* on reference policy pi_ref,
   gated by Cognitive Consistency Verification (`V_CCV`) and final answer correctness.
3. Normalized Foraging Cost (`C(tau_i)`) and Composite Reward (`R(tau_i)`).
4. Std-Free Group-Mean Advantage (`A_i = R_i - mean(R)`) with degenerate-group skipping,
   preventing variance explosion when group standard deviation sigma_R -> 0.
5. Turn-Mean (`TurnWise`) Clipped Policy Gradient + KL Loss, giving Turn-1 tool decisions
   and Turn-2 synthesis equal per-turn gradient weight.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Union

import torch

from forager.protocol import (
    ActionType,
    compute_format_anchor_reward,
    parse_model_response,
    verify_cognitive_consistency,
)


@dataclass(frozen=True)
class IGGRPOConfig:
    """Hyperparameters for Counterfactual Information-Gain GRPO."""

    lambda_ig: float = 0.65
    lambda_fmt: float = 0.2
    lambda_cost: float = 0.03
    wrong_mcq_penalty: float = 0.5
    ig_clip_max: float = 1.0
    max_forage_tokens: int = 512
    clip_eps: float = 0.2
    kl_beta: float = 0.01
    degenerate_group_eps: float = 1e-3
    proactive_time_tolerance: float = 2.0


def _normalize_answer_str(text: str) -> str:
    cleaned = text.strip().lower()
    # If single choice letter like "(A)" or "A." or "A: ...", extract canonical letter
    m_choice = re.match(r"^\(?([a-d])\)?(?:[\.\:\)\s]|$)", cleaned)
    if m_choice is not None and len(cleaned) <= 4:
        return m_choice.group(1)
    cleaned = re.sub(r"[^a-z0-9\u4e00-\u9fff\s]", " ", cleaned)
    return " ".join(cleaned.split())


def _ngram_f1(pred_tokens: list[str], gt_tokens: list[str], n: int = 1) -> float:
    if len(pred_tokens) < n or len(gt_tokens) < n:
        return 0.0
    pred_ngrams = [tuple(pred_tokens[i : i + n]) for i in range(len(pred_tokens) - n + 1)]
    gt_ngrams = [tuple(gt_tokens[i : i + n]) for i in range(len(gt_tokens) - n + 1)]
    gt_counts: dict[tuple[str, ...], int] = {}
    for ng in gt_ngrams:
        gt_counts[ng] = gt_counts.get(ng, 0) + 1
    overlap = 0
    for ng in pred_ngrams:
        if gt_counts.get(ng, 0) > 0:
            overlap += 1
            gt_counts[ng] -= 1
    if overlap == 0:
        return 0.0
    prec = overlap / len(pred_ngrams)
    rec = overlap / len(gt_ngrams)
    return 2.0 * prec * rec / (prec + rec)


def _lcs_f1(pred_tokens: list[str], gt_tokens: list[str]) -> float:
    if not pred_tokens or not gt_tokens:
        return 0.0
    m, n = len(pred_tokens), len(gt_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if pred_tokens[i] == gt_tokens[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    lcs = dp[m][n]
    if lcs == 0:
        return 0.0
    prec = lcs / m
    rec = lcs / n
    return 2.0 * prec * rec / (prec + rec)


def compute_task_accuracy_reward(
    pred_answer: str,
    gt_answer: str,
    task_type: Literal["mcq", "open", "proactive"] = "mcq",
    wrong_mcq_penalty: float = 0.5,
    pred_time: Optional[float] = None,
    gt_time: Optional[float] = None,
    time_tolerance: float = 2.0,
) -> tuple[float, bool]:
    """
    Compute task accuracy reward R_acc and boolean correctness flag `is_correct`.

    - `mcq`: +1.0 if exact match, else `-wrong_mcq_penalty` (-0.5) to penalize blind guessing.
    - `open`: Average of ROUGE-1, ROUGE-2, and ROUGE-L F1 scores in [0, 1].
    - `proactive`: `max(0, 1 - |t_pred - t_gt| / tau_tol) * I(y_hat == y^*)`.
    """
    norm_pred = _normalize_answer_str(pred_answer)
    norm_gt = _normalize_answer_str(gt_answer)

    if not norm_pred:
        return (-wrong_mcq_penalty if task_type == "mcq" else 0.0), False

    if task_type == "mcq":
        # Check exact match or leading option letter match
        is_correct = (norm_pred == norm_gt) or (
            len(norm_gt) == 1
            and norm_pred.startswith(norm_gt + " ")
        )
        reward = 1.0 if is_correct else -float(wrong_mcq_penalty)
        return float(reward), bool(is_correct)

    if task_type == "proactive":
        is_ans_correct = (norm_pred == norm_gt) or (norm_gt in norm_pred)
        if not is_ans_correct:
            return 0.0, False
        if pred_time is None or gt_time is None:
            return 1.0, True
        time_factor = max(0.0, 1.0 - abs(float(pred_time) - float(gt_time)) / max(1e-6, float(time_tolerance)))
        is_correct = time_factor > 0.0
        return float(time_factor), bool(is_correct)

    if task_type == "open":
        p_toks = norm_pred.split()
        g_toks = norm_gt.split()
        r1 = _ngram_f1(p_toks, g_toks, n=1)
        r2 = _ngram_f1(p_toks, g_toks, n=2) if len(g_toks) >= 2 else r1
        rl = _lcs_f1(p_toks, g_toks)
        rouge_avg = (r1 + r2 + rl) / 3.0
        is_correct = rouge_avg >= 0.5 or (norm_pred == norm_gt)
        return float(rouge_avg), bool(is_correct)

    raise ValueError(f"Unsupported task_type: {task_type}")


def compute_counterfactual_ig_reward(
    logprob_with_evidence: Union[float, torch.Tensor],
    logprob_without_evidence: Union[float, torch.Tensor],
    is_correct: Union[bool, torch.Tensor],
    ccv_valid: Union[bool, torch.Tensor],
    answer_token_len: Union[int, torch.Tensor] = 1,
    clip_max: float = 1.0,
) -> Union[float, torch.Tensor]:
    """
    Compute Counterfactual Visual Information Gain Reward R_IG(tau_i):

    Delta I(tau_i) = (1 / |y^*|) * (log pi_ref(y^* | W_t^rec, E_i, q_t) - log pi_ref(y^* | W_t^rec, q_t))
    R_IG(tau_i)    = V_CCV(tau_i) * I(y_hat_i == y^*) * clip(Delta I(tau_i), 0.0, clip_max)

    Supports both scalar Python floats and batched PyTorch tensors.
    """
    if isinstance(logprob_with_evidence, torch.Tensor) or isinstance(logprob_without_evidence, torch.Tensor):
        lp_with = torch.as_tensor(logprob_with_evidence, dtype=torch.float32)
        lp_without = torch.as_tensor(
            logprob_without_evidence, dtype=torch.float32, device=lp_with.device
        )
        lengths = torch.clamp(
            torch.as_tensor(answer_token_len, dtype=torch.float32, device=lp_with.device),
            min=1.0,
        )
        delta_i = (lp_with - lp_without) / lengths
        clipped_ig = torch.clamp(delta_i, min=0.0, max=float(clip_max))
        gate_correct = torch.as_tensor(is_correct, dtype=torch.float32, device=lp_with.device)
        gate_ccv = torch.as_tensor(ccv_valid, dtype=torch.float32, device=lp_with.device)
        return gate_ccv * gate_correct * clipped_ig

    denom = float(max(1, int(answer_token_len)))
    delta_i_scalar = (float(logprob_with_evidence) - float(logprob_without_evidence)) / denom
    clipped_scalar = min(float(clip_max), max(0.0, delta_i_scalar))
    if not bool(is_correct) or not bool(ccv_valid):
        return 0.0
    return float(clipped_scalar)


def compute_lookahead_wait_ig_reward(
    logprob_after_wait: Union[float, torch.Tensor],
    logprob_before_wait: Union[float, torch.Tensor],
    is_correct: Union[bool, torch.Tensor],
    ccv_valid: Union[bool, torch.Tensor],
    answer_token_len: Union[int, torch.Tensor] = 1,
    clip_max: float = 1.0,
) -> Union[float, torch.Tensor]:
    """
    Compute Temporal Lookahead Counterfactual Information Gain Reward Delta I_wait(tau_i) for `<Wait>`:

    Delta I_wait(tau_i) = (1 / |y^*|) * (log pi_ref(y^* | W_{t_q + Delta t}^rec, q_t) - log pi_ref(y^* | W_{t_q}^rec, q_t))
    R_IG_wait(tau_i)    = V_CCV(tau_i) * I(y_hat_i == y^*) * clip(Delta I_wait(tau_i), 0.0, clip_max)

    Unifies Past (<Recall>), Present (<ZoomIn>), and Future (<Wait>) foraging under a single
    3D Spatio-Temporal Counterfactual Information Gain objective.
    """
    return compute_counterfactual_ig_reward(
        logprob_with_evidence=logprob_after_wait,
        logprob_without_evidence=logprob_before_wait,
        is_correct=is_correct,
        ccv_valid=ccv_valid,
        answer_token_len=answer_token_len,
        clip_max=clip_max,
    )


def compute_foraging_cost(
    foraged_visual_tokens: int,
    max_forage_budget: int = 512,
) -> float:
    """
    Compute Normalized Foraging Cost C(tau_i) = clip(Tokens(E_{i,t}) / B_max_forage, 0.0, 1.0).
    Turn-1 direct `<Answer>` has 0 foraged tokens -> C(tau_i) = 0.0.
    """
    if foraged_visual_tokens <= 0:
        return 0.0
    return float(min(1.0, max(0.0, float(foraged_visual_tokens) / max(1, int(max_forage_budget)))))


@dataclass(frozen=True)
class RolloutSample:
    """Container for a single 1-turn or 2-turn trajectory rollout in IG-GRPO."""

    turn1_response: str
    final_answer: str
    gt_answer: str
    task_type: Literal["mcq", "open", "proactive"] = "mcq"
    turn2_response: Optional[str] = None
    foraged_visual_tokens: int = 0
    logprob_with_evidence: float = 0.0
    logprob_without_evidence: float = 0.0
    answer_token_len: int = 1
    pred_time: Optional[float] = None
    gt_time: Optional[float] = None


@dataclass(frozen=True)
class RewardBreakdown:
    """Detailed breakdown of composite reward components for logging and diagnostics."""

    total_reward: float
    r_acc: float
    r_ig: float
    r_anchor: float
    cost: float
    is_correct: bool
    ccv_valid: bool
    used_foraging_tool: bool


def compute_composite_reward(
    sample: RolloutSample,
    config: Optional[IGGRPOConfig] = None,
) -> RewardBreakdown:
    """
    Compute total verifiable reward for a rollout trajectory:
    R(tau_i) = R_acc + lambda_ig * R_IG + lambda_fmt * R_anchor - lambda_cost * C(tau_i)
    """
    cfg = config or IGGRPOConfig()

    r_acc, is_correct = compute_task_accuracy_reward(
        pred_answer=sample.final_answer,
        gt_answer=sample.gt_answer,
        task_type=sample.task_type,
        wrong_mcq_penalty=cfg.wrong_mcq_penalty,
        pred_time=sample.pred_time,
        gt_time=sample.gt_time,
        time_tolerance=cfg.proactive_time_tolerance,
    )

    parsed_t1 = parse_model_response(sample.turn1_response)
    r_anchor_t1 = compute_format_anchor_reward(sample.turn1_response)
    if sample.turn2_response is not None:
        r_anchor_t2 = compute_format_anchor_reward(sample.turn2_response)
        r_anchor = 0.5 * (r_anchor_t1 + r_anchor_t2)
    else:
        r_anchor = r_anchor_t1

    ccv_valid = verify_cognitive_consistency(parsed_t1.think_text, parsed_t1.action)
    used_foraging_tool = parsed_t1.action_type in (
        ActionType.ZOOM_IN,
        ActionType.RECALL,
        ActionType.WAIT,
    )

    if parsed_t1.action_type == ActionType.WAIT:
        r_ig = float(
            compute_lookahead_wait_ig_reward(
                logprob_after_wait=sample.logprob_with_evidence,
                logprob_before_wait=sample.logprob_without_evidence,
                is_correct=is_correct,
                ccv_valid=ccv_valid,
                answer_token_len=sample.answer_token_len,
                clip_max=cfg.ig_clip_max,
            )
        )
    elif used_foraging_tool:
        r_ig = float(
            compute_counterfactual_ig_reward(
                logprob_with_evidence=sample.logprob_with_evidence,
                logprob_without_evidence=sample.logprob_without_evidence,
                is_correct=is_correct,
                ccv_valid=ccv_valid,
                answer_token_len=sample.answer_token_len,
                clip_max=cfg.ig_clip_max,
            )
        )
    else:
        r_ig = 0.0

    cost = compute_foraging_cost(
        foraged_visual_tokens=sample.foraged_visual_tokens,
        max_forage_budget=cfg.max_forage_tokens,
    )

    total = (
        r_acc
        + cfg.lambda_ig * r_ig
        + cfg.lambda_fmt * r_anchor
        - cfg.lambda_cost * cost
    )
    return RewardBreakdown(
        total_reward=float(total),
        r_acc=float(r_acc),
        r_ig=float(r_ig),
        r_anchor=float(r_anchor),
        cost=float(cost),
        is_correct=bool(is_correct),
        ccv_valid=bool(ccv_valid),
        used_foraging_tool=bool(used_foraging_tool),
    )


def compute_std_free_group_advantages(
    rewards: Union[Sequence[float], torch.Tensor],
    degenerate_eps: float = 1e-3,
) -> tuple[torch.Tensor, bool]:
    """
    Compute Std-Free Group-Mean Baseline advantages:
        A_i = R_i - (1 / G) * sum_{j=1}^G R_j
    Omits division by group standard deviation sigma_R to prevent gradient explosion
    on easy/hard prompts where sigma_R -> 0.

    Returns:
        `(advantages, is_degenerate)`:
        If `max(R) - min(R) < degenerate_eps`, returns all-zero advantages and `is_degenerate=True`.
    """
    r_tensor = torch.as_tensor(rewards, dtype=torch.float32)
    if r_tensor.numel() == 0:
        raise ValueError("rewards must be non-empty")

    span = float((r_tensor.max() - r_tensor.min()).item())
    if span < float(degenerate_eps):
        return torch.zeros_like(r_tensor), True

    mean_r = r_tensor.mean()
    advantages = r_tensor - mean_r
    return advantages, False


def compute_turn_mean_grpo_loss(
    turn_logprobs: Sequence[Sequence[torch.Tensor]],
    old_turn_logprobs: Sequence[Sequence[torch.Tensor]],
    advantages: torch.Tensor,
    ref_turn_logprobs: Optional[Sequence[Sequence[torch.Tensor]]] = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.01,
) -> torch.Tensor:
    """
    Compute Turn-Mean (`TurnWise`) IG-GRPO policy loss across a group of G trajectories.

    Mathematical formulation (Sec. 3.4 Step 3):
    For each rollout i in {1..G} with K_i in {1, 2} turns, we average token surrogate
    objectives within each turn k first (`1 / |y_{i,k}|`), then across turns (`1 / K_i`),
    and finally across the group (`1 / G`). This ensures a concise Turn-1 `<ZoomIn>`/`<Recall>`
    decision receives equal gradient weight to a longer Turn-2 `<Answer>`.

    Args:
        turn_logprobs: List of length G; each element is a list of K_i 1D tensors of token
            log-probabilities under current policy pi_theta.
        old_turn_logprobs: Matching list of token log-probabilities under rollout policy pi_old.
        advantages: 1D tensor of shape `(G,)` from `compute_std_free_group_advantages`.
        ref_turn_logprobs: Optional matching list of token log-probabilities under pi_ref for KL penalty.
        clip_eps: PPO/GRPO clipping epsilon (default 0.2).
        kl_beta: Coefficient on the Schulman k3 KL estimator (default 0.01).

    Returns:
        Scalar loss tensor (to be minimized via `.backward()`).
    """
    num_rollouts = len(turn_logprobs)
    if num_rollouts == 0:
        raise ValueError("turn_logprobs cannot be empty")
    if len(old_turn_logprobs) != num_rollouts or advantages.numel() != num_rollouts:
        raise ValueError("Mismatched group size G across turn_logprobs, old_turn_logprobs, and advantages")

    first_tensor = turn_logprobs[0][0]
    device = first_tensor.device
    dtype = first_tensor.dtype
    adv_tensor = advantages.to(device=device, dtype=dtype)

    rollout_objectives: list[torch.Tensor] = []

    for i in range(num_rollouts):
        turns_cur = turn_logprobs[i]
        turns_old = old_turn_logprobs[i]
        if len(turns_cur) == 0 or len(turns_cur) != len(turns_old):
            raise ValueError(f"Invalid turn structure for rollout {i}")

        adv_i = adv_tensor[i]
        turn_surrogates: list[torch.Tensor] = []

        for k in range(len(turns_cur)):
            lp_cur = turns_cur[k]
            lp_old = turns_old[k].detach()
            if lp_cur.numel() == 0:
                continue

            ratio = torch.exp(lp_cur - lp_old)
            surr1 = ratio * adv_i
            surr2 = torch.clamp(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * adv_i
            token_obj = torch.minimum(surr1, surr2)

            if ref_turn_logprobs is not None:
                lp_ref = ref_turn_logprobs[i][k].detach()
                log_r_ref = lp_ref - lp_cur
                # Non-negative unbiased KL estimator: exp(log_r) - log_r - 1
                kl_tokens = torch.exp(log_r_ref) - log_r_ref - 1.0
                token_obj = token_obj - float(kl_beta) * kl_tokens

            # Average within turn k first (1 / |y_{i,k}|)
            turn_surrogates.append(token_obj.mean())

        # Average across K_i turns (1 / K_i)
        rollout_obj = torch.stack(turn_surrogates).mean()
        rollout_objectives.append(rollout_obj)

    # Loss is negative expected objective (-J_{IG-GRPO})
    group_objective = torch.stack(rollout_objectives).mean()
    return -group_objective
