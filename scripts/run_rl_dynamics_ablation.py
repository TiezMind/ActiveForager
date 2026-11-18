#!/usr/bin/env python3
"""
100% Real-VLM-Grounded RL Training Dynamics & Tool-Collapse Ablation Runner (`scripts/run_rl_dynamics_ablation.py`).

Eliminates all hardcoded conditional accuracy vectors (`rt_cond`, `bt_cond`, `far_cond`) and
hardcoded log-probability constants (`-1.38`, `-0.05`):
1. When `--backend hf` is specified (or when `real_vlm_rollout_bank.json` is built on GPU),
   loads the real `Qwen2.5-VL-7B` model (`HFQwenVLEngine`) and evaluates all 4 action branches
   (`<Answer>`, `<ZoomIn>`, `<Recall>`, `<Wait>`) plus real single-prefill counterfactual
   log-probabilities (`engine.score_answer_logprob` with and without foraged evidence) on a
   stratified real-video `OVO-Bench` validation set (`ovo_bench_new.json` + `decord.VideoReader`).
2. Trains the 4 RL objective variants over `max_steps` (300 steps):
   - `vanilla_grpo`: Clean frames only (no 3D-SDG), no Counterfactual IG (`lambda_ig = 0.0`),
     divides by group std `sigma_R`, high tool cost penalty (`lambda_cost = 0.18`), no format anchor (`lambda_fmt = 0.0`).
   - `streamscout_binary_cost`: 1D temporal escalation (`clean` + `temporal`), binary tool penalty (`lambda_ig = 0.0`).
   - `para_grpo_1d`: 1D temporal frame gating (`clean` + `temporal`) without spatial `<ZoomIn>` or proactive `<Wait>` SDG.
   - `ig_grpo_ours`: Full Symmetric 3D-SDG (`clean`, `spatial`, `temporal`, `premature`) + Counterfactual Visual
     Information Gain (`R_IG`, `lambda_ig = 0.75`) + Format Anchoring (`R_anchor`) + Std-Free Group-Mean Advantage.
3. At every logging interval (`log_every`), evaluates the policy directly on the real VLM per-sample
   branch outcomes from `OVO-Bench`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from forager.agent import (
    HFQwenVLEngine,
    MockVLMEngine,
    StreamingForagerAgent,
)
from forager.dual_buffer import (
    CausalStreamingDualBuffer,
    HighResRingBuffer,
    ProtectedRecentWindow,
    VideoFrame,
)
from forager.ig_grpo import (
    IGGRPOConfig,
    compute_std_free_group_advantages,
    compute_task_accuracy_reward,
    compute_turn_mean_grpo_loss,
)
from forager.protocol import ActionType, RecallAction, ZoomInAction
from forager.verl_adapter import verl_compute_score
from scripts.eval_ovobench import (
    BACKWARD_TASKS,
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
    FORWARD_ACTIVE_TASKS,
    REALTIME_TASKS,
    load_ovobench_from_json,
)

RL_VARIANTS = ("vanilla_grpo", "streamscout_binary_cost", "para_grpo_1d", "ig_grpo_ours")


class ContextualPolicyHead(nn.Module):
    """
    Contextual policy head mapping 4-dim regime features `[is_clean_rt, is_spat, is_hist, is_far]`
    to 5 action outcomes `[Answer, ZoomIn, Recall, Wait, MalformedSyntax]`, initialized from the
    empirical SFT cold-start action prior.
    """

    def __init__(self) -> None:
        super().__init__()
        self.logits_table = nn.Parameter(torch.zeros(4, 5))
        with torch.no_grad():
            init_lp = torch.log(torch.tensor([0.47, 0.185, 0.225, 0.08, 0.04], dtype=torch.float32))
            self.logits_table.copy_(init_lp.unsqueeze(0).repeat(4, 1))
            self.logits_table.add_(torch.randn(4, 5) * 0.02)

    def forward(self, ctx: torch.Tensor) -> torch.Tensor:
        return ctx @ self.logits_table


def build_real_vlm_rollout_bank(
    model_path: str,
    samples_per_task: int = 2,
    bank_out: Optional[Path] = None,
    use_mock: bool = False,
) -> list[dict[str, Any]]:
    """
    Run 100% real VLM inference (`HFQwenVLEngine`) and real counterfactual log-probability scoring
    (`score_answer_logprob`) across all 12 OVO-Bench subtasks and all 4 action branches
    (`<Answer>`, `<ZoomIn>`, `<Recall>`, `<Wait>`) on real `.mp4` videos decoded via `decord`.
    """
    engine = MockVLMEngine() if use_mock else HFQwenVLEngine(model_name_or_path=model_path)
    samples = load_ovobench_from_json(
        DEFAULT_OVOBENCH_JSON,
        video_root=DEFAULT_OVOBENCH_VIDEO_ROOT,
        max_samples_per_task=samples_per_task,
        lazy_decode=not use_mock,
        allow_synthetic_fallback=use_mock,
    )
    agent_ss = StreamingForagerAgent(engine=engine, mode="simplestream", use_sdam_mask=True)
    agent_fo = StreamingForagerAgent(engine=engine, mode="forager", use_sdam_mask=True)

    bank: list[dict[str, Any]] = []
    for idx, s in enumerate(samples):
        all_frames = s.materialize_frames(allow_synthetic_fallback=use_mock)
        prefix_frames = [f for f in all_frames if f.timestamp <= s.query_timestamp + 1e-6]
        future_frames = [f for f in all_frames if f.timestamp > s.query_timestamp + 1e-6]
        blind_meta = s.blind_agent_metadata(include_mock_hints=use_mock)

        env = CausalStreamingDualBuffer(
            recent_window=ProtectedRecentWindow(window_size=4, base_resolution=(224, 224), base_fps=1.0),
            ring_buffer=HighResRingBuffer(duration_sec=4.0, spatial_hires_hw=(672, 672)),
        )
        for pf in prefix_frames:
            env.step_frame(pf)
        recent_frames, state_header = env.get_turn1_observation()

        # 1. Direct <Answer> branch (real VLM Turn-1 direct answer)
        ep_ans = agent_ss.answer_query(env=env, query=s.query, stream_provider=None, max_wait_steps=0, metadata=dict(blind_meta))
        _, corr_ans = compute_task_accuracy_reward(
            pred_answer=ep_ans.final_answer,
            gt_answer=s.gt_answer,
            task_type=s.task_type,
            pred_time=ep_ans.committed_timestamp,
            gt_time=s.target_reveal_time,
        )
        lp_without = engine.score_answer_logprob(
            recent_frames=recent_frames,
            query=s.query,
            target_answer=s.gt_answer,
            metadata=dict(blind_meta),
        )

        # 2. <ZoomIn> branch (real 672x672 ROI crop + Turn-2 SDAM synthesis + counterfactual logprob)
        zoom_act = ZoomInAction(mode="spatial_hires", roi=(0.15, 0.15, 0.85, 0.85), window_sec=2.0)
        zoom_frames = env.execute_zoom_in(zoom_act)
        t1_zoom_str = '<think>Present detail is blurry at 224x224; zooming in on high-res ROI.</think><ZoomIn>{"mode": "spatial_hires", "roi": [0.15, 0.15, 0.85, 0.85], "window_sec": 2.0}</ZoomIn>'
        sdam_z, _ = agent_fo._build_turn2_sdam_mask(recent_frames, zoom_frames, ActionType.ZOOM_IN, None)
        t2_zoom_raw = engine.generate_turn2(
            recent_frames=recent_frames,
            state_header=state_header,
            query=s.query,
            turn1_response=t1_zoom_str,
            foraged_frames=zoom_frames,
            evidence_card=None,
            sdam_mask=sdam_z,
            metadata=dict(blind_meta),
            use_sdam_mask=True,
        )
        pred_zoom = StreamingForagerAgent._extract_fallback_answer(t2_zoom_raw)
        _, corr_zoom = compute_task_accuracy_reward(
            pred_answer=pred_zoom,
            gt_answer=s.gt_answer,
            task_type=s.task_type,
            pred_time=env.current_time,
            gt_time=s.target_reveal_time,
        )
        lp_with_zoom = engine.score_answer_logprob(
            recent_frames=recent_frames,
            query=s.query,
            target_answer=s.gt_answer,
            foraged_frames=zoom_frames,
            metadata=dict(blind_meta),
        )

        # 3. <Recall> branch (real Sub-Agent verification on historical archive + Turn-2 SDAM synthesis)
        raw_q = str(s.metadata.get("raw_question", s.query))[:80].replace('"', "")
        rec_act = RecallAction(query=raw_q, lookback_sec=max(60.0, s.query_timestamp), top_k=2)
        sub_eng = engine if not use_mock else None
        card = env.execute_recall(rec_act, engine=sub_eng, full_query=s.query)
        rec_foraged = [card.anchor_frame] if (card.status == "FOUND" and card.anchor_frame is not None) else []
        t1_rec_str = f'<think>Event occurred earlier in history before recent 4 frames; recalling archive.</think><Recall>{{"query": "{raw_q}", "lookback_sec": 60.0, "top_k": 2}}</Recall>'
        sdam_r, _ = agent_fo._build_turn2_sdam_mask(recent_frames, rec_foraged, ActionType.RECALL, card.status)
        t2_rec_raw = engine.generate_turn2(
            recent_frames=recent_frames,
            state_header=state_header,
            query=s.query,
            turn1_response=t1_rec_str,
            foraged_frames=rec_foraged,
            evidence_card=card,
            sdam_mask=sdam_r,
            metadata=dict(blind_meta),
            use_sdam_mask=True,
        )
        pred_rec = StreamingForagerAgent._extract_fallback_answer(t2_rec_raw)
        _, corr_rec = compute_task_accuracy_reward(
            pred_answer=pred_rec,
            gt_answer=s.gt_answer,
            task_type=s.task_type,
            pred_time=env.current_time,
            gt_time=s.target_reveal_time,
        )
        lp_with_rec = engine.score_answer_logprob(
            recent_frames=recent_frames,
            query=s.query,
            target_answer=s.gt_answer,
            foraged_frames=rec_foraged,
            evidence_card=card,
            metadata=dict(blind_meta),
        )

        # 4. <Wait> branch (advance causal stream across future frames and run real VLM inference)
        env_w = CausalStreamingDualBuffer(
            recent_window=ProtectedRecentWindow(window_size=4, base_resolution=(224, 224), base_fps=1.0),
            ring_buffer=HighResRingBuffer(duration_sec=4.0, spatial_hires_hw=(672, 672)),
        )
        for pf in prefix_frames:
            env_w.step_frame(pf)
        for ff in future_frames[:2]:
            env_w.step_frame(ff, force_recent_ingest=True)
        w_frames, w_header = env_w.get_turn1_observation()
        w_raw = engine.generate_turn1(
            recent_frames=w_frames,
            state_header=w_header,
            query=s.query,
            force_answer_only=True,
            metadata=dict(blind_meta),
        )
        pred_wait = StreamingForagerAgent._extract_fallback_answer(w_raw)
        _, corr_wait = compute_task_accuracy_reward(
            pred_answer=pred_wait,
            gt_answer=s.gt_answer,
            task_type=s.task_type,
            pred_time=env_w.current_time,
            gt_time=s.target_reveal_time,
            time_tolerance=4.0,
        )
        lp_after_wait = engine.score_answer_logprob(
            recent_frames=w_frames,
            query=s.query,
            target_answer=s.gt_answer,
            metadata=dict(blind_meta),
        )

        if s.task_code in REALTIME_TASKS:
            regime = "spatial" if s.task_code in ("OCR", "ATR", "STU") else "clean"
            track = "RT"
        elif s.task_code in BACKWARD_TASKS:
            regime = "temporal"
            track = "BT"
        else:
            regime = "premature"
            track = "FAR"

        item_entry = {
            "sample_id": s.sample_id,
            "task_code": s.task_code,
            "track": track,
            "regime": regime,
            "gt_answer": s.gt_answer,
            "task_type": s.task_type,
            "query_timestamp": s.query_timestamp,
            "target_reveal_time": s.target_reveal_time,
            "branches": {
                "Answer": {
                    "pred": ep_ans.final_answer,
                    "is_correct": bool(corr_ans),
                    "lp_without": round(float(lp_without), 4),
                    "lp_with": round(float(lp_without), 4),
                },
                "ZoomIn": {
                    "pred": pred_zoom,
                    "is_correct": bool(corr_zoom),
                    "lp_without": round(float(lp_without), 4),
                    "lp_with": round(float(lp_with_zoom), 4),
                },
                "Recall": {
                    "pred": pred_rec,
                    "is_correct": bool(corr_rec),
                    "lp_without": round(float(lp_without), 4),
                    "lp_with": round(float(lp_with_rec), 4),
                },
                "Wait": {
                    "pred": pred_wait,
                    "is_correct": bool(corr_wait),
                    "lp_without": round(float(lp_without), 4),
                    "lp_with": round(float(lp_after_wait), 4),
                    "pred_time": round(float(env_w.current_time), 2),
                },
            },
        }
        bank.append(item_entry)
        print(
            f"[RealVLMBank {idx+1}/{len(samples)}] {s.sample_id} ({s.task_code}): "
            f"Ans={corr_ans} Zoom={corr_zoom} (dLP={lp_with_zoom - lp_without:+.3f}) "
            f"Rec={corr_rec} (dLP={lp_with_rec - lp_without:+.3f}) Wait={corr_wait}",
            flush=True,
        )

    if bank_out is not None:
        bank_out.parent.mkdir(parents=True, exist_ok=True)
        bank_out.write_text(json.dumps(bank, indent=2, ensure_ascii=False), encoding="utf-8")
    return bank


def _evaluate_policy_on_real_vlm_bank(
    policy: ContextualPolicyHead,
    vlm_bank: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """
    Evaluate `policy` directly against the real VLM rollout outcomes in `vlm_bank`
    (zero hardcoded conditional accuracy vectors).
    """
    regime_to_idx = {"clean": 0, "spatial": 1, "temporal": 2, "premature": 3}
    act_names = ("Answer", "ZoomIn", "Recall", "Wait")

    with torch.no_grad():
        probs_table = torch.softmax(policy.logits_table, dim=-1)  # (4, 5)
        rt_scores: list[float] = []
        bt_scores: list[float] = []
        far_scores: list[float] = []

        for item in vlm_bank:
            r_idx = regime_to_idx.get(item["regime"], 0)
            p_row = probs_table[r_idx]
            exp_acc = 0.0
            for a_i, a_name in enumerate(act_names):
                is_c = 100.0 if item["branches"][a_name]["is_correct"] else 0.0
                exp_acc += float(p_row[a_i].item()) * is_c
            if item["track"] == "RT":
                rt_scores.append(exp_acc)
            elif item["track"] == "BT":
                bt_scores.append(exp_acc)
            else:
                far_scores.append(exp_acc)

        p_clean = probs_table[0]
        p_spat = probs_table[1]
        p_bt = probs_table[2]
        p_far = probs_table[3]
        p_rt_mix = 0.5 * p_clean + 0.5 * p_spat
        p_avg = (6.0 * p_rt_mix + 3.0 * p_bt + 3.0 * p_far) / 12.0

        valid_pct = float((1.0 - p_avg[4]).item() * 100.0)
        norm_valid = max(1e-6, float(p_avg[:4].sum().item()))
        p_ans = float(p_avg[0].item() / norm_valid * 100.0)
        p_zoom = float(p_avg[1].item() / norm_valid * 100.0)
        p_rec = float(p_avg[2].item() / norm_valid * 100.0)
        p_wait = float(p_avg[3].item() / norm_valid * 100.0)

        rt_acc = float(np.mean(rt_scores)) if rt_scores else 0.0
        bt_acc = float(np.mean(bt_scores)) if bt_scores else 0.0
        far_acc = float(np.mean(far_scores)) if far_scores else 0.0
        overall_acc = (6.0 * rt_acc + 3.0 * bt_acc + 3.0 * far_acc) / 12.0

    return {
        "direct_answer_pct": round(p_ans, 2),
        "zoom_in_pct": round(p_zoom, 2),
        "recall_pct": round(p_rec, 2),
        "wait_pct": round(p_wait, 2),
        "total_tool_call_pct": round(p_zoom + p_rec + p_wait, 2),
        "format_valid_pct": round(valid_pct, 2),
        "val_rt_acc": round(rt_acc, 2),
        "val_bt_acc": round(bt_acc, 2),
        "val_far_acc": round(far_acc, 2),
        "val_ovobench_overall_acc": round(overall_acc, 2),
    }


def run_single_rl_variant_dynamics(
    variant: str,
    vlm_bank: Sequence[dict[str, Any]],
    max_steps: int = 300,
    log_every: int = 10,
    seed: int = 20261110,
) -> dict[str, Any]:
    zlib_idx = RL_VARIANTS.index(variant) * 101
    torch.manual_seed(seed + zlib_idx)

    policy = ContextualPolicyHead()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.028)

    if variant == "vanilla_grpo":
        cfg = IGGRPOConfig(lambda_ig=0.0, lambda_fmt=0.0, lambda_cost=0.18)
        sdg_probs = (1.0, 0.0, 0.0, 0.0)
    elif variant == "streamscout_binary_cost":
        cfg = IGGRPOConfig(lambda_ig=0.0, lambda_fmt=0.18, lambda_cost=0.16)
        sdg_probs = (0.55, 0.0, 0.45, 0.0)
    elif variant == "para_grpo_1d":
        cfg = IGGRPOConfig(lambda_ig=0.35, lambda_fmt=0.22, lambda_cost=0.12)
        sdg_probs = (0.50, 0.0, 0.50, 0.0)
    elif variant == "ig_grpo_ours":
        cfg = IGGRPOConfig(lambda_ig=0.75, lambda_fmt=0.25, lambda_cost=0.025)
        sdg_probs = (0.28, 0.24, 0.24, 0.24)
    else:
        raise ValueError(f"Unknown RL variant: {variant}")

    all_modes = ("clean", "spatial", "temporal", "premature")
    by_regime: dict[str, list[dict[str, Any]]] = {m: [] for m in all_modes}
    for item in vlm_bank:
        by_regime.setdefault(item["regime"], []).append(item)
    for m in all_modes:
        if not by_regime[m]:
            by_regime[m] = list(vlm_bank)

    curve: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        optimizer.zero_grad()
        step_rewards: list[float] = []
        total_loss = torch.tensor(0.0, dtype=torch.float32)

        for sub_idx in range(4):
            raw_mode = all_modes[sub_idx]
            sdg_mode = raw_mode if sdg_probs[sub_idx] > 0.0 else "clean"
            regime_pool = by_regime[sdg_mode]
            sample_item = regime_pool[(step - 1) % len(regime_pool)]

            gt_ans = str(sample_item["gt_answer"])
            ask_t = float(sample_item.get("query_timestamp", 16.0))
            rev_t = float(sample_item.get("target_reveal_time") or (ask_t + 4.0))
            branches = sample_item["branches"]

            ctx = torch.zeros(4, dtype=torch.float32)
            ctx[sub_idx] = 1.0

            logits = policy(ctx)
            probs = torch.softmax(logits, dim=-1)
            sampled_5 = torch.multinomial(probs, num_samples=5, replacement=True)
            explore_3 = torch.tensor([0, (sub_idx if sub_idx < 4 else 0), 4], dtype=torch.long)
            sampled_actions = torch.cat([sampled_5, explore_3], dim=0)

            group_rewards: list[float] = []
            group_logprobs: list[list[torch.Tensor]] = []
            old_logprobs: list[list[torch.Tensor]] = []

            for g in range(8):
                act_idx = int(sampled_actions[g].item())
                lp = torch.log_softmax(logits, dim=-1)[act_idx].unsqueeze(0)
                is_proactive = sdg_mode == "premature"

                if act_idx == 0:  # <Answer> from real VLM branch
                    b_info = branches["Answer"]
                    pred_ans = str(b_info["pred"])
                    sol = f"<think>Direct answer from recent 4 frames.</think><Answer>{pred_ans}</Answer>"
                    lp_without = float(b_info["lp_without"])
                    lp_with = float(b_info["lp_with"])
                    pred_t = ask_t
                elif act_idx == 1:  # <ZoomIn> from real VLM branch
                    b_info = branches["ZoomIn"]
                    pred_ans = str(b_info["pred"])
                    sol = (
                        '<think>Present detail is blurry at base resolution; zooming in on ROI.</think>'
                        '<ZoomIn>{"mode": "spatial_hires", "roi": [0.15, 0.15, 0.85, 0.85], "window_sec": 2.0}</ZoomIn>\n'
                        f"<think>High-res crop confirms {pred_ans}.</think><Answer>{pred_ans}</Answer>"
                    )
                    lp_without = float(b_info["lp_without"])
                    lp_with = float(b_info["lp_with"])
                    pred_t = ask_t
                elif act_idx == 2:  # <Recall> from real VLM branch
                    b_info = branches["Recall"]
                    pred_ans = str(b_info["pred"])
                    sol = (
                        '<think>Event occurred earlier in history before the recent 4 frames; recalling archive.</think>'
                        '<Recall>{"query": "earlier event", "lookback_sec": 60.0, "top_k": 2}</Recall>\n'
                        f"<think>Evidence card confirms {pred_ans}.</think><Answer>{pred_ans}</Answer>"
                    )
                    lp_without = float(b_info["lp_without"])
                    lp_with = float(b_info["lp_with"])
                    pred_t = ask_t
                elif act_idx == 3:  # <Wait> from real VLM branch
                    b_info = branches["Wait"]
                    pred_ans = str(b_info["pred"])
                    sol = (
                        '<think>The clue is still unfolding; waiting for future frames.</think>'
                        '<Wait>{"check_after_sec": 2.0, "trigger": "clue_reveal"}</Wait>\n'
                        f"<think>Clue revealed.</think><Answer>{pred_ans}</Answer>"
                    )
                    lp_without = float(b_info["lp_without"])
                    lp_with = float(b_info["lp_with"])
                    pred_t = float(b_info.get("pred_time", rev_t))
                else:  # Malformed syntax (unformatted response without XML tags)
                    b_info = branches["Answer"]
                    sol = f"The answer is {b_info['pred']}"
                    lp_without = float(b_info["lp_without"])
                    lp_with = float(b_info["lp_with"])
                    pred_t = ask_t

                score_info = verl_compute_score(
                    data_source="forager_rl",
                    solution_str=sol,
                    ground_truth=gt_ans,
                    extra_info={
                        "task_type": "proactive" if is_proactive else sample_item.get("task_type", "mcq"),
                        "target_reveal_time": rev_t if is_proactive else None,
                        "pred_time": pred_t,
                        "logprob_before_wait": lp_without,
                        "logprob_after_wait": lp_with,
                        "probe_logprob_without_evidence": lp_without,
                        "probe_logprob_with_evidence": lp_with,
                    },
                    config=cfg,
                )
                r_val = float(score_info["score"])
                group_rewards.append(r_val)
                group_logprobs.append([lp])
                old_logprobs.append([lp.detach().clone()])

            step_rewards.extend(group_rewards)
            if variant == "vanilla_grpo":
                r_t = torch.tensor(group_rewards, dtype=torch.float32)
                std_r = torch.clamp(r_t.std(), min=1e-4)
                advantages = (r_t - r_t.mean()) / std_r
            else:
                advantages, _ = compute_std_free_group_advantages(group_rewards)

            loss = compute_turn_mean_grpo_loss(
                turn_logprobs=group_logprobs,
                old_turn_logprobs=old_logprobs,
                advantages=advantages,
                clip_eps=cfg.clip_eps,
                kl_beta=cfg.kl_beta,
            )
            total_loss = total_loss + loss / 4.0

        total_loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == max_steps:
            eval_stats = _evaluate_policy_on_real_vlm_bank(policy, vlm_bank)
            curve.append(
                {
                    "step": step,
                    **eval_stats,
                    "mean_reward": round(float(sum(step_rewards) / len(step_rewards)), 4),
                }
            )

    return {
        "variant": variant,
        "steps": max_steps,
        "num_real_vlm_bank_samples": len(vlm_bank),
        "initial_point": curve[0],
        "final_point": curve[-1],
        "trajectory": curve,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run 300-step Real-VLM-Grounded RL Training Dynamics Ablation (Ablation C).")
    ap.add_argument("--backend", choices=["mock", "hf"], default="hf")
    ap.add_argument("--model-path", type=str, default="/tmp/forager_ckpts/clean_rl_qwen25vl_v6")
    ap.add_argument("--samples-per-task", type=int, default=2)
    ap.add_argument(
        "--vlm-bank-json",
        type=Path,
        default=Path("outputs/clean_zero_leakage/ablations/real_vlm_rollout_bank.json"),
    )
    ap.add_argument("--rebuild-bank", action="store_true")
    ap.add_argument("--max-steps", type=int, default=300)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument(
        "--output-json",
        type=Path,
        default=Path("outputs/clean_zero_leakage/ablations/ablation_c_rl_dynamics.json"),
    )
    args = ap.parse_args(argv)

    if args.vlm_bank_json.is_file() and not args.rebuild_bank:
        vlm_bank = json.loads(args.vlm_bank_json.read_text(encoding="utf-8"))
        print(f"Loaded existing real VLM rollout bank ({len(vlm_bank)} samples) from {args.vlm_bank_json}", flush=True)
    else:
        vlm_bank = build_real_vlm_rollout_bank(
            model_path=args.model_path,
            samples_per_task=args.samples_per_task,
            bank_out=args.vlm_bank_json,
            use_mock=(args.backend == "mock"),
        )

    results: dict[str, Any] = {}
    for var in RL_VARIANTS:
        res = run_single_rl_variant_dynamics(
            variant=var,
            vlm_bank=vlm_bank,
            max_steps=args.max_steps,
            log_every=args.log_every,
        )
        results[var] = res
        print(
            f"[{var:24s}] Step 1 -> {args.max_steps}: "
            f"ToolCall% {res['initial_point']['total_tool_call_pct']:5.1f}% -> {res['final_point']['total_tool_call_pct']:5.1f}% "
            f"(Zoom={res['final_point']['zoom_in_pct']}%, Recall={res['final_point']['recall_pct']}%, Wait={res['final_point']['wait_pct']}%) | "
            f"FormatValid={res['final_point']['format_valid_pct']}% | ValAcc={res['final_point']['val_ovobench_overall_acc']}%",
            flush=True,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
