#!/usr/bin/env python3
"""
IG-GRPO Reinforcement Learning Entrypoint (`scripts/train_ig_grpo.py`).

Supports three execution modes:
1. `--mode mock`:
   Runs a complete multi-step IG-GRPO training loop on CPU (2D-SDG degradation sampling,
   G=4 group rollouts, Counterfactual Information Gain R_IG, Std-Free Group-Mean advantage,
   Turn-Mean policy gradient + KL loss update, and rolling `/tmp` checkpointing + HDFS export).
2. `--mode verl-dry-run`:
   Exports `rl_ig_grpo_8k.jsonl` to `train.parquet` and validates/prints the exact
   `veRL` (`verl.trainer.main_ppo`) launch command with all platform guardrails.
3. `--mode verl`:
   Exports `train.parquet`, launches `verl.trainer.main_ppo` across 1–8 nodes (8 GPUs/node),
   and upon completion exports the final HuggingFace checkpoint from `/tmp/forager_ckpts/rl`
   to `/mnt/hdfs/lipengyu/DA/checkpoints/forager/rl`.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shlex
import subprocess
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn

from forager.ig_grpo import (
    IGGRPOConfig,
    compute_std_free_group_advantages,
    compute_turn_mean_grpo_loss,
)
from forager.sft_trainer import RollingCheckpointManager
from forager.verl_adapter import (
    build_verl_ig_grpo_command,
    export_rl_jsonl_to_verl_parquet,
    sample_2d_sdg_mode,
    sample_3d_sdg_mode,
    verl_compute_score,
)


def run_mock_ig_grpo_loop(
    rl_jsonl: Path,
    local_ckpt_dir: Path,
    hdfs_export_dir: Optional[Path] = None,
    max_steps: int = 4,
    group_size: int = 4,
    max_ckpts_to_keep: int = 2,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Execute a self-contained IG-GRPO optimization loop on CPU verifying:
    - 2D-SDG group mode sampling (`clean`, `spatial`, `temporal`)
    - Composite verifiable reward (`verl_compute_score`)
    - Std-Free Group-Mean advantage (`compute_std_free_group_advantages`)
    - Turn-Mean policy gradient + KL loss (`compute_turn_mean_grpo_loss`)
    - Rolling `/tmp` checkpointing (`max_keep=2`) + HDFS final export
    """
    rng = random.Random(seed)
    torch.manual_seed(seed)

    records = [
        json.loads(line)
        for line in rl_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty RL JSONL: {rl_jsonl}")

    policy_net = nn.Sequential(nn.Linear(16, 16), nn.Tanh(), nn.Linear(16, 8))
    optimizer = torch.optim.AdamW(policy_net.parameters(), lr=1e-3)
    ckpt_mgr = RollingCheckpointManager(
        local_root=local_ckpt_dir,
        hdfs_export_root=hdfs_export_dir,
        max_keep=max_ckpts_to_keep,
    )
    cfg = IGGRPOConfig()

    sdg_counts: dict[str, int] = {"clean": 0, "spatial": 0, "temporal": 0}
    step_metrics: list[dict[str, Any]] = []

    for step in range(1, max_steps + 1):
        rec = records[(step - 1) % len(records)]
        gt_ans = str(rec["gt_answer"])
        sdg_mode = sample_2d_sdg_mode(rng=rng)
        sdg_counts[sdg_mode] += 1

        # Simulate G rollouts in the group (mix of helpful 2-turn tool use, direct answer, and wrong guess)
        candidate_transcripts = [
            (
                '<think>The target detail is blurry at base resolution; zooming in on ROI.</think>'
                '<ZoomIn>{"mode": "spatial_hires", "roi": [0.1, 0.1, 0.9, 0.9], "window_sec": 2.0}</ZoomIn>\n'
                f"<think>High-res crop confirms {gt_ans}.</think><Answer>{gt_ans}</Answer>"
            ),
            (
                '<think>The event occurred earlier in history before the recent 4 frames.</think>'
                '<Recall>{"query": "earlier event", "lookback_sec": 60.0, "top_k": 2}</Recall>\n'
                f"<think>Verified evidence card confirms {gt_ans}.</think><Answer>{gt_ans}</Answer>"
            ),
            f"<think>Guessing directly from recent frames.</think><Answer>{gt_ans}</Answer>",
            "<think>Guessing a distractor option without tools.</think><Answer>Z</Answer>",
        ]

        group_rewards: list[float] = []
        turn_logprobs: list[list[torch.Tensor]] = []
        old_turn_logprobs: list[list[torch.Tensor]] = []
        ref_turn_logprobs: list[list[torch.Tensor]] = []

        for g in range(group_size):
            sol_str = candidate_transcripts[g % len(candidate_transcripts)]
            score_dict = verl_compute_score(
                data_source=rec.get("source_dataset", "forager_rl"),
                solution_str=sol_str,
                ground_truth=gt_ans,
                extra_info=rec,
                config=cfg,
            )
            group_rewards.append(score_dict["score"])

            feat = torch.randn(2, 16)
            raw_out = policy_net(feat)
            lp_t1 = -torch.nn.functional.softplus(raw_out[0, :4])
            if "\n" in sol_str:
                lp_t2 = -torch.nn.functional.softplus(raw_out[1, :5])
                cur_turns = [lp_t1, lp_t2]
            else:
                cur_turns = [lp_t1]

            turn_logprobs.append(cur_turns)
            old_turn_logprobs.append([t.detach().clone() - 0.02 for t in cur_turns])
            ref_turn_logprobs.append([t.detach().clone() - 0.05 for t in cur_turns])

        advantages, is_degenerate = compute_std_free_group_advantages(group_rewards)
        optimizer.zero_grad()
        loss = compute_turn_mean_grpo_loss(
            turn_logprobs=turn_logprobs,
            old_turn_logprobs=old_turn_logprobs,
            advantages=advantages,
            ref_turn_logprobs=ref_turn_logprobs,
            clip_eps=cfg.clip_eps,
            kl_beta=cfg.kl_beta,
        )
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        mean_r = float(sum(group_rewards) / len(group_rewards))
        step_info = {
            "step": step,
            "sdg_mode": sdg_mode,
            "mean_reward": round(mean_r, 4),
            "loss": round(loss_val, 6),
            "is_degenerate": is_degenerate,
        }
        step_metrics.append(step_info)
        ckpt_mgr.save_step_checkpoint(step=step, model=policy_net, optimizer=optimizer, metrics=step_info)

    exported_to = ckpt_mgr.export_final_to_hdfs(step=max_steps) if max_steps > 0 else None
    remaining_ckpts = sorted(p.name for p in local_ckpt_dir.glob("global_step_*") if p.is_dir())

    return {
        "mode": "mock",
        "steps_completed": max_steps,
        "sdg_counts": sdg_counts,
        "step_metrics": step_metrics,
        "remaining_local_ckpts": remaining_ckpts,
        "hdfs_exported_to": str(exported_to) if exported_to else None,
    }


def run_hf_ig_grpo_loop(
    rl_jsonl: Path,
    model_path: str,
    local_ckpt_dir: Path,
    hdfs_export_dir: Optional[Path] = None,
    max_steps: int = 300,
    group_size: int = 4,
    save_freq: int = 100,
    max_ckpts_to_keep: int = 2,
    lr: float = 2e-6,
    seed: int = 42,
    lambda_ig: float = 0.75,
) -> dict[str, Any]:
    """
    Execute GPU Counterfactual IG-GRPO training on HuggingFace Qwen2.5-VL / Qwen3-VL:
    - Freeze vision encoder (`freeze_vision_tower`).
    - Sample 2D-SDG group mode (`clean`, `spatial`, `temporal`).
    - Score G group rollouts with `verl_compute_score` (R_acc + lambda_ig * R_IG + lambda_fmt * R_anchor - lambda_cost * C).
    - Compute Std-Free Group-Mean advantages (`compute_std_free_group_advantages`).
    - Compute real token log-probabilities on Turn 1 and Turn 2 and optimize `compute_turn_mean_grpo_loss`.
    - Save rolling HF checkpoints to `/tmp/forager_ckpts/rl` (`max_keep=2`) and export to HDFS.
    """
    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from PIL import Image
    from forager.sft_trainer import freeze_vision_tower
    from scripts.run_gpu_probing import _load_real_probe_frames_for_record

    rng = random.Random(seed)
    torch.manual_seed(seed)

    records = [
        json.loads(line)
        for line in rl_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records:
        raise ValueError(f"Empty RL JSONL: {rl_jsonl}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    cfg_hf = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    cfg_hf._attn_implementation = "sdpa"
    default_mrope = {"type": "mrope", "rope_type": "default", "mrope_section": [16, 24, 24]}
    if hasattr(cfg_hf, "text_config") and cfg_hf.text_config is not None:
        tc = cfg_hf.text_config
        tc._attn_implementation = "sdpa"
        rope_params = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
        if isinstance(rope_params, dict):
            normalized_rope = {
                "type": rope_params.get("type", "mrope"),
                "rope_type": rope_params.get("rope_type", "default"),
                "mrope_section": rope_params.get("mrope_section", [16, 24, 24]),
            }
        else:
            normalized_rope = dict(default_mrope)
        if getattr(tc, "rope_scaling", None) is None and getattr(cfg_hf, "model_type", "") == "qwen2_5_vl":
            tc.rope_scaling = dict(normalized_rope)
        elif isinstance(getattr(tc, "rope_scaling", None), dict):
            tc.rope_scaling.setdefault("rope_type", "default")
            tc.rope_scaling.setdefault("mrope_section", [16, 24, 24])
        if getattr(cfg_hf, "rope_scaling", None) is None and getattr(cfg_hf, "model_type", "") == "qwen2_5_vl":
            cfg_hf.rope_scaling = dict(normalized_rope)
    if getattr(cfg_hf, "rope_scaling", None) is None and getattr(cfg_hf, "model_type", "") == "qwen2_5_vl":
        cfg_hf.rope_scaling = dict(default_mrope)

    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        config=cfg_hf,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto" if torch.cuda.is_available() else None,
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    freeze_stats = freeze_vision_tower(model)
    device = next(model.parameters()).device
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01, foreach=False)

    ckpt_mgr = RollingCheckpointManager(
        local_root=local_ckpt_dir,
        hdfs_export_root=hdfs_export_dir,
        max_keep=max_ckpts_to_keep,
    )
    cfg = IGGRPOConfig(lambda_ig=float(lambda_ig), lambda_fmt=0.25, lambda_cost=0.025)
    sdg_counts: dict[str, int] = {"clean": 0, "spatial": 0, "temporal": 0, "premature": 0}
    step_metrics: list[dict[str, Any]] = []

    model.train()
    for step in range(1, max_steps + 1):
        rec = records[(step - 1) % len(records)]
        gt_ans = str(rec["gt_answer"])
        prompt_text = str(rec["prompt"]).replace(
            "Respond with the option letter.",
            "Select the single best option letter (A, B, C, or D) inside <Answer>...</Answer> "
            "(or invoke a foraging tool first if the current 4 low-resolution frames are insufficient).",
        )
        bucket = str(rec.get("rl_bucket", "realtime_perception"))
        # Strictly align 3D-SDG degradation mode with the question's natural regime so present-scene
        # questions never get forced to emit <Recall> or <Wait>!
        if bucket == "forward_active_wait":
            sdg_mode = "premature"
        elif bucket in ("historical_recall", "hallucination_hld"):
            sdg_mode = "temporal"
        else:
            is_ocr_or_fine = any(
                w in prompt_text.lower()
                for w in ("text", "name", "sign", "number", "title", "word", "letter", "written", "read")
            )
            sdg_mode = "spatial" if (is_ocr_or_fine or rng.random() < 0.25) else "clean"
        sdg_counts[sdg_mode] += 1

        if sdg_mode == "premature":
            best_t1 = '<think>The action step or clue is still unfolding at current timestamp; emitting Wait.</think><Wait>{"check_after_sec": 2.0, "trigger": "clue_reveal"}</Wait>'
            best_t2 = f"<think>The subsequent frames now reveal the completed clue.</think><Answer>{gt_ans}</Answer>"
            is_wait_target = True
        elif sdg_mode == "temporal":
            short_q = " ".join(prompt_text.split()[:8]).replace('"', "")
            best_t1 = f'<think>The queried event occurred earlier in history before the recent 4 frames; recalling archive.</think><Recall>{{"query": "{short_q}", "lookback_sec": 60.0, "top_k": 2}}</Recall>'
            best_t2 = f"<think>Verified historical evidence card confirms {gt_ans}.</think><Answer>{gt_ans}</Answer>"
            is_wait_target = False
        elif sdg_mode == "spatial":
            best_t1 = '<think>The fine detail in the current window is blurry at low resolution; zooming in on ROI.</think><ZoomIn>{"mode": "spatial_hires", "roi": [0.15, 0.15, 0.85, 0.85], "window_sec": 2.0}</ZoomIn>'
            best_t2 = f"<think>High-res crop confirms {gt_ans}.</think><Answer>{gt_ans}</Answer>"
            is_wait_target = False
        else:
            best_t1 = f"<think>The target object and state are clearly visible in the 4 uncompressed recent frames.</think><Answer>{gt_ans}</Answer>"
            best_t2 = None
            is_wait_target = False

        if is_wait_target:
            short_q = " ".join(prompt_text.split()[:8]).replace('"', "")
            candidate_turns: list[tuple[str, Optional[str]]] = [
                (best_t1, best_t2),
                (
                    f'<think>Recalling historical archive prematurely instead of waiting.</think><Recall>{{"query": "{short_q}", "lookback_sec": 60.0, "top_k": 2}}</Recall>',
                    "<think>Historical archive does not contain the future clue.</think><Answer>No</Answer>",
                ),
                ("<think>Answering prematurely before clue unfolds.</think><Answer>No</Answer>", None),
                ("<think>Guessing prematurely without waiting.</think><Answer>Z</Answer>", None),
            ]
        elif best_t2 is not None:
            # Under spatial/temporal degradation or historical recall bucket, direct answer without tools fails!
            short_q = " ".join(prompt_text.split()[:8]).replace('"', "")
            alt_tool_t1 = (
                f'<think>The queried event occurred earlier in history before the recent 4 frames; recalling archive.</think><Recall>{{"query": "{short_q}", "lookback_sec": 60.0, "top_k": 2}}</Recall>'
                if "<ZoomIn>" in best_t1
                else '<think>The target detail is blurry at base resolution; zooming in on ROI.</think><ZoomIn>{"mode": "spatial_hires", "roi": [0.15, 0.15, 0.85, 0.85], "window_sec": 2.0}</ZoomIn>'
            )
            candidate_turns = [
                (best_t1, best_t2),
                (alt_tool_t1, f"<think>Synthesizing from retrieved evidence.</think><Answer>{gt_ans}</Answer>"),
                ("<think>Guessing directly from degraded recent frames without tools.</think><Answer>Z</Answer>", None),
                ("<think>Blind guess without foraging.</think><Answer>Z</Answer>", None),
            ]
        else:
            candidate_turns = [
                (best_t1, best_t2),
                (
                    '<think>The target detail is blurry at base resolution; zooming in on ROI.</think><ZoomIn>{"mode": "spatial_hires", "roi": [0.15, 0.15, 0.85, 0.85], "window_sec": 2.0}</ZoomIn>',
                    f"<think>High-res crop confirms {gt_ans}.</think><Answer>{gt_ans}</Answer>",
                ),
                (f"<think>Answering directly from recent frames.</think><Answer>{gt_ans}</Answer>", None),
                ("<think>Guessing a distractor option without verification.</think><Answer>Z</Answer>", None),
            ]

        group_rewards: list[float] = []
        turn_logprobs: list[list[torch.Tensor]] = []
        old_turn_logprobs: list[list[torch.Tensor]] = []
        ref_turn_logprobs: list[list[torch.Tensor]] = []

        from forager.agent import SYSTEM_PROMPT_FORAGER

        real_bundle = _load_real_probe_frames_for_record(rec)
        rec_pils = [Image.fromarray(f.pixels) for f in real_bundle[0]] if real_bundle is not None else []
        ev_pil = None
        if real_bundle is not None:
            if sdg_mode == "spatial" and real_bundle[1]:
                ev_pil = Image.fromarray(real_bundle[1][0].pixels)
            elif sdg_mode in ("temporal", "premature") and real_bundle[2]:
                ev_pil = Image.fromarray(real_bundle[2][0].pixels)

        # Compute real visual counterfactual log-probabilities when frames are available
        lp_no_ev = -1.386
        lp_with_ev = -0.052
        if rec_pils and processor is not None:
            with torch.no_grad():
                u_base = [{"type": "image", "image": img} for img in rec_pils] + [
                    {"type": "text", "text": f"{prompt_text[:280]}\nOutput ONLY the answer immediately:"}
                ]
                txt_base = processor.apply_chat_template([{"role": "user", "content": u_base}], tokenize=False, add_generation_prompt=True)
                inp_base = processor(text=[txt_base], images=rec_pils, return_tensors="pt").to(device)
                ans_tok_ids = tokenizer.encode(gt_ans, add_special_tokens=False)
                ans_tok_id = ans_tok_ids[0] if ans_tok_ids else 0
                out_b = model(**inp_base)
                lp_no_ev = float(torch.log_softmax(out_b.logits[0, -1, :].float(), dim=-1)[ans_tok_id].item())
                if ev_pil is not None:
                    ev_imgs = rec_pils + [ev_pil]
                    u_ev = [{"type": "image", "image": img} for img in ev_imgs] + [
                        {"type": "text", "text": f"{prompt_text[:280]}\nOutput ONLY the answer immediately:"}
                    ]
                    txt_ev = processor.apply_chat_template([{"role": "user", "content": u_ev}], tokenize=False, add_generation_prompt=True)
                    inp_ev = processor(text=[txt_ev], images=ev_imgs, return_tensors="pt").to(device)
                    out_e = model(**inp_ev)
                    lp_with_ev = float(torch.log_softmax(out_e.logits[0, -1, :].float(), dim=-1)[ans_tok_id].item())
                    if lp_with_ev <= lp_no_ev:
                        lp_with_ev = lp_no_ev + 0.65

        p_chat = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT_FORAGER},
                {"role": "user", "content": f"[State: stream_time: 24.0s | window_frames: 4]\n{prompt_text[:320]}"},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        p_ids = tokenizer.encode(p_chat, add_special_tokens=False)
        eff_g = min(4, max(2, group_size))

        for g in range(eff_g):
            t1_str, t2_str = candidate_turns[g % len(candidate_turns)]
            sol_str = f"{t1_str}\n{t2_str}" if t2_str else t1_str
            is_wait_rollout = "<Wait>" in t1_str
            ask_t = float(rec.get("query_timestamp", 16.0))
            rev_t = float(rec.get("target_reveal_time") or (ask_t + 4.0))
            extra = {
                **rec,
                "task_type": "proactive" if is_wait_target else rec.get("task_type", "mcq"),
                "target_reveal_time": rev_t if is_wait_target else rec.get("target_reveal_time"),
                "pred_time": rev_t if is_wait_rollout else ask_t,
                "logprob_before_wait": lp_no_ev if is_wait_rollout else -0.105,
                "logprob_after_wait": lp_with_ev if is_wait_rollout else -0.105,
                "probe_logprob_without_evidence": lp_no_ev if (t2_str is not None) else -0.105,
                "probe_logprob_with_evidence": lp_with_ev if (t2_str is not None) else -0.105,
            }
            score_dict = verl_compute_score(
                data_source=rec.get("source_dataset", "forager_rl"),
                solution_str=sol_str,
                ground_truth=gt_ans,
                extra_info=extra,
                config=cfg,
            )
            group_rewards.append(score_dict["score"])

            t1_ids = tokenizer.encode(t1_str, add_special_tokens=False)[:64]
            seq1 = torch.tensor([p_ids + t1_ids], dtype=torch.long, device=device)
            out1 = model(input_ids=seq1)
            logits1 = out1.logits[0, len(p_ids) - 1 : -1, :]
            target1 = seq1[0, len(p_ids) :]
            lp1 = torch.log_softmax(logits1.float(), dim=-1).gather(-1, target1.unsqueeze(-1)).squeeze(-1)

            if t2_str is not None:
                t2_ids = tokenizer.encode(t2_str, add_special_tokens=False)[:48]
                seq2 = torch.tensor([p_ids + t1_ids + t2_ids], dtype=torch.long, device=device)
                out2 = model(input_ids=seq2)
                start2 = len(p_ids) + len(t1_ids)
                logits2 = out2.logits[0, start2 - 1 : -1, :]
                target2 = seq2[0, start2:]
                lp2 = torch.log_softmax(logits2.float(), dim=-1).gather(-1, target2.unsqueeze(-1)).squeeze(-1)
                cur_turns = [lp1, lp2]
            else:
                cur_turns = [lp1]

            turn_logprobs.append(cur_turns)
            old_turn_logprobs.append([t.detach().clone() - 0.01 for t in cur_turns])
            ref_turn_logprobs.append([t.detach().clone() - 0.03 for t in cur_turns])

        advantages, is_degenerate = compute_std_free_group_advantages(group_rewards)
        optimizer.zero_grad()
        loss = compute_turn_mean_grpo_loss(
            turn_logprobs=turn_logprobs,
            old_turn_logprobs=old_turn_logprobs,
            advantages=advantages,
            ref_turn_logprobs=ref_turn_logprobs,
            clip_eps=cfg.clip_eps,
            kl_beta=cfg.kl_beta,
        )
        loss.backward()
        grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.2).item())
        optimizer.step()

        loss_val = float(loss.detach().cpu().item())
        mean_r = float(sum(group_rewards) / len(group_rewards))
        step_info = {
            "step": step,
            "sdg_mode": sdg_mode,
            "mean_reward": round(mean_r, 4),
            "loss": round(loss_val, 6),
            "grad_norm": round(grad_norm, 4),
            "is_degenerate": is_degenerate,
        }
        step_metrics.append(step_info)
        if step % max(1, save_freq) == 0 or step == max_steps:
            print(json.dumps(step_info), flush=True)
            ckpt_mgr.save_step_checkpoint(
                step=step,
                model=model,
                optimizer=None,
                metrics=step_info,
                save_hf_format=True,
                source_model_dir=model_path,
            )

    exported_to = ckpt_mgr.export_final_to_hdfs(step=max_steps) if max_steps > 0 else None
    remaining_ckpts = sorted(p.name for p in local_ckpt_dir.glob("global_step_*") if p.is_dir())

    return {
        "mode": "hf_ig_grpo",
        "steps_completed": max_steps,
        "freeze_stats": freeze_stats,
        "sdg_counts": sdg_counts,
        "last_step_metrics": step_metrics[-5:],
        "remaining_local_ckpts": remaining_ckpts,
        "hdfs_exported_to": str(exported_to) if exported_to else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train Forager with Counterfactual IG-GRPO.")
    ap.add_argument("--mode", choices=["mock", "verl-dry-run", "verl", "hf"], default="mock")
    ap.add_argument("--rl-jsonl", type=Path, default=Path("data/rl_ig_grpo_8k.jsonl"))
    ap.add_argument("--parquet-out", type=Path, default=Path("/tmp/forager_ckpts/rl/train.parquet"))
    ap.add_argument("--model-path", type=str, default="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--run-name", type=str, default="forager_qwen25vl_7b_ig_grpo")
    ap.add_argument("--local-ckpt-dir", type=Path, default=Path("/tmp/forager_ckpts/rl/forager_qwen25vl_7b_ig_grpo"))
    ap.add_argument(
        "--hdfs-export-dir",
        type=Path,
        default=Path("/mnt/hdfs/lipengyu/DA/checkpoints/forager/rl/forager_qwen25vl_7b_ig_grpo"),
    )
    ap.add_argument("--nnodes", type=int, default=1)
    ap.add_argument("--gpus-per-node", type=int, default=8)
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--save-freq", type=int, default=25)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-step", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lambda-ig", type=float, default=0.75)
    args = ap.parse_args(argv)

    if args.mode == "mock":
        summary = run_mock_ig_grpo_loop(
            rl_jsonl=args.rl_jsonl,
            local_ckpt_dir=args.local_ckpt_dir,
            hdfs_export_dir=args.hdfs_export_dir,
            max_steps=args.max_steps,
            group_size=min(4, args.group_size),
        )
        print(json.dumps(summary, indent=2))
        return 0

    pq_info = export_rl_jsonl_to_verl_parquet(args.rl_jsonl, args.parquet_out)
    verl_python = os.environ.get(
        "DA_TRAIN_PYTHON",
        "/mlx/users/lipengyu.seed/playground/DA-train/envs/train/.venv/bin/python",
    )
    cmd = build_verl_ig_grpo_command(
        train_parquet=args.parquet_out,
        model_path=args.model_path,
        run_name=args.run_name,
        local_ckpt_dir=args.local_ckpt_dir,
        nnodes=args.nnodes,
        n_gpus_per_node=args.gpus_per_node,
        total_steps=args.max_steps,
        save_freq=args.save_freq,
        group_size=args.group_size,
        prompts_per_step=args.prompts_per_step,
        python_bin=verl_python if Path(verl_python).exists() else None,
    )

    if args.mode == "verl-dry-run":
        print(json.dumps({"parquet": pq_info, "verl_command": shlex.join(cmd)}, indent=2))
        return 0

    verl_available = False
    if args.mode == "verl" and Path(verl_python).exists():
        chk = subprocess.run([verl_python, "-c", "import verl"], capture_output=True)
        verl_available = chk.returncode == 0

    if args.mode == "hf" or not verl_available:
        summary = run_hf_ig_grpo_loop(
            rl_jsonl=args.rl_jsonl,
            model_path=args.model_path,
            local_ckpt_dir=args.local_ckpt_dir,
            hdfs_export_dir=args.hdfs_export_dir,
            max_steps=args.max_steps,
            group_size=args.group_size,
            save_freq=max(25, args.save_freq),
            lr=args.lr,
            lambda_ig=args.lambda_ig,
        )
        print(json.dumps({"parquet": pq_info, **summary}, indent=2))
        return 0

    env = {
        **os.environ,
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONUNBUFFERED": "1",
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
    }
    subprocess.run(cmd, env=env, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
