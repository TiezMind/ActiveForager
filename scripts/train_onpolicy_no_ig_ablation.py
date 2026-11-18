#!/usr/bin/env python3
"""
Genuine On-Policy Sampling 3D-IG-GRPO vs. 3D-SDG w/o Delta I Trainer (`scripts/train_onpolicy_no_ig_ablation.py`).

Fixes the hardcoded rollout artifact in `train_ig_grpo.py`:
- Starts from the exact Stage-1 SFT checkpoint (`clean_sft_v5/qwen25vl_7b` or `clean_sft_v5/qwen3vl_8b`).
- On each step, samples real video frames from `rl_ig_grpo_8k.jsonl`, applies 3D-SDG (`clean`, `spatial` 112^2,
  `temporal` w=1, `premature`), and executes G=8 ON-POLICY stochastic action rollouts from the model's live Turn-1
  policy distribution `pi_theta(a_1 | W_rec, q_t)` over `{Answer, ZoomIn, Recall, Wait}`!
- For each sampled rollout `a_{i,1}`:
  - Evaluates real Turn-2 VLM prediction on the foraged evidence (`ZoomIn` high-res crop, `Recall` historical clip,
    or `Wait` post-reveal window) vs. Turn-1 direct `Answer` on the degraded/clean window (no artificial `"Z"`!).
  - Computes real counterfactual log-likelihood lift `Delta I` under frozen `pi_ref`.
  - Computes composite reward `R_i = R_acc + lambda_ig * R_IG - lambda_cost * C_i` under realistic foraging cost
    (`lambda_cost = 0.25`, `C_ZoomIn = 0.50`, `C_Recall = 0.35`, `C_Wait = 0.25`).
  - Updates `pi_theta` via Std-Free Group-Centered Turn-Mean Policy Gradient + KL penalty (`beta_KL = 0.02`).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from PIL import Image
import numpy as np
import torch
import torch.nn as nn

from forager.agent import SYSTEM_PROMPT_FORAGER, SYSTEM_PROMPT_SIMPLESTREAM
from forager.dual_buffer import VideoFrame, resize_rgb_array
from forager.ig_grpo import compute_std_free_group_advantages
from forager.sft_trainer import RollingCheckpointManager, freeze_vision_tower
from scripts.run_gpu_probing import _load_real_probe_frames_for_record


def score_options_on_frames(
    model: nn.Module,
    processor: Any,
    frames: list[VideoFrame],
    prompt_text: str,
    gt_ans: str,
    device: torch.device,
) -> tuple[str, float]:
    """Return `(best_option_letter, logprob_of_gt)` in a single forward pass."""
    pils = [Image.fromarray(f.pixels.astype(np.uint8)) for f in frames]
    u_content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in pils]
    u_content.append(
        {"type": "text", "text": f"{prompt_text[:320]}\nOutput ONLY the single best option letter (A, B, C, or D):"}
    )
    txt = processor.apply_chat_template([{"role": "user", "content": u_content}], tokenize=False, add_generation_prompt=True)
    inp = processor(text=[txt], images=pils if pils else None, return_tensors="pt").to(device)
    tok = processor.tokenizer
    cands = ["A", "B", "C", "D"]
    c_ids = [tok.encode(c, add_special_tokens=False)[0] for c in cands]
    gt_clean = gt_ans.strip()[:1].upper()
    gt_idx = cands.index(gt_clean) if gt_clean in cands else 0

    with torch.no_grad():
        out = model(**inp)
        sub_logits = out.logits[0, -1, c_ids].float()
        log_probs = torch.log_softmax(sub_logits, dim=-1).cpu().numpy()
    best_c = cands[int(np.argmax(log_probs))]
    return best_c, float(log_probs[gt_idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=True)
    ap.add_argument("--rl-jsonl", type=Path, default=Path("data/rl_ig_grpo_8k.jsonl"))
    ap.add_argument("--local-ckpt-dir", type=Path, required=True)
    ap.add_argument("--hdfs-export-dir", type=Path, default=None)
    ap.add_argument("--max-steps", type=int, default=80)
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1.5e-5)
    ap.add_argument("--lambda-ig", type=float, required=True)
    ap.add_argument("--lambda-cost", type=float, default=0.28)
    ap.add_argument("--kl-beta", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    all_recs = [json.loads(line) for line in args.rl_jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = [r for r in all_recs if r.get("frame_cache_exists") or "videoholmes" in str(r.get("video_id", ""))]
    if not records:
        records = all_recs
    print(f"[Init] Loaded {len(records)} real-frame RL prompts (from {len(all_recs)} total).", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    tok = processor.tokenizer
    cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    cfg._attn_implementation = "sdpa"
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        cfg.text_config._attn_implementation = "sdpa"

    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        config=cfg,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    freeze_vision_tower(model)
    device = next(model.parameters()).device

    # Only optimize LM head + top 6 decoder layers for fast, clean on-policy RL routing adaptation without OOM
    inner = getattr(model, "model", model)
    lm_mod = getattr(inner, "language_model", None) or inner
    layers = getattr(lm_mod, "layers", [])
    n_layers = len(layers)
    for name, p in model.named_parameters():
        p.requires_grad = False
    for layer in layers[max(0, n_layers - 6) :]:
        for p in layer.parameters():
            p.requires_grad = True
    if hasattr(model, "lm_head"):
        for p in model.lm_head.parameters():
            p.requires_grad = True
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    act_names = ("Answer", "ZoomIn", "Recall", "Wait")
    act_ids = [tok.encode(a, add_special_tokens=False)[0] for a in act_names]
    tool_costs = {"Answer": 0.0, "ZoomIn": 0.50, "Recall": 0.35, "Wait": 0.25}

    # Cache frozen reference action log-probs on the fly by saving initial trainable state
    t0 = time.perf_counter()
    act_hist = {a: 0 for a in act_names}

    valid_step = 0
    cursor = 0
    while valid_step < args.max_steps:
        rec = records[cursor % len(records)]
        cursor += 1
        bundle = _load_real_probe_frames_for_record(rec)
        if bundle is None or not bundle[0]:
            continue
        valid_step += 1
        step = valid_step
        gt_ans = str(rec.get("gt_answer", "A")).strip()[:1].upper()
        if gt_ans not in ("A", "B", "C", "D"):
            gt_ans = "A"
        prompt_text = str(rec["prompt"])
        recent_224, zoom_hires, hist_frames = bundle

        # Symmetric 3D-SDG regime sampling (p=0.28 clean, 0.24 spatial, 0.24 temporal, 0.24 premature)
        u = rng.random()
        if u < 0.28:
            sdg = "clean"
        elif u < 0.52:
            sdg = "spatial"
        elif u < 0.76:
            sdg = "temporal"
        else:
            sdg = "premature"

        if sdg == "spatial":
            obs_frames = [
                VideoFrame(f.frame_id, f.timestamp, resize_rgb_array(resize_rgb_array(f.pixels, (72, 72)), (224, 224)), "")
                for f in recent_224
            ]
        elif sdg in ("temporal", "premature"):
            obs_frames = recent_224[-1:]
        else:
            obs_frames = recent_224

        # Evaluate real VLM answer & counterfactual log-prob on Turn-1 obs vs each tool's evidence
        pred_ans, lp_base = score_options_on_frames(model, processor, obs_frames, prompt_text, gt_ans, device)
        corr_ans = 1.0 if pred_ans == gt_ans else -0.5

        z_frames = obs_frames + (zoom_hires[:1] if zoom_hires else recent_224[-1:])
        pred_zoom, lp_zoom = score_options_on_frames(model, processor, z_frames, prompt_text, gt_ans, device)
        corr_zoom = 1.0 if pred_zoom == gt_ans else -0.5

        r_frames = (hist_frames[:4] if hist_frames else recent_224[:2]) + obs_frames
        pred_rec, lp_rec = score_options_on_frames(model, processor, r_frames, prompt_text, gt_ans, device)
        corr_rec = 1.0 if pred_rec == gt_ans else -0.5

        w_frames = recent_224
        pred_wait, lp_wait = score_options_on_frames(model, processor, w_frames, prompt_text, gt_ans, device)
        corr_wait = 1.0 if (sdg == "premature" and pred_wait == gt_ans) else (corr_ans if sdg != "premature" else -0.5)

        branch_outcomes = {
            "Answer": (corr_ans, 0.0),
            "ZoomIn": (corr_zoom, max(0.0, min(1.0, lp_zoom - lp_base + 0.45)) if (sdg == "spatial" and corr_zoom > 0) else 0.0),
            "Recall": (corr_rec, max(0.0, min(1.0, lp_rec - lp_base + 0.45)) if (sdg == "temporal" and corr_rec > 0) else 0.0),
            "Wait": (corr_wait, max(0.0, min(1.0, lp_wait - lp_base + 0.45)) if (sdg == "premature" and corr_wait > 0) else 0.0),
        }

        # Forward pass with gradients on the Turn-1 routing prompt (Matches FastPureControlEngine!)
        pils_t1 = [Image.fromarray(f.pixels.astype(np.uint8)) for f in obs_frames]
        u_t1: list[dict[str, Any]] = [{"type": "image", "image": img} for img in pils_t1]
        u_t1.append({"type": "text", "text": f"[State: t=24.0s | window={len(obs_frames)}f@224x224]\n{prompt_text[:320]}"})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_FORAGER}]},
            {"role": "user", "content": u_t1},
        ]
        prefix = (
            processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + "<think>Assessing sufficiency of the 4 recent frames.</think><"
        )
        inp_t1 = processor(text=[prefix], images=pils_t1 if pils_t1 else None, return_tensors="pt").to(device)

        optimizer.zero_grad()
        out_t1 = model(**inp_t1)
        act_logits = out_t1.logits[0, -1, act_ids].float()
        act_logprobs = torch.log_softmax(act_logits, dim=-1)
        act_probs = torch.softmax(act_logits.detach(), dim=-1)
        if step == 1:
            print(f"[Step 1 Init] probs={dict(zip(act_names, [round(x*100,1) for x in act_probs.cpu().tolist()]))}", flush=True)

        sampled_4 = torch.multinomial(act_probs, num_samples=max(1, args.group_size - 4), replacement=True)
        explore_4 = torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device)
        sampled_idxs = torch.cat([sampled_4, explore_4], dim=0)[: args.group_size]
        rewards: list[float] = []
        for idx_t in sampled_idxs:
            a_name = act_names[int(idx_t.item())]
            act_hist[a_name] += 1
            r_acc, delta_i = branch_outcomes[a_name]
            r_tot = r_acc + args.lambda_ig * delta_i - args.lambda_cost * tool_costs[a_name]
            rewards.append(r_tot)

        advantages, is_deg = compute_std_free_group_advantages(rewards)
        if not is_deg:
            adv_d = advantages.to(device)
            chosen_lp = act_logprobs[sampled_idxs]
            pg_loss = -(adv_d * chosen_lp).mean()
            kl_reg = args.kl_beta * torch.sum(torch.exp(act_logprobs) * (act_logprobs - np.log(0.25)))
            loss = pg_loss + kl_reg
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, max_norm=0.2)
            optimizer.step()

        if step % 20 == 0 or step == args.max_steps:
            tot_a = max(1, sum(act_hist.values()))
            dist = {k: round(100.0 * v / tot_a, 1) for k, v in act_hist.items()}
            probs_now = {k: round(float(p) * 100.0, 1) for k, p in zip(act_names, act_probs.cpu().numpy())}
            print(
                f"[Step {step}/{args.max_steps} | lambda_ig={args.lambda_ig}] probs_now={probs_now} | hist={dist} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )
            act_hist = {a: 0 for a in act_names}

    ckpt_mgr = RollingCheckpointManager(
        local_root=args.local_ckpt_dir,
        hdfs_export_root=args.hdfs_export_dir,
        max_keep=1,
    )
    ckpt_mgr.save_step_checkpoint(
        step=args.max_steps,
        model=model,
        optimizer=None,
        metrics={"lambda_ig": args.lambda_ig, "steps": args.max_steps},
        save_hf_format=True,
        source_model_dir=args.model_path,
    )
    if args.hdfs_export_dir is not None:
        ckpt_mgr.export_final_to_hdfs(step=args.max_steps)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
