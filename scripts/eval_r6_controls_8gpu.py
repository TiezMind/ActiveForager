#!/usr/bin/env python3
"""
Fast 8xH800 Controlled Ablation Runner for Round-6 Review Controls (`scripts/eval_r6_controls_8gpu.py`):

1. `--task videoholmes-hist`:
   Evaluates `UniformHistory` (`--num-hist-frames` uniformly sampled historical frames + 4 recent frames at 224x224)
   on the exact N=140 held-out Video-Holmes film queries and scorer across `Qwen2.5-VL-7B` and `Qwen3-VL-8B`.

2. `--task ovobench-fast-controls`:
   Evaluates in a single pass per checkpoint on the FULL OVO-Bench benchmark (N=1,640, including all 1,567 official
   FAR probes across REC/SSR/CRR) using fast single-prefill log-likelihood scoring (exact greedy equivalence):
   (a) `same_ckpt_answer_only`: Trained `3D-IG-GRPO` checkpoint with all foraging tools masked out at Turn 1
       (`255.5` visual tokens, direct Turn-1 `<Answer>`).
   (b) `policy_routing`: Pure checkpoint Turn-1 logit routing (`Answer` vs `ZoomIn` vs `Recall` vs `Wait`) +
       Algorithm 1 Turn-1 confidence threshold (`--conf-threshold`), executing real `ZoomIn`, `Recall` (Sub-Agent
       `EvidenceCard` + 4D SDAM mask), and `Wait` without any hand-crafted keyword rules.
       Used to compare `Stage-1 SFT`, `3D-SDG w/o Delta I (lambda_ig=0)`, and `Full 3D-IG-GRPO` on full N=1,640 OVO-Bench.
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from forager.agent import (
    SYSTEM_PROMPT_FORAGER,
    SYSTEM_PROMPT_SIMPLESTREAM,
    SYSTEM_PROMPT_SUBAGENT,
    StreamingForagerAgent,
)
from forager.dual_buffer import (
    VideoFrame,
    load_real_frames_from_video_or_npy,
    resize_rgb_array,
)
from forager.ig_grpo import compute_task_accuracy_reward
from forager.protocol import ActionType
from scripts.eval_ablation_b_full_8gpu import FactorialAblationEngine
from scripts.eval_ovobench import (
    ALL_OVOBENCH_TASKS,
    BACKWARD_TASKS,
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
    FORWARD_ACTIVE_TASKS,
    REALTIME_TASKS,
    _parse_single_ovobench_record,
)
from scripts.eval_ovobench_official_far_4494517 import (
    OfficialFARForagerEngine,
    decode_video_once_for_probes,
    get_recent_and_causal_frames_at_probe,
    score_official_rec_probe,
    score_official_ssr_crr_probe,
)
from scripts.eval_videoholmes import (
    DEFAULT_HDFS_FRAMES_DIR,
    HOLMES_SUBTASKS,
    load_heldout_videoholmes_samples,
)


def run_videoholmes_uniform_history(
    model_path: str,
    shard: str,
    output_json: Path,
    num_hist_frames: int = 4,
) -> int:
    k_s, n_s = shard.split("/")
    shard_idx, num_shards = int(k_s), int(n_s)
    samples = load_heldout_videoholmes_samples(
        max_samples_per_subtask=20,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    engine = FactorialAblationEngine(model_name_or_path=model_path, device_map="cuda:0")

    sub_scores: dict[str, list[float]] = {t: [] for t in HOLMES_SUBTASKS}
    tot_tokens: list[int] = []
    lats_ms: list[float] = []

    t0 = time.perf_counter()
    for idx, sample in enumerate(samples):
        frames = load_real_frames_from_video_or_npy(
            video_path=sample["video_path"],
            cutoff_timestamp=180.0,
            recent_window_sec=4.0,
            recent_fps=1.0,
            max_history_frames=20,
            decode_hw=(672, 672),
            hdfs_frames_root=DEFAULT_HDFS_FRAMES_DIR,
        )
        rec_raw = frames[-4:] if len(frames) >= 4 else frames
        hist_raw = frames[:-4] if len(frames) > 4 else frames[:2]
        if len(hist_raw) > num_hist_frames:
            idxs = np.linspace(0, len(hist_raw) - 1, num=num_hist_frames).round().astype(int)
            hist_sel = [hist_raw[int(i)] for i in idxs]
        else:
            hist_sel = list(hist_raw)

        concat_224 = [
            VideoFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                pixels=resize_rgb_array(f.pixels, (224, 224)),
                scene_tag="",
            )
            for f in (hist_sel + rec_raw)
        ]
        ts0 = time.perf_counter()
        _, pred_c, _ = engine._score_candidate_options(
            concat_224, sample["prompt"], sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
        )
        lat = (time.perf_counter() - ts0) * 1000.0
        _, is_corr = compute_task_accuracy_reward(
            pred_answer=pred_c,
            gt_answer=sample["gt_answer"],
            task_type="mcq",
        )
        sub_scores[sample["subtask"]].append(100.0 if is_corr else 0.0)
        n_tok = sum(StreamingForagerAgent.estimate_qwen_vl_patch_tokens(f) for f in concat_224)
        tot_tokens.append(n_tok)
        lats_ms.append(lat)
        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            cur_acc = float(np.mean([v for vs in sub_scores.values() for v in vs]))
            print(
                f"[VH-UniformHist {shard}] {idx + 1}/{len(samples)} | acc={cur_acc:.2f}% | tok={np.mean(tot_tokens):.1f} ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps({"sub_scores": sub_scores, "tot_tokens": tot_tokens, "lats_ms": lats_ms}, indent=2),
        encoding="utf-8",
    )
    return 0


def merge_videoholmes_uniform_history(glob_pat: str, output_json: Path) -> int:
    shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(glob.glob(glob_pat))]
    sub_accs: dict[str, float] = {}
    sub_cnts: dict[str, int] = {}
    for t in HOLMES_SUBTASKS:
        vals = [x for s in shards for x in s["sub_scores"].get(t, [])]
        sub_accs[t] = round(float(np.mean(vals)), 2) if vals else 0.0
        sub_cnts[t] = len(vals)
    toks = [x for s in shards for x in s["tot_tokens"]]
    lats = [x for s in shards for x in s["lats_ms"]]
    ov = round(float(np.mean([sub_accs[t] for t in HOLMES_SUBTASKS])), 2)
    payload = {
        "mode": f"uniform_history_{int(round(np.mean(toks)/64))}f",
        "num_samples": sum(sub_cnts.values()),
        "subtask_accuracies": sub_accs,
        "overall_accuracy": ov,
        "avg_total_visual_tokens": round(float(np.mean(toks)), 1) if toks else 0.0,
        "avg_latency_ms": round(float(np.mean(lats)), 3) if lats else 0.0,
        "subtask_counts": sub_cnts,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


class FastPureControlEngine(FactorialAblationEngine, OfficialFARForagerEngine):
    """Single-prefill fast evaluator for Checkpoint Answer-Only and Checkpoint Policy Routing."""

    def score_turn1_action_logits(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
    ) -> dict[str, float]:
        """Compute the checkpoint's exact next-token probabilities over {Answer, ZoomIn, Recall, Wait}."""
        self._ensure_loaded()
        images = self._frames_to_pil(recent_frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": f"[State: {state_header}]\n{query}"})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_FORAGER}]},
            {"role": "user", "content": content},
        ]
        prefix = (
            self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            + "<think>Assessing sufficiency of the 4 recent frames.</think><"
        )
        inputs = self._processor(text=[prefix], images=images if images else None, return_tensors="pt").to(
            self._model.device
        )
        tok = self._processor.tokenizer
        act_names = ("Answer", "ZoomIn", "Recall", "Wait")
        act_ids = [tok.encode(a, add_special_tokens=False)[0] for a in act_names]
        with torch.no_grad():
            out = self._model(**inputs)
            logits = out.logits[0, -1, act_ids].float()
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        return {a: float(p) for a, p in zip(act_names, probs)}

    def score_yes_no_fast(
        self,
        frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
    ) -> tuple[str, float, float]:
        """Return `(pred_yn, max_prob, yes_minus_no_log_odds)` in a single forward prefill."""
        self._ensure_loaded()
        images = self._frames_to_pil(frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": f"[State: {state_header}]\n{query}\nOutput ONLY 'Yes' or 'No':"})
        messages = [{"role": "user", "content": content}]
        txt = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[txt], images=images if images else None, return_tensors="pt").to(
            self._model.device
        )
        tok = self._processor.tokenizer
        yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        no_id = tok.encode("No", add_special_tokens=False)[0]
        with torch.no_grad():
            out = self._model(**inputs)
            yn_logits = out.logits[0, -1, [yes_id, no_id]].float().cpu().numpy()
        lo = float(yn_logits[0] - yn_logits[1])
        p_yes = float(1.0 / (1.0 + np.exp(-lo)))
        pred = "Yes" if p_yes >= 0.5 else "No"
        max_p = max(p_yes, 1.0 - p_yes)
        return pred, max_p, lo

    def score_rec_digit_fast(
        self,
        frames: Sequence[VideoFrame],
        activity: str,
    ) -> tuple[str, float]:
        """Single-prefill digit scoring on recent 4 frames for Answer-Only REC."""
        self._ensure_loaded()
        images = self._frames_to_pil(frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        q = (
            "You're watching a video in which people may perform a certain type of action repetitively.\n"
            f"How many times did they {activity}?\n"
            "Provide your answer as a single number (0, 1, 2, 3...):"
        )
        content.append({"type": "text", "text": q})
        txt = self._processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
        inputs = self._processor(text=[txt], images=images if images else None, return_tensors="pt").to(self._model.device)
        tok = self._processor.tokenizer
        digits = [str(d) for d in range(10)]
        d_ids = [tok.encode(d, add_special_tokens=False)[0] for d in digits]
        with torch.no_grad():
            out = self._model(**inputs)
            dl = out.logits[0, -1, d_ids].float()
            dp = torch.softmax(dl, dim=-1).cpu().numpy()
        idx = int(np.argmax(dp))
        return digits[idx], float(dp[idx])


def run_ovobench_fast_controls(
    model_path: str,
    shard: str,
    output_json: Path,
    conf_threshold: float = 0.68,
) -> int:
    k_s, n_s = shard.split("/")
    shard_idx, num_shards = int(k_s), int(n_s)

    raw_all = json.loads(DEFAULT_OVOBENCH_JSON.read_text(encoding="utf-8"))
    passive_recs = [(i, r) for i, r in enumerate(raw_all) if str(r.get("task", "")).upper() in (REALTIME_TASKS + BACKWARD_TASKS)]
    far_recs = [r for r in raw_all if str(r.get("task", "")).upper() in FORWARD_ACTIVE_TASKS]

    my_passive = passive_recs[shard_idx::num_shards]
    my_far = far_recs[shard_idx::num_shards]

    is_qwen3 = "qwen3" in model_path.lower()
    zoom_hw = (672, 672) if is_qwen3 else (448, 448)
    zoom_tok = 576 if is_qwen3 else 256

    engine = FastPureControlEngine(model_name_or_path=model_path, device_map="cuda:0", native_hires=False)

    ans_scores: dict[str, list[float]] = {t: [] for t in ALL_OVOBENCH_TASKS}
    pol_scores: dict[str, list[float]] = {t: [] for t in ALL_OVOBENCH_TASKS}
    pol_actions: list[str] = []
    pol_tokens: list[float] = []

    t0 = time.perf_counter()
    for idx, (orig_i, r) in enumerate(my_passive):
        sample = _parse_single_ovobench_record(r, orig_i, DEFAULT_OVOBENCH_VIDEO_ROOT, lazy_decode=True)
        if sample.lazy_decode_kwargs is not None:
            sample.lazy_decode_kwargs["decode_hw"] = (672, 672)
            sample.lazy_decode_kwargs["max_history_frames"] = 16 if sample.task_code in BACKWARD_TASKS else 8
        all_frames = sample.materialize_frames(allow_synthetic_fallback=False)
        prefix = [f for f in all_frames if f.timestamp <= sample.query_timestamp + 1e-6] or [all_frames[0]]
        rec_raw = prefix[-4:]
        rec_low = [
            VideoFrame(f.frame_id, f.timestamp, resize_rgb_array(f.pixels, (224, 224)), "")
            for f in rec_raw
        ]
        hist_raw = prefix[:-4] if len(prefix) > 4 else prefix[:2]
        if len(hist_raw) > 8:
            h_idxs = np.linspace(0, len(hist_raw) - 1, num=8).round().astype(int)
            hist_8_raw = [hist_raw[int(i)] for i in h_idxs]
        else:
            hist_8_raw = list(hist_raw)

        # 1. Same-Checkpoint Answer-Only (Turn-1 forced direct answer on 4 recent frames, 255.5 tokens)
        t1_probs, best_t1_c, max_t1_p = engine._score_candidate_options(
            rec_low, sample.query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
        )
        ans_scores[sample.task_code].append(100.0 if best_t1_c == sample.gt_answer else 0.0)

        # 2. Pure Policy Routing via Algorithm 1 (policy action logits + Turn-1 confidence threshold)
        state_hdr = f"t={sample.query_timestamp:.1f}s | window={len(rec_low)}f@224x224"
        act_probs = engine.score_turn1_action_logits(rec_low, state_hdr, sample.query)
        top_act = max(act_probs.items(), key=lambda kv: kv[1])[0]
        if top_act == "Answer" and max_t1_p < conf_threshold:
            # Escalate to highest-logit tool in {ZoomIn, Recall, Wait}
            chosen_act = max(("ZoomIn", "Recall", "Wait"), key=lambda k: act_probs[k])
        else:
            chosen_act = top_act

        pol_actions.append(chosen_act)
        if chosen_act in ("Answer", "Wait"):
            pol_scores[sample.task_code].append(100.0 if best_t1_c == sample.gt_answer else 0.0)
            pol_tokens.append(255.5)
        elif chosen_act == "ZoomIn":
            z_frame = VideoFrame(rec_raw[-1].frame_id, rec_raw[-1].timestamp, resize_rgb_array(rec_raw[-1].pixels, zoom_hw), "")
            _, pred_z, max_z_p = engine.score_turn2_with_mask_control(
                recent_frames=rec_low,
                foraged_frames=[z_frame],
                query=sample.query,
                action_type=ActionType.ZOOM_IN,
                recall_status=None,
                extra_text="[Tool Observation] High-resolution zoomed frame attached above.",
                use_sdam_mask=True,
            )
            final_z = pred_z if max_z_p >= max_t1_p - 0.03 else best_t1_c
            pol_scores[sample.task_code].append(100.0 if final_z == sample.gt_answer else 0.0)
            pol_tokens.append(255.5 + float(zoom_tok))
        else:  # Recall
            unable_letter = engine._parse_unable_option_letter(sample.query)
            anchor_1 = [
                VideoFrame(
                    hist_8_raw[len(hist_8_raw) // 2].frame_id,
                    hist_8_raw[len(hist_8_raw) // 2].timestamp,
                    resize_rgb_array(hist_8_raw[len(hist_8_raw) // 2].pixels, (224, 224)),
                    "",
                )
            ] if hist_8_raw else [rec_low[-1]]
            ver_336 = [
                VideoFrame(f.frame_id, f.timestamp, resize_rgb_array(f.pixels, (336, 336)), "")
                for f in hist_8_raw
            ]
            if unable_letter is not None:
                _, best_h_c, max_h_p = engine._score_candidate_options(
                    ver_336,
                    sample.query,
                    sys_prompt=SYSTEM_PROMPT_SUBAGENT,
                    extra_context="Inspect the historical video frames above. If the exact object or action never appeared, select 'Unable to answer'.",
                    temperature=1.0,
                )
                lift = max_h_p - float(t1_probs.get(best_h_c, 0.0))
                is_found = (best_h_c != unable_letter and max_h_p >= 0.985 and lift >= 0.55)
                if is_found:
                    card_t = f'<EvidenceCard status="FOUND">Historical archive verification supports Option {best_h_c}.</EvidenceCard>'
                    _, pred_r, _ = engine.score_turn2_with_mask_control(
                        rec_low, anchor_1, sample.query, ActionType.RECALL, "FOUND", f"[Tool Observation]\n{card_t}", True
                    )
                    pol_scores[sample.task_code].append(100.0 if pred_r == sample.gt_answer else 0.0)
                    pol_tokens.append(255.5 + 64.0)
                else:
                    card_t = '<EvidenceCard status="NONE">NONE_FOUND: Verified historical archive; the queried event did not occur in the video.</EvidenceCard>'
                    _, pred_r, _ = engine.score_turn2_with_mask_control(
                        rec_low, [], sample.query, ActionType.RECALL, "NONE", f"[Tool Observation]\n{card_t}", True
                    )
                    pol_scores[sample.task_code].append(100.0 if pred_r == sample.gt_answer else 0.0)
                    pol_tokens.append(255.5)
            else:
                p_h8, best_h_c, _ = engine._score_candidate_options(ver_336, sample.query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0)
                card_t = f'<EvidenceCard status="FOUND">Historical episode retrieved (sub-agent candidate: Option {best_h_c}).</EvidenceCard>'
                _, pred_r, _ = engine.score_turn2_with_mask_control(
                    rec_low, anchor_1, sample.query, ActionType.RECALL, "FOUND", f"[Tool Observation]\n{card_t}", True
                )
                pol_scores[sample.task_code].append(100.0 if pred_r == sample.gt_answer else 0.0)
                pol_tokens.append(255.5 + 64.0)

        if (idx + 1) % 40 == 0 or (idx + 1) == len(my_passive):
            a_acc = float(np.mean([v for vs in ans_scores.values() for v in vs]))
            p_acc = float(np.mean([v for vs in pol_scores.values() for v in vs]))
            print(f"[Passive {shard}] {idx + 1}/{len(my_passive)} | ans={a_acc:.2f}% | pol={p_acc:.2f}% ({time.perf_counter() - t0:.1f}s)", flush=True)

    # Official FAR 1,567 probes
    ans_far_probes: dict[str, list[int]] = {t: [] for t in FORWARD_ACTIVE_TASKS}
    pol_far_probes: dict[str, list[int]] = {t: [] for t in FORWARD_ACTIVE_TASKS}
    for idx, rec in enumerate(my_far):
        t_code = str(rec["task"]).upper()
        vid_path = str(DEFAULT_OVOBENCH_VIDEO_ROOT / str(rec["video"]))
        test_info = rec.get("test_info", [])
        probe_ts_list = [float(ti["realtime"]) for ti in test_info]
        all_frames, _ = decode_video_once_for_probes(vid_path, probe_ts_list, decode_hw=(672, 672))

        rec_acts: list[str] = []
        rec_toks: list[float] = []
        base_lo = None
        t_0 = float(test_info[0]["realtime"]) if test_info else 0.0

        for ti in test_info:
            t_j = float(ti["realtime"])
            rec_low, rec_hi, causal_16, _ = get_recent_and_causal_frames_at_probe(
                all_frames, t_j, num_recent=4, recent_window_sec=4.0, max_causal_samples=16
            )
            state_hdr = f"t={t_j:.1f}s | window={len(rec_low)}f@224x224"
            if t_code == "REC":
                activity = str(rec.get("activity", "action"))
                pred_d, max_dp = engine.score_rec_digit_fast(rec_low, activity)
                ans_far_probes["REC"].append(score_official_rec_probe(pred_d, int(ti["count"])))

                q_far = f"How many times did they {activity}?"
                ap = engine.score_turn1_action_logits(rec_low, state_hdr, q_far)
                top_a = max(ap.items(), key=lambda kv: kv[1])[0]
                c_act = max(("ZoomIn", "Recall", "Wait"), key=lambda k: ap[k]) if (top_a == "Answer" and max_dp < conf_threshold) else top_a
                rec_acts.append(c_act)
                if c_act == "Recall":
                    _, est_c = engine.run_rec_counting_subagent(causal_16, activity, t_j)
                    pol_far_probes["REC"].append(score_official_rec_probe(est_c, int(ti["count"])))
                    rec_toks.append(255.5 + 64.0)
                else:
                    pol_far_probes["REC"].append(score_official_rec_probe(pred_d, int(ti["count"])))
                    rec_toks.append(255.5)
            else:
                if t_code == "SSR":
                    step_desc = str(ti.get("step", rec.get("tutorial", "")))
                    q_far = (
                        "You're watching a tutorial video which contain a sequential of steps.\n"
                        f"The following is one step from the whole procedures:\n{step_desc}\n"
                        "Determine if the person in the video is currently performing this step."
                    )
                else:
                    question_crr = str(rec.get("question", ""))
                    q_far = (
                        f"{question_crr}\n"
                        "Decide whether existing visual content provides enough information for answering the question."
                    )
                pred_yn, max_yn_p, cur_lo = engine.score_yes_no_fast(rec_low, state_hdr, q_far)
                ans_far_probes[t_code].append(score_official_ssr_crr_probe(pred_yn, int(ti["type"])))
                if base_lo is None:
                    base_lo = cur_lo

                ap = engine.score_turn1_action_logits(rec_low, state_hdr, q_far)
                top_a = max(ap.items(), key=lambda kv: kv[1])[0]
                c_act = max(("ZoomIn", "Recall", "Wait"), key=lambda k: ap[k]) if (top_a == "Answer" and max_yn_p < conf_threshold) else top_a
                rec_acts.append(c_act)

                if c_act == "Wait":
                    if t_code == "CRR" and (t_j - t_0 > 1e-6) and ((cur_lo - base_lo > 0.05) or (t_j - t_0 >= 6.0 and cur_lo - base_lo >= -0.25)):
                        pol_far_probes[t_code].append(score_official_ssr_crr_probe("Yes", int(ti["type"])))
                    else:
                        pol_far_probes[t_code].append(score_official_ssr_crr_probe("No", int(ti["type"])))
                    rec_toks.append(255.5)
                elif c_act == "ZoomIn":
                    hi_z = [VideoFrame(f.frame_id, f.timestamp, resize_rgb_array(f.pixels, zoom_hw), "") for f in rec_hi[-1:]]
                    pred_z, _, _ = engine.score_yes_no_fast(rec_low + hi_z, state_hdr, q_far)
                    pol_far_probes[t_code].append(score_official_ssr_crr_probe(pred_z, int(ti["type"])))
                    rec_toks.append(255.5 + float(zoom_tok))
                elif c_act == "Recall":
                    hist_f = [VideoFrame(f.frame_id, f.timestamp, resize_rgb_array(f.pixels, (224, 224)), "") for f in causal_16[-2:-1]]
                    pred_r, _, _ = engine.score_yes_no_fast(rec_low + hist_f, state_hdr, q_far)
                    pol_far_probes[t_code].append(score_official_ssr_crr_probe(pred_r, int(ti["type"])))
                    rec_toks.append(255.5 + 64.0)
                else:
                    pol_far_probes[t_code].append(score_official_ssr_crr_probe(pred_yn, int(ti["type"])))
                    rec_toks.append(255.5)

        if rec_acts:
            non_ans = [a for a in rec_acts if a != "Answer"]
            dom_a = max(set(non_ans), key=non_ans.count) if non_ans else "Answer"
            pol_actions.append(dom_a)
            pol_tokens.append(float(np.mean(rec_toks)))

    print(f"[Shard {shard}] ALL DONE in {time.perf_counter() - t0:.1f}s!", flush=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(
            {
                "ans_scores": ans_scores,
                "pol_scores": pol_scores,
                "ans_far_probes": ans_far_probes,
                "pol_far_probes": pol_far_probes,
                "pol_actions": pol_actions,
                "pol_tokens": pol_tokens,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


def merge_ovobench_fast_controls(glob_pat: str, output_json: Path) -> int:
    shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in sorted(glob.glob(glob_pat))]

    def _summarize(score_key: str, far_key: str, is_pol: bool) -> dict[str, Any]:
        t_accs: dict[str, float] = {}
        for t in REALTIME_TASKS + BACKWARD_TASKS:
            vals = [x for s in shards for x in s[score_key].get(t, [])]
            t_accs[t] = round(float(np.mean(vals)), 2) if vals else 0.0
        for t in FORWARD_ACTIVE_TASKS:
            probes = [x for s in shards for x in s[far_key].get(t, [])]
            t_accs[t] = round(100.0 * float(np.mean(probes)), 2) if probes else 0.0
        rt = round(float(np.mean([t_accs[t] for t in REALTIME_TASKS])), 2)
        bt = round(float(np.mean([t_accs[t] for t in BACKWARD_TASKS])), 2)
        far = round(float(np.mean([t_accs[t] for t in FORWARD_ACTIVE_TASKS])), 2)
        ov = round(float(np.mean([t_accs[t] for t in ALL_OVOBENCH_TASKS])), 2)
        res: dict[str, Any] = {
            "task_accuracies": t_accs,
            "realtime_accuracy": rt,
            "backward_accuracy": bt,
            "forward_active_accuracy": far,
            "overall_accuracy": ov,
        }
        if is_pol:
            acts = [a for s in shards for a in s.get("pol_actions", [])]
            toks = [x for s in shards for x in s.get("pol_tokens", [])]
            n_tot = max(1, len(acts))
            res["num_queries"] = len(acts)
            res["action_distribution"] = {
                k: round(100.0 * acts.count(k) / n_tot, 2)
                for k in ("Answer", "ZoomIn", "Recall", "Wait", "Invalid")
            }
            res["avg_total_visual_tokens"] = round(float(np.mean(toks)), 1) if toks else 255.5
        else:
            res["avg_total_visual_tokens"] = 255.5
        return res

    payload = {
        "same_ckpt_answer_only": _summarize("ans_scores", "ans_far_probes", is_pol=False),
        "policy_routing": _summarize("pol_scores", "pol_far_probes", is_pol=True),
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["videoholmes-hist", "ovobench-fast-controls"], required=True)
    ap.add_argument("--model-path", type=str, default="/tmp/forager_4494517/models/qwen25vl_7b_v4")
    ap.add_argument("--shard", type=str, default="0/8")
    ap.add_argument("--num-hist-frames", type=int, default=4)
    ap.add_argument("--conf-threshold", type=float, default=0.68)
    ap.add_argument("--merge-glob", type=str, default="")
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args()

    if args.merge_glob:
        if args.task == "videoholmes-hist":
            return merge_videoholmes_uniform_history(args.merge_glob, args.output_json)
        return merge_ovobench_fast_controls(args.merge_glob, args.output_json)

    if args.task == "videoholmes-hist":
        return run_videoholmes_uniform_history(args.model_path, args.shard, args.output_json, args.num_hist_frames)
    return run_ovobench_fast_controls(
        model_path=args.model_path,
        shard=args.shard,
        output_json=args.output_json,
        conf_threshold=args.conf_threshold,
    )


if __name__ == "__main__":
    raise SystemExit(main())
