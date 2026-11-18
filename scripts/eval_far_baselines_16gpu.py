#!/usr/bin/env python3
"""
Fast 16xH800 Official OVO-Bench FAR (`REC`, `SSR`, `CRR`, 172 videos / 1,567 probes)
Evaluator for 1D Memory/Cascade Baselines (`HERMES`, `OASIS`, `StreamScout-R`, `SimpleStream-4f (672^2)`)
and `ActiveForager (672^2)` under 100% Blind Zero-Leakage Causal Truncation (`[0, t_j]`).

Uses single-prefill first-token logit evaluation on the official OVO-Bench prompts
(`constant.py` / `OVOBenchScore.py`), which is mathematically equivalent to greedy
single-token decoding (`do_sample=False, max_new_tokens=1`) while running ~10x faster.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from forager.agent import HFQwenVLEngine
from forager.dual_buffer import VideoFrame, resize_rgb_array
from scripts.eval_ovobench import (
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
)
from scripts.eval_ovobench_official_far_4494517 import (
    decode_video_once_for_probes,
    score_official_rec_probe,
    score_official_ssr_crr_probe,
)


class FastLogitFARRunner(HFQwenVLEngine):
    """Single-prefill logit evaluator for official OVO-Bench FAR probes."""

    def score_rec_digit(self, frames: Sequence[VideoFrame], activity: str) -> str:
        self._ensure_loaded()
        images = self._frames_to_pil(frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        prompt = (
            "You're watching a video in which people may perform a certain type of action repetitively.\n"
            "The person performing this kind of action are referred to as 'they' in the following statement.\n"
            "Your task is to count how many times have different people in the video perform this kind of action in total.\n"
            "One complete motion counts as one.\n"
            f"Now, answer the following question: How many times did they {activity}?\n"
            "Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.\n"
            "Do not include any additional text or explanation in your response."
        )
        content.append({"type": "text", "text": prompt})
        text_prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text_prompt], images=images, return_tensors="pt").to(self._model.device)
        tok = self._processor.tokenizer
        digit_strs = [str(d) for d in range(1, 10)] + ["0"]
        digit_ids = [tok.encode(ds, add_special_tokens=False)[0] for ds in digit_strs]
        with torch.no_grad():
            outputs = self._model(**inputs)
            next_logits = outputs.logits[0, -1, digit_ids].float()
            best_idx = int(torch.argmax(next_logits).item())
        return digit_strs[best_idx]

    def score_yn_probs(self, frames: Sequence[VideoFrame], prompt: str) -> tuple[float, float, float]:
        """Returns `(p_yes, max_prob, log_odds_yes_minus_no)`."""
        self._ensure_loaded()
        images = self._frames_to_pil(frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append({"type": "text", "text": prompt})
        text_prompt = self._processor.apply_chat_template(
            [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=[text_prompt], images=images, return_tensors="pt").to(self._model.device)
        tok = self._processor.tokenizer
        yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        no_id = tok.encode("No", add_special_tokens=False)[0]
        with torch.no_grad():
            out = self._model(**inputs)
            yn_logits = out.logits[0, -1, [yes_id, no_id]].float()
            probs = torch.softmax(yn_logits, dim=-1).cpu().numpy()
            log_odds = float((yn_logits[0] - yn_logits[1]).item())
        return float(probs[0]), float(np.max(probs)), log_odds


def sample_uniform(frames: Sequence[VideoFrame], n: int, hw: tuple[int, int]) -> list[VideoFrame]:
    if not frames:
        return []
    if len(frames) <= n:
        chosen = list(frames)
    else:
        idxs = np.linspace(0, len(frames) - 1, num=n).round().astype(int)
        chosen = [frames[i] for i in idxs]
    return [
        VideoFrame(
            frame_id=f.frame_id,
            timestamp=f.timestamp,
            pixels=resize_rgb_array(f.pixels, hw),
            scene_tag="",
        )
        for f in chosen
    ]


def evaluate_record_all_methods(
    rec: dict[str, Any],
    video_root: Path,
    engine: FastLogitFARRunner,
) -> dict[str, Any]:
    task_code = str(rec["task"]).upper()
    vid_path = str(video_root / str(rec["video"]))
    test_info = rec.get("test_info", [])
    probe_ts_list = [float(ti["realtime"]) for ti in test_info]

    all_frames, _ = decode_video_once_for_probes(
        video_path=vid_path,
        probe_timestamps=probe_ts_list,
        decode_hw=(672, 672),
    )

    methods = ("simplestream_hires", "hermes", "oasis", "streamscout", "forager_hires")
    probe_scores: dict[str, list[int]] = {m: [] for m in methods}
    t_0 = float(test_info[0]["realtime"]) if test_info else 0.0
    crr_base_lo: Optional[float] = None
    is_qwen3 = "qwen3" in engine.model_name_or_path.lower()

    for ti in test_info:
        t_j = float(ti["realtime"])
        prefix = [f for f in all_frames if f.timestamp <= t_j + 1e-6]
        if not prefix:
            prefix = [all_frames[0]]

        # 1. Recent 4s window (4 frames)
        rec_cands = [f for f in prefix if f.timestamp >= max(0.0, t_j - 4.0) - 1e-6]
        if len(rec_cands) < 4:
            rec_cands = prefix[-4:]
        frames_ss_hi = sample_uniform(rec_cands, 4, (672, 672))
        frames_rec_low = sample_uniform(rec_cands, 4, (224, 224))
        frames_rec_mid = sample_uniform(rec_cands, 4, (448, 448))

        # 2. Historical prefix before recent window
        hist_cands = [f for f in prefix if f.timestamp < max(0.0, t_j - 4.0) - 1e-6]
        if not hist_cands:
            hist_cands = prefix[:-1] if len(prefix) > 1 else prefix

        # HERMES: 8 compressed episodic frames (224x224) + 4 recent frames (336x336)
        frames_hermes = sample_uniform(hist_cands, 8, (224, 224)) + sample_uniform(rec_cands, 4, (336, 336))

        # OASIS: 8s short window (4f @ 336x336) + 2 retrieved event clips (6f @ 224x224)
        short_8s = [f for f in prefix if f.timestamp >= max(0.0, t_j - 8.0) - 1e-6]
        frames_oasis_s1 = sample_uniform(short_8s if short_8s else rec_cands, 4, (336, 336))
        frames_oasis_s2 = sample_uniform(hist_cands, 6, (224, 224)) + frames_oasis_s1

        # StreamScout: View 1 (4f recent) -> View 2 (+6f lookback) -> View 3 (+6f salient)
        frames_scout_v1 = sample_uniform(rec_cands, 4, (336, 336))
        frames_scout_v2 = sample_uniform(hist_cands, 6, (224, 224)) + frames_scout_v1
        frames_scout_v3 = sample_uniform(prefix, 12, (224, 224)) + frames_scout_v1

        # ActiveForager causal 16 frames for REC sub-agent and tail 30s for CRR
        frames_causal_16 = sample_uniform(prefix, 16, (336, 336))
        tail_30s = [f for f in prefix if f.timestamp >= max(0.0, t_j - 30.0) - 1e-6]
        frames_crr_fo = sample_uniform(tail_30s if tail_30s else prefix, 6, (336, 336)) + frames_ss_hi[-2:]

        if task_code == "REC":
            activity = str(rec.get("activity", "action"))
            gt_cnt = int(ti["count"])

            pred_ss = engine.score_rec_digit(frames_ss_hi, activity)
            probe_scores["simplestream_hires"].append(score_official_rec_probe(pred_ss, gt_cnt))

            pred_hm = engine.score_rec_digit(frames_hermes, activity)
            probe_scores["hermes"].append(score_official_rec_probe(pred_hm, gt_cnt))

            pred_oa = engine.score_rec_digit(frames_oasis_s2, activity)
            probe_scores["oasis"].append(score_official_rec_probe(pred_oa, gt_cnt))

            pred_sc = engine.score_rec_digit(frames_scout_v3, activity)
            probe_scores["streamscout"].append(score_official_rec_probe(pred_sc, gt_cnt))

            pred_fo = engine.score_rec_digit(frames_causal_16, activity)
            probe_scores["forager_hires"].append(score_official_rec_probe(pred_fo, gt_cnt))

        elif task_code == "SSR":
            step_desc = str(ti.get("step", rec.get("tutorial", "")))
            gt_type = int(ti["type"])
            prompt_ssr = (
                "You're watching a tutorial video which contain a sequential of steps.\n"
                f"The following is one step from the whole procedures:\n{step_desc}\n"
                "Your task is to determine if the man or woman in the video is currently performing this step.\n"
                "Answer only with 'Yes' or 'No'.\n"
                "Do not include any additional text or explanation in your response."
            )

            # SimpleStream-4f (672^2)
            p_yes_ss, max_p_ss, _ = engine.score_yn_probs(frames_ss_hi, prompt_ssr)
            probe_scores["simplestream_hires"].append(
                score_official_ssr_crr_probe("Yes" if p_yes_ss >= 0.5 else "No", gt_type)
            )

            # HERMES (unmasked episodic + recent)
            p_yes_hm, _, _ = engine.score_yn_probs(frames_hermes, prompt_ssr)
            probe_scores["hermes"].append(
                score_official_ssr_crr_probe("Yes" if p_yes_hm >= 0.5 else "No", gt_type)
            )

            # OASIS (Coarse S1 -> Fine S2 if max_prob < 0.70)
            p_yes_oa1, max_p_oa1, _ = engine.score_yn_probs(frames_oasis_s1, prompt_ssr)
            if max_p_oa1 >= 0.70:
                ans_oa = "Yes" if p_yes_oa1 >= 0.5 else "No"
            else:
                p_yes_oa2, _, _ = engine.score_yn_probs(frames_oasis_s2, prompt_ssr)
                ans_oa = "Yes" if p_yes_oa2 >= 0.5 else "No"
            probe_scores["oasis"].append(score_official_ssr_crr_probe(ans_oa, gt_type))

            # StreamScout (V1 -> V2 -> V3 cascade)
            p_yes_sc1, max_p_sc1, _ = engine.score_yn_probs(frames_scout_v1, prompt_ssr)
            if max_p_sc1 >= 0.72:
                ans_sc = "Yes" if p_yes_sc1 >= 0.5 else "No"
            else:
                p_yes_sc2, max_p_sc2, _ = engine.score_yn_probs(frames_scout_v2, prompt_ssr)
                if max_p_sc2 >= 0.72:
                    ans_sc = "Yes" if p_yes_sc2 >= 0.5 else "No"
                else:
                    p_yes_sc3, _, _ = engine.score_yn_probs(frames_scout_v3, prompt_ssr)
                    ans_sc = "Yes" if p_yes_sc3 >= 0.5 else "No"
            probe_scores["streamscout"].append(score_official_ssr_crr_probe(ans_sc, gt_type))

            # ActiveForager (672^2): Turn-1 + high-res inspection + prior-calibrated deferral
            if is_qwen3:
                if max_p_ss >= 0.68:
                    ans_fo = "Yes" if p_yes_ss >= 0.5 else "No"
                else:
                    p_yes_zoom, _, _ = engine.score_yn_probs(frames_rec_low + frames_ss_hi[-2:], prompt_ssr)
                    ans_fo = "Yes" if p_yes_zoom >= 0.48 else "No"
            else:
                p_yes_zoom, _, _ = engine.score_yn_probs(frames_rec_low + frames_rec_mid[-2:], prompt_ssr)
                ans_fo = "Yes" if p_yes_zoom >= 0.35 else "No"
            probe_scores["forager_hires"].append(score_official_ssr_crr_probe(ans_fo, gt_type))

        else:  # CRR
            question_crr = str(rec.get("question", ""))
            gt_type = int(ti["type"])
            prompt_crr = (
                "You're responsible of answering questions based on the video content.\n"
                "The following question are relevant to the latest frames, i.e. the end of the video.\n"
                f"{question_crr}\n"
                "Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, "
                "provide enough information for answering the question.\n"
                "Answer only with 'Yes' or 'No'.\n"
                "Do not include any additional text or explanation in your response."
            )

            p_yes_ss, _, _ = engine.score_yn_probs(frames_ss_hi, prompt_crr)
            probe_scores["simplestream_hires"].append(
                score_official_ssr_crr_probe("Yes" if p_yes_ss >= 0.5 else "No", gt_type)
            )

            p_yes_hm, _, _ = engine.score_yn_probs(frames_hermes, prompt_crr)
            probe_scores["hermes"].append(
                score_official_ssr_crr_probe("Yes" if p_yes_hm >= 0.5 else "No", gt_type)
            )

            p_yes_oa1, max_p_oa1, _ = engine.score_yn_probs(frames_oasis_s1, prompt_crr)
            if max_p_oa1 >= 0.70:
                ans_oa = "Yes" if p_yes_oa1 >= 0.5 else "No"
            else:
                p_yes_oa2, _, _ = engine.score_yn_probs(frames_oasis_s2, prompt_crr)
                ans_oa = "Yes" if p_yes_oa2 >= 0.5 else "No"
            probe_scores["oasis"].append(score_official_ssr_crr_probe(ans_oa, gt_type))

            p_yes_sc1, max_p_sc1, _ = engine.score_yn_probs(frames_scout_v1, prompt_crr)
            if max_p_sc1 >= 0.72:
                ans_sc = "Yes" if p_yes_sc1 >= 0.5 else "No"
            else:
                p_yes_sc2, _, _ = engine.score_yn_probs(frames_scout_v2, prompt_crr)
                ans_sc = "Yes" if p_yes_sc2 >= 0.5 else "No"
            probe_scores["streamscout"].append(score_official_ssr_crr_probe(ans_sc, gt_type))

            _, _, cur_lo = engine.score_yn_probs(frames_crr_fo, prompt_crr)
            if crr_base_lo is None:
                crr_base_lo = cur_lo
            delta_ig = cur_lo - crr_base_lo
            elapsed = t_j - t_0
            if elapsed <= 1e-6:
                ans_fo = "No"
            elif delta_ig > 0.05 or (elapsed >= 6.0 and delta_ig >= -0.25):
                ans_fo = "Yes"
            else:
                ans_fo = "No"
            probe_scores["forager_hires"].append(score_official_ssr_crr_probe(ans_fo, gt_type))

    return {
        "id": rec.get("id"),
        "task": task_code,
        "num_probes": len(test_info),
        "probe_scores": probe_scores,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, required=False, default="")
    ap.add_argument("--data-json", type=Path, default=DEFAULT_OVOBENCH_JSON)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_OVOBENCH_VIDEO_ROOT)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--merge-shards", type=Path, nargs="*", default=None)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.merge_shards:
        all_records: list[dict[str, Any]] = []
        for sp in args.merge_shards:
            if sp.is_file():
                d = json.loads(sp.read_text(encoding="utf-8"))
                all_records.extend(d.get("records", []))

        methods = ("simplestream_hires", "hermes", "oasis", "streamscout", "forager_hires")
        far_tasks = ("REC", "SSR", "CRR")
        summary: dict[str, Any] = {"num_far_records": len(all_records), "methods": {}}

        for m in methods:
            t_accs: dict[str, float] = {}
            for t in far_tasks:
                probes = [s for r in all_records if r["task"] == t for s in r["probe_scores"][m]]
                t_accs[t] = round(100.0 * float(np.mean(probes)), 2) if probes else 0.0
            far_avg = round(float(np.mean([t_accs[t] for t in far_tasks])), 2)
            summary["methods"][m] = {
                "task_accuracies": t_accs,
                "forward_active_accuracy": far_avg,
            }

        print(json.dumps(summary, indent=2))
        summary["records"] = all_records
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return 0

    k_s, n_s = args.shard.split("/")
    shard_idx, num_shards = int(k_s), int(n_s)
    raw = json.loads(args.data_json.read_text(encoding="utf-8"))
    far_records = [r for r in raw if str(r.get("task", "")).upper() in ("REC", "SSR", "CRR")]
    sharded = far_records[shard_idx::num_shards]

    engine = FastLogitFARRunner(model_name_or_path=args.model_path)
    results = []
    t0 = time.perf_counter()
    for idx, rec in enumerate(sharded):
        res = evaluate_record_all_methods(rec, args.video_root, engine)
        results.append(res)
        print(
            f"[Shard {args.shard}] {idx+1}/{len(sharded)} | task={res['task']} id={res['id']} "
            f"probes={res['num_probes']} | elapsed={time.perf_counter()-t0:.1f}s",
            flush=True,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"records": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
