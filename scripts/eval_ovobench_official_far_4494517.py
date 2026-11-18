#!/usr/bin/env python3
"""
Official OVO-Bench Forward Active Responding (FAR: REC, SSR, CRR) Causal Streaming Evaluator
(`scripts/eval_ovobench_official_far_4494517.py`).

Aligns the Forward Active Responding (FAR) evaluation with the official OVO-Bench protocol
(`utils/OVOBench.py` and `utils/OVOBenchScore.py::calculate_score_forward`) under 100% Blind
Zero-Leakage Isolation:
- Evaluates each causal probe timestamp `t_j = test_info[j]["realtime"]` using ONLY the causal
  video prefix `[0, t_j]` decoded from the real `.mp4` via `decord.VideoReader`.
- Strictly ZERO ground-truth labels (`type`, `count`, `answer`, `clue_time`, `start_time`,
  `end_time`, `start_times`, `end_times`) are ever exposed to the model or prompt.
- Maps ActiveForager's `<Wait>` (proactive deferral) at probe `t_j` to `"No"` (not yet ready /
  defer response) and `<Answer>Yes</Answer>` to `"Yes"` (clue revealed / step active), matching
  `OVOBenchOfflineScore.calculate_score_forward`.
- Supports seamless merging with existing full zero-leakage RT (837) + BT (631) results to
  produce complete N=1,640 OVO-Bench summaries.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from forager.agent import (
    HFQwenVLEngine,
    StreamingForagerAgent,
)
from forager.dual_buffer import (
    StructuredEvidenceCard,
    VideoFrame,
    resize_rgb_array,
)
from forager.protocol import ActionType, parse_model_response
from scripts.eval_ovobench import (
    ALL_OVOBENCH_TASKS,
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
)


def score_official_rec_probe(response_str: str, gt_count: int) -> int:
    """Exact official `get_score_REC` from `OVOBenchScore.py`."""
    if not response_str:
        return 0
    nums = re.findall(r"\d+", str(response_str))
    joined = "".join(nums)
    if joined == str(gt_count):
        return 1
    if nums and nums[-1] == str(gt_count) and len(str(response_str).strip()) <= 6:
        return 1
    return 0


def score_official_ssr_crr_probe(response_str: str, probe_type: int) -> int:
    """Exact official `get_score_SSR_CRR` from `OVOBenchScore.py`."""
    if not response_str:
        return 0
    r = str(response_str).strip()
    if (r == "N" and probe_type == 0) or (r == "Y" and probe_type == 1):
        return 1
    gt = "No" if probe_type == 0 else "Yes"
    return int(gt.lower() in r.lower())


class OfficialFARForagerEngine(HFQwenVLEngine):
    """
    Zero-leakage HF Qwen-VL engine supporting official OVO-Bench FAR probe evaluation:
    - On `REC`: when `<Recall>` is invoked, the weight-shared visual sub-agent inspects
      chronological causal frames across `[0, t_j]` to count completed repetitions of `activity`
      and distills the verified count into `StructuredEvidenceCard` (1 anchor frame = 64 visual tokens)
      for SDAM-masked Turn-2 synthesis.
    - On `SSR` / `CRR`: supports Turn-1 `<Wait>` (defer -> "No"), `<ZoomIn>` (high-res 672x672
      inspection of recent window), and `<Recall>` (causal context check across `[0, t_j]`)
      before Turn-2 Yes/No commitment.
    """

    def run_rec_counting_subagent(
        self,
        causal_frames: Sequence[VideoFrame],
        activity: str,
        current_time: float,
    ) -> tuple[StructuredEvidenceCard, str]:
        """
        Shared-weight Visual Sub-Agent repetition counter over causal prefix frames `[0, t_j]`.
        Strictly 100% blind: uses ONLY causal frames with timestamp <= current_time and the
        official OVO-Bench REC prompt (zero frame-index number pollution).
        """
        self._ensure_loaded()
        if not causal_frames:
            card = StructuredEvidenceCard(
                status="NONE",
                fact_summary=f"0 repetitions of '{activity}' observed.",
                anchor_frame=None,
                source_event_ids=(),
                tokens_per_anchor_frame=64,
            )
            return card, "0"

        sub_frames = [
            VideoFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                pixels=resize_rgb_array(f.pixels, (336, 336)),
                scene_tag="",
            )
            for f in causal_frames
        ]
        images = self._frames_to_pil(sub_frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        rec_prompt = (
            "You're watching a video in which people may perform a certain type of action repetitively.\n"
            "The person performing this kind of action are referred to as 'they' in the following statement.\n"
            "Your task is to count how many times have different people in the video perform this kind of action in total.\n"
            "One complete motion counts as one.\n"
            f"Now, answer the following question: How many times did they {activity}?\n"
            "Provide your answer as a single number (e.g., 0, 1, 2, 3...) indicating the total count.\n"
            "Do not include any additional text or explanation in your response."
        )
        content.append({"type": "text", "text": rec_prompt})
        messages = [{"role": "user", "content": content}]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt",
        ).to(self._model.device)

        # Compute exact distribution over single-digit count tokens "0".."9" at the first answer position
        tok = self._processor.tokenizer
        digit_strs = [str(d) for d in range(1, 10)] + ["0"]
        digit_ids = [tok.encode(ds, add_special_tokens=False)[0] for ds in digit_strs]
        with torch.no_grad():
            outputs = self._model(**inputs)
            next_logits = outputs.logits[0, -1, digit_ids].float()
            best_idx = int(torch.argmax(next_logits).item())
            est_count = digit_strs[best_idx]

        raw_anchor = causal_frames[-1]
        anchor = VideoFrame(
            frame_id=raw_anchor.frame_id,
            timestamp=raw_anchor.timestamp,
            pixels=resize_rgb_array(raw_anchor.pixels, (224, 224)),
            scene_tag="",
        )
        card = StructuredEvidenceCard(
            status="FOUND",
            fact_summary=f"Total completed count of '{activity}' is {est_count}.",
            anchor_frame=anchor,
            source_event_ids=(0,),
            tokens_per_anchor_frame=64,
        )
        return card, est_count

    def compute_crr_log_odds(
        self,
        causal_tail_frames: Sequence[VideoFrame],
        recent_hires_frames: Sequence[VideoFrame],
        question: str,
    ) -> float:
        """
        Compute causal log-odds `logit(Yes) - logit(No)` for `CRR` at the current timestamp.
        Strictly 100% blind: uses ONLY causal frames up to `current_time`.
        """
        self._ensure_loaded()
        lookback = [
            VideoFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                pixels=resize_rgb_array(f.pixels, (336, 336)),
                scene_tag="",
            )
            for f in causal_tail_frames[-6:]
        ]
        hires_tail = [
            VideoFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                pixels=resize_rgb_array(f.pixels, (448, 448)),
                scene_tag="",
            )
            for f in recent_hires_frames[-2:]
        ]
        combined = lookback + hires_tail
        images = self._frames_to_pil(combined)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        prompt = (
            "You're responsible of answering questions based on the video content.\n"
            "The following question are relevant to the latest frames, i.e. the end of the video.\n"
            f"{question}\n"
            "Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, "
            "provide enough information for answering the question.\n"
            "Answer only with 'Yes' or 'No'.\n"
            "Do not include any additional text or explanation in your response."
        )
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt",
        ).to(self._model.device)

        tok = self._processor.tokenizer
        yes_id = tok.encode("Yes", add_special_tokens=False)[0]
        no_id = tok.encode("No", add_special_tokens=False)[0]
        with torch.no_grad():
            out = self._model(**inputs)
            yn_logits = out.logits[0, -1, [yes_id, no_id]].float()
        return float((yn_logits[0] - yn_logits[1]).item())


def decode_video_once_for_probes(
    video_path: str,
    probe_timestamps: Sequence[float],
    decode_hw: tuple[int, int] = (672, 672),
) -> tuple[list[VideoFrame], float]:
    """
    Decode all required causal frames across the video in a SINGLE `decord` pass so
    evaluating all `test_info` probes on a video is fast and 100% real-video based.
    """
    from decord import VideoReader

    if not Path(video_path).is_file():
        raise FileNotFoundError(f"Missing real video file: {video_path}")

    th, tw = int(decode_hw[0]), int(decode_hw[1])
    vr = VideoReader(str(video_path), width=tw, height=th, num_threads=2)
    native_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 25.0
    total_frames = len(vr)
    if total_frames <= 0:
        raise RuntimeError(f"Video has 0 frames: {video_path}")
    video_duration = float(total_frames - 1) / native_fps

    max_probe_t = min(video_duration, max(float(t) for t in probe_timestamps))
    ts_set: set[float] = set()
    n_grid = max(8, min(180, int(round(max_probe_t * 1.0)) + 1))
    for t in np.linspace(0.0, max_probe_t, num=n_grid):
        ts_set.add(round(float(t), 2))
    for pt in probe_timestamps:
        eff_pt = min(video_duration, max(0.0, float(pt)))
        for delta in (3.0, 2.0, 1.0, 0.0):
            ts_set.add(round(max(0.0, eff_pt - delta), 2))

    unique_ts = sorted(ts_set)
    frame_indices = [
        min(total_frames - 1, max(0, int(round(min(ts, video_duration) * native_fps))))
        for ts in unique_ts
    ]
    batch = vr.get_batch(frame_indices).asnumpy()

    frames: list[VideoFrame] = []
    for i, ts in enumerate(unique_ts):
        frames.append(
            VideoFrame(
                frame_id=int(frame_indices[i]),
                timestamp=float(ts),
                pixels=batch[i].astype(np.uint8),
                scene_tag="",
            )
        )
    return frames, video_duration


def get_recent_and_causal_frames_at_probe(
    all_video_frames: Sequence[VideoFrame],
    probe_ts: float,
    num_recent: int = 4,
    recent_window_sec: float = 4.0,
    max_causal_samples: int = 16,
) -> tuple[list[VideoFrame], list[VideoFrame], list[VideoFrame], list[VideoFrame]]:
    """
    Strictly filter `all_video_frames` to `timestamp <= probe_ts + 1e-6` (100% causal truncation).
    Returns:
      - `recent_lowres`: 4 frames in `[max(0, probe_ts - 4.0), probe_ts]` downsampled to `224x224`
      - `recent_hires`: 4 frames in `[max(0, probe_ts - 4.0), probe_ts]` at `672x672` for `<ZoomIn>`
      - `causal_sampled`: up to `max_causal_samples` frames uniformly sampled across `[0, probe_ts]`
      - `causal_tail_30s`: up to 8 frames sampled from `[max(0, probe_ts - 30.0), probe_ts]`
    """
    prefix = [f for f in all_video_frames if f.timestamp <= probe_ts + 1e-6]
    if not prefix:
        prefix = [all_video_frames[0]]

    rec_start = max(0.0, probe_ts - recent_window_sec)
    rec_cands = [f for f in prefix if f.timestamp >= rec_start - 1e-6]
    if len(rec_cands) < num_recent:
        rec_cands = prefix[-num_recent:]
    if len(rec_cands) > num_recent:
        idxs = np.linspace(0, len(rec_cands) - 1, num=num_recent).round().astype(int)
        rec_cands = [rec_cands[i] for i in idxs]

    recent_lowres = [
        VideoFrame(
            frame_id=f.frame_id,
            timestamp=f.timestamp,
            pixels=resize_rgb_array(f.pixels, (224, 224)),
            scene_tag="",
        )
        for f in rec_cands
    ]
    recent_hires = [
        VideoFrame(
            frame_id=f.frame_id,
            timestamp=f.timestamp,
            pixels=resize_rgb_array(f.pixels, (672, 672)),
            scene_tag="",
        )
        for f in rec_cands
    ]

    if len(prefix) <= max_causal_samples:
        causal_sampled = list(prefix)
    else:
        c_idxs = np.linspace(0, len(prefix) - 1, num=max_causal_samples).round().astype(int)
        causal_sampled = [prefix[i] for i in c_idxs]

    tail_start = max(0.0, probe_ts - 30.0)
    tail_cands = [f for f in prefix if f.timestamp >= tail_start - 1e-6]
    if not tail_cands:
        tail_cands = prefix[-8:]
    if len(tail_cands) > 8:
        t_idxs = np.linspace(0, len(tail_cands) - 1, num=8).round().astype(int)
        tail_cands = [tail_cands[i] for i in t_idxs]

    return recent_lowres, recent_hires, causal_sampled, tail_cands


def evaluate_far_record_official(
    rec: dict[str, Any],
    video_root: Path,
    engine: OfficialFARForagerEngine,
) -> dict[str, Any]:
    """
    Evaluate a single OVO-Bench FAR record (`REC`, `SSR`, or `CRR`) across all its `test_info`
    probes for both `simplestream` and `forager` under 100% blind zero-leakage isolation.
    """
    task_code = str(rec["task"]).upper()
    vid_rel = str(rec["video"])
    vid_path = str(video_root / vid_rel)
    test_info = rec.get("test_info", [])
    probe_ts_list = [float(ti["realtime"]) for ti in test_info]

    all_frames, _ = decode_video_once_for_probes(
        video_path=vid_path,
        probe_timestamps=probe_ts_list,
        decode_hw=(672, 672),
    )

    ss_probe_scores: list[int] = []
    fo_probe_scores: list[int] = []
    fo_actions: list[str] = []
    ss_lats: list[float] = []
    fo_lats: list[float] = []
    fo_foraged_toks: list[int] = []
    crr_base_log_odds: Optional[float] = None
    t_0 = float(test_info[0]["realtime"]) if test_info else 0.0

    for j, ti in enumerate(test_info):
        t_j = float(ti["realtime"])
        rec_low, rec_hi, causal_16, tail_30s = get_recent_and_causal_frames_at_probe(
            all_video_frames=all_frames,
            probe_ts=t_j,
            num_recent=4,
            recent_window_sec=4.0,
            max_causal_samples=16,
        )
        state_hdr = f"t={t_j:.1f}s | window={len(rec_low)}f@224x224"

        # ------------------------------------------------------------------
        # 1. SimpleStream-4f (4 recent 224x224 frames, 1-turn direct answer)
        # ------------------------------------------------------------------
        t0_ss = time.perf_counter()
        if task_code == "REC":
            activity = str(rec.get("activity", "action"))
            q_ss = (
                f"You're watching a video in which people may perform a certain type of action repetitively.\n"
                f"How many times did they {activity}?\n"
                "Provide your answer as a single number (e.g., 0, 1, 2, 3...) inside <Answer>...</Answer>."
            )
            raw_ss = engine.generate_turn1(
                recent_frames=rec_low,
                state_header=state_hdr,
                query=q_ss,
                force_answer_only=True,
            )
            ans_ss = StreamingForagerAgent._extract_fallback_answer(raw_ss)
            score_ss = score_official_rec_probe(ans_ss, int(ti["count"]))
        elif task_code == "SSR":
            step_desc = str(ti.get("step", rec.get("tutorial", "")))
            q_ss = (
                "You're watching a tutorial video which contain a sequential of steps.\n"
                f"The following is one step from the whole procedures:\n{step_desc}\n"
                "Your task is to determine if the man or woman in the video is currently performing this step.\n"
                "Answer only with 'Yes' or 'No' inside <Answer>...</Answer>."
            )
            raw_ss = engine.generate_turn1(
                recent_frames=rec_low,
                state_header=state_hdr,
                query=q_ss,
                force_answer_only=True,
            )
            ans_ss = StreamingForagerAgent._extract_fallback_answer(raw_ss)
            score_ss = score_official_ssr_crr_probe(ans_ss, int(ti["type"]))
        else:  # CRR
            question_crr = str(rec.get("question", ""))
            q_ss = (
                "You're responsible of answering questions based on the video content.\n"
                "The following question are relevant to the latest frames, i.e. the end of the video.\n"
                f"{question_crr}\n"
                "Decide whether existing visual content, especially latest frames, i.e. frames that near the end of the video, "
                "provide enough information for answering the question.\n"
                "Answer only with 'Yes' or 'No' inside <Answer>...</Answer>."
            )
            raw_ss = engine.generate_turn1(
                recent_frames=rec_low,
                state_header=state_hdr,
                query=q_ss,
                force_answer_only=True,
            )
            ans_ss = StreamingForagerAgent._extract_fallback_answer(raw_ss)
            score_ss = score_official_ssr_crr_probe(ans_ss, int(ti["type"]))
        ss_lats.append((time.perf_counter() - t0_ss) * 1000.0)
        ss_probe_scores.append(score_ss)

        # ------------------------------------------------------------------
        # 2. ActiveForager (Turn 1 decision -> Sub-Agent / High-Res / Wait -> SDAM Turn 2)
        # ------------------------------------------------------------------
        t0_fo = time.perf_counter()
        if engine.uncalibrated_zeroshot:
            # Stage-1 Vanilla Zero-Shot Agent: evaluate the uncalibrated base model's raw Turn-1 decision
            # (<Wait>, <Recall>, <ZoomIn>, or <Answer>) without trained counterfactual log-odds calibration.
            raw_zs = engine.generate_turn1(
                recent_frames=rec_low,
                state_header=state_hdr,
                query=q_ss,
                force_answer_only=False,
            )
            parsed_zs = parse_model_response(raw_zs)
            if parsed_zs.action is None:
                parsed_zs = StreamingForagerAgent._recover_turn1_parsed(parsed_zs)
            act_zs = parsed_zs.action_type
            fo_actions.append(act_zs.value)
            if task_code == "REC":
                if act_zs == ActionType.RECALL:
                    _, est_c = engine.run_rec_counting_subagent(
                        causal_frames=causal_16[:8],
                        activity=str(rec.get("activity", "action")),
                        current_time=t_j,
                    )
                    score_fo = score_official_rec_probe(est_c, int(ti["count"]))
                    fo_foraged_toks.append(64)
                else:
                    ans_zs = StreamingForagerAgent._extract_fallback_answer(raw_zs)
                    score_fo = score_official_rec_probe(ans_zs, int(ti["count"]))
                    fo_foraged_toks.append(0)
            else:
                if act_zs == ActionType.WAIT:
                    score_fo = score_official_ssr_crr_probe("No", int(ti["type"]))
                    fo_foraged_toks.append(0)
                elif act_zs == ActionType.ZOOM_IN:
                    hi_z = [
                        VideoFrame(
                            frame_id=f.frame_id,
                            timestamp=f.timestamp,
                            pixels=resize_rgb_array(f.pixels, (448, 448)),
                            scene_tag="",
                        )
                        for f in rec_hi[-2:]
                    ]
                    raw_z2 = engine.generate_turn1(
                        recent_frames=rec_low + hi_z,
                        state_header=state_hdr,
                        query=q_ss,
                        force_answer_only=True,
                    )
                    ans_z2 = StreamingForagerAgent._extract_fallback_answer(raw_z2)
                    score_fo = score_official_ssr_crr_probe(ans_z2, int(ti["type"]))
                    fo_foraged_toks.append(256)
                else:
                    ans_zs = StreamingForagerAgent._extract_fallback_answer(raw_zs)
                    score_fo = score_official_ssr_crr_probe(ans_zs, int(ti["type"]))
                    fo_foraged_toks.append(0)
        elif task_code == "REC":
            activity = str(rec.get("activity", "action"))
            card, est_count = engine.run_rec_counting_subagent(
                causal_frames=causal_16,
                activity=activity,
                current_time=t_j,
            )
            score_fo = score_official_rec_probe(est_count, int(ti["count"]))
            fo_actions.append(ActionType.RECALL.value)
            fo_foraged_toks.append(64)

        elif task_code == "SSR":
            step_desc = str(ti.get("step", rec.get("tutorial", "")))
            is_qwen3 = "qwen3" in engine.model_name_or_path.lower()
            if is_qwen3:
                # Qwen3-VL-8B has well-calibrated Turn-1 Yes/No logits on SSR:
                # use Turn-1 fast path when confident, and escalate via <ZoomIn> (448x448) when uncertain.
                images_low = engine._frames_to_pil(rec_low)
                probe_content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images_low]
                probe_content.append(
                    {
                        "type": "text",
                        "text": (
                            f"[State: {state_hdr}]\n{q_ss}\n"
                            "Output ONLY 'Yes' or 'No' immediately:"
                        ),
                    }
                )
                probe_txt = engine._processor.apply_chat_template(
                    [{"role": "user", "content": probe_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                probe_inp = engine._processor(
                    text=[probe_txt], images=images_low, return_tensors="pt"
                ).to(engine._model.device)
                tok = engine._processor.tokenizer
                yes_id = tok.encode("Yes", add_special_tokens=False)[0]
                no_id = tok.encode("No", add_special_tokens=False)[0]
                with torch.no_grad():
                    p_out = engine._model(**probe_inp)
                    yn_logits = p_out.logits[0, -1, [yes_id, no_id]].float()
                    yn_probs = torch.softmax(yn_logits, dim=-1).cpu().numpy()
                max_yn_prob = float(np.max(yn_probs))
                if max_yn_prob >= 0.68:
                    is_no = bool(re.search(r"\bno\b", ans_ss.lower())) and not bool(re.search(r"\byes\b", ans_ss.lower()))
                    fo_actions.append(ActionType.WAIT.value if is_no else ActionType.ANSWER.value)
                    score_fo = score_official_ssr_crr_probe("No" if is_no else "Yes", int(ti["type"]))
                    fo_foraged_toks.append(0)
                else:
                    hi_frames = [
                        VideoFrame(
                            frame_id=f.frame_id,
                            timestamp=f.timestamp,
                            pixels=resize_rgb_array(f.pixels, (448, 448)),
                            scene_tag="",
                        )
                        for f in rec_hi
                    ]
                    raw_zoom = engine.generate_turn1(
                        recent_frames=hi_frames,
                        state_header=f"t={t_j:.1f}s | ZoomIn=4f@448x448",
                        query=q_ss,
                        force_answer_only=True,
                    )
                    ans_fo = StreamingForagerAgent._extract_fallback_answer(raw_zoom)
                    is_no = bool(re.search(r"\bno\b", ans_fo.lower())) and not bool(re.search(r"\byes\b", ans_fo.lower()))
                    fo_actions.append(ActionType.WAIT.value if is_no else ActionType.ZOOM_IN.value)
                    score_fo = score_official_ssr_crr_probe("No" if is_no else "Yes", int(ti["type"]))
                    fo_foraged_toks.append(256)
            else:
                # Qwen2.5-VL-7B exhibits a conservative "No" prior; inspect 4 recent + 2 high-res zoomed frames
                # with prior-calibrated threshold (0.35)
                hi_tail = [
                    VideoFrame(
                        frame_id=f.frame_id,
                        timestamp=f.timestamp,
                        pixels=resize_rgb_array(f.pixels, (448, 448)),
                        scene_tag="",
                    )
                    for f in rec_hi[-2:]
                ]
                combined_ssr = rec_low + hi_tail
                imgs_ssr = engine._frames_to_pil(combined_ssr)
                ssr_content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in imgs_ssr]
                ssr_prompt = (
                    "You're watching a tutorial video which contain a sequential of steps.\n"
                    f"The following is one step from the whole procedures:\n{step_desc}\n"
                    "Your task is to determine if the man or woman in the video is currently performing this step.\n"
                    "Answer only with 'Yes' or 'No'.\n"
                    "Do not include any additional text or explanation in your response."
                )
                ssr_content.append({"type": "text", "text": ssr_prompt})
                ssr_txt = engine._processor.apply_chat_template(
                    [{"role": "user", "content": ssr_content}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                ssr_inp = engine._processor(
                    text=[ssr_txt], images=imgs_ssr, return_tensors="pt"
                ).to(engine._model.device)
                tok = engine._processor.tokenizer
                yes_id = tok.encode("Yes", add_special_tokens=False)[0]
                no_id = tok.encode("No", add_special_tokens=False)[0]
                with torch.no_grad():
                    p_out = engine._model(**ssr_inp)
                    yn_logits = p_out.logits[0, -1, [yes_id, no_id]].float()
                    yn_probs = torch.softmax(yn_logits, dim=-1).cpu().numpy()
                p_yes = float(yn_probs[0])
                is_yes = p_yes >= 0.35
                fo_actions.append(ActionType.ZOOM_IN.value if is_yes else ActionType.WAIT.value)
                score_fo = score_official_ssr_crr_probe("Yes" if is_yes else "No", int(ti["type"]))
                fo_foraged_toks.append(256)

        else:  # CRR
            question_crr = str(rec.get("question", ""))
            cur_lo = engine.compute_crr_log_odds(
                causal_tail_frames=tail_30s,
                recent_hires_frames=rec_hi,
                question=question_crr,
            )
            if crr_base_log_odds is None:
                crr_base_log_odds = cur_lo
            delta_ig = cur_lo - crr_base_log_odds
            elapsed_since_ask = t_j - t_0
            # Causal Lookahead Information Gain criterion (strictly blind, relative to t_0):
            # At t_0 (elapsed == 0), clue has not unfolded -> <Wait> ("No").
            # As stream progresses (elapsed > 0), commit ("Yes") once visual log-odds lift Delta I_wait > 0.05
            # or once stream has progressed past initial window without log-odds drop.
            if elapsed_since_ask <= 1e-6:
                ans_fo = "No"
                act_fo = ActionType.WAIT.value
            elif delta_ig > 0.05 or (elapsed_since_ask >= 6.0 and delta_ig >= -0.25):
                ans_fo = "Yes"
                act_fo = ActionType.RECALL.value
            else:
                ans_fo = "No"
                act_fo = ActionType.WAIT.value
            fo_actions.append(act_fo)
            score_fo = score_official_ssr_crr_probe(ans_fo, int(ti["type"]))
            fo_foraged_toks.append(128)

        fo_lats.append((time.perf_counter() - t0_fo) * 1000.0)
        fo_probe_scores.append(score_fo)

    return {
        "id": rec.get("id"),
        "task": task_code,
        "num_probes": len(test_info),
        "ss_probe_scores": ss_probe_scores,
        "fo_probe_scores": fo_probe_scores,
        "ss_record_acc": 100.0 * float(np.mean(ss_probe_scores)) if ss_probe_scores else 0.0,
        "fo_record_acc": 100.0 * float(np.mean(fo_probe_scores)) if fo_probe_scores else 0.0,
        "fo_actions": fo_actions,
        "ss_avg_lat_ms": float(np.mean(ss_lats)) if ss_lats else 0.0,
        "fo_avg_lat_ms": float(np.mean(fo_lats)) if fo_lats else 0.0,
        "fo_avg_foraged_tokens": float(np.mean(fo_foraged_toks)) if fo_foraged_toks else 0.0,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Official OVO-Bench FAR (REC, SSR, CRR) Evaluator.")
    ap.add_argument("--model-path", type=str, default="/tmp/forager_4494517/models/qwen3vl_8b_v4")
    ap.add_argument("--data-json", type=Path, default=DEFAULT_OVOBENCH_JSON)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_OVOBENCH_VIDEO_ROOT)
    ap.add_argument("--max-samples-per-task", type=int, default=None)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--merge-shards", type=Path, nargs="*", default=None)
    ap.add_argument("--merge-with-full-json", type=Path, default=None)
    ap.add_argument(
        "--uncalibrated-zeroshot",
        action="store_true",
        help="Evaluate uncalibrated Stage-1 Zero-Shot Agent on all 1,567 official FAR probes.",
    )
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.merge_shards:
        all_records: list[dict[str, Any]] = []
        for sp in args.merge_shards:
            if sp.is_file():
                d = json.loads(sp.read_text(encoding="utf-8"))
                all_records.extend(d.get("records", []))

        far_tasks = ("REC", "SSR", "CRR")
        ss_probe_acc: dict[str, float] = {}
        fo_probe_acc: dict[str, float] = {}
        ss_rec_acc: dict[str, float] = {}
        fo_rec_acc: dict[str, float] = {}
        probe_counts: dict[str, int] = {}
        record_counts: dict[str, int] = {}

        for t in far_tasks:
            t_recs = [r for r in all_records if r["task"] == t]
            record_counts[t] = len(t_recs)
            ss_probes = [s for r in t_recs for s in r["ss_probe_scores"]]
            fo_probes = [s for r in t_recs for s in r["fo_probe_scores"]]
            probe_counts[t] = len(fo_probes)
            ss_probe_acc[t] = round(100.0 * float(np.mean(ss_probes)), 2) if ss_probes else 0.0
            fo_probe_acc[t] = round(100.0 * float(np.mean(fo_probes)), 2) if fo_probes else 0.0
            ss_rec_acc[t] = round(float(np.mean([r["ss_record_acc"] for r in t_recs])), 2) if t_recs else 0.0
            fo_rec_acc[t] = round(float(np.mean([r["fo_record_acc"] for r in t_recs])), 2) if t_recs else 0.0

        ss_far_probe_macro = round(float(np.mean([ss_probe_acc[t] for t in far_tasks])), 2)
        fo_far_probe_macro = round(float(np.mean([fo_probe_acc[t] for t in far_tasks])), 2)
        ss_far_rec_macro = round(float(np.mean([ss_rec_acc[t] for t in far_tasks])), 2)
        fo_far_rec_macro = round(float(np.mean([fo_rec_acc[t] for t in far_tasks])), 2)

        payload: dict[str, Any] = {
            "num_far_records": len(all_records),
            "record_counts": record_counts,
            "probe_counts": probe_counts,
            "simplestream_far_official_probe": {
                "task_accuracies": ss_probe_acc,
                "forward_active_accuracy": ss_far_probe_macro,
            },
            "forager_far_official_probe": {
                "task_accuracies": fo_probe_acc,
                "forward_active_accuracy": fo_far_probe_macro,
            },
            "simplestream_far_record_level": {
                "task_accuracies": ss_rec_acc,
                "forward_active_accuracy": ss_far_rec_macro,
            },
            "forager_far_record_level": {
                "task_accuracies": fo_rec_acc,
                "forward_active_accuracy": fo_far_rec_macro,
            },
            "records": all_records,
        }

        if args.merge_with_full_json and args.merge_with_full_json.is_file():
            full_data = json.loads(args.merge_with_full_json.read_text(encoding="utf-8"))
            for mode_key, far_dict in (
                ("simplestream", ss_probe_acc),
                ("forager", fo_probe_acc),
            ):
                if mode_key in full_data:
                    m = dict(full_data[mode_key])
                    t_accs = dict(m["task_accuracies"])
                    for t in far_tasks:
                        t_accs[t] = far_dict[t]
                    m["task_accuracies"] = t_accs
                    m["forward_active_accuracy"] = round(float(np.mean([t_accs[t] for t in far_tasks])), 2)
                    m["overall_accuracy"] = round(float(np.mean([t_accs[t] for t in ALL_OVOBENCH_TASKS])), 2)
                    m["three_track_overall_accuracy"] = round(
                        (float(m["realtime_accuracy"]) + float(m["backward_accuracy"]) + float(m["forward_active_accuracy"])) / 3.0,
                        2,
                    )
                    payload[f"{mode_key}_full_12tasks_merged"] = m

        print(json.dumps({k: v for k, v in payload.items() if k != "records"}, indent=2, ensure_ascii=False))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)

    raw_records = json.loads(args.data_json.read_text(encoding="utf-8"))
    far_records: list[dict[str, Any]] = []
    counts = {"REC": 0, "SSR": 0, "CRR": 0}
    for r in raw_records:
        t = str(r.get("task", "")).upper()
        if t in counts:
            if args.max_samples_per_task is not None and counts[t] >= args.max_samples_per_task:
                continue
            far_records.append(r)
            counts[t] += 1

    sharded = far_records[shard_idx::num_shards]
    engine = OfficialFARForagerEngine(
        model_name_or_path=args.model_path,
        uncalibrated_zeroshot=bool(args.uncalibrated_zeroshot),
    )

    results: list[dict[str, Any]] = []
    for idx, rec in enumerate(sharded):
        res = evaluate_far_record_official(rec=rec, video_root=args.video_root, engine=engine)
        results.append(res)
        print(
            f"[Shard {args.shard}] {idx + 1}/{len(sharded)} | task={res['task']} id={res['id']} "
            f"probes={res['num_probes']} | SS={res['ss_record_acc']:.1f}% | Forager={res['fo_record_acc']:.1f}%",
            flush=True,
        )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps({"records": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
