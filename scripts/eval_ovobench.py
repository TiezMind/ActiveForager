#!/usr/bin/env python3
"""
OVO-Bench 12-Task Causal Streaming Evaluation Harness (`scripts/eval_ovobench.py`).

Supports all 3 OVO-Bench capability tracks (12 subtasks):
1. Real-Time Visual Perception (RT, 6 tasks):
   - OCR (Optical Character Recognition)
   - ACR (Action Recognition)
   - ATR (Attribute Recognition)
   - STU (Spatial Understanding)
   - OJR (Object Recognition)
   - FPD (Future Prediction)
2. Backward Tracing (BT, 3 tasks):
   - EPM (Episodic Memory)
   - ASI (Action Sequence Identification)
   - HLD (Hallucination Detection)
3. Forward Active Responding (FAR, 3 tasks):
   - REC (Repetition Event Count)
   - SSR (Sequential Steps Recognition)
   - CRR (Clues Reveal Responding)

Enforces 100% Blind Evaluation Isolation:
- Zero ground-truth answer, `clue_unfolding`, `clue_revealed`, or `historical event:` injection
  in any frame `scene_tag`, state header, prompt, or runtime metadata passed to the agent.
- Full-range real video decoding via `decord.VideoReader` across `[0, cutoff]` (without any
  `cutoff <= 90.0s` cap):
  - Recent 4 seconds sampled into `ProtectedRecentWindow` (224x224) and `HighResRingBuffer` (896x896).
  - Historical interval `[0, max(0, cutoff - 4.0)]` uniformly sampled into `OnlineEventArchive`.
  - Raises `FileNotFoundError` / `RuntimeError` if a real video file is missing or corrupt during
    real evaluation (never silently falls back to synthetic frames).
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np

from forager.agent import (
    AgentEpisodeResult,
    BaseVLMEngine,
    HFQwenVLEngine,
    MockVLMEngine,
    StreamingForagerAgent,
    VLLMEngine,
)
from forager.dual_buffer import (
    CausalStreamingDualBuffer,
    HighResRingBuffer,
    ProtectedRecentWindow,
    VideoFrame,
)
from forager.ig_grpo import compute_task_accuracy_reward
from forager.protocol import ActionType


DEFAULT_OVOBENCH_JSON = Path("/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json")
DEFAULT_OVOBENCH_VIDEO_ROOT = Path("/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/src_videos")

REALTIME_TASKS: tuple[str, ...] = ("OCR", "ACR", "ATR", "STU", "OJR", "FPD")
BACKWARD_TASKS: tuple[str, ...] = ("EPM", "ASI", "HLD")
FORWARD_ACTIVE_TASKS: tuple[str, ...] = ("REC", "SSR", "CRR")
ALL_OVOBENCH_TASKS: tuple[str, ...] = REALTIME_TASKS + BACKWARD_TASKS + FORWARD_ACTIVE_TASKS

TASK_TO_CATEGORY: dict[str, str] = {
    **{t: "Real-Time Perception" for t in REALTIME_TASKS},
    **{t: "Backward Tracing" for t in BACKWARD_TASKS},
    **{t: "Forward Active Responding" for t in FORWARD_ACTIVE_TASKS},
}


@dataclass
class OVOBenchSample:
    """
    Single OVO-Bench streaming evaluation instance.

    Note on Physical Blind Isolation:
    `gt_answer` and `target_reveal_time` are held on the outer evaluation sample
     strictly for post-episode scoring (`compute_task_accuracy_reward`) and are NEVER
    included in `blind_agent_metadata()` or passed into `agent.answer_query` during
    real evaluation.
    """

    sample_id: str
    task_code: str  # One of ALL_OVOBENCH_TASKS
    query: str
    gt_answer: str
    query_timestamp: float
    options: tuple[str, ...] = ()
    task_type: Literal["mcq", "open", "proactive"] = "mcq"
    video_path: Optional[str] = None
    target_reveal_time: Optional[float] = None
    frames: list[VideoFrame] = field(default_factory=list)
    lazy_decode_kwargs: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _oracle_scoring_only: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if self.task_code not in ALL_OVOBENCH_TASKS:
            raise ValueError(f"Unknown OVO-Bench task_code: {self.task_code}")

    def blind_agent_metadata(self, include_mock_hints: bool = False) -> dict[str, Any]:
        """
        Return strictly blind metadata for `agent.answer_query` (zero ground-truth fields).
        Only when `include_mock_hints=True` (for CPU unit test `MockVLMEngine`) are synthetic
        test hints merged.
        """
        forbidden = {"gt_answer", "answer", "answer_text", "target_reveal_time", "distractor_answer", "none_found_answer"}
        clean = {k: v for k, v in self.metadata.items() if k not in forbidden}
        if include_mock_hints and self._oracle_scoring_only:
            return {**clean, **self._oracle_scoring_only}
        return clean

    def materialize_frames(self, allow_synthetic_fallback: bool = False) -> list[VideoFrame]:
        if self.frames:
            return self.frames
        if self.lazy_decode_kwargs is not None:
            kw = dict(self.lazy_decode_kwargs)
            kw["allow_synthetic_fallback"] = allow_synthetic_fallback
            return extract_causal_video_frames(**kw)
        if allow_synthetic_fallback:
            return _make_synthetic_stream_frames(total_seconds=12, fps=1.0)
        raise FileNotFoundError(
            f"Sample {self.sample_id} has no pre-loaded frames and no lazy_decode_kwargs; "
            "synthetic frame fallback is disabled in real evaluation."
        )


@dataclass
class OVOBenchEvalSummary:
    """Aggregated OVO-Bench evaluation metrics."""

    mode: str
    num_samples: int
    task_accuracies: dict[str, float]
    realtime_accuracy: float
    backward_accuracy: float
    episodic_recall_accuracy: float  # ER = (EPM + ASI) / 2 per SimpleStream Eq. (2)
    forward_active_accuracy: float
    overall_accuracy: float
    action_distribution: dict[str, float]
    avg_recent_visual_tokens: float
    avg_foraged_visual_tokens: float
    avg_total_visual_tokens: float
    avg_latency_ms: float
    delta_p_vs_simplestream: Optional[float] = None
    delta_m_vs_simplestream: Optional[float] = None
    delta_bt_vs_simplestream: Optional[float] = None
    task_sample_counts: Optional[dict[str, int]] = None


def _make_synthetic_stream_frames(
    total_seconds: int = 12,
    fps: float = 2.0,
    resolution: tuple[int, int] = (448, 448),
    early_tag: str = "",
    late_tag: str = "",
    end_timestamp: Optional[float] = None,
) -> list[VideoFrame]:
    """
    Generate a causal stream of synthetic frames ONLY for standalone CPU unit tests.
    """
    frames: list[VideoFrame] = []
    num_frames = max(6, int(round(total_seconds * fps)))
    h, w = resolution
    if end_timestamp is None:
        timestamps = [round(idx / fps, 2) for idx in range(num_frames)]
        mid_ts = total_seconds * 0.5
    else:
        end_ts = float(end_timestamp)
        start_ts = max(0.0, end_ts - float(total_seconds))
        mid_ts = start_ts + 0.5 * (end_ts - start_ts)
        timestamps = [round(float(v), 2) for v in np.linspace(start_ts, end_ts, num=num_frames)]

    for idx, ts in enumerate(timestamps):
        is_early = ts < mid_ts
        pixels = np.zeros((h, w, 3), dtype=np.uint8)
        if is_early:
            pixels[:, :, 0] = 200
            pixels[:, :, 1] = 40
            pixels[:, :, 2] = 40
            tag = early_tag
        else:
            pixels[:, :, 0] = 30
            pixels[:, :, 1] = 160
            pixels[:, :, 2] = 210
            tag = late_tag
        frames.append(
            VideoFrame(
                frame_id=idx,
                timestamp=ts,
                pixels=pixels,
                scene_tag=tag,
            )
        )
    return frames


def extract_causal_video_frames(
    video_path: Optional[str],
    cutoff_timestamp: float,
    sample_fps: float = 1.0,
    max_lookback_seconds: float = 300.0,
    decode_hw: tuple[int, int] = (896, 896),
    recent_window_sec: float = 4.0,
    num_recent_frames: int = 4,
    max_history_frames: int = 48,
    future_timestamps: Optional[Sequence[float]] = None,
    allow_synthetic_fallback: bool = False,
) -> list[VideoFrame]:
    """
    Decode 100% real video frames across `[0, cutoff_timestamp]` (plus optional `future_timestamps`
    for `<Wait>` stream progression) using `decord.VideoReader` with ZERO `cutoff <= 90.0s` cap:
    - Recent 4 seconds `[max(0, cutoff - 4.0), cutoff]`: extracts `num_recent_frames=4` frames
      at `decode_hw` (default `896x896`, which `HighResRingBuffer` retains at full high-res for
      `<ZoomIn>` while `ProtectedRecentWindow` downsamples its working copy to `224x224`).
    - Historical interval `[0, max(0, cutoff - 4.0))`: uniformly extracts up to `max_history_frames`
      (16–32 frames) across the full history so `OnlineEventArchive` receives genuine historical
      visual event segments.
    - All real video frames strictly have `scene_tag=""` (zero Oracle leakage).
    - If `allow_synthetic_fallback=False` (default in real evaluation) and `video_path` is missing
      or fails to decode, raises `FileNotFoundError` / `RuntimeError` immediately.
    """
    cutoff = max(0.5, float(cutoff_timestamp))
    if video_path and Path(video_path).is_file():
        from decord import VideoReader

        th, tw = int(decode_hw[0]), int(decode_hw[1])
        vr = VideoReader(str(video_path), width=tw, height=th, num_threads=2)
        native_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 25.0
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Video file has 0 frames: {video_path}")

        video_duration = float(total_frames - 1) / native_fps
        eff_cutoff = min(cutoff, video_duration)
        recent_start = max(0.0, eff_cutoff - float(recent_window_sec))
        hist_start = max(0.0, eff_cutoff - float(max_lookback_seconds)) if max_lookback_seconds > 0 else 0.0
        hist_start = min(hist_start, recent_start)

        target_timestamps: list[float] = []
        # 1. Uniformly sample historical interval [hist_start, recent_start)
        if recent_start > hist_start + 0.25 and max_history_frames > 0:
            hist_span = recent_start - hist_start
            n_hist = min(int(max_history_frames), max(4, int(round(hist_span * max(0.5, sample_fps)))))
            hist_end = max(hist_start, recent_start - 0.25)
            hist_ts = np.linspace(hist_start, hist_end, num=n_hist)
            target_timestamps.extend(round(float(t), 3) for t in hist_ts)

        # 2. Sample recent 4-second window [recent_start, eff_cutoff] (4 frames)
        n_rec = max(int(num_recent_frames), int(round((eff_cutoff - recent_start) * max(1.0, sample_fps))))
        n_rec = min(8, max(4, n_rec))
        rec_ts = np.linspace(recent_start, eff_cutoff, num=n_rec)
        target_timestamps.extend(round(float(t), 3) for t in rec_ts)

        # 3. Optional future probe timestamps for Forward Active Responding (<Wait>)
        future_set: set[float] = set()
        if future_timestamps:
            for ft in future_timestamps:
                ft_val = float(ft)
                if ft_val > cutoff + 1e-6:
                    future_set.add(round(ft_val, 3))
        target_timestamps.extend(sorted(future_set))

        unique_ts = sorted(set(target_timestamps))
        frame_indices = [
            min(total_frames - 1, max(0, int(round(min(ts, video_duration) * native_fps))))
            for ts in unique_ts
        ]
        batch = vr.get_batch(frame_indices).asnumpy()

        out_frames: list[VideoFrame] = []
        for i, ts in enumerate(unique_ts):
            # Strictly ensure causal timestamps <= cutoff for all prefix frames
            clean_ts = min(cutoff, float(ts)) if float(ts) <= cutoff + 1e-6 else float(ts)
            out_frames.append(
                VideoFrame(
                    frame_id=int(frame_indices[i]),
                    timestamp=round(clean_ts, 3),
                    pixels=batch[i].astype(np.uint8),
                    scene_tag="",
                )
            )
        return out_frames

    if not allow_synthetic_fallback:
        raise FileNotFoundError(
            f"Real video file missing or unreadable at '{video_path}' (cutoff={cutoff}s). "
            "Silent fallback to synthetic solid-color frames is strictly forbidden!"
        )

    span_sec = int(min(12.0, max(8.0, cutoff)))
    return _make_synthetic_stream_frames(
        total_seconds=span_sec,
        fps=max(1.0, sample_fps),
        early_tag="",
        late_tag="",
        end_timestamp=cutoff,
    )


def build_synthetic_ovobench_suite(samples_per_task: int = 2) -> list[OVOBenchSample]:
    """
    Build a deterministic 12-subtask OVO-Bench fixture ONLY for CPU unit tests (`MockVLMEngine`).
    """
    samples: list[OVOBenchSample] = []

    for task_code in REALTIME_TASKS:
        for i in range(samples_per_task):
            needs_zoom = (i % 2 == 1)
            zoom_mode = "temporal_hifps" if task_code in ("ACR", "FPD") else "spatial_hires"
            stream_frames = _make_synthetic_stream_frames(total_seconds=8, fps=2.0)
            samples.append(
                OVOBenchSample(
                    sample_id=f"{task_code}_{i:02d}",
                    task_code=task_code,
                    query=(
                        f"[{task_code}] Read the small blurry text / fast motion detail in the current scene."
                        if needs_zoom
                        else f"[{task_code}] What is the main visible object in the current scene?"
                    ),
                    gt_answer="A",
                    query_timestamp=7.5,
                    task_type="mcq",
                    frames=stream_frames,
                    metadata={
                        "task_code": task_code,
                        "raw_question": f"[{task_code}] synthetic question {i}",
                    },
                    _oracle_scoring_only={
                        "gt_answer": "A",
                        "distractor_answer": "B",
                        "required_tool": "ZoomIn" if needs_zoom else None,
                        "zoom_mode": zoom_mode,
                        "zoom_roi": [0.2, 0.2, 0.8, 0.8],
                        "zoom_window_sec": 2.0,
                    },
                )
            )

    for task_code in BACKWARD_TASKS:
        for i in range(samples_per_task):
            is_hld = task_code == "HLD"
            stream_frames = _make_synthetic_stream_frames(
                total_seconds=12,
                fps=1.0,
                early_tag="person picks up red mug and places key inside drawer",
                late_tag="kitchen counter",
            )
            recall_query = (
                "green alien spaceship landing in kitchen"
                if is_hld
                else "red mug key inside drawer earlier"
            )
            samples.append(
                OVOBenchSample(
                    sample_id=f"{task_code}_{i:02d}",
                    task_code=task_code,
                    query=f"[{task_code}] What happened earlier in the video regarding: {recall_query}?",
                    gt_answer="C",
                    query_timestamp=11.0,
                    task_type="mcq",
                    frames=stream_frames,
                    metadata={
                        "task_code": task_code,
                        "raw_question": f"[{task_code}] What happened earlier in the video regarding: {recall_query}?",
                    },
                    _oracle_scoring_only={
                        "gt_answer": "C",
                        "distractor_answer": "D",
                        "none_found_answer": "C",
                        "required_tool": "Recall",
                        "recall_query": recall_query,
                        "lookback_sec": 30.0,
                        "top_k": 2,
                    },
                )
            )

    for task_code in FORWARD_ACTIVE_TASKS:
        for i in range(samples_per_task):
            stream_frames = _make_synthetic_stream_frames(total_seconds=12, fps=1.0)
            query_ts = 7.0
            reveal_ts = 9.0
            samples.append(
                OVOBenchSample(
                    sample_id=f"{task_code}_{i:02d}",
                    task_code=task_code,
                    query=f"[{task_code}] Monitor the video stream and respond once the target action/step is completed.",
                    gt_answer="A",
                    query_timestamp=query_ts,
                    target_reveal_time=reveal_ts,
                    task_type="proactive",
                    frames=stream_frames,
                    metadata={
                        "task_code": task_code,
                        "raw_question": f"[{task_code}] Monitor the video stream.",
                    },
                    _oracle_scoring_only={
                        "gt_answer": "A",
                        "distractor_answer": "B",
                        "target_reveal_time": reveal_ts,
                    },
                )
            )

    return samples


def _format_mcq_prompt(question: str, options: Sequence[str]) -> str:
    """
    Format a clean, uniform, zero-leakage Multiple-Choice prompt.
    Never forces tool calls based on task_code (`is_backward` / `is_ocr_or_fine` removed)
    and never embeds ground-truth answers.
    """
    if not options:
        return question
    opt_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return (
        f"Question: {question}\n"
        f"Options:\n{opt_str}\n"
        "Select the single best option letter (A, B, C, or D) inside <Answer>...</Answer> "
        "(or invoke a foraging tool first if the current 4 low-resolution frames are insufficient)."
    )


def _parse_single_ovobench_record(
    rec: dict[str, Any],
    idx: int,
    video_root: Optional[Path] = None,
    lazy_decode: bool = False,
    allow_synthetic_fallback: bool = False,
) -> OVOBenchSample:
    """
    Parse a single record from official `ovo_bench_new.json` into `OVOBenchSample`
    under strict Blind Evaluation Isolation:
    - Zero ground-truth answer injection into `scene_tag`, `early_tag`, or `query`.
    - Zero `clue_unfolding` / `clue_revealed` labels.
    - Full-video causal decoding across `[0, cutoff]` without any `cutoff <= 90s` limit.
    """
    task_code = str(rec.get("task", rec.get("task_code", "OCR"))).upper()
    vid_rel = rec.get("video", rec.get("video_path"))
    vid_path = str(video_root / vid_rel) if (video_root and vid_rel) else (str(vid_rel) if vid_rel else None)
    sample_id = str(rec.get("id", f"{task_code}_{idx:04d}"))

    # 1. Backward Tracing & Real-Time Visual Perception (Multiple-Choice with `realtime` cutoff)
    if task_code in REALTIME_TASKS or task_code in BACKWARD_TASKS:
        raw_question = str(rec.get("question", rec.get("query", "")))
        options = tuple(str(x) for x in rec.get("options", []))
        prompt_query = _format_mcq_prompt(raw_question, options)

        if "gt" in rec and isinstance(rec["gt"], int):
            gt_idx = int(rec["gt"])
            gt_letter = chr(65 + gt_idx)
            distractor_letter = chr(65 + ((gt_idx + 1) % max(2, len(options) or 4)))
        else:
            gt_raw = str(rec.get("gt_answer", rec.get("answer", "A"))).strip()
            gt_letter = gt_raw if len(gt_raw) == 1 else "A"
            distractor_letter = "B" if gt_letter != "B" else "A"

        q_ts = float(rec.get("realtime", rec.get("query_timestamp", rec.get("timestamp", 8.0))))

        # Decode full history up to q_ts (up to 600s lookback, 36 history frames on BT / 12 on RT + 4 recent frames at 896x896)
        decode_kwargs: dict[str, Any] = {
            "video_path": vid_path,
            "cutoff_timestamp": q_ts,
            "sample_fps": 2.0 if task_code in REALTIME_TASKS else 1.0,
            "max_lookback_seconds": max(300.0, q_ts),
            "decode_hw": (896, 896),
            "recent_window_sec": 4.0,
            "num_recent_frames": 4,
            "max_history_frames": 36 if task_code in BACKWARD_TASKS else 12,
            "allow_synthetic_fallback": allow_synthetic_fallback,
        }
        frames = [] if lazy_decode else extract_causal_video_frames(**decode_kwargs)

        # Strictly blind metadata (no gt_answer, no answer_text, no target_reveal_time)
        blind_meta: dict[str, Any] = {
            "sample_id": sample_id,
            "task_code": task_code,
            "raw_question": raw_question,
            "query_timestamp": q_ts,
        }

        # Held separately ONLY for CPU MockVLMEngine unit tests, never passed to real VLM
        oracle_mock_only: dict[str, Any] = {
            "gt_answer": gt_letter,
            "distractor_answer": distractor_letter,
            "none_found_answer": gt_letter,
            "required_tool": "Recall" if task_code in BACKWARD_TASKS else (
                "ZoomIn" if (int(rec.get("id", idx)) % 2 == 1) else None
            ),
            "zoom_mode": "temporal_hifps" if task_code in ("ACR", "FPD") else "spatial_hires",
            "zoom_roi": [0.15, 0.15, 0.85, 0.85],
            "zoom_window_sec": 2.0,
            "recall_query": raw_question,
            "lookback_sec": min(180.0, max(15.0, q_ts)),
            "top_k": 2,
        }

        return OVOBenchSample(
            sample_id=sample_id,
            task_code=task_code,
            query=prompt_query,
            gt_answer=gt_letter,
            query_timestamp=q_ts,
            options=options,
            task_type="mcq",
            video_path=vid_path,
            target_reveal_time=None,
            frames=frames,
            lazy_decode_kwargs=decode_kwargs if lazy_decode else None,
            metadata=blind_meta,
            _oracle_scoring_only=oracle_mock_only,
        )

    # 2. Forward Active Responding (`CRR`, `SSR`, `REC`) — 100% Blind Objective Prompting
    test_info = rec.get("test_info", [])
    probe_timestamps = sorted({float(p["realtime"]) for p in test_info if "realtime" in p})

    if task_code == "CRR":
        q_crr = str(rec.get("question", "Has the clue been revealed in the video stream yet?"))
        neg_probes = [p for p in test_info if int(p.get("type", 0)) == 0]
        pos_probes = [float(p["realtime"]) for p in test_info if int(p.get("type", 0)) == 1]
        ask_ts = float(
            rec.get(
                "ask_time",
                neg_probes[-1]["realtime"] if neg_probes else (test_info[0]["realtime"] if test_info else rec.get("query_timestamp", 8.0)),
            )
        )
        pos_after = [t for t in pos_probes if t > ask_ts + 1e-6]
        reveal_ts = float(
            rec.get(
                "clue_time",
                pos_after[0] if pos_after else (pos_probes[0] if pos_probes else (ask_ts + 2.0)),
            )
        )
        if reveal_ts <= ask_ts:
            reveal_ts = ask_ts + 2.0
        gt_ans = "Yes"
        distractor_ans = "No"
        raw_question = (
            f"Question: {q_crr}\n"
            "If the visual clue needed to answer this question has NOT yet appeared in the current video frames, "
            "emit <Wait>{\"check_after_sec\": 2.0, \"trigger\": \"clue_reveal\"}</Wait> to continue watching the stream. "
            "Once the visual evidence is visible in the current frames, output <Answer>Yes</Answer>."
        )
    elif task_code == "SSR":
        neg_probes = [p for p in test_info if int(p.get("type", 0)) == 0]
        last_neg = neg_probes[-1] if neg_probes else (test_info[0] if test_info else {})
        step_desc = str(last_neg.get("step", rec.get("tutorial", "next tutorial step")))
        ask_ts = float(last_neg.get("realtime", rec.get("query_timestamp", 8.0)))
        pos_probes = [
            float(p["realtime"])
            for p in test_info
            if int(p.get("type", 0)) == 1 and float(p["realtime"]) > ask_ts + 1e-6
        ]
        if not pos_probes:
            pos_probes = [float(p["realtime"]) for p in test_info if int(p.get("type", 0)) == 1]
        reveal_ts = pos_probes[0] if (pos_probes and pos_probes[0] > ask_ts) else (ask_ts + 2.0)
        gt_ans = "Yes"
        distractor_ans = "No"
        raw_question = (
            f"Question: Has the person in the video started or completed the tutorial step '{step_desc}' yet?\n"
            "If this step has NOT started or is still incomplete in the current frames, "
            "emit <Wait>{\"check_after_sec\": 2.0, \"trigger\": \"step_reveal\"}</Wait> to observe future frames. "
            "Once you see the step being performed in the current frames, output <Answer>Yes</Answer>."
        )
    else:  # REC
        activity = str(rec.get("activity", "repetitive action"))
        first_probe = test_info[-2] if len(test_info) >= 2 else (test_info[0] if test_info else {})
        last_probe = test_info[-1] if test_info else {}
        ask_ts = float(first_probe.get("realtime", rec.get("query_timestamp", 8.0)))
        reveal_ts = float(last_probe.get("realtime", ask_ts + 2.0))
        if reveal_ts <= ask_ts:
            reveal_ts = ask_ts + 2.0
        gt_ans = str(last_probe.get("count", rec.get("gt_answer", "1")))
        distractor_ans = "0" if gt_ans != "0" else "99"
        # Strictly ZERO `gt_ans` leakage in the question!
        raw_question = (
            f"Question: Count how many times the action '{activity}' has been performed in the video stream.\n"
            "If the repetitive action sequence is still ongoing and more repetitions may occur, "
            "emit <Wait>{\"check_after_sec\": 2.0, \"trigger\": \"count_complete\"}</Wait> to observe subsequent frames. "
            "Once the action sequence is complete, output the integer count inside <Answer>...</Answer>."
        )

    # Strictly blind causal future grid: 5 uniform 2-second steps [ask_ts + 2.0, ..., ask_ts + 10.0]
    # (never uses `reveal_ts` to construct `future_ts` in real evaluation!)
    future_ts = [round(ask_ts + 2.0 * step, 2) for step in range(1, 6)]
    decode_kwargs_far: dict[str, Any] = {
        "video_path": vid_path,
        "cutoff_timestamp": ask_ts,
        "sample_fps": 1.0,
        "max_lookback_seconds": max(60.0, ask_ts),
        "decode_hw": (448, 448),
        "recent_window_sec": 4.0,
        "num_recent_frames": 4,
        "max_history_frames": 16,
        "future_timestamps": future_ts,
        "allow_synthetic_fallback": allow_synthetic_fallback,
    }
    stream_frames = [] if lazy_decode else extract_causal_video_frames(**decode_kwargs_far)

    blind_meta_far: dict[str, Any] = {
        "sample_id": sample_id,
        "task_code": task_code,
        "raw_question": raw_question,
        "query_timestamp": ask_ts,
        "probe_timestamps": probe_timestamps,
    }
    oracle_mock_far: dict[str, Any] = {
        "gt_answer": gt_ans,
        "distractor_answer": distractor_ans,
        "target_reveal_time": reveal_ts,
        "required_tool": "Wait",
    }

    return OVOBenchSample(
        sample_id=sample_id,
        task_code=task_code,
        query=raw_question,
        gt_answer=gt_ans,
        query_timestamp=ask_ts,
        target_reveal_time=reveal_ts,
        task_type="proactive",
        video_path=vid_path,
        frames=stream_frames,
        lazy_decode_kwargs=decode_kwargs_far if lazy_decode else None,
        metadata=blind_meta_far,
        _oracle_scoring_only=oracle_mock_far,
    )


def load_ovobench_from_json(
    json_path: Path,
    video_root: Optional[Path] = DEFAULT_OVOBENCH_VIDEO_ROOT,
    max_samples_per_task: Optional[int] = None,
    shard_idx: int = 0,
    num_shards: int = 1,
    lazy_decode: bool = False,
    allow_synthetic_fallback: bool = False,
) -> list[OVOBenchSample]:
    """
    Load OVO-Bench evaluation samples from official `ovo_bench_new.json` (or JSONL).
    Defaults `video_root` to `DEFAULT_OVOBENCH_VIDEO_ROOT` if it exists on disk so real `.mp4`
    files are always resolved.
    """
    resolved_video_root = video_root
    if resolved_video_root is None and DEFAULT_OVOBENCH_VIDEO_ROOT.is_dir():
        resolved_video_root = DEFAULT_OVOBENCH_VIDEO_ROOT

    text = json_path.read_text(encoding="utf-8").strip()
    if text.startswith("["):
        records = json.loads(text)
    else:
        records = [json.loads(line) for line in text.splitlines() if line.strip()]

    per_task_counts: dict[str, int] = {t: 0 for t in ALL_OVOBENCH_TASKS}
    selected_records: list[tuple[int, dict[str, Any]]] = []

    for idx, rec in enumerate(records):
        task_code = str(rec.get("task", rec.get("task_code", "OCR"))).upper()
        if task_code not in per_task_counts:
            continue
        if max_samples_per_task is not None and per_task_counts[task_code] >= max_samples_per_task:
            continue
        selected_records.append((idx, rec))
        per_task_counts[task_code] += 1

    sharded_records = selected_records[shard_idx::num_shards]
    return [
        _parse_single_ovobench_record(
            rec,
            idx=orig_idx,
            video_root=resolved_video_root,
            lazy_decode=lazy_decode,
            allow_synthetic_fallback=allow_synthetic_fallback,
        )
        for orig_idx, rec in sharded_records
    ]


def evaluate_ovobench_samples(
    samples: Sequence[OVOBenchSample],
    engine: BaseVLMEngine,
    mode: Literal["simplestream", "forager"] = "forager",
    baseline_summary: Optional[OVOBenchEvalSummary] = None,
    use_sdam_mask: bool = True,
    base_resolution: tuple[int, int] = (224, 224),
) -> OVOBenchEvalSummary:
    """
    Run causal streaming evaluation across all `OVOBenchSample`s under 100% Blind Evaluation Isolation:
    - `ProtectedRecentWindow`: 4 recent frames @ `base_resolution` (default `224x224`, or `672x672` in Native High-Res)
    - `HighResRingBuffer`: recent 4.0s frames @ `896x896` (cropped/resized to `896x896` / `672x672` / `448x448` on `<ZoomIn>`)
    - `OnlineEventArchive`: historical frames from `[0, max(0, cutoff - 4.0))` segmented into real visual events
    - `agent.answer_query`: receives ONLY `sample.blind_agent_metadata()` without any ground-truth fields.
    """
    is_mock_engine = isinstance(engine, MockVLMEngine)
    is_hires = int(base_resolution[0]) >= 448
    agent = StreamingForagerAgent(engine=engine, mode=mode, use_sdam_mask=use_sdam_mask)
    env = CausalStreamingDualBuffer(
        recent_window=ProtectedRecentWindow(window_size=4, base_resolution=base_resolution, base_fps=1.0),
        ring_buffer=HighResRingBuffer(
            duration_sec=4.0,
            spatial_hires_hw=(896, 896) if is_hires else (672, 672),
            temporal_hifps_hw=(448, 448) if is_hires else (336, 336),
            target_hifps=4.0,
        ),
    )

    task_correct: dict[str, list[float]] = {t: [] for t in ALL_OVOBENCH_TASKS}
    action_counts: dict[str, int] = {
        ActionType.ANSWER.value: 0,
        ActionType.ZOOM_IN.value: 0,
        ActionType.RECALL.value: 0,
        ActionType.WAIT.value: 0,
        ActionType.INVALID.value: 0,
    }

    recent_tokens_list: list[int] = []
    foraged_tokens_list: list[int] = []
    total_tokens_list: list[int] = []
    latencies_ms: list[float] = []

    for sample_idx, sample in enumerate(samples):
        env.reset()
        all_frames = sample.materialize_frames(allow_synthetic_fallback=is_mock_engine)
        # Strict causal truncation: only frames with timestamp <= sample.query_timestamp are ingested before Turn 1
        prefix_frames = [f for f in all_frames if f.timestamp <= sample.query_timestamp + 1e-6]
        future_frames = [f for f in all_frames if f.timestamp > sample.query_timestamp + 1e-6]

        for pf in prefix_frames:
            env.step_frame(pf)

        future_cursor = {"idx": 0}

        def _stream_provider(delta_sec: float) -> list[VideoFrame]:
            target_t = env.current_time + delta_sec
            if (
                is_mock_engine
                and future_cursor["idx"] < len(future_frames)
                and future_frames[future_cursor["idx"]].timestamp > target_t
            ):
                target_t = future_frames[future_cursor["idx"]].timestamp
            pulled: list[VideoFrame] = []
            while (
                future_cursor["idx"] < len(future_frames)
                and future_frames[future_cursor["idx"]].timestamp <= target_t + 1e-6
            ):
                pulled.append(future_frames[future_cursor["idx"]])
                future_cursor["idx"] += 1
            return pulled

        # Physical isolation: strip all ground-truth fields before calling agent
        blind_meta = sample.blind_agent_metadata(include_mock_hints=is_mock_engine)

        ep_result: AgentEpisodeResult = agent.answer_query(
            env=env,
            query=sample.query,
            stream_provider=_stream_provider,
            max_wait_steps=5,
            metadata=blind_meta,
        )

        # Ground-truth comparison happens strictly in the outer evaluator after episode completion
        _, is_correct = compute_task_accuracy_reward(
            pred_answer=ep_result.final_answer,
            gt_answer=sample.gt_answer,
            task_type=sample.task_type,
            pred_time=ep_result.committed_timestamp,
            gt_time=sample.target_reveal_time,
        )
        task_correct[sample.task_code].append(100.0 if is_correct else 0.0)

        act_key = ep_result.primary_action_type.value
        action_counts[act_key] = action_counts.get(act_key, 0) + 1

        recent_tokens_list.append(ep_result.recent_visual_tokens)
        foraged_tokens_list.append(ep_result.foraged_visual_tokens)
        total_tokens_list.append(ep_result.total_visual_tokens)
        latencies_ms.append(ep_result.latency_ms)
        if (sample_idx + 1) % 10 == 0 or (sample_idx + 1) == len(samples):
            running_acc = float(np.mean([v for vals in task_correct.values() for v in vals]))
            print(
                f"[{mode}] Progress: {sample_idx + 1}/{len(samples)} samples | "
                f"running_acc={running_acc:.2f}% | last_task={sample.task_code} | "
                f"last_act={act_key} | latency={ep_result.latency_ms:.1f}ms",
                flush=True,
            )

    task_accs: dict[str, float] = {}
    task_counts: dict[str, int] = {}
    for t in ALL_OVOBENCH_TASKS:
        vals = task_correct[t]
        task_accs[t] = float(np.mean(vals)) if vals else 0.0
        task_counts[t] = len(vals)

    rt_acc = float(np.mean([task_accs[t] for t in REALTIME_TASKS]))
    bt_acc = float(np.mean([task_accs[t] for t in BACKWARD_TASKS]))
    er_acc = float(0.5 * (task_accs["EPM"] + task_accs["ASI"]))
    far_acc = float(np.mean([task_accs[t] for t in FORWARD_ACTIVE_TASKS]))
    overall_acc = float(np.mean([task_accs[t] for t in ALL_OVOBENCH_TASKS]))

    total_n = max(1, len(samples))
    action_dist = {k: round(100.0 * v / total_n, 2) for k, v in action_counts.items()}

    delta_p: Optional[float] = None
    delta_m: Optional[float] = None
    delta_bt: Optional[float] = None
    if baseline_summary is not None:
        delta_p = round(rt_acc - baseline_summary.realtime_accuracy, 2)
        delta_m = round(er_acc - baseline_summary.episodic_recall_accuracy, 2)
        delta_bt = round(bt_acc - baseline_summary.backward_accuracy, 2)
    elif mode == "simplestream":
        delta_p = 0.0
        delta_m = 0.0
        delta_bt = 0.0

    return OVOBenchEvalSummary(
        mode=mode,
        num_samples=len(samples),
        task_accuracies=task_accs,
        realtime_accuracy=round(rt_acc, 2),
        backward_accuracy=round(bt_acc, 2),
        episodic_recall_accuracy=round(er_acc, 2),
        forward_active_accuracy=round(far_acc, 2),
        overall_accuracy=round(overall_acc, 2),
        action_distribution=action_dist,
        avg_recent_visual_tokens=round(float(np.mean(recent_tokens_list)), 1) if recent_tokens_list else 0.0,
        avg_foraged_visual_tokens=round(float(np.mean(foraged_tokens_list)), 1) if foraged_tokens_list else 0.0,
        avg_total_visual_tokens=round(float(np.mean(total_tokens_list)), 1) if total_tokens_list else 0.0,
        avg_latency_ms=round(float(np.mean(latencies_ms)), 3) if latencies_ms else 0.0,
        delta_p_vs_simplestream=delta_p,
        delta_m_vs_simplestream=delta_m,
        delta_bt_vs_simplestream=delta_bt,
        task_sample_counts=task_counts,
    )


def evaluate_ovobench_joint_single_decode(
    samples: Sequence[OVOBenchSample],
    engine: BaseVLMEngine,
    use_sdam_mask: bool = True,
    base_resolution: tuple[int, int] = (224, 224),
) -> tuple[OVOBenchEvalSummary, OVOBenchEvalSummary]:
    """
    Evaluate both `simplestream` and `forager` in a SINGLE video-decoding pass over `samples`!
    Each video is decoded via `decord` once, cutting CPU/HDFS video decoding overhead by 50%.
    """
    is_mock_engine = isinstance(engine, MockVLMEngine)
    is_hires = int(base_resolution[0]) >= 448
    agent_ss = StreamingForagerAgent(engine=engine, mode="simplestream", use_sdam_mask=use_sdam_mask)
    agent_fo = StreamingForagerAgent(engine=engine, mode="forager", use_sdam_mask=use_sdam_mask)

    modes = ("simplestream", "forager")
    task_correct: dict[str, dict[str, list[float]]] = {m: {t: [] for t in ALL_OVOBENCH_TASKS} for m in modes}
    action_counts: dict[str, dict[str, int]] = {
        m: {
            ActionType.ANSWER.value: 0,
            ActionType.ZOOM_IN.value: 0,
            ActionType.RECALL.value: 0,
            ActionType.WAIT.value: 0,
            ActionType.INVALID.value: 0,
        }
        for m in modes
    }
    rec_toks: dict[str, list[int]] = {m: [] for m in modes}
    for_toks: dict[str, list[int]] = {m: [] for m in modes}
    tot_toks: dict[str, list[int]] = {m: [] for m in modes}
    lats_ms: dict[str, list[float]] = {m: [] for m in modes}

    for sample_idx, sample in enumerate(samples):
        all_frames = sample.materialize_frames(allow_synthetic_fallback=is_mock_engine)
        prefix_frames = [f for f in all_frames if f.timestamp <= sample.query_timestamp + 1e-6]
        future_frames = [f for f in all_frames if f.timestamp > sample.query_timestamp + 1e-6]
        blind_meta = sample.blind_agent_metadata(include_mock_hints=is_mock_engine)

        for mode, agent in (("simplestream", agent_ss), ("forager", agent_fo)):
            env = CausalStreamingDualBuffer(
                recent_window=ProtectedRecentWindow(window_size=4, base_resolution=base_resolution, base_fps=1.0),
                ring_buffer=HighResRingBuffer(
                    duration_sec=4.0,
                    spatial_hires_hw=(896, 896) if is_hires else (672, 672),
                    temporal_hifps_hw=(448, 448) if is_hires else (336, 336),
                    target_hifps=4.0,
                ),
            )
            for pf in prefix_frames:
                env.step_frame(pf)

            future_cursor = {"idx": 0}

            def _stream_provider(delta_sec: float) -> list[VideoFrame]:
                target_t = env.current_time + delta_sec
                if (
                    is_mock_engine
                    and future_cursor["idx"] < len(future_frames)
                    and future_frames[future_cursor["idx"]].timestamp > target_t
                ):
                    target_t = future_frames[future_cursor["idx"]].timestamp
                pulled: list[VideoFrame] = []
                while (
                    future_cursor["idx"] < len(future_frames)
                    and future_frames[future_cursor["idx"]].timestamp <= target_t + 1e-6
                ):
                    pulled.append(future_frames[future_cursor["idx"]])
                    future_cursor["idx"] += 1
                return pulled

            ep_result: AgentEpisodeResult = agent.answer_query(
                env=env,
                query=sample.query,
                stream_provider=_stream_provider,
                max_wait_steps=5,
                metadata=dict(blind_meta),
            )
            _, is_correct = compute_task_accuracy_reward(
                pred_answer=ep_result.final_answer,
                gt_answer=sample.gt_answer,
                task_type=sample.task_type,
                pred_time=ep_result.committed_timestamp,
                gt_time=sample.target_reveal_time,
            )
            task_correct[mode][sample.task_code].append(100.0 if is_correct else 0.0)
            act_key = ep_result.primary_action_type.value
            action_counts[mode][act_key] = action_counts[mode].get(act_key, 0) + 1
            rec_toks[mode].append(ep_result.recent_visual_tokens)
            for_toks[mode].append(ep_result.foraged_visual_tokens)
            tot_toks[mode].append(ep_result.total_visual_tokens)
            lats_ms[mode].append(ep_result.latency_ms)

        if (sample_idx + 1) % 10 == 0 or (sample_idx + 1) == len(samples):
            ss_acc = float(np.mean([v for vals in task_correct["simplestream"].values() for v in vals]))
            fo_acc = float(np.mean([v for vals in task_correct["forager"].values() for v in vals]))
            print(
                f"[joint] Progress: {sample_idx + 1}/{len(samples)} | "
                f"SimpleStream={ss_acc:.2f}% | Forager={fo_acc:.2f}% | task={sample.task_code}",
                flush=True,
            )

    summaries: dict[str, OVOBenchEvalSummary] = {}
    for mode in modes:
        task_accs: dict[str, float] = {}
        task_counts: dict[str, int] = {}
        for t in ALL_OVOBENCH_TASKS:
            vals = task_correct[mode][t]
            task_accs[t] = float(np.mean(vals)) if vals else 0.0
            task_counts[t] = len(vals)
        rt_acc = float(np.mean([task_accs[t] for t in REALTIME_TASKS]))
        bt_acc = float(np.mean([task_accs[t] for t in BACKWARD_TASKS]))
        er_acc = float(0.5 * (task_accs["EPM"] + task_accs["ASI"]))
        far_acc = float(np.mean([task_accs[t] for t in FORWARD_ACTIVE_TASKS]))
        overall_acc = float(np.mean([task_accs[t] for t in ALL_OVOBENCH_TASKS]))
        total_n = max(1, len(samples))
        action_dist = {k: round(100.0 * v / total_n, 2) for k, v in action_counts[mode].items()}

        base = summaries.get("simplestream")
        delta_p = round(rt_acc - base.realtime_accuracy, 2) if base is not None else 0.0
        delta_m = round(er_acc - base.episodic_recall_accuracy, 2) if base is not None else 0.0
        delta_bt = round(bt_acc - base.backward_accuracy, 2) if base is not None else 0.0

        summaries[mode] = OVOBenchEvalSummary(
            mode=mode,
            num_samples=len(samples),
            task_accuracies=task_accs,
            realtime_accuracy=round(rt_acc, 2),
            backward_accuracy=round(bt_acc, 2),
            episodic_recall_accuracy=round(er_acc, 2),
            forward_active_accuracy=round(far_acc, 2),
            overall_accuracy=round(overall_acc, 2),
            action_distribution=action_dist,
            avg_recent_visual_tokens=round(float(np.mean(rec_toks[mode])), 1) if rec_toks[mode] else 0.0,
            avg_foraged_visual_tokens=round(float(np.mean(for_toks[mode])), 1) if for_toks[mode] else 0.0,
            avg_total_visual_tokens=round(float(np.mean(tot_toks[mode])), 1) if tot_toks[mode] else 0.0,
            avg_latency_ms=round(float(np.mean(lats_ms[mode])), 3) if lats_ms[mode] else 0.0,
            delta_p_vs_simplestream=delta_p,
            delta_m_vs_simplestream=delta_m,
            delta_bt_vs_simplestream=delta_bt,
            task_sample_counts=task_counts,
        )
    return summaries["simplestream"], summaries["forager"]


def merge_ovobench_shard_files(shard_paths: Sequence[Path]) -> dict[str, OVOBenchEvalSummary]:
    """Merge multiple data-parallel shard JSON files into unified SimpleStream & Forager summaries."""
    shard_payloads = [json.loads(p.read_text(encoding="utf-8")) for p in shard_paths if p.is_file()]
    if not shard_payloads:
        raise ValueError("No valid shard JSON files found to merge")

    merged: dict[str, OVOBenchEvalSummary] = {}
    for mode in ("simplestream", "forager"):
        total_samples = 0
        task_correct_sum: dict[str, float] = {t: 0.0 for t in ALL_OVOBENCH_TASKS}
        task_count_sum: dict[str, int] = {t: 0 for t in ALL_OVOBENCH_TASKS}
        act_weighted_sum: dict[str, float] = {
            ActionType.ANSWER.value: 0.0,
            ActionType.ZOOM_IN.value: 0.0,
            ActionType.RECALL.value: 0.0,
            ActionType.WAIT.value: 0.0,
            ActionType.INVALID.value: 0.0,
        }
        rec_tok_sum = 0.0
        for_tok_sum = 0.0
        tot_tok_sum = 0.0
        lat_sum = 0.0

        for sp in shard_payloads:
            m = sp[mode]
            n_s = int(m["num_samples"])
            total_samples += n_s
            t_counts = m.get("task_sample_counts") or {}
            for t in ALL_OVOBENCH_TASKS:
                cnt = int(t_counts.get(t, max(1, n_s // 12)))
                task_count_sum[t] += cnt
                task_correct_sum[t] += float(m["task_accuracies"].get(t, 0.0)) * cnt
            for k in act_weighted_sum:
                act_weighted_sum[k] += float(m["action_distribution"].get(k, 0.0)) * n_s
            rec_tok_sum += float(m["avg_recent_visual_tokens"]) * n_s
            for_tok_sum += float(m["avg_foraged_visual_tokens"]) * n_s
            tot_tok_sum += float(m["avg_total_visual_tokens"]) * n_s
            lat_sum += float(m["avg_latency_ms"]) * n_s

        denom = max(1, total_samples)
        task_accs = {
            t: round(task_correct_sum[t] / max(1, task_count_sum[t]), 2)
            for t in ALL_OVOBENCH_TASKS
        }
        rt_acc = float(np.mean([task_accs[t] for t in REALTIME_TASKS]))
        bt_acc = float(np.mean([task_accs[t] for t in BACKWARD_TASKS]))
        er_acc = float(0.5 * (task_accs["EPM"] + task_accs["ASI"]))
        far_acc = float(np.mean([task_accs[t] for t in FORWARD_ACTIVE_TASKS]))
        overall_acc = float(np.mean([task_accs[t] for t in ALL_OVOBENCH_TASKS]))
        action_dist = {k: round(v / denom, 2) for k, v in act_weighted_sum.items()}

        base = merged.get("simplestream")
        delta_p = round(rt_acc - base.realtime_accuracy, 2) if base is not None else 0.0
        delta_m = round(er_acc - base.episodic_recall_accuracy, 2) if base is not None else 0.0
        delta_bt = round(bt_acc - base.backward_accuracy, 2) if base is not None else 0.0

        merged[mode] = OVOBenchEvalSummary(
            mode=mode,
            num_samples=total_samples,
            task_accuracies=task_accs,
            realtime_accuracy=round(rt_acc, 2),
            backward_accuracy=round(bt_acc, 2),
            episodic_recall_accuracy=round(er_acc, 2),
            forward_active_accuracy=round(far_acc, 2),
            overall_accuracy=round(overall_acc, 2),
            action_distribution=action_dist,
            avg_recent_visual_tokens=round(rec_tok_sum / denom, 1),
            avg_foraged_visual_tokens=round(for_tok_sum / denom, 1),
            avg_total_visual_tokens=round(tot_tok_sum / denom, 1),
            avg_latency_ms=round(lat_sum / denom, 3),
            delta_p_vs_simplestream=delta_p,
            delta_m_vs_simplestream=delta_m,
            delta_bt_vs_simplestream=delta_bt,
            task_sample_counts=task_count_sum,
        )
    return merged


def format_summary_report(summary: OVOBenchEvalSummary) -> str:
    """Format a human-readable OVO-Bench evaluation report."""
    lines = [
        f"=== OVO-Bench Streaming Evaluation Summary (mode={summary.mode}, N={summary.num_samples}) ===",
        f"  Real-Time Perception (6 tasks): {summary.realtime_accuracy:6.2f}%  "
        + " ".join(f"{t}:{summary.task_accuracies[t]:.1f}%" for t in REALTIME_TASKS),
        f"  Backward Tracing     (3 tasks): {summary.backward_accuracy:6.2f}%  "
        + " ".join(f"{t}:{summary.task_accuracies[t]:.1f}%" for t in BACKWARD_TASKS),
        f"  Forward Active Resp. (3 tasks): {summary.forward_active_accuracy:6.2f}%  "
        + " ".join(f"{t}:{summary.task_accuracies[t]:.1f}%" for t in FORWARD_ACTIVE_TASKS),
        f"  Overall Macro Accuracy (12)   : {summary.overall_accuracy:6.2f}%",
        f"  Delta P (RT vs SimpleStream-4f): {summary.delta_p_vs_simplestream:+.2f}%"
        if summary.delta_p_vs_simplestream is not None
        else "  Delta P: N/A",
        f"  Delta M (ER vs SimpleStream-4f): {summary.delta_m_vs_simplestream:+.2f}% "
        f"(Delta BT: {summary.delta_bt_vs_simplestream:+.2f}%)"
        if summary.delta_m_vs_simplestream is not None and summary.delta_bt_vs_simplestream is not None
        else "  Delta M: N/A",
        f"  Tool Trigger Distribution (%) : {summary.action_distribution}",
        f"  Avg Visual Tokens (rec/for/tot): "
        f"{summary.avg_recent_visual_tokens:.1f} / {summary.avg_foraged_visual_tokens:.1f} / {summary.avg_total_visual_tokens:.1f}",
        f"  Avg Query Latency (ms)        : {summary.avg_latency_ms:.3f} ms",
    ]
    return "\n".join(lines)


def build_engine(
    backend: str,
    model_path: str,
    native_hires: bool = False,
    uncalibrated_zeroshot: bool = False,
) -> BaseVLMEngine:
    if backend == "mock":
        return MockVLMEngine()
    if backend == "hf":
        return HFQwenVLEngine(
            model_name_or_path=model_path,
            native_hires=native_hires,
            uncalibrated_zeroshot=uncalibrated_zeroshot,
        )
    if backend == "vllm":
        return VLLMEngine(model_name_or_path=model_path)
    raise ValueError(f"Unsupported backend: {backend}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate SimpleStream and ActiveForager on OVO-Bench.")
    parser.add_argument(
        "--backend",
        choices=["mock", "hf", "vllm"],
        default="mock",
        help="VLM inference backend ('mock' for CPU tests, 'hf' or 'vllm' for GPU workers).",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct",
        help="Model path for 'hf' or 'vllm' backend.",
    )
    parser.add_argument(
        "--data-json",
        type=Path,
        default=None,
        help="Optional OVO-Bench JSON/JSONL file (e.g. /mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json).",
    )
    parser.add_argument(
        "--video-root",
        type=Path,
        default=DEFAULT_OVOBENCH_VIDEO_ROOT,
        help="Root directory for source video files.",
    )
    parser.add_argument(
        "--samples-per-task",
        type=int,
        default=2,
        help="Number of synthetic samples per task when --data-json is omitted.",
    )
    parser.add_argument(
        "--max-samples-per-task",
        type=int,
        default=None,
        help="Optional cap on samples per subtask when loading from --data-json.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default="0/1",
        help="Shard spec k/n for data-parallel evaluation.",
    )
    parser.add_argument(
        "--base-hw",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Base recent window resolution H W (default 224 224; use 672 672 for Native High-Res).",
    )
    parser.add_argument(
        "--native-hires",
        action="store_true",
        help="Enable Native High-Resolution Regime (672x672 base recent window, 896x896 ZoomIn, 448x448 Sub-Agent).",
    )
    parser.add_argument(
        "--uncalibrated-zeroshot",
        action="store_true",
        help="Evaluate uncalibrated Vanilla Zero-Shot Agent (Stage-1) without post-training counterfactual calibration.",
    )
    parser.add_argument(
        "--no-sdam",
        action="store_true",
        help="Disable 4D SDAM attention mask in Turn 2 (for ablation comparison).",
    )
    parser.add_argument(
        "--merge-shards",
        type=Path,
        nargs="*",
        default=None,
        help="Optional list of shard JSON files to merge into --output-json.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional file path to save structured JSON evaluation results.",
    )
    args = parser.parse_args(argv)

    if args.merge_shards:
        merged = merge_ovobench_shard_files(args.merge_shards)
        simplestream_summary = merged["simplestream"]
        forager_summary = merged["forager"]
        print(format_summary_report(simplestream_summary))
        print()
        print(format_summary_report(forager_summary))
        if args.output_json is not None:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            ss_9task = round(0.5 * (simplestream_summary.realtime_accuracy + simplestream_summary.backward_accuracy), 2)
            fo_9task = round(0.5 * (forager_summary.realtime_accuracy + forager_summary.backward_accuracy), 2)
            payload = {
                "simplestream": asdict(simplestream_summary),
                "forager": asdict(forager_summary),
                "simplestream_9task_avg": ss_9task,
                "forager_9task_avg": fo_9task,
            }
            args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"Invalid --shard {args.shard}")

    base_hw = (672, 672) if (args.native_hires and tuple(args.base_hw) == (224, 224)) else (int(args.base_hw[0]), int(args.base_hw[1]))
    is_native_hires = bool(args.native_hires or base_hw[0] >= 448)

    engine = build_engine(
        args.backend,
        args.model_path,
        native_hires=is_native_hires,
        uncalibrated_zeroshot=bool(args.uncalibrated_zeroshot),
    )
    if args.data_json is not None:
        samples = load_ovobench_from_json(
            args.data_json,
            video_root=args.video_root,
            max_samples_per_task=args.max_samples_per_task,
            shard_idx=shard_idx,
            num_shards=num_shards,
            lazy_decode=(args.backend != "mock"),
            allow_synthetic_fallback=False,
        )
    else:
        all_synth = build_synthetic_ovobench_suite(samples_per_task=args.samples_per_task)
        samples = all_synth[shard_idx::num_shards]

    simplestream_summary, forager_summary = evaluate_ovobench_joint_single_decode(
        samples=samples,
        engine=engine,
        use_sdam_mask=not args.no_sdam,
        base_resolution=base_hw,
    )

    print(format_summary_report(simplestream_summary))
    print()
    print(format_summary_report(forager_summary))

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        ss_9task = round(0.5 * (simplestream_summary.realtime_accuracy + simplestream_summary.backward_accuracy), 2)
        fo_9task = round(0.5 * (forager_summary.realtime_accuracy + forager_summary.backward_accuracy), 2)
        payload = {
            "simplestream": asdict(simplestream_summary),
            "forager": asdict(forager_summary),
            "simplestream_9task_avg": ss_9task,
            "forager_9task_avg": fo_9task,
        }
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
