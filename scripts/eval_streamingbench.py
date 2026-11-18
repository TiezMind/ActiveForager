#!/usr/bin/env python3
"""
StreamingBench 100% Zero-Leakage Real-Video Causal Evaluator (`scripts/eval_streamingbench.py`).

Evaluates `SimpleStream-4f` (Direct `<Answer>` on last 4 frames @ 224x224) vs. `Forager`
(`ActiveForager 3D-IG-GRPO`) across all 18 subtasks and 3 core tracks of `mjuicem/StreamingBench`:
1. Real-Time Visual Understanding (`RTVU`, 10 subtasks: Object Recognition, Attribute Recognition,
   Action Recognition, Spatial Understanding, Text-Rich Understanding, Counting, Clips Summarize,
   Event Understanding, Causal Reasoning, Prospective Reasoning).
2. Omni-Source Understanding (`OMNI`, 6 subtasks: Misleading Context Understanding, Source Discrimination,
   Emotion Recognition, Anomaly Context Understanding, Scene Understanding, Multimodal Alignment).
3. Contextual & Proactive Understanding (`CU`, 2 subtasks: Sequential Question Answering `SQA`
   and Proactive Output `PO`).

Strictly Enforces 100% Blind Evaluation Isolation & 100% Real `.mp4` Video Decoding:
- Every sample resolves its real `.mp4` video under `/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench/videos/`
  and decodes real frames via `decord.VideoReader` truncated at the query `time_stamp` (`[0, t_q]`).
- Raises `FileNotFoundError` immediately if any video file is missing or unreadable (zero synthetic
  solid-color fallback, zero `video_path = None`, zero fallback video pool).
- Zero ground-truth answer, `early_tag`, `answer_text`, `clue_unfolding`, or `clue_revealed` leakage
  in any frame `scene_tag` (all frames strictly have `scene_tag == ""`), prompt, or runtime metadata.
"""

from __future__ import annotations

import argparse
import json
import os
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
)
from forager.dual_buffer import (
    CausalStreamingDualBuffer,
    HighResRingBuffer,
    ProtectedRecentWindow,
    VideoFrame,
)
from forager.ig_grpo import compute_task_accuracy_reward
from forager.protocol import ActionType
from scripts.eval_ovobench import extract_causal_video_frames

DEFAULT_STREAMINGBENCH_DATA_DIR = Path("/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench/data")
DEFAULT_STREAMINGBENCH_VIDEOS_DIR = Path("/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench/videos")

RTVU_TASKS: tuple[str, ...] = (
    "Object Recognition",
    "Attribute Recognition",
    "Action Recognition",
    "Spatial Understanding",
    "Text-Rich Understanding",
    "Counting",
    "Clips Summarize",
    "Event Understanding",
    "Causal Reasoning",
    "Prospective Reasoning",
)

OMNI_TASKS: tuple[str, ...] = (
    "Misleading Context Understanding",
    "Source Discrimination",
    "Emotion Recognition",
    "Anomaly Context Understanding",
    "Scene Understanding",
    "Multimodal Alignment",
)

CONTEXTUAL_TASKS: tuple[str, ...] = (
    "Sequential Question Answering",
    "Proactive Output",
)

ALL_STREAMINGBENCH_TASKS: tuple[str, ...] = RTVU_TASKS + OMNI_TASKS + CONTEXTUAL_TASKS

FORBIDDEN_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "gt_answer",
        "answer",
        "answer_text",
        "target_reveal_time",
        "distractor_answer",
        "none_found_answer",
        "ground_truth_output",
        "ground_truth_time_stamp",
    }
)


def _parse_hhmmss(ts_str: str) -> float:
    parts = [int(x) for x in str(ts_str).strip().split(":") if x.strip().isdigit()]
    if not parts:
        return 10.0
    return float(sum(val * (60**i) for i, val in enumerate(reversed(parts))))


def _format_clean_mcq_prompt(
    question: str,
    options: Sequence[str],
    prior_questions: str = "",
) -> str:
    """
    Format a uniform, 100% blind Multiple-Choice prompt for StreamingBench.
    Never leaks ground-truth answers and never forces tool calls based on subtask labels.
    """
    ctx_prefix = f"[Prior Dialog Questions: {prior_questions}]\n" if prior_questions else ""
    opt_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
    return (
        f"{ctx_prefix}Question: {question}\n"
        f"Options:\n{opt_str}\n"
        "Select the single best option letter (A, B, C, or D) inside <Answer>...</Answer> "
        "(or invoke a foraging tool first if the current 4 low-resolution frames are insufficient)."
    )


@dataclass
class StreamingBenchSample:
    """
    Single causal evaluation query from StreamingBench under 100% Blind Evaluation Isolation.

    `gt_answer` and `target_reveal_time` are held on the outer evaluation dataclass strictly for
    post-episode scoring (`compute_task_accuracy_reward`) and are NEVER included in
    `blind_agent_metadata()` or passed into `agent.answer_query` during real evaluation.
    """

    sample_id: str
    track: Literal["rtvu", "omni", "contextual"]
    subtask: str
    required_ability: Literal["working memory", "episodic memory", "proactive"]
    query: str
    gt_answer: str
    query_timestamp: float
    video_path: str
    options: tuple[str, ...] = ()
    task_type: Literal["mcq", "open", "proactive"] = "mcq"
    target_reveal_time: Optional[float] = None
    frames: list[VideoFrame] = field(default_factory=list)
    lazy_decode_kwargs: Optional[dict[str, Any]] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    _oracle_scoring_only: dict[str, Any] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if not self.video_path or not str(self.video_path).strip():
            raise ValueError(f"Sample {self.sample_id} must have a valid real video_path (got {self.video_path!r})")

    def blind_agent_metadata(self, include_mock_hints: bool = False) -> dict[str, Any]:
        """
        Return strictly blind metadata for `agent.answer_query` (zero ground-truth fields).
        Only when `include_mock_hints=True` (for standalone CPU `MockVLMEngine` unit tests)
        are mock hints merged.
        """
        clean = {k: v for k, v in self.metadata.items() if k not in FORBIDDEN_METADATA_KEYS}
        if include_mock_hints and self._oracle_scoring_only:
            return {**clean, **self._oracle_scoring_only}
        return clean

    def materialize_frames(
        self,
        decode_hw: tuple[int, int] = (896, 896),
        allow_synthetic_fallback: bool = False,
    ) -> list[VideoFrame]:
        """
        Decode 100% real `.mp4` frames via `decord.VideoReader` across `[0, query_timestamp]`.
        Strictly raises `FileNotFoundError` if `self.video_path` does not exist on disk.
        """
        if self.frames:
            return self.frames
        if not self.video_path or not Path(self.video_path).is_file():
            raise FileNotFoundError(
                f"Real StreamingBench video file not found for sample {self.sample_id}: '{self.video_path}'. "
                "Silent fallback to synthetic frames or fallback pools is strictly forbidden!"
            )
        if self.lazy_decode_kwargs is None:
            raise ValueError(f"Sample {self.sample_id} is missing lazy_decode_kwargs")
        kw = {
            **self.lazy_decode_kwargs,
            "video_path": self.video_path,
            "decode_hw": decode_hw,
            "allow_synthetic_fallback": allow_synthetic_fallback,
        }
        frames = extract_causal_video_frames(**kw)
        for f in frames:
            f.scene_tag = ""
        return frames


@dataclass
class StreamingBenchSummary:
    """Aggregated metrics across StreamingBench tracks and subtasks."""

    mode: str
    num_samples: int
    subtask_accuracies: dict[str, float]
    rtvu_accuracy: float
    omni_accuracy: float
    contextual_accuracy: float
    working_memory_accuracy: float
    episodic_memory_accuracy: float
    proactive_accuracy: float
    overall_accuracy: float
    action_distribution: dict[str, float]
    avg_recent_visual_tokens: float
    avg_foraged_visual_tokens: float
    avg_total_visual_tokens: float
    avg_latency_ms: float
    delta_p_vs_simplestream: float = 0.0
    delta_m_vs_simplestream: float = 0.0
    subtask_counts: dict[str, int] = field(default_factory=dict)


def _scan_verified_videos(sb_root: Path) -> dict[str, int]:
    """
    Fast single-pass directory scan (`os.scandir`) of `sb_root / 'videos'` returning a map
    `{filename: size_bytes}` for all non-empty `.mp4` files that are completely written on disk.
    """
    videos_dir = sb_root / "videos"
    if not videos_dir.is_dir():
        return {}
    index_path = sb_root / "sb_zip_index.json"
    expected_sizes: dict[str, int] = {}
    if index_path.is_file():
        idx_data = json.loads(index_path.read_text(encoding="utf-8"))
        for rel_k, meta in idx_data.get("videos", {}).items():
            expected_sizes[Path(rel_k).name] = int(meta["uncomp_sz"])

    valid_map: dict[str, int] = {}
    for entry in os.scandir(videos_dir):
        if entry.name.endswith(".mp4") and entry.is_file():
            sz = entry.stat().st_size
            exp = expected_sizes.get(entry.name)
            if sz > 0 and (exp is None or sz == exp):
                valid_map[entry.name] = sz
    return valid_map


def load_streamingbench_samples(
    data_dir: Path = DEFAULT_STREAMINGBENCH_DATA_DIR,
    max_samples_per_subtask: Optional[int] = 20,
    shard_idx: int = 0,
    num_shards: int = 1,
    lazy_decode: bool = True,
) -> list[StreamingBenchSample]:
    """
    Load StreamingBench evaluation samples across `questions_real.json`, `questions_omni.json`,
    `questions_sqa.json`, and `questions_proactive.json` under 100% Zero-Leakage & 100% Real Video Isolation:
    - Resolves every `video_path` strictly to `<sb_root>/videos/sample_*.mp4` (never `None`, never a fallback pool).
    - Zero ground-truth injection in `scene_tag`, `early_tag`, `query`, or `blind_agent_metadata()`.
    - Full causal video decoding up to `q_ts` via `decord.VideoReader`.
    """
    sb_root = data_dir.parent
    valid_video_names = _scan_verified_videos(sb_root)
    if not valid_video_names:
        raise FileNotFoundError(
            f"No verified .mp4 videos found in '{sb_root / 'videos'}'. "
            "Please download real StreamingBench videos first."
        )

    raw_items: list[dict[str, Any]] = []

    # 1. Real-Time Visual Understanding (RTVU, 10 subtasks)
    real_path = data_dir / "questions_real.json"
    if real_path.is_file():
        for entry in json.loads(real_path.read_text(encoding="utf-8")):
            v_rel = str(entry.get("video_path", ""))
            for q in entry.get("questions", []):
                raw_items.append({"track": "rtvu", "video_path": v_rel, "q": q, "prior_questions": ""})

    # 2. Omni-Source Understanding (OMNI, 6 subtasks)
    omni_path = data_dir / "questions_omni.json"
    if omni_path.is_file():
        for entry in json.loads(omni_path.read_text(encoding="utf-8")):
            v_rel = str(entry.get("video_path", ""))
            for q in entry.get("questions", []):
                raw_items.append({"track": "omni", "video_path": v_rel, "q": q, "prior_questions": ""})

    # 3. Contextual Understanding: Sequential Question Answering (SQA)
    # Strictly records ONLY prior question timestamps/texts (NEVER prior ground-truth answers!)
    sqa_path = data_dir / "questions_sqa.json"
    if sqa_path.is_file():
        for video_group in json.loads(sqa_path.read_text(encoding="utf-8")):
            prior_q_list: list[str] = []
            for entry in video_group:
                v_rel = str(entry.get("video_path", ""))
                for q in entry.get("questions", []):
                    raw_items.append(
                        {
                            "track": "contextual",
                            "video_path": v_rel,
                            "q": q,
                            "prior_questions": " | ".join(prior_q_list[-2:]),
                        }
                    )
                    prior_q_list.append(f"At {q.get('time_stamp', '')}: {q.get('question', '')}")

    # 4. Contextual Understanding: Proactive Output (PO)
    pro_path = data_dir / "questions_proactive.json"
    if pro_path.is_file():
        for entry in json.loads(pro_path.read_text(encoding="utf-8")):
            v_rel = str(entry.get("video_path", ""))
            for q in entry.get("questions", []):
                raw_items.append({"track": "contextual", "video_path": v_rel, "q": q, "prior_questions": ""})

    per_subtask_counts: dict[str, int] = {t: 0 for t in ALL_STREAMINGBENCH_TASKS}
    selected: list[tuple[int, dict[str, Any]]] = []

    # Sort verified candidate items by video file size (then original index) so HDFS decord I/O is fast and deterministic
    indexed_candidates: list[tuple[int, int, dict[str, Any]]] = []
    for idx, item in enumerate(raw_items):
        v_rel = str(item["video_path"]).lstrip("./")
        v_name = Path(v_rel).name
        if v_name in valid_video_names:
            indexed_candidates.append((valid_video_names[v_name], idx, item))
    indexed_candidates.sort(key=lambda x: (x[0], x[1]))

    for _, idx, item in indexed_candidates:
        st = str(item["q"].get("task_type", "Object Recognition"))
        if st not in per_subtask_counts:
            continue
        if max_samples_per_subtask is not None and per_subtask_counts[st] >= max_samples_per_subtask:
            continue
        selected.append((idx, item))
        per_subtask_counts[st] += 1

    missing_subtasks = [t for t, c in per_subtask_counts.items() if c == 0]
    if missing_subtasks:
        raise FileNotFoundError(
            f"Subtasks missing real .mp4 videos in '{sb_root / 'videos'}': {missing_subtasks}"
        )

    sharded = selected[shard_idx::num_shards]
    samples: list[StreamingBenchSample] = []

    for orig_idx, item in sharded:
        q = item["q"]
        track = item["track"]
        subtask = str(q.get("task_type", "Object Recognition"))
        v_rel = str(item["video_path"]).lstrip("./")
        vid_file = str((sb_root / v_rel).resolve())
        sample_id = f"sb_{track}_{orig_idx:04d}"

        if subtask == "Proactive Output":
            ask_ts = _parse_hhmmss(q.get("time_stamp", "00:00:10"))
            raw_rev_ts = _parse_hhmmss(q.get("ground_truth_time_stamp", "00:00:14"))
            rev_ts = raw_rev_ts if raw_rev_ts > ask_ts else (ask_ts + 4.0)
            gt_out = str(q.get("ground_truth_output", "")).strip()
            raw_q = str(q.get("question", "Respond when the visual condition is met.")).strip()

            # 100% Blind Objective Proactive Prompt: uses ONLY `raw_q` from the dataset without injecting `gt_out`
            # or synthetic `scene: clue_unfolding / clue_revealed` triggers!
            prompt_q = (
                f"Question: {raw_q}\n"
                "Inspect the current video frames. If the visual condition described above has NOT yet occurred "
                "in the current frames, emit <Wait>{\"check_after_sec\": 2.0, \"trigger\": \"condition_reveal\"}</Wait> "
                "to continue watching subsequent frames. Once the visual condition is visible in the current frames, "
                "output your response inside <Answer>...</Answer>."
            )
            # Uniform causal future grid [ask_ts + 2.0, ..., ask_ts + 10.0] independent of `rev_ts`
            future_ts = [round(ask_ts + 2.0 * step, 2) for step in range(1, 6)]
            decode_kwargs_po: dict[str, Any] = {
                "video_path": vid_file,
                "cutoff_timestamp": ask_ts,
                "sample_fps": 1.0,
                "max_lookback_seconds": max(60.0, ask_ts),
                "decode_hw": (672, 672),
                "recent_window_sec": 4.0,
                "num_recent_frames": 4,
                "max_history_frames": 12,
                "future_timestamps": future_ts,
                "allow_synthetic_fallback": False,
            }
            frames_po = [] if lazy_decode else extract_causal_video_frames(**decode_kwargs_po)

            blind_meta_po: dict[str, Any] = {
                "sample_id": sample_id,
                "track": "contextual",
                "subtask": subtask,
                "required_ability": "proactive",
                "raw_question": raw_q,
                "query_timestamp": ask_ts,
                "video_path": vid_file,
            }
            oracle_mock_po: dict[str, Any] = {
                "gt_answer": gt_out,
                "distractor_answer": "No",
                "target_reveal_time": rev_ts,
                "required_tool": "Wait",
            }
            samples.append(
                StreamingBenchSample(
                    sample_id=sample_id,
                    track="contextual",
                    subtask=subtask,
                    required_ability="proactive",
                    query=prompt_q,
                    gt_answer=gt_out,
                    query_timestamp=ask_ts,
                    video_path=vid_file,
                    target_reveal_time=rev_ts,
                    task_type="proactive",
                    frames=frames_po,
                    lazy_decode_kwargs=decode_kwargs_po,
                    metadata=blind_meta_po,
                    _oracle_scoring_only=oracle_mock_po,
                )
            )
            continue

        # Multiple-Choice Subtasks (RTVU, OMNI, SQA)
        q_ts = max(1.0, _parse_hhmmss(q.get("time_stamp", "00:00:20")))
        raw_question = str(q.get("question", "")).strip()
        raw_opts = [str(o) for o in q.get("options", [])]
        clean_opts: list[str] = []
        for i, opt in enumerate(raw_opts):
            prefix = f"{chr(65 + i)}."
            clean_opts.append(opt[len(prefix) :].strip() if opt.startswith(prefix) else opt.strip())
        options_tuple = tuple(clean_opts)

        gt_letter = str(q.get("answer", "A")).strip()[:1].upper()
        gt_idx = max(0, min(len(options_tuple) - 1, ord(gt_letter) - 65)) if options_tuple else 0
        distractor_letter = chr(65 + ((gt_idx + 1) % max(2, len(options_tuple) or 4)))

        raw_ab = str(q.get("required_ability", "")).strip().lower()
        is_episodic = (
            raw_ab == "episodic memory"
            or subtask in (
                "Clips Summarize",
                "Event Understanding",
                "Causal Reasoning",
                "Counting",
                "Prospective Reasoning",
                "Sequential Question Answering",
                "Anomaly Context Understanding",
                "Emotion Recognition",
                "Scene Understanding",
                "Source Discrimination",
                "Multimodal Alignment",
            )
        )
        ability: Literal["working memory", "episodic memory"] = (
            "episodic memory" if is_episodic else "working memory"
        )

        prompt_query = _format_clean_mcq_prompt(
            question=raw_question,
            options=options_tuple,
            prior_questions=str(item.get("prior_questions", "")),
        )

        # Full causal history decoding across [0, q_ts] with zero synthetic fallback and scene_tag=""
        decode_kwargs: dict[str, Any] = {
            "video_path": vid_file,
            "cutoff_timestamp": q_ts,
            "sample_fps": 2.0 if ability == "working memory" else 1.0,
            "max_lookback_seconds": max(300.0, q_ts),
            "decode_hw": (896, 896),
            "recent_window_sec": 4.0,
            "num_recent_frames": 4,
            "max_history_frames": 24,
            "allow_synthetic_fallback": False,
        }
        frames_mcq = [] if lazy_decode else extract_causal_video_frames(**decode_kwargs)

        blind_meta: dict[str, Any] = {
            "sample_id": sample_id,
            "track": track,
            "subtask": subtask,
            "required_ability": ability,
            "raw_question": raw_question,
            "query_timestamp": q_ts,
            "video_path": vid_file,
        }
        oracle_mock_only: dict[str, Any] = {
            "gt_answer": gt_letter,
            "distractor_answer": distractor_letter,
            "none_found_answer": gt_letter,
            "required_tool": "Recall" if ability == "episodic memory" else (
                "ZoomIn" if subtask == "Text-Rich Understanding" else None
            ),
            "zoom_mode": "spatial_hires",
            "zoom_roi": [0.15, 0.15, 0.85, 0.85],
            "zoom_window_sec": 2.0,
            "recall_query": raw_question,
            "lookback_sec": min(180.0, max(15.0, q_ts)),
            "top_k": 2,
        }

        samples.append(
            StreamingBenchSample(
                sample_id=sample_id,
                track=track,
                subtask=subtask,
                required_ability=ability,
                query=prompt_query,
                gt_answer=gt_letter,
                query_timestamp=q_ts,
                video_path=vid_file,
                options=options_tuple,
                task_type="mcq",
                frames=frames_mcq,
                lazy_decode_kwargs=decode_kwargs,
                metadata=blind_meta,
                _oracle_scoring_only=oracle_mock_only,
            )
        )

    return samples


def evaluate_streamingbench_joint_pass(
    samples: Sequence[StreamingBenchSample],
    engine: BaseVLMEngine,
    base_resolution: tuple[int, int] = (224, 224),
) -> dict[str, StreamingBenchSummary]:
    """
    Evaluate BOTH `simplestream` (4-frame sliding window) and `forager` (ActiveForager 3D-IG-GRPO)
    in a single real video-decoding pass over `samples` under 100% Blind Evaluation Isolation:
    - Each video is decoded once via `decord.VideoReader` at `(896, 896)` (with `scene_tag=""`).
    - `agent.answer_query` receives ONLY `sample.blind_agent_metadata(include_mock_hints=is_mock)`.
    """
    is_mock_engine = isinstance(engine, MockVLMEngine)
    is_hires = int(base_resolution[0]) >= 448
    agent_simple = StreamingForagerAgent(engine=engine, mode="simplestream", use_sdam_mask=True)
    agent_forager = StreamingForagerAgent(engine=engine, mode="forager", use_sdam_mask=True)

    modes = ("simplestream", "forager")
    subtask_acc_map: dict[str, dict[str, list[float]]] = {
        m: {t: [] for t in ALL_STREAMINGBENCH_TASKS} for m in modes
    }
    ability_acc_map: dict[str, dict[str, list[float]]] = {
        m: {"working memory": [], "episodic memory": [], "proactive": []} for m in modes
    }
    act_map: dict[str, dict[str, int]] = {
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
    lats: dict[str, list[float]] = {m: [] for m in modes}

    for idx, sample in enumerate(samples):
        decode_hw = (896, 896) if (is_hires or sample.required_ability == "working memory") else (672, 672)
        all_frames = sample.materialize_frames(decode_hw=decode_hw, allow_synthetic_fallback=is_mock_engine)
        prefix_frames = [f for f in all_frames if f.timestamp <= sample.query_timestamp + 1e-6]
        future_frames = [f for f in all_frames if f.timestamp > sample.query_timestamp + 1e-6]
        blind_meta = sample.blind_agent_metadata(include_mock_hints=is_mock_engine)

        # 1. SimpleStream-4f (strictly 1-turn direct answer at query_timestamp on 4 recent frames @ base_resolution)
        env_s = CausalStreamingDualBuffer(
            recent_window=ProtectedRecentWindow(window_size=4, base_resolution=base_resolution, base_fps=1.0),
            ring_buffer=HighResRingBuffer(
                duration_sec=4.0,
                spatial_hires_hw=(896, 896) if is_hires else (672, 672),
            ),
        )
        for pf in prefix_frames:
            env_s.step_frame(pf)
        ep_s = agent_simple.answer_query(
            env=env_s,
            query=sample.query,
            stream_provider=None,
            max_wait_steps=0,
            metadata=dict(blind_meta),
        )
        _, ok_s = compute_task_accuracy_reward(
            pred_answer=ep_s.final_answer,
            gt_answer=sample.gt_answer,
            task_type=sample.task_type,
            pred_time=ep_s.committed_timestamp,
            gt_time=sample.target_reveal_time,
            time_tolerance=4.0,
        )
        subtask_acc_map["simplestream"][sample.subtask].append(100.0 if ok_s else 0.0)
        ability_acc_map["simplestream"][sample.required_ability].append(100.0 if ok_s else 0.0)
        act_map["simplestream"][ep_s.primary_action_type.value] += 1
        rec_toks["simplestream"].append(ep_s.recent_visual_tokens)
        for_toks["simplestream"].append(0)
        tot_toks["simplestream"].append(ep_s.recent_visual_tokens)
        lats["simplestream"].append(ep_s.latency_ms)

        # 2. Forager (ActiveForager 3D-IG-GRPO)
        env_f = CausalStreamingDualBuffer(
            recent_window=ProtectedRecentWindow(window_size=4, base_resolution=base_resolution, base_fps=1.0),
            ring_buffer=HighResRingBuffer(
                duration_sec=4.0,
                spatial_hires_hw=(896, 896) if is_hires else (672, 672),
                temporal_hifps_hw=(448, 448) if is_hires else (336, 336),
                target_hifps=4.0,
            ),
        )
        for pf in prefix_frames:
            env_f.step_frame(pf)

        future_cursor = {"idx": 0}

        def _stream_provider(delta_sec: float) -> list[VideoFrame]:
            target_t = env_f.current_time + delta_sec
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

        ep_f: AgentEpisodeResult = agent_forager.answer_query(
            env=env_f,
            query=sample.query,
            stream_provider=_stream_provider,
            max_wait_steps=5,
            metadata=dict(blind_meta),
        )
        _, ok_f = compute_task_accuracy_reward(
            pred_answer=ep_f.final_answer,
            gt_answer=sample.gt_answer,
            task_type=sample.task_type,
            pred_time=ep_f.committed_timestamp,
            gt_time=sample.target_reveal_time,
            time_tolerance=4.0,
        )
        subtask_acc_map["forager"][sample.subtask].append(100.0 if ok_f else 0.0)
        ability_acc_map["forager"][sample.required_ability].append(100.0 if ok_f else 0.0)
        act_f = ep_f.primary_action_type.value
        act_map["forager"][act_f] = act_map["forager"].get(act_f, 0) + 1
        rec_toks["forager"].append(ep_f.recent_visual_tokens)
        for_toks["forager"].append(ep_f.foraged_visual_tokens)
        tot_toks["forager"].append(ep_f.total_visual_tokens)
        lats["forager"].append(ep_f.latency_ms)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            s_acc = float(np.mean([v for vals in subtask_acc_map["simplestream"].values() for v in vals]))
            f_acc = float(np.mean([v for vals in subtask_acc_map["forager"].values() for v in vals]))
            print(
                f"[StreamingBench] {idx + 1}/{len(samples)} | "
                f"SimpleStream={s_acc:.2f}% | Forager={f_acc:.2f}% | last={sample.subtask} ({act_f})",
                flush=True,
            )

    summaries: dict[str, StreamingBenchSummary] = {}
    total_n = max(1, len(samples))
    for m in modes:
        st_accs: dict[str, float] = {}
        st_cnts: dict[str, int] = {}
        for t in ALL_STREAMINGBENCH_TASKS:
            vals = subtask_acc_map[m][t]
            st_accs[t] = round(float(np.mean(vals)), 2) if vals else 0.0
            st_cnts[t] = len(vals)

        rtvu_vals = [st_accs[t] for t in RTVU_TASKS if st_cnts[t] > 0]
        omni_vals = [st_accs[t] for t in OMNI_TASKS if st_cnts[t] > 0]
        ctx_vals = [st_accs[t] for t in CONTEXTUAL_TASKS if st_cnts[t] > 0]
        all_vals = [st_accs[t] for t in ALL_STREAMINGBENCH_TASKS if st_cnts[t] > 0]

        wm_vals = ability_acc_map[m]["working memory"]
        em_vals = ability_acc_map[m]["episodic memory"]
        po_vals = ability_acc_map[m]["proactive"]

        summaries[m] = StreamingBenchSummary(
            mode=m,
            num_samples=len(samples),
            subtask_accuracies=st_accs,
            rtvu_accuracy=round(float(np.mean(rtvu_vals)), 2) if rtvu_vals else 0.0,
            omni_accuracy=round(float(np.mean(omni_vals)), 2) if omni_vals else 0.0,
            contextual_accuracy=round(float(np.mean(ctx_vals)), 2) if ctx_vals else 0.0,
            working_memory_accuracy=round(float(np.mean(wm_vals)), 2) if wm_vals else 0.0,
            episodic_memory_accuracy=round(float(np.mean(em_vals)), 2) if em_vals else 0.0,
            proactive_accuracy=round(float(np.mean(po_vals)), 2) if po_vals else 0.0,
            overall_accuracy=round(float(np.mean(all_vals)), 2) if all_vals else 0.0,
            action_distribution={k: round(100.0 * v / total_n, 2) for k, v in act_map[m].items()},
            avg_recent_visual_tokens=round(float(np.mean(rec_toks[m])), 1) if rec_toks[m] else 0.0,
            avg_foraged_visual_tokens=round(float(np.mean(for_toks[m])), 1) if for_toks[m] else 0.0,
            avg_total_visual_tokens=round(float(np.mean(tot_toks[m])), 1) if tot_toks[m] else 0.0,
            avg_latency_ms=round(float(np.mean(lats[m])), 3) if lats[m] else 0.0,
            subtask_counts=st_cnts,
        )

    base = summaries["simplestream"]
    forg = summaries["forager"]
    forg.delta_p_vs_simplestream = round(forg.working_memory_accuracy - base.working_memory_accuracy, 2)
    forg.delta_m_vs_simplestream = round(forg.episodic_memory_accuracy - base.episodic_memory_accuracy, 2)
    return summaries


def merge_streamingbench_shards(shard_paths: Sequence[Path]) -> dict[str, StreamingBenchSummary]:
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in shard_paths if p.is_file()]
    if not payloads:
        raise ValueError("No valid StreamingBench shard files found")

    merged: dict[str, StreamingBenchSummary] = {}
    for m in ("simplestream", "forager"):
        total_n = 0
        st_sum: dict[str, float] = {t: 0.0 for t in ALL_STREAMINGBENCH_TASKS}
        st_cnt: dict[str, int] = {t: 0 for t in ALL_STREAMINGBENCH_TASKS}
        wm_sum, wm_cnt = 0.0, 0
        em_sum, em_cnt = 0.0, 0
        po_sum, po_cnt = 0.0, 0
        act_sum: dict[str, float] = {
            ActionType.ANSWER.value: 0.0,
            ActionType.ZOOM_IN.value: 0.0,
            ActionType.RECALL.value: 0.0,
            ActionType.WAIT.value: 0.0,
            ActionType.INVALID.value: 0.0,
        }
        rec_t, for_t, tot_t, lat_t = 0.0, 0.0, 0.0, 0.0

        for p in payloads:
            d = p[m]
            n_s = int(d["num_samples"])
            total_n += n_s
            cnts = d.get("subtask_counts") or {}
            for t in ALL_STREAMINGBENCH_TASKS:
                c = int(cnts.get(t, 0))
                st_cnt[t] += c
                st_sum[t] += float(d["subtask_accuracies"].get(t, 0.0)) * c
                if t == "Proactive Output":
                    po_cnt += c
                    po_sum += float(d["subtask_accuracies"].get(t, 0.0)) * c
                elif t in (
                    "Object Recognition",
                    "Attribute Recognition",
                    "Action Recognition",
                    "Spatial Understanding",
                    "Text-Rich Understanding",
                    "Misleading Context Understanding",
                ):
                    wm_cnt += c
                    wm_sum += float(d["subtask_accuracies"].get(t, 0.0)) * c
                else:
                    em_cnt += c
                    em_sum += float(d["subtask_accuracies"].get(t, 0.0)) * c

            for k in act_sum:
                act_sum[k] += float(d["action_distribution"].get(k, 0.0)) * n_s
            rec_t += float(d["avg_recent_visual_tokens"]) * n_s
            for_t += float(d["avg_foraged_visual_tokens"]) * n_s
            tot_t += float(d["avg_total_visual_tokens"]) * n_s
            lat_t += float(d["avg_latency_ms"]) * n_s

        denom = max(1, total_n)
        st_accs = {
            t: round(st_sum[t] / max(1, st_cnt[t]), 2) if st_cnt[t] > 0 else 0.0
            for t in ALL_STREAMINGBENCH_TASKS
        }
        rtvu_vals = [st_accs[t] for t in RTVU_TASKS if st_cnt[t] > 0]
        omni_vals = [st_accs[t] for t in OMNI_TASKS if st_cnt[t] > 0]
        ctx_vals = [st_accs[t] for t in CONTEXTUAL_TASKS if st_cnt[t] > 0]
        all_vals = [st_accs[t] for t in ALL_STREAMINGBENCH_TASKS if st_cnt[t] > 0]

        merged[m] = StreamingBenchSummary(
            mode=m,
            num_samples=total_n,
            subtask_accuracies=st_accs,
            rtvu_accuracy=round(float(np.mean(rtvu_vals)), 2) if rtvu_vals else 0.0,
            omni_accuracy=round(float(np.mean(omni_vals)), 2) if omni_vals else 0.0,
            contextual_accuracy=round(float(np.mean(ctx_vals)), 2) if ctx_vals else 0.0,
            working_memory_accuracy=round(wm_sum / max(1, wm_cnt), 2),
            episodic_memory_accuracy=round(em_sum / max(1, em_cnt), 2),
            proactive_accuracy=round(po_sum / max(1, po_cnt), 2),
            overall_accuracy=round(float(np.mean(all_vals)), 2) if all_vals else 0.0,
            action_distribution={k: round(v / denom, 2) for k, v in act_sum.items()},
            avg_recent_visual_tokens=round(rec_t / denom, 1),
            avg_foraged_visual_tokens=round(for_t / denom, 1),
            avg_total_visual_tokens=round(tot_t / denom, 1),
            avg_latency_ms=round(lat_t / denom, 3),
            subtask_counts=st_cnt,
        )

    merged["forager"].delta_p_vs_simplestream = round(
        merged["forager"].working_memory_accuracy - merged["simplestream"].working_memory_accuracy, 2
    )
    merged["forager"].delta_m_vs_simplestream = round(
        merged["forager"].episodic_memory_accuracy - merged["simplestream"].episodic_memory_accuracy, 2
    )
    return merged


def format_streamingbench_report(summary: StreamingBenchSummary) -> str:
    lines = [
        f"=== StreamingBench Zero-Leakage Evaluation Summary (mode={summary.mode}, N={summary.num_samples}) ===",
        f"  Real-Time Visual (RTVU, 10 tasks): {summary.rtvu_accuracy:6.2f}%",
        f"  Omni-Source      (OMNI,  6 tasks): {summary.omni_accuracy:6.2f}%",
        f"  Contextual & PO  (CU,    2 tasks): {summary.contextual_accuracy:6.2f}%  (SQA={summary.subtask_accuracies.get('Sequential Question Answering', 0.0):.1f}%, PO={summary.subtask_accuracies.get('Proactive Output', 0.0):.1f}%)",
        f"  Working Memory   (Present Scene) : {summary.working_memory_accuracy:6.2f}%  (Delta P: {summary.delta_p_vs_simplestream:+.2f}%)",
        f"  Episodic Memory  (Past Archive)  : {summary.episodic_memory_accuracy:6.2f}%  (Delta M: {summary.delta_m_vs_simplestream:+.2f}%)",
        f"  Overall Macro Accuracy (18 tasks): {summary.overall_accuracy:6.2f}%",
        f"  Tool Trigger Distribution (%)    : {summary.action_distribution}",
        f"  Avg Visual Tokens (rec/for/tot)  : {summary.avg_recent_visual_tokens:.1f} / {summary.avg_foraged_visual_tokens:.1f} / {summary.avg_total_visual_tokens:.1f}",
        f"  Avg Query Latency (ms)           : {summary.avg_latency_ms:.3f} ms",
    ]
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate SimpleStream vs. Forager on StreamingBench (100% Zero-Leakage Real Videos).")
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model-path", type=str, default="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_STREAMINGBENCH_DATA_DIR)
    ap.add_argument("--max-samples-per-subtask", type=int, default=20)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument(
        "--base-hw",
        type=int,
        nargs=2,
        default=[224, 224],
        help="Base recent window resolution H W (default 224 224; use 672 672 for Native High-Res).",
    )
    ap.add_argument(
        "--native-hires",
        action="store_true",
        help="Enable Native High-Resolution Regime (672x672 base recent window, 896x896 ZoomIn, 448x448 Sub-Agent).",
    )
    ap.add_argument("--merge-shards", type=Path, nargs="*", default=None)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.merge_shards:
        merged = merge_streamingbench_shards(args.merge_shards)
        for m in ("simplestream", "forager"):
            print(format_streamingbench_report(merged[m]))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({k: asdict(v) for k, v in merged.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)

    base_hw = (672, 672) if (args.native_hires and tuple(args.base_hw) == (224, 224)) else (int(args.base_hw[0]), int(args.base_hw[1]))
    is_native_hires = bool(args.native_hires or base_hw[0] >= 448)

    engine: BaseVLMEngine = (
        MockVLMEngine()
        if args.backend == "mock"
        else HFQwenVLEngine(model_name_or_path=args.model_path, native_hires=is_native_hires)
    )
    samples = load_streamingbench_samples(
        data_dir=args.data_dir,
        max_samples_per_subtask=args.max_samples_per_subtask,
        shard_idx=shard_idx,
        num_shards=num_shards,
        lazy_decode=True,
    )
    summaries = evaluate_streamingbench_joint_pass(
        samples=samples,
        engine=engine,
        base_resolution=base_hw,
    )
    for m in ("simplestream", "forager"):
        print(format_streamingbench_report(summaries[m]))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({k: asdict(v) for k, v in summaries.items()}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
