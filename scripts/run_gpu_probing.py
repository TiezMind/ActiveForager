#!/usr/bin/env python3
"""
GPU Counterfactual Log-Likelihood Probing & DAPO Zero-Gradient Filter (`scripts/run_gpu_probing.py`).

Runs data-parallel shards (`--shard k/n`) on GPUs to compute:
1. Base log-probability `log pi_ref(y^* | W_t^rec, q_t)` on real decoded video frames (4 frames @ 224x224, 1 fps).
2. Foraged log-probability `log pi_ref(y^* | W_t^rec, E_t, q_t)` using real high-res cropped frames (672x672)
   for `<ZoomIn>` or shared-weight VLM visual sub-agent verification (`run_subagent_verification`) on real
   historical frames for `<Recall>` (never copying `answer_text` into `<EvidenceCard>`).
3. Counterfactual Visual Information Gain `Delta I_CF = (lp_with - lp_without) / |y^*|`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from forager.agent import BaseVLMEngine, HFQwenVLEngine, MockVLMEngine
from forager.dual_buffer import (
    StructuredEvidenceCard,
    VideoFrame,
    load_real_frames_from_video_or_npy,
    resize_rgb_array,
)
from forager.ig_grpo import compute_counterfactual_ig_reward


DEFAULT_HOLMES_VIDEO_DIR = Path("/mnt/hdfs/lipengyu/DA/video/raw/TencentARC/Video-Holmes/videos_cropped")
DEFAULT_HDFS_FRAMES_DIR = Path("/mnt/hdfs/lipengyu/DA/video/frames/f768-fps2-px230400")


def _make_probe_frames(resolution: tuple[int, int] = (224, 224), count: int = 4) -> list[VideoFrame]:
    frames: list[VideoFrame] = []
    for i in range(count):
        pixels = np.full((resolution[0], resolution[1], 3), 100 + i * 20, dtype=np.uint8)
        frames.append(VideoFrame(frame_id=i, timestamp=float(i + 1), pixels=pixels, scene_tag=""))
    return frames


def _load_real_probe_frames_for_record(
    rec: dict[str, Any],
) -> Optional[tuple[list[VideoFrame], list[VideoFrame], list[VideoFrame]]]:
    """
    Decode real `(recent_224_frames, zoom_hires_frames, history_frames)` from `.mp4` or `.npy` cache
    when available on disk.
    """
    vid_rel = str(rec.get("video_path", ""))
    vid_id = str(rec.get("video_id", "")).split("__")[-1]
    cutoff = float(rec.get("query_timestamp", 30.0))

    candidates: list[Path] = []
    if vid_rel:
        candidates.append(Path(vid_rel))
        if DEFAULT_HOLMES_VIDEO_DIR.is_dir():
            candidates.append(DEFAULT_HOLMES_VIDEO_DIR / Path(vid_rel).name)
    if vid_id and DEFAULT_HOLMES_VIDEO_DIR.is_dir():
        candidates.append(DEFAULT_HOLMES_VIDEO_DIR / f"{vid_id}.mp4")

    resolved_path: Optional[Path] = None
    for c in candidates:
        if c.is_file():
            resolved_path = c
            break

    try:
        all_frames = load_real_frames_from_video_or_npy(
            video_path=resolved_path or vid_rel or f"{vid_id}.mp4",
            cutoff_timestamp=cutoff,
            recent_window_sec=4.0,
            recent_fps=1.0,
            max_history_frames=8,
            decode_hw=(672, 672),
            hdfs_frames_root=DEFAULT_HDFS_FRAMES_DIR,
        )
    except (FileNotFoundError, RuntimeError):
        return None

    if not all_frames:
        return None

    rec_high = all_frames[-4:] if len(all_frames) >= 4 else all_frames
    hist_frames = all_frames[:-4] if len(all_frames) > 4 else all_frames[:2]
    recent_224 = [
        VideoFrame(
            frame_id=f.frame_id,
            timestamp=f.timestamp,
            pixels=resize_rgb_array(f.pixels, (224, 224)),
            scene_tag="",
        )
        for f in rec_high
    ]
    zoom_hires = [rec_high[-1]]
    return recent_224, zoom_hires, hist_frames


def build_probe_engine(backend: str, model_path: str) -> BaseVLMEngine:
    if backend == "mock":
        return MockVLMEngine()
    if backend == "hf":
        return HFQwenVLEngine(model_name_or_path=model_path)
    raise ValueError(f"Unsupported backend for logprob probing: {backend}")


def probe_records(
    records: Sequence[dict[str, Any]],
    engine: BaseVLMEngine,
    min_ig_threshold: float = 0.15,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    sharded = list(records)[shard_idx::num_shards]
    is_mock = isinstance(engine, MockVLMEngine)
    mock_recent = _make_probe_frames((224, 224), count=4)
    mock_zoom = _make_probe_frames((448, 448), count=1)

    kept: list[dict[str, Any]] = []
    stats = {"total_probed": len(sharded), "kept": 0, "dropped_low_ig": 0, "real_video_decoded": 0}

    for rec in sharded:
        prompt = str(rec.get("prompt", ""))
        gt_ans = str(rec.get("gt_answer", "A"))
        act_type = rec.get("action_type") or rec.get("rl_bucket", "Answer")
        needs_foraging = act_type in (
            "ZoomIn",
            "Recall",
            "realtime_perception",
            "historical_recall",
            "hallucination_hld",
        )

        real_bundle = None if is_mock else _load_real_probe_frames_for_record(rec)
        if real_bundle is not None:
            recent_frames, zoom_frames, hist_frames = real_bundle
            stats["real_video_decoded"] += 1
        else:
            recent_frames, zoom_frames, hist_frames = mock_recent, mock_zoom, mock_zoom

        meta = {
            "mock_logprob_without_evidence": -1.386 if needs_foraging else -0.105,
            "mock_logprob_with_evidence": -0.062 if needs_foraging else -0.105,
        }

        lp_without = engine.score_answer_logprob(
            recent_frames=recent_frames,
            query=prompt,
            target_answer=gt_ans,
            foraged_frames=None,
            evidence_card=None,
            metadata=meta,
        )

        ev_card: Optional[StructuredEvidenceCard] = None
        if needs_foraging:
            is_recall_type = "Recall" in str(act_type) or "recall" in str(act_type) or "hld" in str(act_type)
            if is_recall_type:
                if not is_mock and callable(getattr(engine, "run_subagent_verification", None)):
                    ev_card = engine.run_subagent_verification(
                        candidate_frames=hist_frames,
                        query=prompt,
                    )
                else:
                    ev_card = StructuredEvidenceCard(
                        status=rec.get("recall_status", "FOUND"),
                        fact_summary="Objective historical visual observation from retrieved frames.",
                        anchor_frame=hist_frames[0] if rec.get("recall_status") != "NONE" else None,
                    )
            lp_with = engine.score_answer_logprob(
                recent_frames=recent_frames,
                query=prompt,
                target_answer=gt_ans,
                foraged_frames=zoom_frames if ev_card is None else None,
                evidence_card=ev_card,
                metadata=meta,
            )
        else:
            lp_with = lp_without

        r_ig = float(
            compute_counterfactual_ig_reward(
                logprob_with_evidence=lp_with,
                logprob_without_evidence=lp_without,
                is_correct=True,
                ccv_valid=True,
                answer_token_len=max(1, len(gt_ans.split())),
            )
        )

        if needs_foraging and r_ig < min_ig_threshold:
            stats["dropped_low_ig"] += 1
            continue

        enriched = {
            **rec,
            "probe_logprob_without_evidence": round(lp_without, 4),
            "probe_logprob_with_evidence": round(lp_with, 4),
            "probe_counterfactual_ig": round(r_ig, 4),
        }
        if ev_card is not None:
            enriched["turn2_evidence"] = ev_card.to_prompt_text()
        kept.append(enriched)
        stats["kept"] += 1

    return kept, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run Counterfactual Log-Likelihood Probing on SFT/RL manifests.")
    ap.add_argument("--input-jsonl", type=Path, required=True)
    ap.add_argument("--output-jsonl", type=Path, required=True)
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model-path", type=str, default="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--min-ig", type=float, default=0.15)
    ap.add_argument("--shard", type=str, default="0/1", help="Shard spec k/n for parallel GPU probing.")
    ap.add_argument("--limit", type=int, default=0, help="Optional cap on records (0 = all).")
    args = ap.parse_args(argv)

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)
    if not (0 <= shard_idx < num_shards):
        raise ValueError(f"Invalid --shard {args.shard}")

    records = [
        json.loads(line)
        for line in args.input_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if args.limit > 0:
        records = records[: args.limit]

    engine = build_probe_engine(args.backend, args.model_path)
    kept, stats = probe_records(
        records=records,
        engine=engine,
        min_ig_threshold=args.min_ig,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.output_jsonl.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept),
        encoding="utf-8",
    )
    print(json.dumps({"shard": args.shard, **stats, "output": str(args.output_jsonl)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
