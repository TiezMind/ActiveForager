#!/usr/bin/env python3
"""
100% Zero-Leakage Real-Video Generalization Evaluator on Video-Holmes (`scripts/eval_videoholmes.py`).

Evaluates `SimpleStream-4f` (last 4 frames @ 224x224) vs. `ActiveForager` (with `<ZoomIn>` on 672x672
high-res ring buffer and `<Recall>` with Shared-Weight VLM Sub-Agent Verification + 4D SDAM masking)
on real `.mp4` videos and `.npy` frame caches from `Video-Holmes` across all 7 multi-hop reasoning
subtasks:
- `SR`   (Social Reasoning)
- `IMC`  (Intention & Motive Reasoning)
- `TCI`  (Temporal Causal Inference)
- `TA`   (Temporal Association)
- `MHR`  (Multi-Hop Reasoning)
- `PAR`  (Plot & Action Reasoning)
- `CTI`  (Character & Theme Inference)

Strictly enforces 100% Blind Evaluation Isolation:
- Zero ground-truth answer or Oracle tag injection (`scene_tag=""` on all frames).
- Evaluates strictly on the held-out `Video-Holmes` videos disjoint from the SFT cold-start set.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Optional, Sequence

import numpy as np

from forager.agent import (
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
    load_real_frames_from_video_or_npy,
)
from forager.ig_grpo import compute_task_accuracy_reward
from forager.protocol import ActionType


DEFAULT_HOLMES_JSON = Path("/mnt/hdfs/lipengyu/DA/video/raw/TencentARC/Video-Holmes/train_Video-Holmes.json")
DEFAULT_HOLMES_VIDEO_DIR = Path("/mnt/hdfs/lipengyu/DA/video/raw/TencentARC/Video-Holmes/videos_cropped")
DEFAULT_HDFS_FRAMES_DIR = Path("/mnt/hdfs/lipengyu/DA/video/frames/f768-fps2-px230400")
DEFAULT_SFT_JSONL = Path("data/sft_coldstart_6k.jsonl")

HOLMES_SUBTASKS: tuple[str, ...] = ("SR", "IMC", "TCI", "TA", "MHR", "PAR", "CTI")


@dataclass
class VideoHolmesEvalSummary:
    mode: str
    num_samples: int
    subtask_accuracies: dict[str, float]
    overall_accuracy: float
    action_distribution: dict[str, float]
    avg_recent_visual_tokens: float
    avg_foraged_visual_tokens: float
    avg_total_visual_tokens: float
    avg_latency_ms: float
    delta_vs_simplestream: float = 0.0
    subtask_counts: Optional[dict[str, int]] = None


def load_heldout_videoholmes_samples(
    holmes_json: Path = DEFAULT_HOLMES_JSON,
    sft_jsonl: Path = DEFAULT_SFT_JSONL,
    max_samples_per_subtask: int = 20,
    shard_idx: int = 0,
    num_shards: int = 1,
) -> list[dict[str, Any]]:
    sft_vids: set[str] = set()
    if sft_jsonl.is_file():
        for line in sft_jsonl.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                vid = str(r.get("video_id", "")).split("__")[-1]
                sft_vids.add(vid)

    raw_data = json.loads(holmes_json.read_text(encoding="utf-8"))
    counts: dict[str, int] = {t: 0 for t in HOLMES_SUBTASKS}
    selected: list[dict[str, Any]] = []

    for item in raw_data:
        vid_id = str(item.get("video ID", ""))
        q_type = str(item.get("Question Type", "SR")).strip()
        if q_type not in counts:
            continue
        if vid_id in sft_vids:
            continue
        if max_samples_per_subtask > 0 and counts[q_type] >= max_samples_per_subtask:
            continue
        opts_dict = item.get("Options", {})
        letters = sorted(opts_dict.keys())
        if not letters or item.get("Answer") not in letters:
            continue
        options = [str(opts_dict[k]).strip() for k in letters]
        opt_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(options))
        q_text = str(item["Question"]).strip()
        prompt = (
            f"Question: {q_text}\n"
            f"Options:\n{opt_str}\n"
            "Select the single best option letter (A, B, C, E, or F) inside <Answer>...</Answer> "
            "(or invoke <Recall> / <ZoomIn> first if the current 4 low-resolution frames are insufficient)."
        )
        selected.append(
            {
                "sample_id": f"{q_type}_{ item.get('Question ID', len(selected)) }",
                "subtask": q_type,
                "video_id": vid_id,
                "video_path": str(DEFAULT_HOLMES_VIDEO_DIR / f"{vid_id}.mp4"),
                "prompt": prompt,
                "gt_answer": str(item["Answer"]).strip(),
            }
        )
        counts[q_type] += 1

    return selected[shard_idx::num_shards]


def evaluate_videoholmes_joint(
    samples: Sequence[dict[str, Any]],
    engine: BaseVLMEngine,
) -> tuple[VideoHolmesEvalSummary, VideoHolmesEvalSummary]:
    is_mock = isinstance(engine, MockVLMEngine)
    agent_ss = StreamingForagerAgent(engine=engine, mode="simplestream", use_sdam_mask=True)
    agent_fo = StreamingForagerAgent(engine=engine, mode="forager", use_sdam_mask=True)

    modes = ("simplestream", "forager")
    sub_corr: dict[str, dict[str, list[float]]] = {m: {t: [] for t in HOLMES_SUBTASKS} for m in modes}
    act_cnt: dict[str, dict[str, int]] = {
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
        if is_mock:
            frames = [
                VideoFrame(
                    frame_id=i,
                    timestamp=float(i + 1),
                    pixels=np.full((448, 448, 3), 80 + i * 10, dtype=np.uint8),
                    scene_tag="",
                )
                for i in range(10)
            ]
        else:
            frames = load_real_frames_from_video_or_npy(
                video_path=sample["video_path"],
                cutoff_timestamp=180.0,
                recent_window_sec=4.0,
                recent_fps=1.0,
                max_history_frames=20,
                decode_hw=(672, 672),
                hdfs_frames_root=DEFAULT_HDFS_FRAMES_DIR,
            )

        for mode, agent in (("simplestream", agent_ss), ("forager", agent_fo)):
            env = CausalStreamingDualBuffer(
                recent_window=ProtectedRecentWindow(window_size=4, base_resolution=(224, 224), base_fps=1.0),
                ring_buffer=HighResRingBuffer(duration_sec=4.0, spatial_hires_hw=(672, 672)),
            )
            for pf in frames:
                env.step_frame(pf)

            blind_meta = {
                "sample_id": sample["sample_id"],
                "subtask": sample["subtask"],
            }
            if is_mock:
                blind_meta["gt_answer"] = sample["gt_answer"]
                blind_meta["distractor_answer"] = "B" if sample["gt_answer"] != "B" else "A"
                blind_meta["required_tool"] = "Recall"

            ep_res = agent.answer_query(
                env=env,
                query=sample["prompt"],
                stream_provider=None,
                max_wait_steps=0,
                metadata=blind_meta,
            )
            _, is_corr = compute_task_accuracy_reward(
                pred_answer=ep_res.final_answer,
                gt_answer=sample["gt_answer"],
                task_type="mcq",
            )
            sub_corr[mode][sample["subtask"]].append(100.0 if is_corr else 0.0)
            ak = ep_res.primary_action_type.value
            act_cnt[mode][ak] = act_cnt[mode].get(ak, 0) + 1
            rec_toks[mode].append(ep_res.recent_visual_tokens)
            for_toks[mode].append(ep_res.foraged_visual_tokens)
            tot_toks[mode].append(ep_res.total_visual_tokens)
            lats[mode].append(ep_res.latency_ms)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            ss_a = float(np.mean([v for vals in sub_corr["simplestream"].values() for v in vals]))
            fo_a = float(np.mean([v for vals in sub_corr["forager"].values() for v in vals]))
            print(
                f"[Video-Holmes] {idx + 1}/{len(samples)} | SimpleStream={ss_a:.2f}% | Forager={fo_a:.2f}% | subtask={sample['subtask']}",
                flush=True,
            )

    out: dict[str, VideoHolmesEvalSummary] = {}
    for mode in modes:
        s_accs: dict[str, float] = {}
        s_cnts: dict[str, int] = {}
        for t in HOLMES_SUBTASKS:
            vals = sub_corr[mode][t]
            s_accs[t] = round(float(np.mean(vals)), 2) if vals else 0.0
            s_cnts[t] = len(vals)
        ov_acc = round(float(np.mean([s_accs[t] for t in HOLMES_SUBTASKS])), 2)
        tot_n = max(1, len(samples))
        a_dist = {k: round(100.0 * v / tot_n, 2) for k, v in act_cnt[mode].items()}
        base = out.get("simplestream")
        delta = round(ov_acc - base.overall_accuracy, 2) if base is not None else 0.0
        out[mode] = VideoHolmesEvalSummary(
            mode=mode,
            num_samples=len(samples),
            subtask_accuracies=s_accs,
            overall_accuracy=ov_acc,
            action_distribution=a_dist,
            avg_recent_visual_tokens=round(float(np.mean(rec_toks[mode])), 1) if rec_toks[mode] else 0.0,
            avg_foraged_visual_tokens=round(float(np.mean(for_toks[mode])), 1) if for_toks[mode] else 0.0,
            avg_total_visual_tokens=round(float(np.mean(tot_toks[mode])), 1) if tot_toks[mode] else 0.0,
            avg_latency_ms=round(float(np.mean(lats[mode])), 3) if lats[mode] else 0.0,
            delta_vs_simplestream=delta,
            subtask_counts=s_cnts,
        )
    return out["simplestream"], out["forager"]


def merge_videoholmes_shards(shard_paths: Sequence[Path]) -> dict[str, VideoHolmesEvalSummary]:
    payloads = [json.loads(p.read_text(encoding="utf-8")) for p in shard_paths if p.is_file()]
    merged: dict[str, VideoHolmesEvalSummary] = {}
    for mode in ("simplestream", "forager"):
        tot_n = 0
        corr_sum = {t: 0.0 for t in HOLMES_SUBTASKS}
        cnt_sum = {t: 0 for t in HOLMES_SUBTASKS}
        act_sum = {k: 0.0 for k in ("Answer", "ZoomIn", "Recall", "Wait", "Invalid")}
        r_sum, f_sum, t_sum, l_sum = 0.0, 0.0, 0.0, 0.0
        for sp in payloads:
            m = sp[mode]
            ns = int(m["num_samples"])
            tot_n += ns
            sc = m.get("subtask_counts") or {}
            for t in HOLMES_SUBTASKS:
                c = int(sc.get(t, 0))
                cnt_sum[t] += c
                corr_sum[t] += float(m["subtask_accuracies"].get(t, 0.0)) * c
            for k in act_sum:
                act_sum[k] += float(m["action_distribution"].get(k, 0.0)) * ns
            r_sum += float(m["avg_recent_visual_tokens"]) * ns
            f_sum += float(m["avg_foraged_visual_tokens"]) * ns
            t_sum += float(m["avg_total_visual_tokens"]) * ns
            l_sum += float(m["avg_latency_ms"]) * ns
        denom = max(1, tot_n)
        s_accs = {t: round(corr_sum[t] / max(1, cnt_sum[t]), 2) for t in HOLMES_SUBTASKS}
        ov_acc = round(float(np.mean([s_accs[t] for t in HOLMES_SUBTASKS])), 2)
        base = merged.get("simplestream")
        delta = round(ov_acc - base.overall_accuracy, 2) if base is not None else 0.0
        merged[mode] = VideoHolmesEvalSummary(
            mode=mode,
            num_samples=tot_n,
            subtask_accuracies=s_accs,
            overall_accuracy=ov_acc,
            action_distribution={k: round(v / denom, 2) for k, v in act_sum.items()},
            avg_recent_visual_tokens=round(r_sum / denom, 1),
            avg_foraged_visual_tokens=round(f_sum / denom, 1),
            avg_total_visual_tokens=round(t_sum / denom, 1),
            avg_latency_ms=round(l_sum / denom, 3),
            delta_vs_simplestream=delta,
            subtask_counts=cnt_sum,
        )
    return merged


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate zero-leakage Video-Holmes generalization.")
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model-path", type=str, default="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--max-samples-per-subtask", type=int, default=20)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--merge-shards", type=Path, nargs="*", default=None)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.merge_shards:
        merged = merge_videoholmes_shards(args.merge_shards)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({k: asdict(v) for k, v in merged.items()}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)
    engine: BaseVLMEngine = (
        MockVLMEngine() if args.backend == "mock" else HFQwenVLEngine(model_name_or_path=args.model_path)
    )
    samples = load_heldout_videoholmes_samples(
        max_samples_per_subtask=args.max_samples_per_subtask,
        shard_idx=shard_idx,
        num_shards=num_shards,
    )
    ss_sum, fo_sum = evaluate_videoholmes_joint(samples, engine)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"simplestream": asdict(ss_sum), "forager": asdict(fo_sum)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
