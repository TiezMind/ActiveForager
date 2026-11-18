#!/usr/bin/env python3
"""
Fast 8-GPU Single-Decode Backward Tracing (`EPM`, `ASI`, `HLD`) & Multi-Resolution Evaluator.
Decodes each video ONCE per shard and evaluates both `672^2 Native Hi-Res` and `224^2 Budgeted`
modes under 100% blind zero-leakage isolation with zero duplicate video decoding.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

from forager.agent import StreamingForagerAgent
from forager.dual_buffer import (
    CausalStreamingDualBuffer,
    HighResRingBuffer,
    ProtectedRecentWindow,
)
from forager.ig_grpo import compute_task_accuracy_reward
from scripts.eval_ovobench import (
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
    _parse_single_ovobench_record,
    build_engine,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="")
    ap.add_argument("--shard", type=str, default="0/8")
    ap.add_argument("--tasks", type=str, nargs="*", default=["EPM", "HLD"])
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--merge-glob", type=str, default="")
    args = ap.parse_args()

    if args.merge_glob:
        paths = sorted(glob.glob(args.merge_glob))
        shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
        merged: dict[str, dict[str, float]] = {"hires": {}, "lowres": {}}
        for mode in ("hires", "lowres"):
            for t in args.tasks:
                num = sum(
                    float(s[mode]["task_accuracies"].get(t, 0.0))
                    * int((s[mode].get("task_sample_counts") or {}).get(t, 0))
                    for s in shards
                )
                den = sum(int((s[mode].get("task_sample_counts") or {}).get(t, 0)) for s in shards)
                merged[mode][t] = round(num / max(1, den), 2)
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(json.dumps(merged, indent=2), flush=True)
        return 0

    k_s, n_s = args.shard.split("/")
    shard_idx, num_shards = int(k_s), int(n_s)
    target_tasks = [t.upper() for t in args.tasks]

    raw = json.loads(DEFAULT_OVOBENCH_JSON.read_text(encoding="utf-8"))
    bt_recs = [(i, r) for i, r in enumerate(raw) if str(r.get("task", "")).upper() in target_tasks]
    my_recs = bt_recs[shard_idx::num_shards]

    samples = [
        _parse_single_ovobench_record(r, i, DEFAULT_OVOBENCH_VIDEO_ROOT, lazy_decode=True)
        for i, r in my_recs
    ]
    for s in samples:
        if s.lazy_decode_kwargs is not None:
            s.lazy_decode_kwargs["decode_hw"] = (672, 672)
            s.lazy_decode_kwargs["max_history_frames"] = 24

    engine = build_engine("hf", args.model_path, native_hires=True)
    agent = StreamingForagerAgent(engine=engine, mode="forager", use_sdam_mask=True)

    task_correct: dict[str, dict[str, list[float]]] = {
        "hires": {t: [] for t in target_tasks},
        "lowres": {t: [] for t in target_tasks},
    }

    for idx, sample in enumerate(samples):
        all_frames = sample.materialize_frames(allow_synthetic_fallback=False)
        prefix_frames = [f for f in all_frames if f.timestamp <= sample.query_timestamp + 1e-6]
        blind_meta = sample.blind_agent_metadata(include_mock_hints=False)

        for mode_key, is_hi, base_hw in (
            ("hires", True, (672, 672)),
            ("lowres", False, (224, 224)),
        ):
            engine.native_hires = is_hi
            env = CausalStreamingDualBuffer(
                recent_window=ProtectedRecentWindow(window_size=4, base_resolution=base_hw, base_fps=1.0),
                ring_buffer=HighResRingBuffer(
                    duration_sec=4.0,
                    spatial_hires_hw=(672, 672),
                    temporal_hifps_hw=(448, 448) if is_hi else (336, 336),
                    target_hifps=4.0,
                ),
            )
            for pf in prefix_frames:
                env.step_frame(pf)

            ep_res = agent.answer_query(
                env=env,
                query=sample.query,
                max_wait_steps=5,
                metadata=dict(blind_meta),
            )
            _, is_corr = compute_task_accuracy_reward(
                pred_answer=ep_res.final_answer,
                gt_answer=sample.gt_answer,
                task_type=sample.task_type,
                pred_time=ep_res.committed_timestamp,
                gt_time=sample.target_reveal_time,
            )
            task_correct[mode_key][sample.task_code].append(100.0 if is_corr else 0.0)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            print(f"[Shard {args.shard}] {idx + 1}/{len(samples)} done", flush=True)

    out_payload = {"hires": {}, "lowres": {}}
    for mode_key in ("hires", "lowres"):
        t_accs = {
            t: round(float(np.mean(task_correct[mode_key][t])), 2) if task_correct[mode_key][t] else 0.0
            for t in target_tasks
        }
        t_cnts = {t: len(task_correct[mode_key][t]) for t in target_tasks}
        out_payload[mode_key] = {"task_accuracies": t_accs, "task_sample_counts": t_cnts}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out_payload, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
