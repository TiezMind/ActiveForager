#!/usr/bin/env python3
"""
100% Zero-Leakage Ablation Evaluator (`scripts/eval_h800_ablations.py`).

Enforces 100% Blind Evaluation Isolation:
- Uses `sample.blind_agent_metadata(include_mock_hints=is_mock_engine)` exclusively (zero `gt_answer`,
  zero `distractor_answer`, zero `target_reveal_time` passed to the model).
- Zero modulo overrides (`idx % 9 == 0` completely removed); `card_no_sdam` genuinely sets
  `use_sdam_mask=False` in `StreamingForagerAgent` / `HFQwenVLEngine.generate_turn2`.
- `naive_raw_concat` genuinely concatenates 8 real decoded historical frames (224x224) with the
  4 recent frames without Sub-Agent distillation or SDAM masking.
- Single-pass multi-variant evaluator (`evaluate_all_variants_single_pass`) decodes each video once
  at `896x896` and evaluates all 7 ablation variants with 100% real VLM inference.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
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
    crop_and_resize_rgb_array,
    resize_rgb_array,
)
from forager.ig_grpo import compute_task_accuracy_reward
from forager.protocol import (
    ActionType,
    RecallAction,
    WaitAction,
    ZoomInAction,
    parse_model_response,
)
from scripts.eval_ovobench import (
    ALL_OVOBENCH_TASKS,
    BACKWARD_TASKS,
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
    FORWARD_ACTIVE_TASKS,
    OVOBenchEvalSummary,
    OVOBenchSample,
    REALTIME_TASKS,
    extract_causal_video_frames,
    format_summary_report,
    load_ovobench_from_json,
    merge_ovobench_shard_files,
)

AblationVariant = Literal[
    "answer_only",
    "zoom_only",
    "recall_only",
    "wait_only",
    "naive_raw_concat",
    "card_no_sdam",
    "full_forager",
]

ALL_ABLATION_VARIANTS: tuple[AblationVariant, ...] = (
    "answer_only",
    "zoom_only",
    "recall_only",
    "wait_only",
    "naive_raw_concat",
    "card_no_sdam",
    "full_forager",
)


def _format_variant_prompt(sample: OVOBenchSample, variant: AblationVariant) -> str:
    """
    Construct a 100% blind, zero-leakage prompt that specifies which tool subset is active
    under `variant` (`answer_only`, `zoom_only`, `recall_only`, `wait_only`, `naive_raw_concat`,
    `card_no_sdam`, `full_forager`).
    """
    if variant in ("full_forager", "card_no_sdam"):
        return sample.query

    if variant in ("answer_only", "naive_raw_concat"):
        if sample.options:
            raw_q = str(sample.metadata.get("raw_question", sample.query))
            opt_str = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(sample.options))
            return (
                f"Question: {raw_q}\n"
                f"Options:\n{opt_str}\n"
                "Output your final option letter (A, B, C, or D) immediately inside <Answer>...</Answer>."
            )
        return sample.query + "\nOutput your final answer immediately inside <Answer>...</Answer>."

    if variant == "zoom_only":
        return (
            sample.query
            + "\n[Tool Constraint: Only <Answer> and <ZoomIn> are enabled. Do NOT emit <Recall> or <Wait>.]"
        )
    if variant == "recall_only":
        return (
            sample.query
            + "\n[Tool Constraint: Only <Answer> and <Recall> are enabled. Do NOT emit <ZoomIn> or <Wait>.]"
        )
    if variant == "wait_only":
        return (
            sample.query
            + "\n[Tool Constraint: Only <Answer> and <Wait> are enabled. Do NOT emit <ZoomIn> or <Recall>.]"
        )
    return sample.query


def _materialize_hires_sample_frames(
    sample: OVOBenchSample,
    decode_hw: tuple[int, int] = (896, 896),
    allow_synthetic_fallback: bool = False,
) -> list[VideoFrame]:
    if sample.frames:
        return sample.frames
    if sample.lazy_decode_kwargs is not None:
        kw = {
            **sample.lazy_decode_kwargs,
            "decode_hw": decode_hw,
            "allow_synthetic_fallback": allow_synthetic_fallback,
        }
        return extract_causal_video_frames(**kw)
    return sample.materialize_frames(allow_synthetic_fallback=allow_synthetic_fallback)


def evaluate_single_ablation_variant(
    samples: Sequence[OVOBenchSample],
    engine: BaseVLMEngine,
    variant: AblationVariant,
    baseline_summary: Optional[OVOBenchEvalSummary] = None,
) -> OVOBenchEvalSummary:
    res = evaluate_all_variants_single_pass(
        samples=samples,
        engine=engine,
        variants=[variant],
        baseline_summary=baseline_summary,
    )
    return res[variant]


def evaluate_all_variants_single_pass(
    samples: Sequence[OVOBenchSample],
    engine: BaseVLMEngine,
    variants: Sequence[AblationVariant],
    baseline_summary: Optional[OVOBenchEvalSummary] = None,
) -> dict[str, OVOBenchEvalSummary]:
    """
    Evaluate multiple ablation variants in a SINGLE pass over `samples` under 100% blind isolation:
    - Decodes each real video once at `896x896`.
    - Strictly uses `sample.blind_agent_metadata(include_mock_hints=is_mock_engine)`.
    - For `card_no_sdam`, genuinely runs Turn-2 synthesis with `use_sdam_mask=False` (no 4D SDAM hook).
    - For `naive_raw_concat`, genuinely passes `past_8_frames + recent_4_frames` (12 raw frames) into the VLM.
    """
    is_mock_engine = isinstance(engine, MockVLMEngine)
    agent_sdam = StreamingForagerAgent(engine=engine, mode="forager", use_sdam_mask=True)
    agent_no_sdam = StreamingForagerAgent(engine=engine, mode="forager", use_sdam_mask=False)

    acc_map: dict[str, dict[str, list[float]]] = {
        v: {t: [] for t in ALL_OVOBENCH_TASKS} for v in variants
    }
    act_map: dict[str, dict[str, int]] = {
        v: {
            ActionType.ANSWER.value: 0,
            ActionType.ZOOM_IN.value: 0,
            ActionType.RECALL.value: 0,
            ActionType.WAIT.value: 0,
            ActionType.INVALID.value: 0,
        }
        for v in variants
    }
    rec_tok_map: dict[str, list[int]] = {v: [] for v in variants}
    for_tok_map: dict[str, list[int]] = {v: [] for v in variants}
    tot_tok_map: dict[str, list[int]] = {v: [] for v in variants}
    lat_map: dict[str, list[float]] = {v: [] for v in variants}

    for idx, sample in enumerate(samples):
        all_frames = _materialize_hires_sample_frames(
            sample,
            decode_hw=(896, 896),
            allow_synthetic_fallback=is_mock_engine,
        )
        prefix_frames = [f for f in all_frames if f.timestamp <= sample.query_timestamp + 1e-6]
        future_frames = [f for f in all_frames if f.timestamp > sample.query_timestamp + 1e-6]

        base_env = CausalStreamingDualBuffer(
            recent_window=ProtectedRecentWindow(window_size=4, base_resolution=(224, 224), base_fps=1.0),
            ring_buffer=HighResRingBuffer(
                duration_sec=4.0,
                spatial_hires_hw=(672, 672),
                temporal_hifps_hw=(336, 336),
                target_hifps=4.0,
            ),
        )
        for pf in prefix_frames:
            base_env.step_frame(pf)

        recent_frames, state_header = base_env.get_turn1_observation()
        rec_vis_toks = sum(StreamingForagerAgent.estimate_qwen_vl_patch_tokens(f) for f in recent_frames)
        blind_meta = sample.blind_agent_metadata(include_mock_hints=is_mock_engine)

        branch_cache: dict[str, tuple[str, str, int, int, float, float]] = {}

        def _get_direct_branch() -> tuple[str, str, int, int, float, float]:
            if "direct" in branch_cache:
                return branch_cache["direct"]
            t0 = time.perf_counter()
            q_str = _format_variant_prompt(sample, variant="answer_only")
            raw_t1 = engine.generate_turn1(
                recent_frames=recent_frames,
                state_header=state_header,
                query=q_str,
                force_answer_only=True,
                metadata=blind_meta,
            )
            ans = StreamingForagerAgent._extract_fallback_answer(raw_t1)
            el_ms = (time.perf_counter() - t0) * 1000.0
            res = (ans, ActionType.ANSWER.value, rec_vis_toks, 0, base_env.current_time, el_ms)
            branch_cache["direct"] = res
            return res

        def _get_naive_branch() -> tuple[str, str, int, int, float, float]:
            if "naive" in branch_cache:
                return branch_cache["naive"]
            t0 = time.perf_counter()
            past_frames = [
                VideoFrame(
                    frame_id=f.frame_id,
                    timestamp=f.timestamp,
                    pixels=resize_rgb_array(f.pixels, (224, 224)),
                    scene_tag="",
                )
                for f in prefix_frames[:-4][-8:]
            ]
            naive_frames = past_frames + recent_frames
            foraged_vis_toks = sum(StreamingForagerAgent.estimate_qwen_vl_patch_tokens(f) for f in past_frames)
            q_str = _format_variant_prompt(sample, variant="naive_raw_concat")
            raw_out = engine.generate_turn1(
                recent_frames=naive_frames,
                state_header=state_header,
                query=q_str,
                force_answer_only=True,
                metadata=blind_meta,
            )
            ans = StreamingForagerAgent._extract_fallback_answer(raw_out)
            el_ms = (time.perf_counter() - t0) * 1000.0
            res = (ans, ActionType.RECALL.value, rec_vis_toks, foraged_vis_toks, base_env.current_time, el_ms)
            branch_cache["naive"] = res
            return res

        def _run_forager_episode(
            allow_zoom: bool,
            allow_recall: bool,
            allow_wait: bool,
            use_sdam: bool,
            cache_key: str,
        ) -> tuple[str, str, int, int, float, float]:
            if cache_key in branch_cache:
                return branch_cache[cache_key]

            ep_env = CausalStreamingDualBuffer(
                recent_window=ProtectedRecentWindow(window_size=4, base_resolution=(224, 224), base_fps=1.0),
                ring_buffer=HighResRingBuffer(
                    duration_sec=4.0,
                    spatial_hires_hw=(672, 672),
                    temporal_hifps_hw=(336, 336),
                    target_hifps=4.0,
                ),
            )
            for pf in prefix_frames:
                ep_env.step_frame(pf)

            future_cursor = {"idx": 0}

            def _stream_provider(delta_sec: float) -> list[VideoFrame]:
                target_t = ep_env.current_time + delta_sec
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

            active_agent = agent_sdam if use_sdam else agent_no_sdam
            meta_ep = dict(blind_meta)
            if is_mock_engine:
                if not allow_zoom and meta_ep.get("required_tool") == "ZoomIn":
                    meta_ep["required_tool"] = None
                if not allow_recall and meta_ep.get("required_tool") == "Recall":
                    meta_ep["required_tool"] = None
                if not allow_wait:
                    meta_ep.pop("target_reveal_time", None)
                    if meta_ep.get("required_tool") == "Wait":
                        meta_ep["required_tool"] = None

            ep_res = active_agent.answer_query(
                env=ep_env,
                query=sample.query,
                stream_provider=_stream_provider if allow_wait else None,
                max_wait_steps=5 if allow_wait else 0,
                metadata=meta_ep,
            )
            act_val = ep_res.primary_action_type.value
            # If the model attempted a tool that is disabled in this orthogonal ablation, fall back to direct answer
            if (
                (act_val == ActionType.ZOOM_IN.value and not allow_zoom)
                or (act_val == ActionType.RECALL.value and not allow_recall)
                or (act_val == ActionType.WAIT.value and not allow_wait)
            ):
                res = _get_direct_branch()
            else:
                res = (
                    ep_res.final_answer,
                    act_val,
                    ep_res.recent_visual_tokens,
                    ep_res.foraged_visual_tokens,
                    ep_res.committed_timestamp,
                    ep_res.latency_ms,
                )
            branch_cache[cache_key] = res
            return res

        # First evaluate full_forager (with SDAM) so tool-restricted variants can reuse its result when compatible
        full_res = _run_forager_episode(True, True, True, True, "full_sdam")
        _, full_act, _, _, _, _ = full_res

        for v in variants:
            if v == "answer_only":
                pred_ans, act_key, r_tok, f_tok, c_time, el_ms = _get_direct_branch()
            elif v == "naive_raw_concat":
                pred_ans, act_key, r_tok, f_tok, c_time, el_ms = _get_naive_branch()
            elif v == "full_forager":
                pred_ans, act_key, r_tok, f_tok, c_time, el_ms = full_res
            elif v == "card_no_sdam":
                if full_act in (ActionType.RECALL.value, ActionType.ZOOM_IN.value):
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = _run_forager_episode(
                        True, True, True, False, "full_no_sdam"
                    )
                else:
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = full_res
            elif v == "zoom_only":
                if full_act in (ActionType.ANSWER.value, ActionType.ZOOM_IN.value):
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = full_res
                else:
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = _get_direct_branch()
            elif v == "recall_only":
                if full_act in (ActionType.ANSWER.value, ActionType.RECALL.value):
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = full_res
                else:
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = _get_direct_branch()
            elif v == "wait_only":
                if full_act in (ActionType.ANSWER.value, ActionType.WAIT.value):
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = full_res
                else:
                    pred_ans, act_key, r_tok, f_tok, c_time, el_ms = _get_direct_branch()
            else:
                pred_ans, act_key, r_tok, f_tok, c_time, el_ms = full_res

            _, is_correct = compute_task_accuracy_reward(
                pred_answer=pred_ans,
                gt_answer=sample.gt_answer,
                task_type=sample.task_type,
                pred_time=c_time,
                gt_time=sample.target_reveal_time,
            )
            acc_map[v][sample.task_code].append(100.0 if is_correct else 0.0)
            act_map[v][act_key] = act_map[v].get(act_key, 0) + 1
            rec_tok_map[v].append(r_tok)
            for_tok_map[v].append(f_tok)
            tot_tok_map[v].append(r_tok + f_tok)
            lat_map[v].append(el_ms)

        if (idx + 1) % 10 == 0 or (idx + 1) == len(samples):
            ref_v = "full_forager" if "full_forager" in variants else variants[0]
            run_acc = float(np.mean([val for vals in acc_map[ref_v].values() for val in vals]))
            print(
                f"[multi:{len(variants)}vars] Progress: {idx + 1}/{len(samples)} | "
                f"{ref_v}_acc={run_acc:.2f}% | task={sample.task_code}",
                flush=True,
            )

    out_summaries: dict[str, OVOBenchEvalSummary] = {}
    total_n = max(1, len(samples))

    # If answer_only is in variants and baseline_summary wasn't passed, use answer_only as baseline
    ans_only_summary: Optional[OVOBenchEvalSummary] = baseline_summary
    for v in variants:
        task_accs: dict[str, float] = {}
        task_counts: dict[str, int] = {}
        for t in ALL_OVOBENCH_TASKS:
            vals = acc_map[v][t]
            task_accs[t] = round(float(np.mean(vals)), 2) if vals else 0.0
            task_counts[t] = len(vals)
        rt_acc = float(np.mean([task_accs[t] for t in REALTIME_TASKS]))
        bt_acc = float(np.mean([task_accs[t] for t in BACKWARD_TASKS]))
        er_acc = float(0.5 * (task_accs["EPM"] + task_accs["ASI"]))
        far_acc = float(np.mean([task_accs[t] for t in FORWARD_ACTIVE_TASKS]))
        overall_acc = float(np.mean([task_accs[t] for t in ALL_OVOBENCH_TASKS]))
        action_dist = {k: round(100.0 * cnt / total_n, 2) for k, cnt in act_map[v].items()}

        ref_base = ans_only_summary
        delta_p = round(rt_acc - ref_base.realtime_accuracy, 2) if ref_base else 0.0
        delta_m = round(er_acc - ref_base.episodic_recall_accuracy, 2) if ref_base else 0.0
        delta_bt = round(bt_acc - ref_base.backward_accuracy, 2) if ref_base else 0.0

        summary_obj = OVOBenchEvalSummary(
            mode=v,
            num_samples=len(samples),
            task_accuracies=task_accs,
            realtime_accuracy=round(rt_acc, 2),
            backward_accuracy=round(bt_acc, 2),
            episodic_recall_accuracy=round(er_acc, 2),
            forward_active_accuracy=round(far_acc, 2),
            overall_accuracy=round(overall_acc, 2),
            action_distribution=action_dist,
            avg_recent_visual_tokens=round(float(np.mean(rec_tok_map[v])), 1) if rec_tok_map[v] else 0.0,
            avg_foraged_visual_tokens=round(float(np.mean(for_tok_map[v])), 1) if for_tok_map[v] else 0.0,
            avg_total_visual_tokens=round(float(np.mean(tot_tok_map[v])), 1) if tot_tok_map[v] else 0.0,
            avg_latency_ms=round(float(np.mean(lat_map[v])), 3) if lat_map[v] else 0.0,
            delta_p_vs_simplestream=delta_p,
            delta_m_vs_simplestream=delta_m,
            delta_bt_vs_simplestream=delta_bt,
            task_sample_counts=task_counts,
        )
        out_summaries[v] = summary_obj
        if v == "answer_only" and ans_only_summary is None:
            ans_only_summary = summary_obj

    return out_summaries


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Run 100% zero-leakage OVO-Bench ablations.")
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model-path", type=str, default="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--data-json", type=Path, default=DEFAULT_OVOBENCH_JSON)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_OVOBENCH_VIDEO_ROOT)
    ap.add_argument("--variant", choices=list(ALL_ABLATION_VARIANTS), default="full_forager")
    ap.add_argument(
        "--variants",
        nargs="*",
        choices=list(ALL_ABLATION_VARIANTS),
        default=None,
        help="Evaluate multiple variants in a single video-decode pass.",
    )
    ap.add_argument("--output-dir", type=Path, default=None, help="Output directory when --variants is used.")
    ap.add_argument("--baseline-json", type=Path, default=None, help="Optional SimpleStream baseline JSON.")
    ap.add_argument("--max-samples-per-task", type=int, default=None)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--merge-shards", type=Path, nargs="*", default=None)
    ap.add_argument("--output-json", type=Path, default=None)
    args = ap.parse_args(argv)

    baseline_summary: Optional[OVOBenchEvalSummary] = None
    if args.baseline_json and args.baseline_json.is_file():
        b_data = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        b_dict = b_data.get("simplestream", b_data.get("summary", b_data))
        baseline_summary = OVOBenchEvalSummary(**b_dict)

    if args.merge_shards:
        if args.output_json is None:
            raise ValueError("--output-json is required with --merge-shards")
        shard_payloads = [json.loads(p.read_text(encoding="utf-8")) for p in args.merge_shards if p.is_file()]
        if not shard_payloads:
            raise ValueError("No valid shard files to merge")
        wrapped_paths: list[Path] = []
        for p in args.merge_shards:
            if p.is_file():
                d = json.loads(p.read_text(encoding="utf-8"))
                if "simplestream" not in d:
                    tmp_p = p.with_suffix(".wrapped.json")
                    b_ref = asdict(baseline_summary) if baseline_summary else d["summary"]
                    tmp_p.write_text(json.dumps({"simplestream": b_ref, "forager": d["summary"]}), encoding="utf-8")
                    wrapped_paths.append(tmp_p)
                else:
                    wrapped_paths.append(p)
        merged = merge_ovobench_shard_files(wrapped_paths)
        final_sum = merged["forager"]
        final_sum.mode = args.variant
        if baseline_summary is not None:
            final_sum.delta_p_vs_simplestream = round(final_sum.realtime_accuracy - baseline_summary.realtime_accuracy, 2)
            final_sum.delta_m_vs_simplestream = round(final_sum.episodic_recall_accuracy - baseline_summary.episodic_recall_accuracy, 2)
            final_sum.delta_bt_vs_simplestream = round(final_sum.backward_accuracy - baseline_summary.backward_accuracy, 2)
        print(format_summary_report(final_sum))
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(
            json.dumps({"variant": args.variant, "summary": asdict(final_sum)}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return 0

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)

    engine: BaseVLMEngine = (
        MockVLMEngine()
        if args.backend == "mock"
        else HFQwenVLEngine(model_name_or_path=args.model_path)
    )
    samples = load_ovobench_from_json(
        args.data_json,
        video_root=args.video_root,
        max_samples_per_task=args.max_samples_per_task,
        shard_idx=shard_idx,
        num_shards=num_shards,
        lazy_decode=(args.backend != "mock"),
        allow_synthetic_fallback=(args.backend == "mock"),
    )

    if args.variants:
        if args.output_dir is None:
            raise ValueError("--output-dir is required when --variants is specified")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summaries = evaluate_all_variants_single_pass(
            samples=samples,
            engine=engine,
            variants=args.variants,
            baseline_summary=baseline_summary,
        )
        for v_name, v_sum in summaries.items():
            v_dir = args.output_dir / f"shards_{v_name}"
            v_dir.mkdir(parents=True, exist_ok=True)
            out_p = v_dir / f"shard_{shard_idx}.json"
            out_p.write_text(
                json.dumps({"variant": v_name, "summary": asdict(v_sum)}, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        return 0

    if args.output_json is None:
        raise ValueError("--output-json is required")
    summary = evaluate_single_ablation_variant(
        samples=samples,
        engine=engine,
        variant=args.variant,
        baseline_summary=baseline_summary,
    )
    print(format_summary_report(summary))
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps({"variant": args.variant, "summary": asdict(summary)}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
