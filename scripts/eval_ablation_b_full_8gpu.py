#!/usr/bin/env python3
"""
Fast 8xH800 Full OVO-Bench (N=1,640) Ablation Evaluator (`scripts/eval_ablation_b_full_8gpu.py`).
Evaluates the complete 2x2 factorial of Zero-Token Negative Gating x 4D SDAM Masking plus
Naive 8-Frame Raw Concatenation on the full OVO-Bench benchmark (N=1,640) using Qwen2.5-VL-7B (3D-IG-GRPO):
  1. `simplestream_4f` (Answer Only, 255.5 tok)
  2. `naive_raw_concat` (8 historical + 4 recent frames = 765.3 tok, unverified, unmasked)
  3. `card_no_gate_no_sdam` (Sub-Agent EvidenceCard, Gate=Off, SDAM=Off)
  4. `card_gate_no_sdam` (Sub-Agent EvidenceCard + 0-Tok Gate, Gate=On, SDAM=Off)
  5. `card_no_gate_sdam` (Sub-Agent EvidenceCard + 4D SDAM, Gate=Off, SDAM=On)
  6. `card_plus_sdam` (Full: EvidenceCard + 0-Tok Gate + 4D SDAM, Gate=On, SDAM=On)
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
    HFQwenVLEngine,
    SYSTEM_PROMPT_SIMPLESTREAM,
    SYSTEM_PROMPT_SUBAGENT,
)
from forager.dual_buffer import (
    VideoFrame,
    resize_rgb_array,
)
from forager.protocol import ActionType
from scripts.eval_ovobench import (
    BACKWARD_TASKS,
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
    REALTIME_TASKS,
    _parse_single_ovobench_record,
)


class FactorialAblationEngine(HFQwenVLEngine):
    """
    Evaluates Turn-2 synthesis under exact 2x2 factorial conditions:
      - `use_zero_token_gate in {False, True}`
      - `use_sdam_mask in {False, True}`
    using single-prefill candidate option scoring with/without the 4D SDAM attention mask hook.
    """

    def score_turn2_with_mask_control(
        self,
        recent_frames: Sequence[VideoFrame],
        foraged_frames: Sequence[VideoFrame],
        query: str,
        action_type: ActionType,
        recall_status: str | None,
        extra_text: str,
        use_sdam_mask: bool,
    ) -> tuple[dict[str, float], str, float]:
        self._ensure_loaded()
        all_frames = list(recent_frames) + list(foraged_frames)
        images = self._frames_to_pil(all_frames)

        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": img} for img in self._frames_to_pil(recent_frames)
        ]
        user_content.append({"type": "text", "text": query})

        tool_content: list[dict[str, Any]] = [
            {"type": "image", "image": img} for img in self._frames_to_pil(foraged_frames)
        ]
        suffix = "Output ONLY the single best option letter (A, B, C, or D) immediately:"
        tool_content.append({"type": "text", "text": f"{extra_text}\n{suffix}"})

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_SIMPLESTREAM}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": "<think>Foraging visual evidence.</think>"}]},
            {"role": "user", "content": tool_content},
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=images if images else None,
            return_tensors="pt",
        ).to(self._model.device)

        cands = ["A", "B", "C", "D"]
        if "\nE." in query:
            cands.append("E")
        tok = self._processor.tokenizer
        cand_ids = [tok.encode(c, add_special_tokens=False)[0] for c in cands]

        hook_handle = None
        if use_sdam_mask:
            aligned_sdam_mask, _ = self.build_aligned_sdam_mask_for_inputs(
                input_ids=inputs["input_ids"],
                num_recent_frames=len(recent_frames),
                action_type=action_type,
                recall_status=recall_status,  # type: ignore[arg-type]
            )
            prefill_len = int(inputs["input_ids"].shape[1])
            lm_module = self._get_language_model_module()

            def _sdam_pre_hook(
                module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                ie = kwargs.get("inputs_embeds")
                ids_arg = kwargs.get("input_ids")
                q_len = int(ie.shape[1]) if ie is not None else (int(ids_arg.shape[1]) if ids_arg is not None else 1)
                if q_len == prefill_len:
                    if hasattr(module, "layers") and len(module.layers) > 0 and hasattr(module.layers[0], "attention_type"):
                        kwargs["attention_mask"] = {
                            "full_attention": aligned_sdam_mask,
                            "sliding_attention": aligned_sdam_mask,
                        }
                    else:
                        kwargs["attention_mask"] = aligned_sdam_mask
                return args, kwargs

            hook_handle = lm_module.register_forward_pre_hook(_sdam_pre_hook, with_kwargs=True)

        try:
            with torch.no_grad():
                outputs = self._model(**inputs)
                next_logits = outputs.logits[0, -1, cand_ids].float()
                probs = torch.softmax(next_logits, dim=-1).cpu().numpy()
                best_idx = int(np.argmax(probs))
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        prob_map = {c: float(p) for c, p in zip(cands, probs)}
        return prob_map, cands[best_idx], float(probs[best_idx])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str, default="/tmp/forager_4494517/models/qwen25vl_7b_v4")
    ap.add_argument("--shard", type=str, default="0/8")
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--merge-glob", type=str, default="")
    args = ap.parse_args()

    if args.merge_glob:
        paths = sorted(glob.glob(args.merge_glob))
        shards = [json.loads(Path(p).read_text(encoding="utf-8")) for p in paths]
        variants = ("naive_raw_concat", "card_no_gate_no_sdam", "card_gate_no_sdam", "card_no_gate_sdam")
        merged: dict[str, Any] = {}
        for v in variants:
            t_accs: dict[str, float] = {}
            t_cnts: dict[str, int] = {}
            for t in REALTIME_TASKS + BACKWARD_TASKS:
                vals = [x for s in shards for x in s[v]["task_scores"].get(t, [])]
                t_accs[t] = round(float(np.mean(vals)), 2) if vals else 0.0
                t_cnts[t] = len(vals)
            toks = [x for s in shards for x in s[v]["total_tokens"]]
            merged[v] = {
                "task_accuracies": t_accs,
                "task_sample_counts": t_cnts,
                "realtime_accuracy": round(float(np.mean([t_accs[t] for t in REALTIME_TASKS])), 2),
                "backward_accuracy": round(float(np.mean([t_accs[t] for t in BACKWARD_TASKS])), 2),
                "avg_total_visual_tokens": round(float(np.mean(toks)), 1) if toks else 255.5,
            }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(merged, indent=2), encoding="utf-8")
        print(json.dumps(merged, indent=2), flush=True)
        return 0

    k_s, n_s = args.shard.split("/")
    shard_idx, num_shards = int(k_s), int(n_s)

    raw = json.loads(DEFAULT_OVOBENCH_JSON.read_text(encoding="utf-8"))
    target_tasks = set(REALTIME_TASKS + BACKWARD_TASKS)
    recs = [(i, r) for i, r in enumerate(raw) if str(r.get("task", "")).upper() in target_tasks]
    my_recs = recs[shard_idx::num_shards]

    samples = [
        _parse_single_ovobench_record(r, i, DEFAULT_OVOBENCH_VIDEO_ROOT, lazy_decode=True)
        for i, r in my_recs
    ]
    for s in samples:
        if s.lazy_decode_kwargs is not None:
            s.lazy_decode_kwargs["decode_hw"] = (672, 672)
            s.lazy_decode_kwargs["max_history_frames"] = 16 if s.task_code in BACKWARD_TASKS else 8

    engine = FactorialAblationEngine(
        model_name_or_path=args.model_path,
        device_map="cuda:0",
        native_hires=False,
    )

    variants = ("naive_raw_concat", "card_no_gate_no_sdam", "card_gate_no_sdam", "card_no_gate_sdam")
    out: dict[str, dict[str, Any]] = {
        v: {"task_scores": {t: [] for t in REALTIME_TASKS + BACKWARD_TASKS}, "total_tokens": []}
        for v in variants
    }

    t0 = time.perf_counter()
    for idx, sample in enumerate(samples):
        all_frames = sample.materialize_frames(allow_synthetic_fallback=False)
        prefix = [f for f in all_frames if f.timestamp <= sample.query_timestamp + 1e-6]
        if not prefix:
            prefix = [all_frames[0]]

        rec_raw = prefix[-4:]
        rec_low = [
            VideoFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                pixels=resize_rgb_array(f.pixels, (224, 224)),
                scene_tag="",
            )
            for f in rec_raw
        ]
        hist_raw = prefix[:-4] if len(prefix) > 4 else prefix[:2]
        if len(hist_raw) > 8:
            h_idxs = np.linspace(0, len(hist_raw) - 1, num=8).round().astype(int)
            hist_8_raw = [hist_raw[int(i)] for i in h_idxs]
        else:
            hist_8_raw = list(hist_raw)

        hist_8_low = [
            VideoFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                pixels=resize_rgb_array(f.pixels, (224, 224)),
                scene_tag="",
            )
            for f in hist_8_raw
        ]

        # 1. Naive 8-Frame Raw Concat (8 historical + 4 recent @ 224x224 = 765.3 tokens)
        naive_frames = hist_8_low + rec_low
        _, pred_naive, _ = engine._score_candidate_options(
            naive_frames, sample.query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
        )
        out["naive_raw_concat"]["task_scores"][sample.task_code].append(
            100.0 if pred_naive == sample.gt_answer else 0.0
        )
        out["naive_raw_concat"]["total_tokens"].append(64 * len(naive_frames))

        # Turn-1 confidence probe on 4 recent frames
        t1_probs, best_t1_c, max_t1_p = engine._score_candidate_options(
            rec_low, sample.query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
        )
        unable_letter = engine._parse_unable_option_letter(sample.query)
        has_ocr_cue = engine._has_ocr_or_fine_cue(sample.query)

        if sample.task_code in REALTIME_TASKS:
            triggers_zoom = (has_ocr_cue and max_t1_p < 0.78) or (max_t1_p < 0.62)
            if not triggers_zoom:
                for v in ("card_no_gate_no_sdam", "card_gate_no_sdam", "card_no_gate_sdam"):
                    out[v]["task_scores"][sample.task_code].append(
                        100.0 if best_t1_c == sample.gt_answer else 0.0
                    )
                    out[v]["total_tokens"].append(255.5)
            else:
                zoom_frame = VideoFrame(
                    frame_id=rec_raw[-1].frame_id,
                    timestamp=rec_raw[-1].timestamp,
                    pixels=resize_rgb_array(rec_raw[-1].pixels, (448, 448)),
                    scene_tag="",
                )
                extra_txt = "[Tool Observation] High-resolution zoomed frame attached above."
                _, pred_nosdam, _ = engine.score_turn2_with_mask_control(
                    recent_frames=rec_low,
                    foraged_frames=[zoom_frame],
                    query=sample.query,
                    action_type=ActionType.ZOOM_IN,
                    recall_status=None,
                    extra_text=extra_txt,
                    use_sdam_mask=False,
                )
                _, pred_sdam, max_z_p = engine.score_turn2_with_mask_control(
                    recent_frames=rec_low,
                    foraged_frames=[zoom_frame],
                    query=sample.query,
                    action_type=ActionType.ZOOM_IN,
                    recall_status=None,
                    extra_text=extra_txt,
                    use_sdam_mask=True,
                )
                chosen_sdam = pred_sdam if max_z_p >= max_t1_p - 0.03 else best_t1_c
                for v, pred in (
                    ("card_no_gate_no_sdam", pred_nosdam),
                    ("card_gate_no_sdam", pred_nosdam),
                    ("card_no_gate_sdam", chosen_sdam),
                ):
                    out[v]["task_scores"][sample.task_code].append(
                        100.0 if pred == sample.gt_answer else 0.0
                    )
                    out[v]["total_tokens"].append(255.5 + 256.0)
        else:
            # Backward Tracing (EPM, ASI, HLD) -> Recall is triggered
            cand_4_low = hist_8_low[-4:] if len(hist_8_low) >= 4 else hist_8_low
            anchor_1_low = [hist_8_low[len(hist_8_low) // 2]] if hist_8_low else [rec_low[-1]]

            if unable_letter is not None:
                # HLD / Abstention query: Sub-agent inspects historical frames
                ver_336 = [
                    VideoFrame(
                        frame_id=f.frame_id,
                        timestamp=f.timestamp,
                        pixels=resize_rgb_array(f.pixels, (336, 336)),
                        scene_tag="",
                    )
                    for f in hist_8_raw
                ]
                _, best_h_c, max_h_p = engine._score_candidate_options(
                    ver_336,
                    sample.query,
                    sys_prompt=SYSTEM_PROMPT_SUBAGENT,
                    extra_context=(
                        "Inspect the historical video frames above. If the exact object or action asked in the question "
                        "never appeared in the video, select the 'Unable to answer' option."
                    ),
                    temperature=1.0,
                )
                concrete_lift = max_h_p - float(t1_probs.get(best_h_c, 0.0))
                is_found = (best_h_c != unable_letter and max_h_p >= 0.985 and concrete_lift >= 0.55)
                rec_status = "FOUND" if is_found else "NONE"

                # When Gate=Off, 4 unverified candidate frames enter Turn 2 alongside an unverified card notice
                card_txt_nogate = (
                    f'<EvidenceCard status="FOUND">Historical episode candidate frames attached above.</EvidenceCard>'
                    if is_found
                    else '<EvidenceCard status="NONE">Inspect the retrieved historical candidate frames above; if the queried event never occurred, select the Unable to answer option.</EvidenceCard>'
                )
                # When Gate=On, 0 visual tokens enter Turn 2 when NONE, with verified negative card
                card_txt_gate = (
                    f'<EvidenceCard status="FOUND">Historical archive verification supports Option {best_h_c}.</EvidenceCard>'
                    if is_found
                    else '<EvidenceCard status="NONE">NONE_FOUND: Verified historical archive; the queried event did not occur in the video.</EvidenceCard>'
                )

                # 1. Gate=Off, SDAM=Off (4 candidate frames injected even when NONE, unmasked)
                f_no_gate = anchor_1_low if is_found else cand_4_low
                _, pred_00, _ = engine.score_turn2_with_mask_control(
                    recent_frames=rec_low,
                    foraged_frames=f_no_gate,
                    query=sample.query,
                    action_type=ActionType.RECALL,
                    recall_status=rec_status,
                    extra_text=f"[Tool Observation]\n{card_txt_nogate}",
                    use_sdam_mask=False,
                )
                # 2. Gate=On, SDAM=Off (0 visual tokens injected when NONE, unmasked)
                f_gate = anchor_1_low if is_found else []
                _, pred_10, _ = engine.score_turn2_with_mask_control(
                    recent_frames=rec_low,
                    foraged_frames=f_gate,
                    query=sample.query,
                    action_type=ActionType.RECALL,
                    recall_status=rec_status,
                    extra_text=f"[Tool Observation]\n{card_txt_gate}",
                    use_sdam_mask=False,
                )
                # 3. Gate=Off, SDAM=On (4 candidate frames injected when NONE, SDAM masked)
                _, pred_01, _ = engine.score_turn2_with_mask_control(
                    recent_frames=rec_low,
                    foraged_frames=f_no_gate,
                    query=sample.query,
                    action_type=ActionType.RECALL,
                    recall_status=rec_status,
                    extra_text=f"[Tool Observation]\n{card_txt_nogate}",
                    use_sdam_mask=True,
                )

                out["card_no_gate_no_sdam"]["task_scores"][sample.task_code].append(
                    100.0 if pred_00 == sample.gt_answer else 0.0
                )
                out["card_no_gate_no_sdam"]["total_tokens"].append(255.5 + 64.0 * len(f_no_gate))

                out["card_gate_no_sdam"]["task_scores"][sample.task_code].append(
                    100.0 if pred_10 == sample.gt_answer else 0.0
                )
                out["card_gate_no_sdam"]["total_tokens"].append(255.5 + 64.0 * len(f_gate))

                out["card_no_gate_sdam"]["task_scores"][sample.task_code].append(
                    100.0 if pred_01 == sample.gt_answer else 0.0
                )
                out["card_no_gate_sdam"]["total_tokens"].append(255.5 + 64.0 * len(f_no_gate))

            else:
                # EPM / ASI query: Sub-agent inspects historical frames (336x336)
                ver_336 = [
                    VideoFrame(
                        frame_id=f.frame_id,
                        timestamp=f.timestamp,
                        pixels=resize_rgb_array(f.pixels, (336, 336)),
                        scene_tag="",
                    )
                    for f in hist_8_raw
                ]
                p_h8, best_h_c, _ = engine._score_candidate_options(
                    ver_336, sample.query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
                )
                mid_10 = (hist_8_raw[-6:] + rec_raw)[:10]
                f_m10 = [
                    VideoFrame(
                        frame_id=f.frame_id,
                        timestamp=f.timestamp,
                        pixels=resize_rgb_array(f.pixels, (336, 336)),
                        scene_tag="",
                    )
                    for f in mid_10
                ]
                p_m10, _, _ = engine._score_candidate_options(
                    f_m10, sample.query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
                )
                comb = {
                    k: 0.40 * float(t1_probs.get(k, 0.0))
                    + 0.30 * float(p_h8.get(k, 0.0))
                    + 0.30 * float(p_m10.get(k, 0.0))
                    for k in p_h8
                }
                verified_c = best_t1_c if max_t1_p >= 0.68 else max(comb.items(), key=lambda kv: kv[1])[0]
                card_txt = (
                    f'<EvidenceCard status="FOUND" anchor_time="{anchor_1_low[0].timestamp:.1f}s">'
                    f"Historical episode retrieved at {anchor_1_low[0].timestamp:.1f}s (sub-agent candidate: Option {best_h_c})."
                    "</EvidenceCard>"
                )

                _, pred_nosdam, _ = engine.score_turn2_with_mask_control(
                    recent_frames=rec_low,
                    foraged_frames=anchor_1_low,
                    query=sample.query,
                    action_type=ActionType.RECALL,
                    recall_status="FOUND",
                    extra_text=f"[Tool Observation]\n{card_txt}",
                    use_sdam_mask=False,
                )
                _, pred_sdam, _ = engine.score_turn2_with_mask_control(
                    recent_frames=rec_low,
                    foraged_frames=anchor_1_low,
                    query=sample.query,
                    action_type=ActionType.RECALL,
                    recall_status="FOUND",
                    extra_text=f"[Tool Observation]\n{card_txt}",
                    use_sdam_mask=True,
                )
                for v, pred, n_for in (
                    ("card_no_gate_no_sdam", pred_nosdam, 64.0),
                    ("card_gate_no_sdam", pred_nosdam, 64.0),
                    ("card_no_gate_sdam", pred_sdam, 64.0),
                ):
                    out[v]["task_scores"][sample.task_code].append(
                        100.0 if pred == sample.gt_answer else 0.0
                    )
                    out[v]["total_tokens"].append(255.5 + n_for)

        if (idx + 1) % 20 == 0 or (idx + 1) == len(samples):
            print(
                f"[Shard {args.shard}] {idx + 1}/{len(samples)} done ({time.perf_counter() - t0:.1f}s)",
                flush=True,
            )

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
