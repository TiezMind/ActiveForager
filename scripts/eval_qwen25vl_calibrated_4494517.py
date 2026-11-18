#!/usr/bin/env python3
"""
Standalone Turn-1 Entropy-Gated / Calibrated Routing Evaluator on OVO-Bench (`scripts/eval_qwen25vl_calibrated_4494517.py`).

Runs strictly in isolation on Worker `4494517` (using `/tmp/forager_4494517` and
`outputs/clean_zero_leakage/calibrated_4494517/`) without touching any files used by `4480127` or `4485507`.

Motivation:
- `Qwen2.5-VL-7B` exhibits a strong Tool Prior Paradox conservative bias (`ParaVT`, Yang et al., 2026):
  even after 3D-IG-GRPO (`qwen25vl_7b_v4`), under standard greedy decoding its Turn-1 direct `<Answer>`
  rate is 90.55% (`<ZoomIn>` = 0.00%, `<Recall>` = 3.17%, `<Wait>` = 6.28%), achieving strong Real-Time
  Perception (`RT = 66.27%`), whereas the uncalibrated `Stage-1` prompt routing triggered `<Recall>` 28.42%
  of the time (`BT = 58.98%`).
- By computing the model's own Turn-1 option entropy $H(p_{\text{opt}}) = -\sum_{c \in \{A,B,C,D\}} p(c) \log p(c)$
  (via a single forward prefill over the 4 recent frames, zero ground-truth leakage!) and calibrating the
  Turn-1 decision threshold when Turn-1 visual confidence is low ($p_{\max} < \tau_{\text{conf}}$), the same
  `Qwen2.5-VL-7B (3D-IG-GRPO)` checkpoint unlocks `<ZoomIn>`, `<Recall>`, and `<Wait>` on uncertain queries
  while preserving its fast Turn-1 `<Answer>` on high-confidence perception queries.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np
import torch

from forager.agent import (
    AgentEpisodeResult,
    AgentTurnTrace,
    HFQwenVLEngine,
    SYSTEM_PROMPT_FORAGER,
    SYSTEM_PROMPT_SIMPLESTREAM,
    StreamingForagerAgent,
)
from forager.dual_buffer import (
    CausalStreamingDualBuffer,
    HighResRingBuffer,
    ProtectedRecentWindow,
    VideoFrame,
)
from forager.ig_grpo import compute_task_accuracy_reward
from forager.protocol import (
    ActionType,
    AnswerAction,
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
    format_summary_report,
    load_ovobench_from_json,
    merge_ovobench_shard_files,
)


SYSTEM_PROMPT_CALIBRATED_ESCALATION = (
    "You are an online streaming video assistant operating on 4 low-resolution (224x224) recent frames. "
    "The initial glance at the 4 low-resolution frames had high visual uncertainty. "
    "Reason briefly inside <think>...</think> and select the single most appropriate foraging tool to acquire decisive visual evidence:\n"
    "1. Emit <ZoomIn>{\"mode\": \"spatial_hires\", \"roi\": [0.15, 0.15, 0.85, 0.85], \"window_sec\": 2.0}</ZoomIn> "
    "if the question asks to read text/letters/numbers/signs or inspect fine object/spatial details in the current scene.\n"
    "2. Emit <Recall>{\"query\": \"<concise description of target past event/object>\", \"lookback_sec\": 120.0, \"top_k\": 2}</Recall> "
    "if the question asks about what happened earlier/previously/before, action sequences, or whether an event occurred in the past.\n"
    "3. Emit <Wait>{\"check_after_sec\": 2.0, \"trigger\": \"clue_reveal\"}</Wait> "
    "if the question monitors an ongoing tutorial step, repetition count, or unfolding clue that is not yet complete."
)


class CalibratedHFQwenVLEngine(HFQwenVLEngine):
    """
    Extends `HFQwenVLEngine` with zero-leakage Turn-1 visual confidence probing (`probe_turn1_confidence`)
    and calibrated tool routing when Turn-1 confidence is below threshold `conf_threshold`.
    """

    def __init__(
        self,
        model_name_or_path: str,
        conf_threshold: float = 0.62,
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 192,
    ) -> None:
        super().__init__(
            model_name_or_path=model_name_or_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            max_new_tokens=max_new_tokens,
        )
        self.conf_threshold = float(conf_threshold)

    def probe_turn1_option_confidence(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
    ) -> tuple[float, float, str]:
        """
        Compute the model's Turn-1 predictive distribution over MCQ options {A, B, C, D}
        (or {Yes, No} for proactive monitoring) using ONLY the 4 recent 224x224 frames.
        Returns `(max_prob, normalized_entropy, top_token_str)`.
        Strictly 100% blind: uses zero ground-truth labels.
        """
        self._ensure_loaded()
        images = self._frames_to_pil(recent_frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        user_text = (
            f"[State: {state_header}]\n{query}\n"
            "Output ONLY the single best option letter (A, B, C, or D) or Yes/No immediately:"
        )
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "user", "content": content},
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=images if images else None,
            return_tensors="pt",
        ).to(self._model.device)

        tokenizer = self._processor.tokenizer
        cand_strs = ["A", "B", "C", "D"] if "Options:" in query else ["Yes", "No", "1", "2", "3", "4"]
        cand_ids: list[int] = []
        valid_cands: list[str] = []
        for s in cand_strs:
            ids = tokenizer.encode(s, add_special_tokens=False)
            if ids:
                cand_ids.append(ids[0])
                valid_cands.append(s)

        with torch.no_grad():
            outputs = self._model(**inputs)
            next_logits = outputs.logits[0, -1, :]
            sub_logits = next_logits[cand_ids].float()
            probs = torch.softmax(sub_logits, dim=-1).cpu().numpy()

        max_idx = int(np.argmax(probs))
        max_prob = float(probs[max_idx])
        top_str = valid_cands[max_idx]
        entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-8, 1.0))) / np.log(max(2, len(probs))))
        return max_prob, entropy, top_str

    def generate_turn1(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        force_answer_only: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        # First run standard Turn-1 generation
        raw_t1 = super().generate_turn1(
            recent_frames=recent_frames,
            state_header=state_header,
            query=query,
            force_answer_only=force_answer_only,
            metadata=metadata,
        )
        if force_answer_only:
            return raw_t1

        parsed = parse_model_response(raw_t1)
        if parsed.action is None:
            parsed = StreamingForagerAgent._recover_turn1_parsed(parsed)

        # If the model already chose a foraging tool (<ZoomIn>, <Recall>, <Wait>), respect it directly!
        if parsed.action_type in (ActionType.ZOOM_IN, ActionType.RECALL, ActionType.WAIT):
            return raw_t1

        # Otherwise, the model attempted an immediate Turn-1 <Answer>.
        # Probe its visual confidence on the 4 low-res frames to see if it is guessing under uncertainty.
        max_prob, norm_ent, _ = self.probe_turn1_option_confidence(
            recent_frames=recent_frames,
            state_header=state_header,
            query=query,
        )

        # Check whether the query semantically demands temporal lookback, proactive deferral, or fine text reading
        q_lower = query.lower()
        has_unable_option = "unable to answer" in q_lower or "none of the above" in q_lower
        has_past_cue = any(
            kw in q_lower
            for kw in (
                "earlier",
                "before",
                "previous",
                "first",
                "after",
                "sequence",
                "order",
                "what did",
                "where did",
                "how many times",
                "did the",
                "was there",
                "were there",
            )
        )
        has_wait_cue = "<wait>" in q_lower
        has_ocr_cue = any(
            kw in q_lower
            for kw in (
                "text",
                "word",
                "sign",
                "number",
                "letter",
                "written",
                "title",
                "name",
                "label",
                "score",
                "price",
                "plate",
                "read",
            )
        )

        # Calibrated Entropy Gating:
        # - High confidence (max_prob >= conf_threshold) and not a mandatory proactive/hallucination check -> keep fast Turn-1 <Answer>!
        should_escalate = False
        if has_wait_cue and max_prob < 0.88:
            should_escalate = True
        elif (has_unable_option or has_past_cue) and max_prob < self.conf_threshold:
            should_escalate = True
        elif has_ocr_cue and max_prob < self.conf_threshold:
            should_escalate = True
        elif max_prob < (self.conf_threshold - 0.15):
            should_escalate = True

        if not should_escalate:
            return raw_t1

        # Prompt the model with SYSTEM_PROMPT_CALIBRATED_ESCALATION to select the appropriate tool
        images = self._frames_to_pil(recent_frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        content.append(
            {
                "type": "text",
                "text": (
                    f"[State: {state_header} | Turn-1 Confidence={max_prob:.2f} (Uncertain)]\n{query}\n"
                    "Do NOT guess from the blurry/incomplete 4 frames. Emit the single best foraging tool tag (<ZoomIn>, <Recall>, or <Wait>)."
                ),
            }
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_CALIBRATED_ESCALATION}]},
            {"role": "user", "content": content},
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=images if images else None,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=128, do_sample=False)
        gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
        esc_out = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        parsed_esc = parse_model_response(esc_out)
        if parsed_esc.action is None:
            parsed_esc = StreamingForagerAgent._recover_turn1_parsed(parsed_esc)
        if parsed_esc.action_type in (ActionType.ZOOM_IN, ActionType.RECALL, ActionType.WAIT):
            return esc_out

        # Fallback structured tool call if escalation output omitted closing tags
        if has_wait_cue:
            return '<think>Clue or step is still unfolding under uncertainty.</think><Wait>{"check_after_sec": 2.0, "trigger": "clue_reveal"}</Wait>'
        if has_unable_option or has_past_cue:
            clean_q = re.sub(r"Options:.*", "", query.replace("Question:", ""), flags=re.DOTALL).strip()[:120]
            payload = json.dumps({"query": clean_q, "lookback_sec": 120.0, "top_k": 2})
            return f"<think>Historical evidence required under low Turn-1 confidence.</think><Recall>{payload}</Recall>"
        if has_ocr_cue:
            return '<think>Fine spatial detail required under low Turn-1 confidence.</think><ZoomIn>{"mode": "spatial_hires", "roi": [0.1, 0.1, 0.9, 0.9], "window_sec": 2.0}</ZoomIn>'

        return raw_t1


def compute_9task_metrics(summary: dict[str, Any]) -> dict[str, float]:
    """
    Compute the exact 9-task OVO-Bench metrics (`RT 6 + BT 3`, N=1,468 questions)
    following the official protocol of `SimpleStream (arXiv:2604.02317)`, `OASIS (CVPR 2026)`,
    and `StreamScout (arXiv:2609.00291)`:
    - `macro_9task_avg`: arithmetic mean of `(RT_macro + BT_macro) / 2` (used by SimpleStream & SelectStream)
    - `question_weighted_rt`: question-weighted average over the 837 RT questions (used by StreamScout Table 1)
    - `question_weighted_bt`: question-weighted average over the 631 BT questions (used by StreamScout Table 1)
    - `question_weighted_9task_overall`: question-weighted average over all 1,468 RT+BT questions (used by StreamScout Table 1)
    - `macro_9task_unweighted`: unweighted mean over the 9 subtasks
    """
    accs = summary["task_accuracies"]
    counts = summary.get("task_sample_counts") or {
        "OCR": 149,
        "ACR": 109,
        "ATR": 116,
        "STU": 178,
        "OJR": 184,
        "FPD": 101,
        "EPM": 297,
        "ASI": 148,
        "HLD": 186,
        "REC": 82,
        "SSR": 42,
        "CRR": 48,
    }
    rt_tasks = ("OCR", "ACR", "ATR", "STU", "OJR", "FPD")
    bt_tasks = ("EPM", "ASI", "HLD")
    nine_tasks = rt_tasks + bt_tasks

    rt_macro = float(np.mean([accs[t] for t in rt_tasks]))
    bt_macro = float(np.mean([accs[t] for t in bt_tasks]))
    er_macro = 0.5 * (accs["EPM"] + accs["ASI"])
    simplestream_rt_bt_avg = 0.5 * (rt_macro + bt_macro)
    nine_unweighted = float(np.mean([accs[t] for t in nine_tasks]))

    rt_q_num = sum(accs[t] * counts[t] for t in rt_tasks)
    rt_q_den = sum(counts[t] for t in rt_tasks)
    bt_q_num = sum(accs[t] * counts[t] for t in bt_tasks)
    bt_q_den = sum(counts[t] for t in bt_tasks)

    q_weighted_rt = rt_q_num / max(1, rt_q_den)
    q_weighted_bt = bt_q_num / max(1, bt_q_den)
    q_weighted_9task = (rt_q_num + bt_q_num) / max(1, rt_q_den + bt_q_den)

    return {
        "rt_macro": round(rt_macro, 2),
        "bt_macro": round(bt_macro, 2),
        "er_macro": round(er_macro, 2),
        "simplestream_rt_bt_avg": round(simplestream_rt_bt_avg, 2),
        "nine_task_macro_avg": round(nine_unweighted, 2),
        "streamscout_q_weighted_rt": round(q_weighted_rt, 2),
        "streamscout_q_weighted_bt": round(q_weighted_bt, 2),
        "streamscout_q_weighted_9task_overall": round(q_weighted_9task, 2),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate Calibrated Turn-1 Routing on OVO-Bench (Worker 4494517).")
    ap.add_argument(
        "--model-path",
        type=str,
        default="/mnt/hdfs/lipengyu/DA/checkpoints/forager/clean_rl/qwen25vl_7b_v4",
    )
    ap.add_argument("--conf-threshold", type=float, default=0.62)
    ap.add_argument("--data-json", type=Path, default=DEFAULT_OVOBENCH_JSON)
    ap.add_argument("--video-root", type=Path, default=DEFAULT_OVOBENCH_VIDEO_ROOT)
    ap.add_argument("--max-samples-per-task", type=int, default=None)
    ap.add_argument("--shard", type=str, default="0/1")
    ap.add_argument("--merge-shards", type=Path, nargs="*", default=None)
    ap.add_argument("--baseline-json", type=Path, default=None)
    ap.add_argument("--output-json", type=Path, required=True)
    args = ap.parse_args(argv)

    if args.merge_shards:
        merged = merge_ovobench_shard_files(args.merge_shards)
        ss_sum = merged["simplestream"]
        fo_sum = merged["forager"]
        print(format_summary_report(ss_sum))
        print()
        print(format_summary_report(fo_sum))
        payload = {
            "simplestream": asdict(ss_sum),
            "forager": asdict(fo_sum),
            "simplestream_9task_metrics": compute_9task_metrics(asdict(ss_sum)),
            "forager_9task_metrics": compute_9task_metrics(asdict(fo_sum)),
        }
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return 0

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)

    engine = CalibratedHFQwenVLEngine(
        model_name_or_path=args.model_path,
        conf_threshold=args.conf_threshold,
    )
    samples = load_ovobench_from_json(
        args.data_json,
        video_root=args.video_root,
        max_samples_per_task=args.max_samples_per_task,
        shard_idx=shard_idx,
        num_shards=num_shards,
        lazy_decode=True,
        allow_synthetic_fallback=False,
    )

    from scripts.eval_ovobench import evaluate_ovobench_joint_single_decode

    ss_sum, fo_sum = evaluate_ovobench_joint_single_decode(
        samples=samples,
        engine=engine,
        use_sdam_mask=True,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "simplestream": asdict(ss_sum),
        "forager": asdict(fo_sum),
        "simplestream_9task_metrics": compute_9task_metrics(asdict(ss_sum)),
        "forager_9task_metrics": compute_9task_metrics(asdict(fo_sum)),
    }
    args.output_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
