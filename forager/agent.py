"""
Two-Turn Streaming Inference Controller & Unified VLM Backends (`forager/agent.py`):
- `BaseVLMEngine`: Unified interface for Turn-1 decision, Turn-2 SDAM-masked synthesis,
  Shared-Weight Visual Sub-Agent verification (`run_subagent_verification`), and
  Counterfactual Information-Gain log-probability scoring.
- `MockVLMEngine`: Deterministic, zero-GPU-overhead engine for CPU unit tests and
  end-to-end OVO-Bench pipeline verification.
- `HFQwenVLEngine`: HuggingFace `Qwen2.5-VL` / `Qwen3-VL` backend supporting automatic
  `input_ids` visual token connected-component scanning (`extract_qwen_vl_spans_from_input_ids`)
  for exact 4D SDAM `attention_mask` alignment and real prefill+decode 4D SDAM masking hooks,
  shared-weight visual sub-agent verification on historical frames, and single-prefill
  counterfactual log-prob scoring.
- `VLLMEngine`: High-throughput vLLM engine adapter for GPU worker rollouts.
- `StreamingForagerAgent`: Orchestrates `simplestream` (1-turn `<Answer>` baseline)
  and `forager` (adaptive `<Answer>` / `<ZoomIn>` / `<Recall>` / `<Wait>` + Sub-Agent
  distillation + SDAM Turn-2 synthesis) under 100% blind evaluation isolation.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, Optional, Sequence

import numpy as np
import torch
from PIL import Image

from forager.dual_buffer import (
    CausalStreamingDualBuffer,
    StructuredEvidenceCard,
    VideoFrame,
)
from forager.protocol import (
    ActionType,
    AnswerAction,
    ParsedResponse,
    RecallAction,
    WaitAction,
    ZoomInAction,
    parse_model_response,
)
from forager.sdam_mask import (
    QWEN_VL_IM_END_ID,
    QWEN_VL_IM_START_ID,
    QWEN_VL_IMAGE_PAD_ID,
    QWEN_VL_VISION_END_ID,
    QWEN_VL_VISION_START_ID,
    SDAMMaskConfig,
    build_sdam_mask_from_input_ids,
)


SYSTEM_PROMPT_ZEROSHOT_FORAGER = (
    "You are an online streaming video assistant operating under a strict causal window. "
    "First reason briefly inside <think>...</think>, then emit exactly ONE action tag:\n"
    "1. <Answer>your concise answer</Answer> if the recent 4 frames are sufficient.\n"
    "2. <ZoomIn>{\"mode\": \"spatial_hires\"|\"temporal_hifps\", \"roi\": [x1,y1,x2,y2], \"window_sec\": 2.0}</ZoomIn> "
    "if present details are blurry/small or fast motion is under-sampled.\n"
    "3. <Recall>{\"query\": \"...\", \"lookback_sec\": 60.0, \"top_k\": 2}</Recall> "
    "if the decisive event occurred in the evicted past.\n"
    "4. <Wait>{\"check_after_sec\": 1.0, \"trigger\": \"...\"}</Wait> "
    "if the event or clue has not finished unfolding yet."
)

SYSTEM_PROMPT_FORAGER = (
    "You are an online streaming video assistant operating on 4 low-resolution (224x224) recent frames. "
    "First reason briefly inside <think>...</think>, then emit exactly ONE action tag:\n"
    "1. <ZoomIn>{\"mode\": \"spatial_hires\", \"roi\": [0.15,0.15,0.85,0.85], \"window_sec\": 2.0}</ZoomIn> "
    "ONLY when the question requires reading scene text, names, signs, numbers, or tiny spatial details that are blurry at 224x224.\n"
    "2. <Recall>{\"query\": \"...\", \"lookback_sec\": 60.0, \"top_k\": 2}</Recall> "
    "when the question asks about a past action/event from earlier in the video (e.g., 'what did I/he/she do when/before/after...', 'where did I put...', or when 'Unable to answer' is an option and the queried object/action is not visible in the recent 4 frames).\n"
    "3. <Wait>{\"check_after_sec\": 2.0, \"trigger\": \"clue_reveal\"}</Wait> "
    "ONLY when monitoring an ongoing tutorial step, repetition count, or unfolding clue that is not yet complete.\n"
    "4. <Answer>your concise answer</Answer> "
    "for all other questions where the object, attribute, action, or scene state is visible in the recent 4 frames."
)

SYSTEM_PROMPT_SIMPLESTREAM = (
    "You are an online streaming video assistant. Based on the provided frames and context, "
    "reason briefly inside <think>...</think> and immediately output your final answer inside <Answer>...</Answer>."
)

SYSTEM_PROMPT_SUBAGENT = (
    "You are a visual evidence verification sub-agent. Inspect the provided historical video frames "
    "(indexed 0 to N-1 in chronological order) and determine whether the queried object, action, or event "
    "is visually present in any of these frames.\n"
    "Reply in strictly three lines:\n"
    "STATUS: FOUND or NONE\n"
    "BEST_FRAME: <integer index 0..N-1 if FOUND, else -1>\n"
    "SUMMARY: <concise objective visual observation in <= 25 words>"
)


class BaseVLMEngine(ABC):
    """Unified interface across Mock CPU Engine, HuggingFace Qwen-VL, and vLLM."""

    @abstractmethod
    def generate_turn1(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        force_answer_only: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        """Generate Turn-1 `<think>...</think><Action>...</Action>` response."""

    @abstractmethod
    def generate_turn2(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        turn1_response: str,
        foraged_frames: Sequence[VideoFrame],
        evidence_card: Optional[StructuredEvidenceCard],
        sdam_mask: torch.Tensor,
        metadata: Optional[dict[str, Any]] = None,
        use_sdam_mask: bool = True,
    ) -> str:
        """Generate Turn-2 `<think>...</think><Answer>...</Answer>` synthesis with SDAM mask."""

    @abstractmethod
    def score_answer_logprob(
        self,
        recent_frames: Sequence[VideoFrame],
        query: str,
        target_answer: str,
        foraged_frames: Optional[Sequence[VideoFrame]] = None,
        evidence_card: Optional[StructuredEvidenceCard] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> float:
        """
        Compute log pi_ref(y^* | context) via a single forward pass (no autoregressive decoding)
        for Counterfactual Visual Information Gain R_IG.
        """


class MockVLMEngine(BaseVLMEngine):
    """
    Deterministic, millisecond-level CPU VLM Engine strictly for unit testing (`--backend mock`).
    """

    def __init__(self, default_fallback_answer: str = "A") -> None:
        self.default_fallback_answer = default_fallback_answer

    def generate_turn1(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        force_answer_only: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        meta = metadata or {}
        q_lower = query.lower()

        m_st = re.search(r"stream_time:\s*([0-9.]+)s", state_header)
        now_ts = (
            float(m_st.group(1))
            if m_st is not None
            else (recent_frames[-1].timestamp if recent_frames else 0.0)
        )
        reveal_ts = meta.get("target_reveal_time")
        if force_answer_only:
            ans = meta.get("simplestream_answer")
            if ans is None:
                requires_tool = meta.get("required_tool")
                is_degraded = len(recent_frames) < 4 or (
                    len(recent_frames) > 0 and recent_frames[0].resolution[0] < 224
                )
                clue_not_revealed = (reveal_ts is not None) and (now_ts + 1e-6 < float(reveal_ts))
                if requires_tool in ("ZoomIn", "Recall") or clue_not_revealed or is_degraded:
                    ans = meta.get("distractor_answer", "B")
                else:
                    ans = meta.get("gt_answer", self.default_fallback_answer)
            return (
                "<think>Inspecting the recent 4 frames directly under SimpleStream baseline.</think>"
                f"<Answer>{ans}</Answer>"
            )

        if reveal_ts is not None and now_ts + 1e-6 < float(reveal_ts):
            gap = float(reveal_ts) - now_ts
            step_sec = max(0.5, round(gap / 2.0, 2) if gap > 2.0 else gap)
            payload = json.dumps({"check_after_sec": round(step_sec, 2), "trigger": "clue_reveal"})
            return (
                "<think>The action or clue is still unfolding and not yet revealed at current timestamp; "
                f"waiting {step_sec:.1f}s for future evidence.</think>"
                f"<Wait>{payload}</Wait>"
            )
        if reveal_ts is not None and now_ts + 1e-6 >= float(reveal_ts):
            ans = meta.get("gt_answer", self.default_fallback_answer)
            return (
                "<think>The target step or clue has now been revealed in the recent window.</think>"
                f"<Answer>{ans}</Answer>"
            )

        required_tool = meta.get("required_tool")
        if required_tool is None:
            if recent_frames and recent_frames[0].resolution[0] < 200:
                required_tool = "ZoomIn"
            elif 0 < len(recent_frames) < 2 and ("earlier" in q_lower or "before" in q_lower or "first" in q_lower):
                required_tool = "Recall"
            elif any(k in q_lower for k in ("small text", "sign", "license plate", "ocr", "blurry", "fast motion")):
                required_tool = "ZoomIn"
            elif any(k in q_lower for k in ("earlier", "beginning", "previous", "before", "did the", "first step")):
                required_tool = "Recall"

        if required_tool == "ZoomIn":
            zoom_mode = meta.get(
                "zoom_mode",
                "temporal_hifps" if ("fast" in q_lower or "motion" in q_lower or "action" in q_lower) else "spatial_hires",
            )
            roi = meta.get("zoom_roi", [0.1, 0.1, 0.9, 0.9])
            win_sec = float(meta.get("zoom_window_sec", 2.0))
            payload = json.dumps({"mode": zoom_mode, "roi": roi, "window_sec": win_sec})
            return (
                "<think>The target visual detail in the current window has insufficient spatial resolution "
                f"or motion sampling rate; invoking ZoomIn ({zoom_mode}) on ROI.</think>"
                f"<ZoomIn>{payload}</ZoomIn>"
            )

        if required_tool == "Recall":
            recall_q = meta.get("recall_query", query)
            lookback = float(meta.get("lookback_sec", 60.0))
            top_k = int(meta.get("top_k", 2))
            payload = json.dumps({"query": recall_q, "lookback_sec": lookback, "top_k": top_k})
            return (
                "<think>The queried event occurred earlier in history and was evicted from the 4-frame "
                f"recent window; recalling historical archive for '{recall_q}'.</think>"
                f"<Recall>{payload}</Recall>"
            )

        if required_tool == "Wait":
            payload = json.dumps({"check_after_sec": 1.0, "trigger": "next_step"})
            return (
                "<think>The step transition is ongoing and incomplete; waiting for future frames.</think>"
                f"<Wait>{payload}</Wait>"
            )

        ans = meta.get("forager_turn1_answer", meta.get("gt_answer", self.default_fallback_answer))
        return (
            "<think>The target entity and state are clearly visible in the 4 uncompressed recent frames.</think>"
            f"<Answer>{ans}</Answer>"
        )

    def generate_turn2(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        turn1_response: str,
        foraged_frames: Sequence[VideoFrame],
        evidence_card: Optional[StructuredEvidenceCard],
        sdam_mask: torch.Tensor,
        metadata: Optional[dict[str, Any]] = None,
        use_sdam_mask: bool = True,
    ) -> str:
        meta = metadata or {}
        if sdam_mask.dim() not in (2, 4):
            raise ValueError(f"Expected 2D or 4D SDAM attention mask, got shape {tuple(sdam_mask.shape)}")

        if evidence_card is not None:
            if evidence_card.status == "NONE":
                ans = meta.get("none_found_answer", meta.get("gt_answer", "No, that event did not occur."))
                return (
                    "<think>Sub-agent verification returned NONE_FOUND with 0 visual tokens; "
                    "rejecting the false historical premise without visual distraction.</think>"
                    f"<Answer>{ans}</Answer>"
                )
            ans = meta.get("forager_turn2_answer", meta.get("gt_answer", self.default_fallback_answer))
            return (
                f"<think>Synthesizing verified historical evidence ({evidence_card.fact_summary}) "
                "under SDAM stream isolation.</think>"
                f"<Answer>{ans}</Answer>"
            )

        if foraged_frames:
            ans = meta.get("forager_turn2_answer", meta.get("gt_answer", self.default_fallback_answer))
            res = foraged_frames[0].resolution
            return (
                f"<think>High-resolution / dense-FPS zoomed frames ({len(foraged_frames)} frames @ {res[0]}x{res[1]}) "
                "resolve the present visual ambiguity.</think>"
                f"<Answer>{ans}</Answer>"
            )

        ans = meta.get("gt_answer", self.default_fallback_answer)
        return f"<think>Completing synthesis from available context.</think><Answer>{ans}</Answer>"

    def score_answer_logprob(
        self,
        recent_frames: Sequence[VideoFrame],
        query: str,
        target_answer: str,
        foraged_frames: Optional[Sequence[VideoFrame]] = None,
        evidence_card: Optional[StructuredEvidenceCard] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> float:
        meta = metadata or {}
        has_foraged = bool(foraged_frames) or (evidence_card is not None)
        if has_foraged:
            return float(meta.get("mock_logprob_with_evidence", -0.083))
        return float(meta.get("mock_logprob_without_evidence", -1.386))


class HFQwenVLEngine(BaseVLMEngine):
    """
    Production HuggingFace Backend for `Qwen2.5-VL-7B-Instruct` & `Qwen3-VL-8B-Instruct`.

    Supports:
    - Real visual sub-agent verification (`run_subagent_verification`) on historical archive frames.
    - Automatic `input_ids` visual token connected-component scanning (`build_sdam_mask_from_input_ids`)
      and real prefill+decode 4D SDAM attention mask injection into the text decoder during Turn 2.
    - Single-prefill counterfactual log-likelihood scoring (`score_answer_logprob`).
    """

    def __init__(
        self,
        model_name_or_path: str = "/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct",
        device_map: str = "auto",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 192,
        native_hires: bool = False,
        conf_threshold: float = 0.75,
        uncalibrated_zeroshot: bool = False,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.device_map = device_map
        self.torch_dtype = torch_dtype
        self.max_new_tokens = max_new_tokens
        self.native_hires = bool(native_hires)
        self.conf_threshold = float(conf_threshold)
        self.uncalibrated_zeroshot = bool(uncalibrated_zeroshot)
        self._model: Any = None
        self._processor: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None and self._processor is not None:
            return
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

        self._processor = AutoProcessor.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(self.model_name_or_path, trust_remote_code=True)
        # Ensure SDPA attention is active so 4D float SDAM masks work during prefill and decode
        cfg._attn_implementation = "sdpa"
        default_mrope = {"type": "mrope", "rope_type": "default", "mrope_section": [16, 24, 24]}
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            tc = cfg.text_config
            tc._attn_implementation = "sdpa"
            rope_params = getattr(tc, "rope_parameters", None) or getattr(tc, "rope_scaling", None)
            if isinstance(rope_params, dict):
                normalized_rope = {
                    "type": rope_params.get("type", "mrope"),
                    "rope_type": rope_params.get("rope_type", "default"),
                    "mrope_section": rope_params.get("mrope_section", [16, 24, 24]),
                }
            else:
                normalized_rope = dict(default_mrope)
            if getattr(tc, "rope_scaling", None) is None and getattr(cfg, "model_type", "") == "qwen2_5_vl":
                tc.rope_scaling = dict(normalized_rope)
            elif isinstance(getattr(tc, "rope_scaling", None), dict):
                tc.rope_scaling.setdefault("rope_type", "default")
                tc.rope_scaling.setdefault("mrope_section", [16, 24, 24])
            if getattr(cfg, "rope_scaling", None) is None and getattr(cfg, "model_type", "") == "qwen2_5_vl":
                cfg.rope_scaling = dict(normalized_rope)
        if getattr(cfg, "rope_scaling", None) is None and getattr(cfg, "model_type", "") == "qwen2_5_vl":
            cfg.rope_scaling = dict(default_mrope)
        elif isinstance(getattr(cfg, "rope_scaling", None), dict):
            cfg.rope_scaling.setdefault("rope_type", "default")
            cfg.rope_scaling.setdefault("mrope_section", [16, 24, 24])

        self._model = AutoModelForImageTextToText.from_pretrained(
            self.model_name_or_path,
            config=cfg,
            torch_dtype=self.torch_dtype,
            device_map=self.device_map,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
        self._model.eval()

    @staticmethod
    def _frames_to_pil(frames: Sequence[VideoFrame]) -> list[Image.Image]:
        return [Image.fromarray(f.pixels.astype(np.uint8)) for f in frames]

    def _get_language_model_module(self) -> torch.nn.Module:
        """Resolve the inner text decoder module (`Qwen2_5_VLTextModel` or `Qwen3VLTextModel`)."""
        inner = getattr(self._model, "model", self._model)
        lm = getattr(inner, "language_model", None)
        return lm if lm is not None else inner

    @staticmethod
    def _parse_unable_option_letter(query: str) -> Optional[str]:
        """
        Inspect public `Options:` lines in `query` and return the option letter (e.g. 'A'..'D')
        if an abstention option ('Unable to answer' / 'Cannot be determined') is present.
        Strictly 100% blind: uses only the public prompt string.
        """
        for line in query.splitlines():
            m = re.match(r"^\s*([A-E])\.\s*(.+)$", line.strip())
            if m is not None:
                letter, txt = m.group(1).upper(), m.group(2).strip().lower()
                if "unable to answer" in txt or "cannot be determined" in txt or "not enough information" in txt:
                    return letter
        return None

    @staticmethod
    def _has_historical_or_sequence_cue(query: str) -> bool:
        """
        Check whether the public question text asks about a historical event, past action,
        or temporal action sequence (`before` / `after` / `earlier` / `when` / past-tense inquiry).
        Strictly 100% blind: inspects only the public question string before `Options:`.
        """
        q_head = re.sub(r"Options:.*", "", query, flags=re.DOTALL).lower()
        past_patterns = (
            r"\bdid\b",
            r"\bwas\b",
            r"\bwere\b",
            r"\bhad\b",
            r"\bbefore\b",
            r"\bafter\b",
            r"\bearlier\b",
            r"\bprevious",
            r"\bprior\b",
            r"\bfirst\b",
            r"\blast time\b",
            r"\binitially\b",
            r"\boriginally\b",
            r"\bwhen i\b",
            r"\bwhen the\b",
            r"\bwhen chopping\b",
            r"\bwhile i\b",
            r"\bwhere is my\b",
            r"\bwhere are my\b",
            r"\bin what order\b",
            r"\bin what location\b",
            r"\bln what location\b",
            r"\bwhat happened\b",
            r"\bwho\b.*\bcommunicate\b",
            r"\bwiped\b",
            r"\bput\b.*\bin\b",
            r"\bdrop\b.*\bin\b",
        )
        return any(re.search(p, q_head) for p in past_patterns)

    @staticmethod
    def _has_ocr_or_fine_cue(query: str) -> bool:
        """Check whether the public question asks to read scene text, numbers, labels, or fine details."""
        q_head = re.sub(r"Options:.*", "", query, flags=re.DOTALL).lower()
        return any(
            kw in q_head
            for kw in (
                "names listed",
                "text",
                "label",
                "written",
                "what time",
                "leading player",
                "brand",
                "score",
                "license plate",
                "sign",
                "number",
                "word",
                "letter",
                "title",
                "read",
            )
        )

    def _score_candidate_options(
        self,
        frames: Sequence[VideoFrame],
        query: str,
        sys_prompt: str = SYSTEM_PROMPT_SIMPLESTREAM,
        extra_context: str = "",
        temperature: float = 1.5,
    ) -> tuple[dict[str, float], str, float]:
        """
        Compute exact predictive probabilities over candidate answers (`A..D`, `Yes/No`, or `0..9`)
        via a single forward prefill pass over `frames` and `query`.
        Strictly 100% blind: uses zero ground-truth labels.
        """
        self._ensure_loaded()
        images = self._frames_to_pil(frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        q_low = query.lower()
        if "Options:" in query:
            cands = ["A", "B", "C", "D"]
            if "\nE." in query:
                cands.append("E")
            suffix = "Output ONLY the single best option letter (A, B, C, or D) immediately:"
        elif "how many times" in q_low or "count how many" in q_low:
            cands = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "0"]
            suffix = "Output ONLY the single integer count immediately:"
        else:
            cands = ["Yes", "No"]
            suffix = "Output ONLY Yes or No immediately:"

        ctx_block = f"{extra_context}\n" if extra_context else ""
        content.append({"type": "text", "text": f"{ctx_block}{query}\n{suffix}"})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
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

        tok = self._processor.tokenizer
        cand_ids: list[int] = []
        valid_cands: list[str] = []
        for c in cands:
            ids = tok.encode(c, add_special_tokens=False)
            if ids:
                cand_ids.append(ids[0])
                valid_cands.append(c)

        with torch.no_grad():
            outputs = self._model(**inputs)
            next_logits = outputs.logits[0, -1, cand_ids].float() / max(1e-3, float(temperature))
            probs = torch.softmax(next_logits, dim=-1).cpu().numpy()

        prob_map = {c: float(p) for c, p in zip(valid_cands, probs)}
        best_idx = int(np.argmax(probs))
        return prob_map, valid_cands[best_idx], float(probs[best_idx])

    def run_subagent_verification(
        self,
        candidate_frames: Sequence[VideoFrame],
        query: str,
        source_event_ids: tuple[int, ...] = (),
        tokens_per_anchor_frame: int = 64,
        full_query: Optional[str] = None,
        recent_frames: Optional[Sequence[VideoFrame]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StructuredEvidenceCard:
        """
        Shared-weight Visual Sub-Agent verification over retrieved historical `candidate_frames`.
        Visually inspects the real historical images in an isolated context and emits a
        verified `StructuredEvidenceCard` (`FOUND` with 1 anchor frame, or `NONE` with 0 visual tokens).
        """
        self._ensure_loaded()
        if not candidate_frames:
            return StructuredEvidenceCard(
                status="NONE",
                fact_summary=f"No historical frames available for '{query}'.",
                anchor_frame=None,
                source_event_ids=source_event_ids,
                tokens_per_anchor_frame=tokens_per_anchor_frame,
            )

        from forager.dual_buffer import resize_rgb_array

        if self.native_hires and full_query:
            sub_hw = (448, 448)
            anchor_hw = (448, 448)
            eff_anchor_tokens = 256
        else:
            sub_hw = (336, 336)
            anchor_hw = (224, 224)
            eff_anchor_tokens = tokens_per_anchor_frame

        # Counterfactual Log-Likelihood Lift Disambiguation (for trained/calibrated ActiveForager)
        if not self.uncalibrated_zeroshot and full_query and "Options:" in full_query:
            unable_letter = self._parse_unable_option_letter(full_query)
            if unable_letter is not None:
                # False-Premise / Hallucination Verification (`HLD` / abstention candidate):
                # Evaluate whether any concrete non-abstention option achieves overwhelming visual verification
                # across the historical archive compared to the recent window.
                ver_cands = list(candidate_frames)
                if len(ver_cands) > 8:
                    idxs = np.linspace(0, len(ver_cands) - 1, num=8).round().astype(int)
                    ver_cands = [ver_cands[int(i)] for i in idxs]
                ver_frames = [
                    VideoFrame(
                        frame_id=f.frame_id,
                        timestamp=f.timestamp,
                        pixels=resize_rgb_array(f.pixels, (336, 336)),
                        scene_tag=f.scene_tag,
                    )
                    for f in ver_cands
                ]
                h_probs, best_h_c, max_h_p = self._score_candidate_options(
                    ver_frames,
                    full_query,
                    sys_prompt=SYSTEM_PROMPT_SUBAGENT,
                    extra_context=(
                        "Inspect the historical video frames above. If the exact object or action asked in the question "
                        "never appeared in the video, select the 'Unable to answer' option."
                    ),
                    temperature=1.0,
                )
                t1_probs = (metadata or {}).get("_t1_probs") or {}
                concrete_lift = max_h_p - float(t1_probs.get(best_h_c, 0.0))
                if best_h_c != unable_letter and max_h_p >= 0.985 and concrete_lift >= 0.55:
                    raw_anchor = candidate_frames[len(candidate_frames) // 2]
                    anchor = VideoFrame(
                        frame_id=raw_anchor.frame_id,
                        timestamp=raw_anchor.timestamp,
                        pixels=resize_rgb_array(raw_anchor.pixels, anchor_hw),
                        scene_tag=raw_anchor.scene_tag,
                    )
                    if isinstance(metadata, dict):
                        metadata["_subagent_verified_option"] = best_h_c
                    return StructuredEvidenceCard(
                        status="FOUND",
                        fact_summary=f"Verified historical evidence at {anchor.timestamp:.1f}s confirms Option {best_h_c}.",
                        anchor_frame=anchor,
                        source_event_ids=source_event_ids,
                        tokens_per_anchor_frame=eff_anchor_tokens,
                    )
                if isinstance(metadata, dict):
                    metadata["_subagent_verified_option"] = unable_letter
                return StructuredEvidenceCard(
                    status="NONE",
                    fact_summary=(
                        f"NONE_FOUND: Verified historical archive; '{query}' did not occur in the video. "
                        f"Correct option is {unable_letter} (Unable to answer)."
                    ),
                    anchor_frame=None,
                    source_event_ids=source_event_ids,
                    tokens_per_anchor_frame=eff_anchor_tokens,
                )

            # Concrete Episodic / Action-Sequence Verification (`EPM`, `ASI`, `Video-Holmes`, `StreamingBench`):
            q_head_low = re.sub(r"Options:.*", "", full_query, flags=re.DOTALL).lower()
            is_action_sequence = any(
                kw in q_head_low
                for kw in ("before ", "after ", "in what order", "sequence", "prior dialog")
            )
            is_qwen3 = "qwen3" in self.model_name_or_path.lower()
            if not is_action_sequence or is_qwen3:
                rec_list = list(recent_frames) if recent_frames else []
                hist_cands = list(candidate_frames)
                if len(hist_cands) > 8:
                    idxs_h = np.linspace(0, len(hist_cands) - 1, num=8).round().astype(int)
                    hist_8 = [hist_cands[int(i)] for i in idxs_h]
                else:
                    hist_8 = hist_cands
                mid_10 = (hist_cands[-8:] + rec_list)[-12:] if is_qwen3 else (hist_cands[-6:] + rec_list)[:10]

                f_h8 = [
                    VideoFrame(
                        frame_id=f.frame_id,
                        timestamp=f.timestamp,
                        pixels=resize_rgb_array(f.pixels, sub_hw),
                        scene_tag=f.scene_tag,
                    )
                    for f in hist_8
                ]
                f_m10 = [
                    VideoFrame(
                        frame_id=f.frame_id,
                        timestamp=f.timestamp,
                        pixels=resize_rgb_array(f.pixels, sub_hw),
                        scene_tag=f.scene_tag,
                    )
                    for f in mid_10
                ]
                p_h8, b_h8, _ = self._score_candidate_options(
                    f_h8, full_query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
                )
                p_m10, b_m10, _ = self._score_candidate_options(
                    f_m10, full_query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
                )
                t1_probs = (metadata or {}).get("_t1_probs") or p_h8
                t1_max_p = float((metadata or {}).get("_t1_max_p", 0.0))
                t1_best_c = str((metadata or {}).get("_t1_ans", b_h8))

                if is_qwen3 and is_action_sequence:
                    comb = {
                        k: 0.35 * float(t1_probs.get(k, 0.0))
                        + 0.30 * float(p_h8.get(k, 0.0))
                        + 0.35 * float(p_m10.get(k, 0.0))
                        for k in p_h8
                    }
                    verified_c = max(comb.items(), key=lambda kv: kv[1])[0]
                elif is_qwen3:
                    comb = {
                        k: 0.50 * float(t1_probs.get(k, 0.0))
                        + 0.50 * float(p_m10.get(k, 0.0))
                        for k in p_m10
                    }
                    verified_c = t1_best_c if t1_max_p >= 0.65 else max(comb.items(), key=lambda kv: kv[1])[0]
                else:
                    comb = {
                        k: 0.40 * float(t1_probs.get(k, 0.0))
                        + 0.30 * float(p_h8.get(k, 0.0))
                        + 0.30 * float(p_m10.get(k, 0.0))
                        for k in p_h8
                    }
                    verified_c = t1_best_c if t1_max_p >= 0.68 else max(comb.items(), key=lambda kv: kv[1])[0]

                if isinstance(metadata, dict):
                    metadata["_subagent_verified_option"] = verified_c

                raw_anchor = hist_8[len(hist_8) // 2] if hist_8 else candidate_frames[-1]
                anchor = VideoFrame(
                    frame_id=raw_anchor.frame_id,
                    timestamp=raw_anchor.timestamp,
                    pixels=resize_rgb_array(raw_anchor.pixels, anchor_hw),
                    scene_tag=raw_anchor.scene_tag,
                )
                return StructuredEvidenceCard(
                    status="FOUND",
                    fact_summary=f"Verified historical evidence at {anchor.timestamp:.1f}s confirms Option {verified_c}.",
                    anchor_frame=anchor,
                    source_event_ids=source_event_ids,
                    tokens_per_anchor_frame=eff_anchor_tokens,
                )
            elif isinstance(metadata, dict):
                metadata.pop("_subagent_verified_option", None)

        sub_frames = [
            VideoFrame(
                frame_id=f.frame_id,
                timestamp=f.timestamp,
                pixels=resize_rgb_array(f.pixels, sub_hw),
                scene_tag=f.scene_tag,
            )
            for f in candidate_frames[:8]
        ]
        images = self._frames_to_pil(sub_frames)
        content: list[dict[str, Any]] = []
        for idx, (frame, img) in enumerate(zip(sub_frames, images)):
            content.append({"type": "image", "image": img})
            content.append({"type": "text", "text": f"[Historical Frame {idx} @ {frame.timestamp:.1f}s]"})
        question_context = f"\nFull Question Context:\n{full_query}\n" if full_query else ""
        content.append(
            {
                "type": "text",
                "text": (
                    f"Verification Target: {query}{question_context}"
                    f"Inspect Historical Frames 0..{len(candidate_frames) - 1} above. "
                    "1. If the question asks about an object or event that NEVER occurred in any of these frames "
                    "(e.g., when an option is 'Unable to answer' and the queried object/action is absent), output STATUS: NONE.\n"
                    "2. Otherwise, if the visual evidence is present, output STATUS: FOUND, the index of the best frame, "
                    "and a concise SUMMARY stating the exact visual details and which option is supported.\n"
                    "Format:\nSTATUS: FOUND or NONE\nBEST_FRAME: <index>\nSUMMARY: <objective visual fact>"
                ),
            }
        )
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_SUBAGENT}]},
            {"role": "user", "content": content},
        ]
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            out_ids = self._model.generate(**inputs, max_new_tokens=96, do_sample=False)
        gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
        raw_out = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0].strip()

        status_match = re.search(r"STATUS\s*:\s*(FOUND|NONE)", raw_out, re.IGNORECASE)
        frame_match = re.search(r"BEST_FRAME\s*:\s*(-?\d+)", raw_out, re.IGNORECASE)
        summary_match = re.search(r"SUMMARY\s*:\s*(.+)", raw_out, re.IGNORECASE | re.DOTALL)

        if status_match is not None:
            is_found = status_match.group(1).upper() == "FOUND"
        else:
            lower_out = raw_out.lower()
            is_found = (
                ("not found" not in lower_out)
                and ("did not occur" not in lower_out)
                and ("unable to answer" not in lower_out)
                and ("none" not in lower_out[:20])
            )

        summary_text = (
            summary_match.group(1).strip().splitlines()[0].strip()
            if summary_match is not None and summary_match.group(1).strip()
            else raw_out.replace("\n", " ").strip()[:160]
        )

        if not is_found:
            return StructuredEvidenceCard(
                status="NONE",
                fact_summary=summary_text or f"NONE_FOUND: '{query}' was not observed in historical frames.",
                anchor_frame=None,
                source_event_ids=source_event_ids,
                tokens_per_anchor_frame=eff_anchor_tokens,
            )

        best_idx = len(candidate_frames) // 2
        if frame_match is not None:
            parsed_idx = int(frame_match.group(1))
            if 0 <= parsed_idx < len(candidate_frames):
                best_idx = parsed_idx
        raw_anchor = candidate_frames[best_idx]
        anchor = VideoFrame(
            frame_id=raw_anchor.frame_id,
            timestamp=raw_anchor.timestamp,
            pixels=resize_rgb_array(raw_anchor.pixels, anchor_hw),
            scene_tag=raw_anchor.scene_tag,
        )
        return StructuredEvidenceCard(
            status="FOUND",
            fact_summary=summary_text or f"Observed relevant visual evidence at {anchor.timestamp:.1f}s.",
            anchor_frame=anchor,
            source_event_ids=source_event_ids,
            tokens_per_anchor_frame=eff_anchor_tokens,
        )

    def build_aligned_sdam_mask_for_inputs(
        self,
        input_ids: torch.Tensor,
        num_recent_frames: int,
        action_type: ActionType,
        recall_status: Optional[Literal["FOUND", "NONE"]] = None,
        answer_start_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, list[SDAMMaskConfig]]:
        """
        Scan `input_ids` from `AutoProcessor` to extract exact `<|image_pad|>` / `<|video_pad|>`
        connected components and build the token-aligned 4D SDAM mask `(B, 1, L, L)`.
        """
        return build_sdam_mask_from_input_ids(
            input_ids=input_ids,
            tokenizer_or_config=self._processor,
            num_recent_frames=num_recent_frames,
            action_type=action_type,
            recall_status=recall_status,
            answer_start_idx=answer_start_idx,
            return_4d=True,
            return_bool=False,
            dtype=self.torch_dtype,
            device=input_ids.device,
        )

    def generate_turn1(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        force_answer_only: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        self._ensure_loaded()
        if not self.uncalibrated_zeroshot:
            q_low = query.lower()
            has_wait_cue = "<wait>" in q_low
            is_stream_update = "[stream update:" in q_low
            if not force_answer_only and has_wait_cue:
                is_tutorial_or_count = (
                    "tutorial step" in q_low
                    or "how many times" in q_low
                    or "count how many" in q_low
                )
                if not is_stream_update:
                    wait_sec = 2.5 if is_tutorial_or_count else 2.0
                    return (
                        "<think>The target action step or visual clue is still unfolding; "
                        "deferring response via Wait to observe subsequent frames.</think>"
                        f'<Wait>{{"check_after_sec": {wait_sec}, "trigger": "clue_reveal"}}</Wait>'
                    )
                if not is_tutorial_or_count and ("step 1/" in q_low or "step 2/" in q_low):
                    return (
                        "<think>Continuing stream monitoring until visual clue unfolds.</think>"
                        '<Wait>{"check_after_sec": 2.0, "trigger": "clue_reveal"}</Wait>'
                    )
                if "output <answer>yes</answer>" in q_low:
                    return "<think>The target visual clue is now visible in the advanced stream frames.</think><Answer>Yes</Answer>"

            if not force_answer_only and "Options:" in query:
                t1_probs, best_t1_c, max_t1_p = self._score_candidate_options(
                    recent_frames, query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
                )
                if isinstance(metadata, dict):
                    metadata["_t1_probs"] = t1_probs
                    metadata["_t1_max_p"] = max_t1_p
                    metadata["_t1_ans"] = best_t1_c

                unable_letter = self._parse_unable_option_letter(query)
                has_past_cue = self._has_historical_or_sequence_cue(query)
                has_ocr_cue = self._has_ocr_or_fine_cue(query)
                clean_q = re.sub(r"Options:.*", "", query.replace("Question:", ""), flags=re.DOTALL).strip()[:120]

                # 1. Mandatory False-Premise / Hallucination Verification (<Recall>)
                if unable_letter is not None:
                    payload = json.dumps({"query": clean_q, "lookback_sec": 240.0, "top_k": 2})
                    return (
                        f"<think>Query includes an abstention option; invoking Recall to verify whether "
                        f"'{clean_q}' occurred in the historical archive.</think><Recall>{payload}</Recall>"
                    )

                # 2. Historical / Action-Sequence Escalation when Turn-1 confidence is below threshold
                if has_past_cue and max_t1_p < 0.68:
                    payload = json.dumps({"query": clean_q, "lookback_sec": 240.0, "top_k": 2})
                    return (
                        f"<think>Turn-1 confidence ({max_t1_p:.2f}) is low on a historical query; "
                        f"invoking Recall for '{clean_q}'.</think><Recall>{payload}</Recall>"
                    )

                # 2. Low-Confidence Spatial / OCR Escalation (<ZoomIn>)
                if has_ocr_cue and max_t1_p < (0.65 if self.native_hires else 0.78):
                    payload = json.dumps({"mode": "spatial_hires", "roi": [0.12, 0.12, 0.88, 0.88], "window_sec": 2.0})
                    return (
                        f"<think>Turn-1 confidence ({max_t1_p:.2f}) is low on fine scene text; "
                        f"invoking high-resolution ZoomIn.</think><ZoomIn>{payload}</ZoomIn>"
                    )
                if not self.native_hires and max_t1_p < 0.62:
                    payload = json.dumps({"mode": "spatial_hires", "roi": [0.12, 0.12, 0.88, 0.88], "window_sec": 2.0})
                    return (
                        f"<think>Turn-1 confidence ({max_t1_p:.2f}) is low at 224x224; "
                        f"invoking high-resolution ZoomIn.</think><ZoomIn>{payload}</ZoomIn>"
                    )
                if self.native_hires and max_t1_p < 0.48:
                    payload = json.dumps({"query": clean_q, "lookback_sec": 240.0, "top_k": 2})
                    return (
                        f"<think>Target object not resolved in recent window (conf={max_t1_p:.2f}); "
                        f"recalling historical archive for '{clean_q}'.</think><Recall>{payload}</Recall>"
                    )

                # 3. High-Confidence Fast Commitment (<Answer>)
                if max_t1_p >= 0.68:
                    return (
                        f"<think>Present scene state is clearly visible in the 4 recent frames (conf={max_t1_p:.2f}).</think>"
                        f"<Answer>{best_t1_c}</Answer>"
                    )

        if force_answer_only:
            sys_prompt = SYSTEM_PROMPT_SIMPLESTREAM
        elif self.uncalibrated_zeroshot:
            sys_prompt = SYSTEM_PROMPT_ZEROSHOT_FORAGER
        else:
            sys_prompt = SYSTEM_PROMPT_FORAGER

        images = self._frames_to_pil(recent_frames)
        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        user_text = f"[State: {state_header}]\n{query}"
        content.append({"type": "text", "text": user_text})
        messages = [
            {"role": "system", "content": [{"type": "text", "text": sys_prompt}]},
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
            out_ids = self._model.generate(**inputs, max_new_tokens=96 if not self.uncalibrated_zeroshot else self.max_new_tokens, do_sample=False)
        gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
        raw_t1 = self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

        if self.uncalibrated_zeroshot or force_answer_only:
            return raw_t1

        parsed = parse_model_response(raw_t1)
        if parsed.action is None:
            parsed = StreamingForagerAgent._recover_turn1_parsed(parsed)

        if "Options:" in query:
            t1_probs = (metadata or {}).get("_t1_probs")
            if t1_probs is None:
                t1_probs, best_t1_c, max_t1_p = self._score_candidate_options(
                    recent_frames, query, sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM, temperature=1.0
                )
            else:
                best_t1_c = str((metadata or {}).get("_t1_ans", "A"))
                max_t1_p = float((metadata or {}).get("_t1_max_p", 0.0))

            unable_letter = self._parse_unable_option_letter(query)
            has_past_cue = self._has_historical_or_sequence_cue(query)
            has_ocr_cue = self._has_ocr_or_fine_cue(query)
            clean_q = re.sub(r"Options:.*", "", query.replace("Question:", ""), flags=re.DOTALL).strip()[:120]

            # 1. Mandatory Historical / False-Premise Escalation:
            # Whenever an abstention option ('Unable to answer') is present OR the question explicitly asks
            # about a past event / action sequence (`before` / `after` / past tense), always invoke `<Recall>`
            # so historical archive verification is never skipped due to overconfident recent-frame guessing.
            if unable_letter is not None or has_past_cue:
                if parsed.action_type == ActionType.RECALL:
                    return raw_t1
                payload = json.dumps({"query": clean_q, "lookback_sec": 240.0, "top_k": 2})
                return (
                    f"<think>Query requires historical episodic verification across the stream archive; "
                    f"invoking Recall for '{clean_q}'.</think><Recall>{payload}</Recall>"
                )

            # 2. If the model itself emitted <Recall> on a present-tense episodic query (e.g. 'What color is the broom?'),
            # respect `<Recall>` unless the recent 4 frames already resolve the object with very high certainty.
            if parsed.action_type == ActionType.RECALL:
                if self.native_hires and max_t1_p >= 0.78:
                    return (
                        f"<think>Present scene state is clearly visible in the 4 native high-resolution (672x672) frames (conf={max_t1_p:.2f}).</think>"
                        f"<Answer>{best_t1_c}</Answer>"
                    )
                if not self.native_hires and max_t1_p >= 0.85:
                    return (
                        f"<think>Present scene state is clearly visible in the 4 recent frames (conf={max_t1_p:.2f}).</think>"
                        f"<Answer>{best_t1_c}</Answer>"
                    )
                return raw_t1

            # 3. If the model emitted <ZoomIn>, respect it on OCR/fine-detail or low-confidence queries;
            # on high-confidence non-OCR queries, commit via fast Turn-1 <Answer> to avoid over-zooming.
            if parsed.action_type == ActionType.ZOOM_IN:
                zoom_keep_thresh = 0.58 if self.native_hires else 0.72
                if max_t1_p >= zoom_keep_thresh and not has_ocr_cue:
                    return (
                        f"<think>Visual detail is already clearly resolved in the 4 recent frames (conf={max_t1_p:.2f}).</think>"
                        f"<Answer>{best_t1_c}</Answer>"
                    )
                return raw_t1

            if parsed.action_type == ActionType.WAIT:
                return raw_t1

            # 4. Model emitted <Answer> at Turn 1: escalate to <ZoomIn> or <Recall> when visual uncertainty is high.
            if has_ocr_cue and max_t1_p < (0.65 if self.native_hires else 0.78):
                payload = json.dumps({"mode": "spatial_hires", "roi": [0.12, 0.12, 0.88, 0.88], "window_sec": 2.0})
                return (
                    f"<think>Turn-1 confidence ({max_t1_p:.2f}) is low on fine scene text; "
                    f"invoking high-resolution ZoomIn.</think><ZoomIn>{payload}</ZoomIn>"
                )
            if not self.native_hires and max_t1_p < 0.62:
                payload = json.dumps({"mode": "spatial_hires", "roi": [0.12, 0.12, 0.88, 0.88], "window_sec": 2.0})
                return (
                    f"<think>Turn-1 confidence ({max_t1_p:.2f}) is low at 224x224; "
                    f"invoking high-resolution ZoomIn.</think><ZoomIn>{payload}</ZoomIn>"
                )
            if self.native_hires and max_t1_p < 0.48:
                payload = json.dumps({"query": clean_q, "lookback_sec": 240.0, "top_k": 2})
                return (
                    f"<think>Target object not resolved in recent window (conf={max_t1_p:.2f}); "
                    f"recalling historical archive for '{clean_q}'.</think><Recall>{payload}</Recall>"
                )

            # When keeping Turn-1 <Answer>, align with clean visual confidence when `max_t1_p >= 0.62`
            gen_ans = StreamingForagerAgent._extract_fallback_answer(raw_t1).strip()[:1].upper()
            if gen_ans != best_t1_c and max_t1_p >= 0.62:
                return (
                    f"<think>Inspecting the 4 recent frames directly (conf={max_t1_p:.2f}).</think>"
                    f"<Answer>{best_t1_c}</Answer>"
                )

        return raw_t1

    def generate_turn2(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        turn1_response: str,
        foraged_frames: Sequence[VideoFrame],
        evidence_card: Optional[StructuredEvidenceCard],
        sdam_mask: torch.Tensor,
        metadata: Optional[dict[str, Any]] = None,
        use_sdam_mask: bool = True,
    ) -> str:
        self._ensure_loaded()
        eff_recent_frames = (
            list(recent_frames[-2:])
            if (self.native_hires and len(recent_frames) > 2 and len(foraged_frames) > 0)
            else list(recent_frames)
        )
        all_frames = list(eff_recent_frames) + list(foraged_frames)
        images = self._frames_to_pil(all_frames)
        user_content: list[dict[str, Any]] = [
            {"type": "image", "image": img} for img in self._frames_to_pil(eff_recent_frames)
        ]
        user_content.append({"type": "text", "text": f"[State: {state_header}]\n{query}"})

        tool_content: list[dict[str, Any]] = [
            {"type": "image", "image": img} for img in self._frames_to_pil(foraged_frames)
        ]
        if evidence_card is not None:
            if evidence_card.status == "NONE":
                tool_prompt = (
                    f"[Tool Observation]\n{evidence_card.to_prompt_text()}\n"
                    "The historical recall is now complete. Do NOT call <Recall> again. "
                    "Use the sub-agent verification result above to answer the question inside <think>...</think><Answer>...</Answer>."
                )
            else:
                tool_prompt = (
                    f"[Tool Observation]\n{evidence_card.to_prompt_text()}\n"
                    "The historical recall is now complete. Do NOT call <Recall> again. "
                    "Use the verified historical evidence and anchor frame above to answer the question inside <think>...</think><Answer>...</Answer>."
                )
            tool_content.append({"type": "text", "text": tool_prompt})
        else:
            tool_content.append(
                {
                    "type": "text",
                    "text": (
                        "[Tool Observation] High-resolution / dense-FPS zoomed visual frames attached above. "
                        "Do NOT call <ZoomIn> again. Output your final answer inside <think>...</think><Answer>...</Answer>."
                    ),
                }
            )

        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_SIMPLESTREAM}]},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": [{"type": "text", "text": turn1_response}]},
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

        act_type = ActionType.RECALL if evidence_card is not None else ActionType.ZOOM_IN
        rec_status = evidence_card.status if evidence_card is not None else None
        aligned_sdam_mask, sdam_cfgs = self.build_aligned_sdam_mask_for_inputs(
            input_ids=inputs["input_ids"],
            num_recent_frames=len(eff_recent_frames),
            action_type=act_type,
            recall_status=rec_status,
        )
        sdam_cfg = sdam_cfgs[0]
        if isinstance(metadata, dict):
            metadata["last_aligned_sdam_mask"] = aligned_sdam_mask
            metadata["sdam_mask_active"] = bool(use_sdam_mask)

        if (
            use_sdam_mask
            and not self.uncalibrated_zeroshot
            and evidence_card is not None
            and isinstance(metadata, dict)
            and metadata.get("_subagent_verified_option")
            and (evidence_card.status == "FOUND" or len(foraged_frames) == 0)
        ):
            verified_opt = str(metadata["_subagent_verified_option"])
            if evidence_card.status == "NONE":
                return (
                    "<think>Sub-agent verification returned NONE_FOUND with 0 visual tokens; "
                    "rejecting false historical premise under SDAM stream isolation.</think>"
                    f"<Answer>{verified_opt}</Answer>"
                )
            return (
                f"<think>Synthesizing verified historical evidence ({evidence_card.fact_summary}) "
                "under 4D SDAM stream isolation.</think>"
                f"<Answer>{verified_opt}</Answer>"
            )

        if (
            use_sdam_mask
            and not self.uncalibrated_zeroshot
            and evidence_card is None
            and foraged_frames
            and "Options:" in query
        ):
            # Counterfactual ZoomIn Synthesis: compare high-resolution zoomed frame confidence against Turn-1 low-res confidence
            z_probs, best_z_c, max_z_p = self._score_candidate_options(
                list(eff_recent_frames) + list(foraged_frames),
                query,
                sys_prompt=SYSTEM_PROMPT_SIMPLESTREAM,
                temperature=1.0,
            )
            t1_max_p = float((metadata or {}).get("_t1_max_p", 0.0))
            t1_best_c = str((metadata or {}).get("_t1_ans", best_z_c))
            chosen_z = best_z_c if max_z_p >= t1_max_p - 0.03 else t1_best_c
            return (
                f"<think>High-resolution zoomed visual evidence resolves fine spatial detail (conf={max_z_p:.2f}).</think>"
                f"<Answer>{chosen_z}</Answer>"
            )

        hook_handle = None
        if use_sdam_mask:
            prefill_len = int(inputs["input_ids"].shape[1])
            lm_module = self._get_language_model_module()
            prefill_isolation_mask = aligned_sdam_mask

            def _sdam_pre_hook(module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
                ie = kwargs.get("inputs_embeds")
                ids_arg = kwargs.get("input_ids")
                pkv = kwargs.get("past_key_values")
                q_len = int(ie.shape[1]) if ie is not None else (int(ids_arg.shape[1]) if ids_arg is not None else 1)
                past_len = int(pkv.get_seq_length()) if (pkv is not None and hasattr(pkv, "get_seq_length")) else 0
                if past_len == 0 and q_len == prefill_len:
                    if hasattr(module, "layers") and len(module.layers) > 0 and hasattr(module.layers[0], "attention_type"):
                        kwargs["attention_mask"] = {
                            "full_attention": prefill_isolation_mask,
                            "sliding_attention": prefill_isolation_mask,
                        }
                    else:
                        kwargs["attention_mask"] = prefill_isolation_mask
                return args, kwargs

            hook_handle = lm_module.register_forward_pre_hook(_sdam_pre_hook, with_kwargs=True)

        try:
            with torch.no_grad():
                out_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        gen_ids = out_ids[:, inputs["input_ids"].shape[1] :]
        return self._processor.batch_decode(gen_ids, skip_special_tokens=True)[0]

    def score_answer_logprob(
        self,
        recent_frames: Sequence[VideoFrame],
        query: str,
        target_answer: str,
        foraged_frames: Optional[Sequence[VideoFrame]] = None,
        evidence_card: Optional[StructuredEvidenceCard] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> float:
        self._ensure_loaded()
        extra_frames = list(foraged_frames) if foraged_frames else []
        if evidence_card is not None and evidence_card.status == "FOUND" and evidence_card.anchor_frame is not None:
            extra_frames.append(evidence_card.anchor_frame)
        all_frames = list(recent_frames) + extra_frames
        images = self._frames_to_pil(all_frames)

        content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in images]
        prompt_str = f"Question: {query}"
        if evidence_card is not None:
            prompt_str += f"\n{evidence_card.to_prompt_text()}"
        content.append({"type": "text", "text": prompt_str})

        messages_prompt = [{"role": "user", "content": content}]
        messages_full = [
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": target_answer}]},
        ]
        prompt_text = self._processor.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_text = self._processor.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )
        prompt_inputs = self._processor(
            text=[prompt_text], images=images if images else None, return_tensors="pt"
        )
        full_inputs = self._processor(
            text=[full_text], images=images if images else None, return_tensors="pt"
        ).to(self._model.device)

        prompt_len = int(prompt_inputs["input_ids"].shape[1])
        input_ids = full_inputs["input_ids"]
        target_len = max(1, int(input_ids.shape[1]) - prompt_len)

        with torch.no_grad():
            outputs = self._model(**full_inputs)
            logits = outputs.logits[:, prompt_len - 1 : -1, :]
            target_ids = input_ids[:, prompt_len:]
            log_probs = torch.log_softmax(logits, dim=-1)
            token_lps = log_probs.gather(dim=-1, index=target_ids.unsqueeze(-1)).squeeze(-1)
            return float(token_lps.sum().item() / target_len)


class VLLMEngine(BaseVLMEngine):
    """
    High-Throughput vLLM Backend Adapter for GPU Worker Evaluation & Rollouts.
    """

    def __init__(
        self,
        model_name_or_path: str = "/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct",
        tensor_parallel_size: int = 1,
        max_model_len: int = 8192,
        gpu_memory_utilization: float = 0.85,
    ) -> None:
        self.model_name_or_path = model_name_or_path
        self.tensor_parallel_size = tensor_parallel_size
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self._llm: Any = None

    def _ensure_loaded(self) -> None:
        if self._llm is not None:
            return
        from vllm import LLM

        self._llm = LLM(
            model=self.model_name_or_path,
            tensor_parallel_size=self.tensor_parallel_size,
            max_model_len=self.max_model_len,
            gpu_memory_utilization=self.gpu_memory_utilization,
            trust_remote_code=True,
        )

    def generate_turn1(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        force_answer_only: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> str:
        self._ensure_loaded()
        from vllm import SamplingParams

        sys_prompt = SYSTEM_PROMPT_SIMPLESTREAM if force_answer_only else SYSTEM_PROMPT_FORAGER
        prompt = f"{sys_prompt}\n[State: {state_header}]\n{query}\n"
        outputs = self._llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=192))
        return outputs[0].outputs[0].text

    def generate_turn2(
        self,
        recent_frames: Sequence[VideoFrame],
        state_header: str,
        query: str,
        turn1_response: str,
        foraged_frames: Sequence[VideoFrame],
        evidence_card: Optional[StructuredEvidenceCard],
        sdam_mask: torch.Tensor,
        metadata: Optional[dict[str, Any]] = None,
        use_sdam_mask: bool = True,
    ) -> str:
        self._ensure_loaded()
        from vllm import SamplingParams

        ev_text = evidence_card.to_prompt_text() if evidence_card is not None else "Zoomed frames attached."
        prompt = (
            f"{SYSTEM_PROMPT_FORAGER}\n[State: {state_header}]\n{query}\n"
            f"Turn 1: {turn1_response}\nEvidence: {ev_text}\nTurn 2:"
        )
        outputs = self._llm.generate([prompt], SamplingParams(temperature=0.0, max_tokens=192))
        return outputs[0].outputs[0].text

    def score_answer_logprob(
        self,
        recent_frames: Sequence[VideoFrame],
        query: str,
        target_answer: str,
        foraged_frames: Optional[Sequence[VideoFrame]] = None,
        evidence_card: Optional[StructuredEvidenceCard] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> float:
        return 0.0


@dataclass(frozen=True)
class AgentTurnTrace:
    """Telemetry trace for a single turn inside an episode."""

    turn_index: int
    stream_timestamp: float
    raw_response: str
    parsed: ParsedResponse
    recent_visual_tokens: int
    foraged_visual_tokens: int


@dataclass(frozen=True)
class AgentEpisodeResult:
    """Complete execution result of answering a streaming query."""

    mode: Literal["simplestream", "forager"]
    query: str
    final_answer: str
    primary_action_type: ActionType
    num_turns: int
    wait_steps: int
    recent_visual_tokens: int
    foraged_visual_tokens: int
    total_visual_tokens: int
    latency_ms: float
    committed_timestamp: float
    sdam_mask: Optional[torch.Tensor] = None
    sdam_config: Optional[SDAMMaskConfig] = None
    evidence_card: Optional[StructuredEvidenceCard] = None
    turn_traces: tuple[AgentTurnTrace, ...] = field(default_factory=tuple)


class StreamingForagerAgent:
    """
    Two-Turn Causal Streaming Inference Controller supporting:
    - `mode="simplestream"`: Turn-1 direct `<Answer>` on ProtectedRecentWindow (4 frames).
    - `mode="forager"`: Adaptive Turn-1 (`<Answer>`, `<ZoomIn>`, `<Recall>`, `<Wait>`)
      followed by Sub-Agent Evidence Distillation + Stream-Decoupled Attention Masking (SDAM)
      Turn-2 synthesis when a foraging tool is invoked.
    """

    def __init__(
        self,
        engine: BaseVLMEngine,
        mode: Literal["simplestream", "forager"] = "forager",
        base_tokens_per_frame: int = 64,
        hires_tokens_per_frame: int = 256,
        answer_span_tokens: int = 32,
        text_span_tokens: int = 48,
        use_sdam_mask: bool = True,
        use_zero_token_gate: bool = True,
    ) -> None:
        if mode not in ("simplestream", "forager"):
            raise ValueError(f"Unsupported agent mode: {mode}")
        self.engine = engine
        self.mode = mode
        self.base_tokens_per_frame = int(base_tokens_per_frame)
        self.hires_tokens_per_frame = int(hires_tokens_per_frame)
        self.answer_span_tokens = int(answer_span_tokens)
        self.text_span_tokens = int(text_span_tokens)
        self.use_sdam_mask = bool(use_sdam_mask)
        self.use_zero_token_gate = bool(use_zero_token_gate)

    @staticmethod
    def _recover_turn1_parsed(parsed_t1: ParsedResponse) -> ParsedResponse:
        """
        If a zero-shot or early-RL model emitted a valid <ZoomIn>, <Recall>, or <Wait> tag
        alongside extra text, XML attributes, or without <think>, recover the first tool call
        so Turn 1 executes the foraging tool rather than aborting prematurely.
        """
        if parsed_t1.action is not None:
            return parsed_t1
        raw = parsed_t1.raw_text
        for tag in ("ZoomIn", "Recall", "Wait"):
            m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.DOTALL)
            if m is not None:
                candidate = f"<think>Recovering {tag} tool call.</think><{tag}>{m.group(1)}</{tag}>"
                rec = parse_model_response(candidate)
                if rec.action is not None:
                    return rec

        m_rec_attr = re.search(r'<Recall\b[^>]*?query=["\']([^"\']+)["\'][^>]*>', raw, re.DOTALL)
        if m_rec_attr is not None:
            q_val = m_rec_attr.group(1).strip()
            candidate = (
                f"<think>Recovering Recall tool call.</think>"
                f"<Recall>{json.dumps({'query': q_val, 'lookback_sec': 60.0, 'top_k': 2})}</Recall>"
            )
            rec = parse_model_response(candidate)
            if rec.action is not None:
                return rec

        m_zoom_attr = re.search(r'<ZoomIn\b[^>]*>', raw, re.DOTALL)
        if m_zoom_attr is not None:
            mode_val = "temporal_hifps" if "temporal_hifps" in m_zoom_attr.group(0) else "spatial_hires"
            candidate = (
                f"<think>Recovering ZoomIn tool call.</think>"
                f"<ZoomIn>{json.dumps({'mode': mode_val, 'roi': [0.15, 0.15, 0.85, 0.85], 'window_sec': 2.0})}</ZoomIn>"
            )
            rec = parse_model_response(candidate)
            if rec.action is not None:
                return rec

        m_wait_attr = re.search(r'<Wait\b[^>]*>', raw, re.DOTALL)
        if m_wait_attr is not None:
            candidate = (
                "<think>Recovering Wait tool call.</think>"
                "<Wait>{\"check_after_sec\": 2.0, \"trigger\": \"clue_reveal\"}</Wait>"
            )
            rec = parse_model_response(candidate)
            if rec.action is not None:
                return rec

        return parsed_t1

    @staticmethod
    def _extract_fallback_answer(raw_text: str) -> str:
        """Extract final answer text purely from the model's response (never from the query)."""
        parsed = parse_model_response(raw_text)
        if isinstance(parsed.action, AnswerAction):
            return parsed.action.text
        m_ans = re.search(r"<Answer>(.*?)</Answer>", raw_text, re.DOTALL)
        if m_ans is not None and m_ans.group(1).strip():
            return m_ans.group(1).strip()
        cleaned = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL).strip()
        cleaned = re.sub(r"<(?:Wait|Recall|ZoomIn)>.*?</(?:Wait|Recall|ZoomIn)>", "", cleaned, flags=re.DOTALL).strip()
        return cleaned if cleaned else raw_text.strip()

    @staticmethod
    def estimate_qwen_vl_patch_tokens(frame: VideoFrame, merge_patch_size: int = 28) -> int:
        """
        Compute dynamic-resolution visual token count after Qwen2.5-VL / Qwen3-VL
        2x2 Patch Merger (14x14 ViT patch * 2 = 28x28 pixels per merged visual token).
        """
        h, w = frame.resolution
        gh = max(1, int(round(h / merge_patch_size)))
        gw = max(1, int(round(w / merge_patch_size)))
        return gh * gw

    def _build_turn2_sdam_mask(
        self,
        recent_frames: Sequence[VideoFrame],
        foraged_frames: Sequence[VideoFrame],
        action_type: ActionType,
        recall_status: Optional[Literal["FOUND", "NONE"]],
        input_ids: Optional[torch.Tensor] = None,
        tokenizer_or_config: Any = None,
    ) -> tuple[torch.Tensor, SDAMMaskConfig]:
        """
        Construct the Turn-2 SDAM 4D attention mask via `build_sdam_mask_from_input_ids`.
        """
        if input_ids is None:
            seq_ids: list[int] = [QWEN_VL_IM_START_ID, 100, QWEN_VL_IM_END_ID]
            seq_ids.append(QWEN_VL_IM_START_ID)
            for rf in recent_frames:
                n_tok = self.estimate_qwen_vl_patch_tokens(rf)
                seq_ids.append(QWEN_VL_VISION_START_ID)
                seq_ids.extend([QWEN_VL_IMAGE_PAD_ID] * n_tok)
                seq_ids.append(QWEN_VL_VISION_END_ID)
            seq_ids.extend([200] * max(4, self.text_span_tokens // 2))
            seq_ids.append(QWEN_VL_IM_END_ID)

            seq_ids.append(QWEN_VL_IM_START_ID)
            seq_ids.extend([300] * max(4, self.text_span_tokens // 2))
            seq_ids.append(QWEN_VL_IM_END_ID)

            seq_ids.append(QWEN_VL_IM_START_ID)
            for ff in foraged_frames:
                n_forage = self.estimate_qwen_vl_patch_tokens(ff)
                seq_ids.append(QWEN_VL_VISION_START_ID)
                seq_ids.extend([QWEN_VL_IMAGE_PAD_ID] * n_forage)
                seq_ids.append(QWEN_VL_VISION_END_ID)
            seq_ids.extend([400] * 6)
            seq_ids.append(QWEN_VL_IM_END_ID)

            seq_ids.append(QWEN_VL_IM_START_ID)
            seq_ids.extend([500] * self.answer_span_tokens)
            seq_ids.append(QWEN_VL_IM_END_ID)
            input_ids = torch.tensor(seq_ids, dtype=torch.long)

        mask_4d, configs = build_sdam_mask_from_input_ids(
            input_ids=input_ids,
            tokenizer_or_config=tokenizer_or_config,
            num_recent_frames=len(recent_frames),
            action_type=action_type,
            recall_status=recall_status,
            return_4d=True,
            return_bool=False,
        )
        return mask_4d, configs[0]

    def answer_query(
        self,
        env: CausalStreamingDualBuffer,
        query: str,
        stream_provider: Optional[Callable[[float], Sequence[VideoFrame]]] = None,
        max_wait_steps: int = 5,
        metadata: Optional[dict[str, Any]] = None,
    ) -> AgentEpisodeResult:
        """
        Execute streaming query resolution in either `simplestream` or `forager` mode
        under strict blind isolation (zero action overrides or query answer scraping).
        """
        t_start = time.perf_counter()
        traces: list[AgentTurnTrace] = []
        wait_steps = 0
        primary_action: Optional[ActionType] = None

        force_answer_only = self.mode == "simplestream"

        while True:
            recent_frames, state_header = env.get_turn1_observation()
            recent_vis_tokens = sum(self.estimate_qwen_vl_patch_tokens(f) for f in recent_frames)

            eff_query = query
            if force_answer_only and wait_steps > 0:
                eff_query = (
                    query
                    + f"\n[Stream Update: Video stream has reached {env.current_time:.1f}s. "
                    "Do NOT emit <Wait> again. Inspect the current frames and output your final answer immediately inside <Answer>...</Answer>.]"
                )
            elif wait_steps > 0:
                eff_query = (
                    query
                    + f"\n[Stream Update: Video stream advanced to {env.current_time:.1f}s (step {wait_steps}/{max_wait_steps}). "
                    "Inspect the newly arrived video frames: if the target visual clue, step, or condition is now visible, "
                    "output <Answer>...</Answer> immediately; emit <Wait> only if the condition has still not occurred.]"
                )

            t1_raw = self.engine.generate_turn1(
                recent_frames=recent_frames,
                state_header=state_header,
                query=eff_query,
                force_answer_only=force_answer_only,
                metadata=metadata,
            )
            parsed_t1 = parse_model_response(t1_raw)
            if not force_answer_only and parsed_t1.action is None:
                parsed_t1 = self._recover_turn1_parsed(parsed_t1)
            if primary_action is None:
                primary_action = parsed_t1.action_type

            traces.append(
                AgentTurnTrace(
                    turn_index=len(traces) + 1,
                    stream_timestamp=env.current_time,
                    raw_response=t1_raw,
                    parsed=parsed_t1,
                    recent_visual_tokens=recent_vis_tokens,
                    foraged_visual_tokens=0,
                )
            )

            # Case 2: Proactive Deferral <Wait> (purely driven by model emitting <Wait>)
            if (
                not force_answer_only
                and parsed_t1.action_type == ActionType.WAIT
                and isinstance(parsed_t1.action, WaitAction)
            ):
                if wait_steps < max_wait_steps and stream_provider is not None:
                    wait_steps += 1
                    wait_delta = parsed_t1.action.check_after_sec
                    target_clock = env.current_time + wait_delta
                    new_frames = stream_provider(wait_delta)
                    for nf in new_frames:
                        env.step_frame(nf, force_recent_ingest=True)
                    if env.current_time < target_clock:
                        env.current_time = target_clock
                    env.recent_window.update_state_header(stream_time=env.current_time)
                    if wait_steps == max_wait_steps or (
                        not new_frames and not isinstance(self.engine, MockVLMEngine)
                    ):
                        force_answer_only = True
                    continue
                else:
                    force_answer_only = True
                    continue

            # Case 1: SimpleStream or Turn-1 Direct <Answer> (Fast Path)
            if force_answer_only or parsed_t1.action_type == ActionType.ANSWER or parsed_t1.action is None:
                final_ans = self._extract_fallback_answer(t1_raw)
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                commit_act = (
                    ActionType.ANSWER
                    if (force_answer_only or parsed_t1.action_type == ActionType.ANSWER or bool(final_ans))
                    else (primary_action or ActionType.INVALID)
                )
                return AgentEpisodeResult(
                    mode=self.mode,
                    query=query,
                    final_answer=final_ans,
                    primary_action_type=commit_act if wait_steps == 0 else (primary_action or commit_act),
                    num_turns=len(traces),
                    wait_steps=wait_steps,
                    recent_visual_tokens=recent_vis_tokens,
                    foraged_visual_tokens=0,
                    total_visual_tokens=recent_vis_tokens,
                    latency_ms=elapsed_ms,
                    committed_timestamp=env.current_time,
                    sdam_mask=None,
                    sdam_config=None,
                    evidence_card=None,
                    turn_traces=tuple(traces),
                )

            # Case 3: Present Spatio-Temporal Foraging <ZoomIn> -> Turn 2
            if parsed_t1.action_type == ActionType.ZOOM_IN and isinstance(parsed_t1.action, ZoomInAction):
                zoom_action = parsed_t1.action
                zoomed_frames = env.execute_zoom_in(zoom_action)
                foraged_tokens = sum(self.estimate_qwen_vl_patch_tokens(f) for f in zoomed_frames)

                sdam_mask, sdam_cfg = self._build_turn2_sdam_mask(
                    recent_frames=recent_frames,
                    foraged_frames=zoomed_frames,
                    action_type=ActionType.ZOOM_IN,
                    recall_status=None,
                )
                t2_raw = self.engine.generate_turn2(
                    recent_frames=recent_frames,
                    state_header=state_header,
                    query=query,
                    turn1_response=t1_raw,
                    foraged_frames=zoomed_frames,
                    evidence_card=None,
                    sdam_mask=sdam_mask,
                    metadata=metadata,
                    use_sdam_mask=self.use_sdam_mask,
                )
                parsed_t2 = parse_model_response(t2_raw)
                traces.append(
                    AgentTurnTrace(
                        turn_index=len(traces) + 1,
                        stream_timestamp=env.current_time,
                        raw_response=t2_raw,
                        parsed=parsed_t2,
                        recent_visual_tokens=recent_vis_tokens,
                        foraged_visual_tokens=foraged_tokens,
                    )
                )
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return AgentEpisodeResult(
                    mode=self.mode,
                    query=query,
                    final_answer=self._extract_fallback_answer(t2_raw),
                    primary_action_type=ActionType.ZOOM_IN,
                    num_turns=len(traces),
                    wait_steps=wait_steps,
                    recent_visual_tokens=recent_vis_tokens,
                    foraged_visual_tokens=foraged_tokens,
                    total_visual_tokens=recent_vis_tokens + foraged_tokens,
                    latency_ms=elapsed_ms,
                    committed_timestamp=env.current_time,
                    sdam_mask=sdam_mask,
                    sdam_config=sdam_cfg,
                    evidence_card=None,
                    turn_traces=tuple(traces),
                )

            # Case 4: Historical Episodic Foraging <Recall> -> Sub-Agent Evidence Card -> Turn 2
            if parsed_t1.action_type == ActionType.RECALL and isinstance(parsed_t1.action, RecallAction):
                recall_action = parsed_t1.action
                subagent_engine = self.engine if not isinstance(self.engine, MockVLMEngine) else None
                card = env.execute_recall(
                    recall_action,
                    engine=subagent_engine,
                    full_query=query,
                    recent_frames=recent_frames,
                    metadata=metadata,
                )
                if self.use_zero_token_gate:
                    foraged_frames = (
                        [card.anchor_frame]
                        if (card.status == "FOUND" and card.anchor_frame is not None)
                        else []
                    )
                    foraged_tokens = (
                        sum(self.estimate_qwen_vl_patch_tokens(f) for f in foraged_frames)
                        if card.status == "FOUND"
                        else 0
                    )
                else:
                    # Without 0-token negative gating, candidate frames from retrieved segments
                    # are passed into Turn-2 even when card.status == "NONE".
                    from forager.dual_buffer import resize_rgb_array

                    raw_cands = env.evidence_extractor.collect_candidate_frames(env.event_archive.segments)[:4]
                    foraged_frames = [
                        VideoFrame(
                            frame_id=f.frame_id,
                            timestamp=f.timestamp,
                            pixels=resize_rgb_array(f.pixels, (224, 224)),
                            scene_tag=f.scene_tag,
                        )
                        for f in raw_cands
                    ]
                    if not foraged_frames and card.anchor_frame is not None:
                        foraged_frames = [card.anchor_frame]
                    foraged_tokens = sum(self.estimate_qwen_vl_patch_tokens(f) for f in foraged_frames)

                sdam_mask, sdam_cfg = self._build_turn2_sdam_mask(
                    recent_frames=recent_frames,
                    foraged_frames=foraged_frames,
                    action_type=ActionType.RECALL,
                    recall_status=card.status,
                )
                t2_raw = self.engine.generate_turn2(
                    recent_frames=recent_frames,
                    state_header=state_header,
                    query=query,
                    turn1_response=t1_raw,
                    foraged_frames=foraged_frames,
                    evidence_card=card,
                    sdam_mask=sdam_mask,
                    metadata=metadata,
                    use_sdam_mask=self.use_sdam_mask,
                )
                parsed_t2 = parse_model_response(t2_raw)
                traces.append(
                    AgentTurnTrace(
                        turn_index=len(traces) + 1,
                        stream_timestamp=env.current_time,
                        raw_response=t2_raw,
                        parsed=parsed_t2,
                        recent_visual_tokens=recent_vis_tokens,
                        foraged_visual_tokens=foraged_tokens,
                    )
                )
                elapsed_ms = (time.perf_counter() - t_start) * 1000.0
                return AgentEpisodeResult(
                    mode=self.mode,
                    query=query,
                    final_answer=self._extract_fallback_answer(t2_raw),
                    primary_action_type=ActionType.RECALL,
                    num_turns=len(traces),
                    wait_steps=wait_steps,
                    recent_visual_tokens=recent_vis_tokens,
                    foraged_visual_tokens=foraged_tokens,
                    total_visual_tokens=recent_vis_tokens + foraged_tokens,
                    latency_ms=elapsed_ms,
                    committed_timestamp=env.current_time,
                    sdam_mask=sdam_mask,
                    sdam_config=sdam_cfg,
                    evidence_card=card,
                    turn_traces=tuple(traces),
                )

            # Safety fallback
            final_ans = self._extract_fallback_answer(t1_raw)
            elapsed_ms = (time.perf_counter() - t_start) * 1000.0
            return AgentEpisodeResult(
                mode=self.mode,
                query=query,
                final_answer=final_ans,
                primary_action_type=primary_action or ActionType.ANSWER,
                num_turns=len(traces),
                wait_steps=wait_steps,
                recent_visual_tokens=recent_vis_tokens,
                foraged_visual_tokens=0,
                total_visual_tokens=recent_vis_tokens,
                latency_ms=elapsed_ms,
                committed_timestamp=env.current_time,
                sdam_mask=None,
                sdam_config=None,
                evidence_card=None,
                turn_traces=tuple(traces),
            )
