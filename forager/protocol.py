"""
Action protocol parser, format anchoring reward (R_anchor), and
Cognitive Consistency Verification (V_CCV) for Spatio-Temporal Foraging Agent.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, Optional, Union


class ActionType(str, Enum):
    """Orthogonal four-way action space plus invalid fallback."""

    ANSWER = "Answer"
    ZOOM_IN = "ZoomIn"
    RECALL = "Recall"
    WAIT = "Wait"
    INVALID = "Invalid"


@dataclass(frozen=True)
class AnswerAction:
    """Direct commitment action: <Answer>...</Answer>."""

    text: str
    action_type: ActionType = field(default=ActionType.ANSWER, init=False)


@dataclass(frozen=True)
class ZoomInAction:
    """
    Present spatio-temporal foraging action:
    <ZoomIn>{"mode": "spatial_hires" | "temporal_hifps", "roi": [x1, y1, x2, y2], "window_sec": float}</ZoomIn>
    """

    mode: Literal["spatial_hires", "temporal_hifps"] = "spatial_hires"
    roi: tuple[float, float, float, float] = (0.0, 0.0, 1.0, 1.0)
    window_sec: float = 2.0
    offset_sec: float = 0.0
    action_type: ActionType = field(default=ActionType.ZOOM_IN, init=False)

    def __post_init__(self) -> None:
        if self.mode not in ("spatial_hires", "temporal_hifps"):
            raise ValueError(f"Invalid ZoomIn mode: {self.mode}")
        if len(self.roi) != 4:
            raise ValueError(f"ROI must have 4 coordinates [x1, y1, x2, y2], got {self.roi}")
        x1, y1, x2, y2 = [float(v) for v in self.roi]
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            raise ValueError(f"ROI coordinates must satisfy 0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, got {self.roi}")
        if self.window_sec <= 0.0:
            raise ValueError(f"window_sec must be positive, got {self.window_sec}")
        object.__setattr__(self, "roi", (x1, y1, x2, y2))
        object.__setattr__(self, "window_sec", float(self.window_sec))
        object.__setattr__(self, "offset_sec", float(self.offset_sec))


@dataclass(frozen=True)
class RecallAction:
    """
    Historical episodic foraging action:
    <Recall>{"query": str, "lookback_sec": float, "top_k": int}</Recall>
    """

    query: str
    lookback_sec: float = 60.0
    top_k: int = 2
    start_sec: Optional[float] = None
    end_sec: Optional[float] = None
    action_type: ActionType = field(default=ActionType.RECALL, init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.query, str) or not self.query.strip():
            raise ValueError("Recall query must be a non-empty string")
        if self.lookback_sec <= 0.0:
            raise ValueError(f"lookback_sec must be positive, got {self.lookback_sec}")
        if int(self.top_k) < 1:
            raise ValueError(f"top_k must be >= 1, got {self.top_k}")
        object.__setattr__(self, "query", self.query.strip())
        object.__setattr__(self, "lookback_sec", float(self.lookback_sec))
        object.__setattr__(self, "top_k", int(self.top_k))


@dataclass(frozen=True)
class WaitAction:
    """
    Proactive deferral action for Forward Active Responding:
    <Wait>{"check_after_sec": float, "trigger": str}</Wait>
    """

    check_after_sec: float = 1.0
    trigger: str = "next_event"
    action_type: ActionType = field(default=ActionType.WAIT, init=False)

    def __post_init__(self) -> None:
        if self.check_after_sec <= 0.0:
            raise ValueError(f"check_after_sec must be positive, got {self.check_after_sec}")
        if not isinstance(self.trigger, str) or not self.trigger.strip():
            raise ValueError("trigger must be a non-empty string")
        object.__setattr__(self, "check_after_sec", float(self.check_after_sec))
        object.__setattr__(self, "trigger", self.trigger.strip())


ParsedAction = Union[AnswerAction, ZoomInAction, RecallAction, WaitAction]


@dataclass(frozen=True)
class ParsedResponse:
    """Structured result of parsing a single-turn model output."""

    raw_text: str
    think_text: Optional[str]
    action_type: ActionType
    action: Optional[ParsedAction]
    has_valid_think: bool
    has_valid_action_tag: bool
    has_valid_payload: bool
    parse_error: Optional[str] = None

    @property
    def is_valid(self) -> bool:
        return self.has_valid_think and self.has_valid_action_tag and self.has_valid_payload


_THINK_FULL_RE = re.compile(r"^\s*<think>(.*?)</think>\s*(.*)$", re.DOTALL)
_THINK_PINNED_RE = re.compile(r"^\s*(.*?)</think>\s*(.*)$", re.DOTALL)
_ACTION_TAG_RE = re.compile(
    r"^\s*<(Answer|ZoomIn|Recall|Wait)>(.*?)</\1>\s*$",
    re.DOTALL,
)
_ANY_ACTION_OPEN_RE = re.compile(r"<(Answer|ZoomIn|Recall|Wait)>")


def _build_action_from_tag(tag_name: str, inner_text: str) -> ParsedAction:
    payload = inner_text.strip()
    if tag_name == "Answer":
        if not payload:
            raise ValueError("Empty <Answer> content")
        return AnswerAction(text=payload)

    if tag_name == "Wait" and not payload:
        return WaitAction()

    data: dict[str, Any] = json.loads(payload)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object inside <{tag_name}>, got {type(data).__name__}")

    if tag_name == "ZoomIn":
        roi_raw = data.get("roi", [0.0, 0.0, 1.0, 1.0])
        if not isinstance(roi_raw, (list, tuple)) or len(roi_raw) != 4:
            raise ValueError("ZoomIn 'roi' must be a 4-element list [x1, y1, x2, y2]")
        return ZoomInAction(
            mode=data.get("mode", "spatial_hires"),
            roi=(float(roi_raw[0]), float(roi_raw[1]), float(roi_raw[2]), float(roi_raw[3])),
            window_sec=float(data.get("window_sec", 2.0)),
            offset_sec=float(data.get("offset_sec", 0.0)),
        )

    if tag_name == "Recall":
        if "query" not in data:
            raise ValueError("Recall JSON must contain 'query'")
        return RecallAction(
            query=str(data["query"]),
            lookback_sec=float(data.get("lookback_sec", 60.0)),
            top_k=int(data.get("top_k", 2)),
            start_sec=float(data["start_sec"]) if "start_sec" in data and data["start_sec"] is not None else None,
            end_sec=float(data["end_sec"]) if "end_sec" in data and data["end_sec"] is not None else None,
        )

    if tag_name == "Wait":
        return WaitAction(
            check_after_sec=float(data.get("check_after_sec", 1.0)),
            trigger=str(data.get("trigger", "next_event")),
        )

    raise ValueError(f"Unsupported action tag: {tag_name}")


def parse_model_response(
    response_text: str,
    allow_pinned_think_prefix: bool = True,
) -> ParsedResponse:
    """
    Parse model output into <think> reasoning and one orthogonal action
    (<Answer>, <ZoomIn>, <Recall>, or <Wait>).

    Supports both full `<think>...</think>` and ParaVT-style pinned prefix
    where `<think>\\n` is part of the prompt and generation closes with `</think>`.
    """
    if not isinstance(response_text, str) or not response_text.strip():
        return ParsedResponse(
            raw_text=response_text or "",
            think_text=None,
            action_type=ActionType.INVALID,
            action=None,
            has_valid_think=False,
            has_valid_action_tag=False,
            has_valid_payload=False,
            parse_error="Empty response",
        )

    think_text: Optional[str] = None
    has_valid_think = False
    remainder = response_text

    m_full = _THINK_FULL_RE.match(response_text)
    if m_full is not None:
        candidate_think = m_full.group(1).strip()
        remainder = m_full.group(2)
        if candidate_think and "<think>" not in candidate_think and "</think>" not in candidate_think:
            think_text = candidate_think
            has_valid_think = True
    elif allow_pinned_think_prefix and "<think>" not in response_text:
        m_pinned = _THINK_PINNED_RE.match(response_text)
        if m_pinned is not None:
            candidate_think = m_pinned.group(1).strip()
            remainder = m_pinned.group(2)
            if candidate_think and "</think>" not in candidate_think:
                think_text = candidate_think
                has_valid_think = True

    # Ensure exactly one action tag in the remainder
    open_matches = _ANY_ACTION_OPEN_RE.findall(remainder)
    m_action = _ACTION_TAG_RE.match(remainder)
    if m_action is None or len(open_matches) != 1:
        return ParsedResponse(
            raw_text=response_text,
            think_text=think_text,
            action_type=ActionType.INVALID,
            action=None,
            has_valid_think=has_valid_think,
            has_valid_action_tag=False,
            has_valid_payload=False,
            parse_error="Missing, unclosed, or multiple action tags",
        )

    tag_name = m_action.group(1)
    inner_text = m_action.group(2)
    action_type = ActionType(tag_name)

    try:
        action_obj = _build_action_from_tag(tag_name, inner_text)
        return ParsedResponse(
            raw_text=response_text,
            think_text=think_text,
            action_type=action_type,
            action=action_obj,
            has_valid_think=has_valid_think,
            has_valid_action_tag=True,
            has_valid_payload=True,
            parse_error=None,
        )
    except Exception as exc:
        return ParsedResponse(
            raw_text=response_text,
            think_text=think_text,
            action_type=action_type,
            action=None,
            has_valid_think=has_valid_think,
            has_valid_action_tag=True,
            has_valid_payload=False,
            parse_error=str(exc),
        )


def compute_format_anchor_reward(
    response_text: str,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
    allow_pinned_think_prefix: bool = True,
) -> float:
    """
    Compute Parseability-Anchored Format Reward R_anchor in [0.0, 1.0].

    Decomposition (default weights = (0.4, 0.3, 0.3)):
    - alpha (0.4): Valid `<think>...</think>` closure with non-empty reasoning.
    - beta  (0.3): Valid single orthogonal action tag closure (`<Answer>`, `<ZoomIn>`, `<Recall>`, `<Wait>`).
    - gamma (0.3): Valid action payload (valid JSON + domain constraints for tools, or non-empty text for `<Answer>`).
    """
    alpha, beta, gamma = weights
    parsed = parse_model_response(
        response_text=response_text,
        allow_pinned_think_prefix=allow_pinned_think_prefix,
    )
    score = 0.0
    if parsed.has_valid_think:
        score += alpha
    if parsed.has_valid_action_tag:
        score += beta
    if parsed.has_valid_payload:
        score += gamma
    return round(float(score), 6)


_ZOOM_KEYWORDS = (
    "zoom",
    "crop",
    "high-res",
    "high res",
    "resolution",
    "blur",
    "blurry",
    "small",
    "tiny",
    "text",
    "ocr",
    "read",
    "detail",
    "fine-grained",
    "fast",
    "motion",
    "fps",
    "temporal_hifps",
    "spatial_hires",
    "roi",
    "closer",
    "unclear",
    "分辨",
    "模糊",
    "放大",
    "细节",
    "文字",
    "高帧率",
    "看清",
)

_RECALL_KEYWORDS = (
    "recall",
    "history",
    "historical",
    "past",
    "earlier",
    "before",
    "previous",
    "ago",
    "evicted",
    "memory",
    "beginning",
    "first",
    "sequence",
    "happened",
    "archive",
    "lookback",
    "之前",
    "历史",
    "回溯",
    "早些",
    "过去",
    "记忆",
    "刚才",
)

_WAIT_KEYWORDS = (
    "wait",
    "future",
    "not yet",
    "premature",
    "ongoing",
    "unfolding",
    "upcoming",
    "defer",
    "later",
    "next step",
    "unrevealed",
    "incomplete",
    "counting",
    "等待",
    "尚未",
    "还没",
    "未出现",
    "继续观察",
    "未来",
)

_DIRECT_SUFFICIENT_CONTRADICTIONS = (
    "no tool needed",
    "clearly visible right now without tools",
    "can answer directly without zooming or recalling",
    "不需要调用工具",
    "直接回答即可",
)


def verify_cognitive_consistency(
    think_text: Optional[str],
    action: Optional[ParsedAction],
) -> bool:
    """
    Verify Cognitive Consistency (V_CCV) between diagnostic `<think>` text
    and the chosen orthogonal action.

    Returns True iff the thought rationale aligns with the action regime:
    - ZoomInAction: thought diagnoses present spatial/motion perceptual uncertainty
      and does not claim tools are unnecessary.
    - RecallAction: thought diagnoses historical/episodic uncertainty or shares
      semantic tokens with `action.query`.
    - WaitAction: thought diagnoses causal prematurity / ongoing future event.
    - AnswerAction: thought is non-empty and does not exclusively demand a tool call.
    """
    if think_text is None or action is None:
        return False

    thought_lower = think_text.strip().lower()
    if len(thought_lower) < 4:
        return False

    if isinstance(action, ZoomInAction):
        if any(phrase in thought_lower for phrase in _DIRECT_SUFFICIENT_CONTRADICTIONS):
            return False
        has_zoom_cue = any(kw in thought_lower for kw in _ZOOM_KEYWORDS)
        # If mode is temporal_hifps, also check motion/temporal cues; if spatial_hires, spatial/detail cues
        return has_zoom_cue

    if isinstance(action, RecallAction):
        if any(phrase in thought_lower for phrase in _DIRECT_SUFFICIENT_CONTRADICTIONS):
            return False
        has_recall_cue = any(kw in thought_lower for kw in _RECALL_KEYWORDS)
        # Also check lexical overlap between query terms (len >= 3) and thought
        query_tokens = [
            tok for tok in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", action.query.lower()) if len(tok) >= 2
        ]
        has_query_overlap = any(tok in thought_lower for tok in query_tokens)
        return has_recall_cue and (has_query_overlap or len(query_tokens) == 0)

    if isinstance(action, WaitAction):
        if any(phrase in thought_lower for phrase in _DIRECT_SUFFICIENT_CONTRADICTIONS):
            return False
        return any(kw in thought_lower for kw in _WAIT_KEYWORDS)

    if isinstance(action, AnswerAction):
        # Must not explicitly state that it must call a tool right now instead of answering
        contradictory_answer = (
            "must call <zoomin>" in thought_lower
            or "must call <recall>" in thought_lower
            or "must emit <wait>" in thought_lower
            or "cannot answer without calling" in thought_lower
        )
        return not contradictory_answer

    return False
