"""
Stream-Decoupled Attention Masking (`forager/sdam_mask.py`):
Constructs 2D / 4D Causal + Block-Diagonal Attention Masks (M^SDAM) for
HuggingFace `Qwen2_5_VLForConditionalGeneration` / `Qwen3VLForConditionalGeneration` and PyTorch SDPA.

Enforces two structural invariants during Turn-2 synthesis:
1. Block-Diagonal Visual Stream Isolation:
   Tokens in `foraged_vis_spans` (I_forage) and tokens in `recent_vis_spans` (I_rec)
   cannot attend to each other (`M_{i,j} = -inf`), preventing cross-stream visual
   representation contamination.
2. Recency-Override Prevention on Verified Recall:
   When `action_type == ActionType.RECALL` and `recall_status == "FOUND"`,
   Turn-2 answer synthesis tokens (`answer_spans`, I_ans) are masked from attending
   to `recent_vis_spans` (I_rec) so present background clutter cannot override
   verified historical facts.

Also provides `extract_qwen_vl_spans_from_input_ids` and `build_sdam_mask_from_input_ids`
to automatically scan `input_ids` for `<|image_pad|>` / `<|video_pad|>` / `<|vision_pad|>`
connected components under arbitrary dynamic resolutions and Patch Merger token lengths.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn

from forager.protocol import ActionType

TokenSpan = tuple[int, int]

# Canonical special token IDs shared by Qwen2.5-VL and Qwen3-VL tokenizers
QWEN_VL_IM_START_ID: int = 151644      # <|im_start|>
QWEN_VL_IM_END_ID: int = 151645        # <|im_end|>
QWEN_VL_VISION_START_ID: int = 151652  # <|vision_start|>
QWEN_VL_VISION_END_ID: int = 151653    # <|vision_end|>
QWEN_VL_VISION_PAD_ID: int = 151654    # <|vision_pad|>
QWEN_VL_IMAGE_PAD_ID: int = 151655     # <|image_pad|>
QWEN_VL_VIDEO_PAD_ID: int = 151656     # <|video_pad|>


@dataclass(frozen=True)
class QwenVLSpecialTokenIds:
    """Resolved special token IDs for Qwen2.5-VL / Qwen3-VL sequence scanning."""

    im_start_id: int = QWEN_VL_IM_START_ID
    im_end_id: int = QWEN_VL_IM_END_ID
    vision_start_id: int = QWEN_VL_VISION_START_ID
    vision_end_id: int = QWEN_VL_VISION_END_ID
    vision_pad_ids: tuple[int, ...] = (
        QWEN_VL_VISION_PAD_ID,
        QWEN_VL_IMAGE_PAD_ID,
        QWEN_VL_VIDEO_PAD_ID,
    )


@dataclass(frozen=True)
class SDAMMaskConfig:
    """Configuration and extracted token intervals for Stream-Decoupled Attention Mask."""

    seq_len: int
    recent_vis_spans: tuple[TokenSpan, ...] = ()
    foraged_vis_spans: tuple[TokenSpan, ...] = ()
    text_spans: tuple[TokenSpan, ...] = ()
    answer_spans: tuple[TokenSpan, ...] = ()
    action_type: Union[ActionType, str] = ActionType.ANSWER
    recall_status: Optional[Literal["FOUND", "NONE"]] = None

    @property
    def num_recent_vis_tokens(self) -> int:
        return sum(e - s for s, e in self.recent_vis_spans)

    @property
    def num_foraged_vis_tokens(self) -> int:
        return sum(e - s for s, e in self.foraged_vis_spans)


def resolve_qwen_vl_special_tokens(tokenizer_or_config: Any = None) -> QwenVLSpecialTokenIds:
    """
    Resolve Qwen2.5-VL / Qwen3-VL special token IDs from a HuggingFace Processor,
    Tokenizer, PretrainedConfig, or dict. Falls back to canonical Qwen-VL IDs (151644..151656).
    """
    if tokenizer_or_config is None:
        return QwenVLSpecialTokenIds()

    if isinstance(tokenizer_or_config, QwenVLSpecialTokenIds):
        return tokenizer_or_config

    im_start = QWEN_VL_IM_START_ID
    im_end = QWEN_VL_IM_END_ID
    vis_start = QWEN_VL_VISION_START_ID
    vis_end = QWEN_VL_VISION_END_ID
    pad_ids: set[int] = {
        QWEN_VL_VISION_PAD_ID,
        QWEN_VL_IMAGE_PAD_ID,
        QWEN_VL_VIDEO_PAD_ID,
    }

    # Case 1: Dictionary mapping
    if isinstance(tokenizer_or_config, dict):
        im_start = int(tokenizer_or_config.get("im_start_token_id", im_start))
        im_end = int(tokenizer_or_config.get("im_end_token_id", im_end))
        vis_start = int(tokenizer_or_config.get("vision_start_token_id", vis_start))
        vis_end = int(tokenizer_or_config.get("vision_end_token_id", vis_end))
        custom_pads: set[int] = set()
        for k in ("vision_token_id", "vision_pad_token_id", "image_token_id", "video_token_id"):
            if k in tokenizer_or_config and tokenizer_or_config[k] is not None:
                custom_pads.add(int(tokenizer_or_config[k]))
        if "vision_pad_ids" in tokenizer_or_config:
            custom_pads.update(int(x) for x in tokenizer_or_config["vision_pad_ids"])
        if custom_pads:
            pad_ids = custom_pads
        return QwenVLSpecialTokenIds(
            im_start_id=im_start,
            im_end_id=im_end,
            vision_start_id=vis_start,
            vision_end_id=vis_end,
            vision_pad_ids=tuple(sorted(pad_ids)),
        )

    # Case 2: Processor wrapping a `.tokenizer`
    tok = getattr(tokenizer_or_config, "tokenizer", tokenizer_or_config)
    convert_fn = getattr(tok, "convert_tokens_to_ids", None)
    unk_id = getattr(tok, "unk_token_id", None)

    if callable(convert_fn):
        def _safe_id(token_str: str, default_val: int) -> int:
            val = convert_fn(token_str)
            if isinstance(val, int) and val >= 0 and (unk_id is None or val != unk_id):
                return val
            return default_val

        im_start = _safe_id("<|im_start|>", im_start)
        im_end = _safe_id("<|im_end|>", im_end)
        vis_start = _safe_id("<|vision_start|>", vis_start)
        vis_end = _safe_id("<|vision_end|>", vis_end)
        resolved_pads: set[int] = set()
        for t_str, d_val in (
            ("<|vision_pad|>", QWEN_VL_VISION_PAD_ID),
            ("<|image_pad|>", QWEN_VL_IMAGE_PAD_ID),
            ("<|video_pad|>", QWEN_VL_VIDEO_PAD_ID),
        ):
            resolved_pads.add(_safe_id(t_str, d_val))
        pad_ids = resolved_pads

    # Case 3: PretrainedConfig attributes (may exist on model.config or processor)
    for attr, target in (
        ("vision_start_token_id", "vis_start"),
        ("vision_end_token_id", "vis_end"),
    ):
        val = getattr(tokenizer_or_config, attr, None)
        if isinstance(val, int) and val >= 0:
            if target == "vis_start":
                vis_start = val
            else:
                vis_end = val

    for attr in ("vision_token_id", "image_token_id", "video_token_id"):
        val = getattr(tokenizer_or_config, attr, None)
        if isinstance(val, int) and val >= 0:
            pad_ids.add(val)

    return QwenVLSpecialTokenIds(
        im_start_id=im_start,
        im_end_id=im_end,
        vision_start_id=vis_start,
        vision_end_id=vis_end,
        vision_pad_ids=tuple(sorted(pad_ids)),
    )


def extract_qwen_vl_spans_from_input_ids(
    input_ids: Union[torch.Tensor, np.ndarray, Sequence[int]],
    tokenizer_or_config: Any = None,
    num_recent_frames: int = 4,
    action_type: Union[ActionType, str] = ActionType.ANSWER,
    recall_status: Optional[Literal["FOUND", "NONE"]] = None,
    answer_start_idx: Optional[int] = None,
    answer_token_len: Optional[int] = None,
    include_vision_delimiters: bool = False,
) -> SDAMMaskConfig:
    """
    Scan a Qwen2.5-VL / Qwen3-VL `input_ids` sequence to extract exact token spans:
    - `recent_vis_spans`: First `num_recent_frames` visual connected components (before Turn-1
      assistant boundary if present), supporting arbitrary per-frame dynamic resolutions.
    - `foraged_vis_spans`: Subsequent visual connected components injected after Turn-1
      (from `<ZoomIn>` crops/bursts or `<Recall>(FOUND)` anchor frame).
    - `answer_spans`: Turn-2 assistant generation / synthesis region `[ans_start, seq_len)`.
    - `text_spans`: All remaining prompt, thought, tool-call, and evidence-card text intervals.

    Args:
        input_ids: 1D sequence of token IDs (or 2D tensor with batch_size=1).
        tokenizer_or_config: Optional HF Tokenizer, Processor, Config, or dict of token IDs.
        num_recent_frames: Number of frames in ProtectedRecentWindow (default 4; may be 1 under 2D-SDG).
        action_type: Orthogonal action executed in Turn 1.
        recall_status: "FOUND" or "NONE" when `action_type` is Recall.
        answer_start_idx: Optional explicit start index of Turn-2 answer tokens.
        answer_token_len: Optional fallback length of trailing answer tokens when chat tags are absent.
        include_vision_delimiters: If True, includes enclosing `<|vision_start|>` and `<|vision_end|>`
            tokens in visual spans; if False (default), spans cover the inner `<|image_pad|>` /
            `<|video_pad|>` / `<|vision_pad|>` patch tokens.

    Returns:
        `SDAMMaskConfig` containing exact `(start, end)` intervals covering `[0, seq_len)`.
    """
    if isinstance(input_ids, torch.Tensor):
        ids_t = input_ids.detach().cpu()
        if ids_t.dim() == 2:
            if ids_t.shape[0] != 1:
                raise ValueError(
                    f"extract_qwen_vl_spans_from_input_ids expects 1D input_ids or (1, L), got shape {tuple(ids_t.shape)}"
                )
            ids_t = ids_t[0]
        elif ids_t.dim() != 1:
            raise ValueError(f"Expected 1D or 2D input_ids tensor, got dim={ids_t.dim()}")
        ids_list: list[int] = [int(x) for x in ids_t.tolist()]
    elif isinstance(input_ids, np.ndarray):
        arr = np.asarray(input_ids)
        if arr.ndim == 2:
            if arr.shape[0] != 1:
                raise ValueError(f"Expected 1D array or (1, L), got shape {arr.shape}")
            arr = arr[0]
        ids_list = [int(x) for x in arr.tolist()]
    else:
        ids_list = [int(x) for x in input_ids]

    seq_len = len(ids_list)
    if seq_len <= 0:
        raise ValueError("input_ids cannot be empty")
    if num_recent_frames < 0:
        raise ValueError(f"num_recent_frames must be >= 0, got {num_recent_frames}")

    spec = resolve_qwen_vl_special_tokens(tokenizer_or_config)
    pad_set = set(spec.vision_pad_ids)

    # Step 1: Identify all visual connected components in chronological order.
    # Supports both `<|vision_start|> ... <|vision_end|>` delimited blocks and
    # bare contiguous runs of `<|image_pad|>` / `<|video_pad|>` / `<|vision_pad|>`.
    covered = [False] * seq_len
    all_vis_spans: list[TokenSpan] = []

    idx = 0
    while idx < seq_len:
        if ids_list[idx] == spec.vision_start_id:
            # Find matching vision_end_id before any next vision_start_id
            end_idx = idx + 1
            while end_idx < seq_len and ids_list[end_idx] != spec.vision_end_id and ids_list[end_idx] != spec.vision_start_id:
                end_idx += 1
            if end_idx < seq_len and ids_list[end_idx] == spec.vision_end_id and end_idx > idx + 1:
                # Check if there are multiple contiguous pad runs separated inside, or one block
                inner_s = idx + 1
                inner_e = end_idx
                span_s = idx if include_vision_delimiters else inner_s
                span_e = (end_idx + 1) if include_vision_delimiters else inner_e
                all_vis_spans.append((span_s, span_e))
                for k in range(idx, end_idx + 1):
                    covered[k] = True
                idx = end_idx + 1
                continue
        idx += 1

    # Also scan for any bare contiguous runs of vision_pad_ids not enclosed by vision_start/end
    idx = 0
    while idx < seq_len:
        if not covered[idx] and ids_list[idx] in pad_set:
            run_s = idx
            while idx < seq_len and not covered[idx] and ids_list[idx] in pad_set:
                covered[idx] = True
                idx += 1
            run_e = idx
            if include_vision_delimiters:
                if run_s > 0 and ids_list[run_s - 1] == spec.vision_start_id:
                    run_s -= 1
                    covered[run_s] = True
                if run_e < seq_len and ids_list[run_e] == spec.vision_end_id:
                    covered[run_e] = True
                    run_e += 1
            all_vis_spans.append((run_s, run_e))
        else:
            idx += 1

    all_vis_spans.sort(key=lambda sp: sp[0])

    # Step 2: Partition visual connected components into `recent_vis_spans` and `foraged_vis_spans`.
    # If chat template `<|im_start|>` boundaries exist:
    # Message 0 is system (or user), first assistant message is Turn 1 response.
    # Any visual block appearing after the first `<|im_end|>` of the user Turn-1 prompt
    # (and after Turn-1 assistant start) is strictly a Turn-2 foraged visual block.
    im_start_positions = [i for i, t in enumerate(ids_list) if t == spec.im_start_id]

    recent_vis_spans: list[TokenSpan] = []
    foraged_vis_spans: list[TokenSpan] = []

    # Determine Turn-1 user message end if chat template is used with >= 3 messages
    turn1_cutoff: Optional[int] = None
    if len(im_start_positions) >= 3:
        # Typical chat structure:
        # im_start[0]: system, im_start[1]: user (Turn 1), im_start[2]: assistant (Turn 1)
        # Or if no system message: im_start[0]: user (Turn 1), im_start[1]: assistant (Turn 1)
        first_vis_pos = all_vis_spans[0][0] if all_vis_spans else 0
        for pos in im_start_positions:
            if pos > first_vis_pos:
                turn1_cutoff = pos
                break

    for sp_idx, span in enumerate(all_vis_spans):
        is_before_cutoff = turn1_cutoff is None or span[0] < turn1_cutoff
        if sp_idx < num_recent_frames and is_before_cutoff:
            recent_vis_spans.append(span)
        else:
            foraged_vis_spans.append(span)

    # Step 3: Determine `answer_spans` (Turn-2 assistant generation / synthesis region).
    last_vis_end = all_vis_spans[-1][1] if all_vis_spans else 0
    resolved_ans_start: Optional[int] = None

    if answer_start_idx is not None:
        resolved_ans_start = max(0, min(seq_len, int(answer_start_idx)))
    elif im_start_positions:
        last_im_start = im_start_positions[-1]
        if last_im_start >= last_vis_end:
            resolved_ans_start = last_im_start
    elif answer_token_len is not None and answer_token_len > 0:
        resolved_ans_start = max(last_vis_end, seq_len - int(answer_token_len))

    answer_spans: list[TokenSpan] = []
    if resolved_ans_start is not None and resolved_ans_start < seq_len:
        answer_spans.append((resolved_ans_start, seq_len))

    # Step 4: Group all remaining uncovered non-visual, non-answer positions into `text_spans`.
    non_text_mask = [False] * seq_len
    for s, e in recent_vis_spans:
        for k in range(s, e):
            non_text_mask[k] = True
    for s, e in foraged_vis_spans:
        for k in range(s, e):
            non_text_mask[k] = True
    for s, e in answer_spans:
        for k in range(s, e):
            non_text_mask[k] = True

    text_spans: list[TokenSpan] = []
    idx = 0
    while idx < seq_len:
        if not non_text_mask[idx]:
            ts = idx
            while idx < seq_len and not non_text_mask[idx]:
                idx += 1
            text_spans.append((ts, idx))
        else:
            idx += 1

    return SDAMMaskConfig(
        seq_len=seq_len,
        recent_vis_spans=tuple(recent_vis_spans),
        foraged_vis_spans=tuple(foraged_vis_spans),
        text_spans=tuple(text_spans),
        answer_spans=tuple(answer_spans),
        action_type=action_type,
        recall_status=recall_status,
    )


def _validate_and_build_index_mask(
    seq_len: int,
    spans: Sequence[TokenSpan],
    device: torch.device,
) -> torch.Tensor:
    """Convert a sequence of `[start, end)` spans into a 1D boolean mask of length `seq_len`."""
    mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
    for start, end in spans:
        s = int(start)
        e = int(end)
        if s < 0 or e > seq_len or s > e:
            raise ValueError(f"Invalid span ({s}, {e}) for seq_len={seq_len}")
        if s < e:
            mask[s:e] = True
    return mask


def build_sdam_attention_mask(
    seq_len: int,
    recent_vis_spans: Sequence[TokenSpan] = (),
    foraged_vis_spans: Sequence[TokenSpan] = (),
    text_spans: Sequence[TokenSpan] = (),
    answer_spans: Sequence[TokenSpan] = (),
    action_type: Union[ActionType, str] = ActionType.ANSWER,
    recall_status: Optional[Literal["FOUND", "NONE"]] = None,
    return_4d: bool = True,
    batch_size: int = 1,
    return_bool: bool = False,
    dtype: torch.dtype = torch.float32,
    neg_inf_value: Optional[float] = None,
    device: Union[torch.device, str] = "cpu",
) -> torch.Tensor:
    """
    Construct the Stream-Decoupled Attention Mask M^SDAM.

    Args:
        seq_len: Total sequence length L.
        recent_vis_spans: Token intervals `[s, e)` belonging to W_t^rec (I_rec).
        foraged_vis_spans: Token intervals `[s, e)` belonging to E_t (I_forage).
        text_spans: Token intervals `[s, e)` belonging to text prompt / thought / tool tags.
        answer_spans: Token intervals `[s, e)` belonging to Turn-2 synthesis / answer (I_ans).
        action_type: ActionType or string ("Answer", "ZoomIn", "Recall", "Wait").
        recall_status: "FOUND" or "NONE" when `action_type` is Recall.
        return_4d: If True, returns shape `(batch_size, 1, seq_len, seq_len)` ready for
            HF `Qwen2_5_VLForConditionalGeneration` / SDPA. If False, returns `(seq_len, seq_len)`.
        batch_size: Batch dimension B when `return_4d=True`.
        return_bool: If True, returns boolean tensor (`True` = attend, `False` = masked).
            If False, returns additive float tensor (`0.0` = attend, `-inf` = masked).
        dtype: Floating-point dtype when `return_bool=False`.
        neg_inf_value: Explicit mask fill value when `return_bool=False` (defaults to `-inf`).
        device: Torch device.

    Returns:
        `attention_mask` tensor of shape `(B, 1, L, L)` or `(L, L)`.
    """
    if seq_len <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")

    dev = torch.device(device)
    rec_mask_1d = _validate_and_build_index_mask(seq_len, recent_vis_spans, dev)
    forage_mask_1d = _validate_and_build_index_mask(seq_len, foraged_vis_spans, dev)
    _ = _validate_and_build_index_mask(seq_len, text_spans, dev)
    ans_mask_1d = _validate_and_build_index_mask(seq_len, answer_spans, dev)

    if torch.any(rec_mask_1d & forage_mask_1d):
        raise ValueError("recent_vis_spans and foraged_vis_spans must be disjoint")

    # 1. Base lower-triangular causal mask: query i can attend to key j <= i
    causal_allowed = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=dev))

    # 2. Block-diagonal visual stream isolation:
    # Block (i in I_forage, j in I_rec) AND symmetric (i in I_rec, j in I_forage)
    forage_to_rec = forage_mask_1d.unsqueeze(1) & rec_mask_1d.unsqueeze(0)
    rec_to_forage = rec_mask_1d.unsqueeze(1) & forage_mask_1d.unsqueeze(0)
    allowed = causal_allowed & (~forage_to_rec) & (~rec_to_forage)

    # 3. Prevent recency override when Recall succeeds (status == "FOUND"):
    # Block (i in I_ans, j in I_rec)
    act_str = action_type.value if isinstance(action_type, ActionType) else str(action_type)
    if act_str == ActionType.RECALL.value and recall_status == "FOUND":
        ans_to_rec = ans_mask_1d.unsqueeze(1) & rec_mask_1d.unsqueeze(0)
        allowed = allowed & (~ans_to_rec)

    if return_bool:
        out_2d = allowed
    else:
        fill_val = float("-inf") if neg_inf_value is None else float(neg_inf_value)
        out_2d = torch.zeros((seq_len, seq_len), dtype=dtype, device=dev)
        out_2d = out_2d.masked_fill(~allowed, fill_val)

    if not return_4d:
        return out_2d

    out_4d = out_2d.unsqueeze(0).unsqueeze(0)
    if batch_size > 1:
        out_4d = out_4d.expand(batch_size, 1, seq_len, seq_len).contiguous()
    return out_4d


def build_sdam_mask_from_input_ids(
    input_ids: Union[torch.Tensor, np.ndarray, Sequence[int]],
    tokenizer_or_config: Any = None,
    num_recent_frames: int = 4,
    action_type: Union[ActionType, str] = ActionType.ANSWER,
    recall_status: Optional[Literal["FOUND", "NONE"]] = None,
    answer_start_idx: Optional[int] = None,
    answer_token_len: Optional[int] = None,
    include_vision_delimiters: bool = False,
    return_4d: bool = True,
    return_bool: bool = False,
    dtype: torch.dtype = torch.float32,
    neg_inf_value: Optional[float] = None,
    device: Optional[Union[torch.device, str]] = None,
) -> tuple[torch.Tensor, list[SDAMMaskConfig]]:
    """
    End-to-end helper that scans 1D `(L,)` or 2D `(B, L)` `input_ids` for Qwen2.5-VL / Qwen3-VL
    visual connected components and builds the exact token-aligned SDAM attention mask.

    Returns:
        `(sdam_mask, configs_per_batch_item)`
    """
    if isinstance(input_ids, torch.Tensor):
        dev = input_ids.device if device is None else torch.device(device)
        ids_2d = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
    else:
        dev = torch.device("cpu" if device is None else device)
        arr = np.asarray(input_ids, dtype=np.int64)
        ids_2d = torch.from_numpy(arr).unsqueeze(0) if arr.ndim == 1 else torch.from_numpy(arr)

    if ids_2d.dim() != 2:
        raise ValueError(f"Expected 1D or 2D input_ids, got shape {tuple(ids_2d.shape)}")

    batch_size, seq_len = int(ids_2d.shape[0]), int(ids_2d.shape[1])
    configs: list[SDAMMaskConfig] = []
    masks_2d: list[torch.Tensor] = []

    for b in range(batch_size):
        cfg = extract_qwen_vl_spans_from_input_ids(
            input_ids=ids_2d[b],
            tokenizer_or_config=tokenizer_or_config,
            num_recent_frames=num_recent_frames,
            action_type=action_type,
            recall_status=recall_status,
            answer_start_idx=answer_start_idx,
            answer_token_len=answer_token_len,
            include_vision_delimiters=include_vision_delimiters,
        )
        configs.append(cfg)
        m2d = build_sdam_attention_mask(
            seq_len=seq_len,
            recent_vis_spans=cfg.recent_vis_spans,
            foraged_vis_spans=cfg.foraged_vis_spans,
            text_spans=cfg.text_spans,
            answer_spans=cfg.answer_spans,
            action_type=action_type,
            recall_status=recall_status,
            return_4d=False,
            return_bool=return_bool,
            dtype=dtype,
            neg_inf_value=neg_inf_value,
            device=dev,
        )
        masks_2d.append(m2d)

    if not return_4d:
        if batch_size == 1:
            return masks_2d[0], configs
        return torch.stack(masks_2d, dim=0), configs

    mask_4d = torch.stack(masks_2d, dim=0).unsqueeze(1)  # (B, 1, L, L)
    return mask_4d, configs


class EvidenceGate(nn.Module):
    """
    Learnable Evidence Gate Scalar gamma_gate in (0, 1) (initialized to 0.2).
    Scales retrieved historical visual/evidence representations to bound the
    attention distractor denominator in Turn-2 synthesis.
    """

    def __init__(self, init_value: float = 0.2) -> None:
        super().__init__()
        if not (0.0 < init_value < 1.0):
            raise ValueError(f"init_value must be in (0, 1), got {init_value}")
        init_logit = math.log(init_value / (1.0 - init_value))
        self.logit = nn.Parameter(torch.tensor(init_logit, dtype=torch.float32))

    @property
    def gate_value(self) -> torch.Tensor:
        return torch.sigmoid(self.logit)

    def forward(
        self,
        hidden_states: torch.Tensor,
        foraged_vis_spans: Sequence[TokenSpan] = (),
    ) -> torch.Tensor:
        """
        Scale tokens inside `foraged_vis_spans` by `gamma_gate = sigmoid(logit)`.
        Supports `hidden_states` of shape `(B, L, D)` or `(L, D)`.
        """
        if not foraged_vis_spans:
            return hidden_states

        seq_len = hidden_states.shape[-2]
        span_mask = _validate_and_build_index_mask(
            seq_len=seq_len,
            spans=foraged_vis_spans,
            device=hidden_states.device,
        )
        gamma = self.gate_value.to(dtype=hidden_states.dtype, device=hidden_states.device)
        scale = torch.where(span_mask, gamma, torch.ones_like(gamma)).unsqueeze(-1)
        if hidden_states.dim() == 3:
            scale = scale.unsqueeze(0)
        return hidden_states * scale
