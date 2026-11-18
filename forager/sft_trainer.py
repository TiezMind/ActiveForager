"""
SDAM-SFT Dataset, Multi-Modal DataCollator, Vision-Tower Freezer, and Trainer (`forager/sft_trainer.py`).

Key Features:
1. `ForagerSFTDataset`:
   Reads 1-turn (`direct_answer`) and 2-turn (`zoom_in`, `recall_found`, `recall_none_hld`, `wait`)
   trajectories from `sft_coldstart_6k.jsonl` and tokenizes them into Qwen2.5-VL / Qwen3-VL chat format
   with `<|vision_start|><|image_pad|>...<|vision_end|>` blocks.
   Marks `labels = -100` on all System, User, and Tool/Evidence tokens so Cross-Entropy Loss is
   computed **exclusively** on Assistant Turn-1 and Turn-2 tokens.
2. `ForagerSDAMDataCollator`:
   Pads `input_ids` and `labels`, and automatically calls `build_sdam_mask_from_input_ids` on each
   sample to produce a batch-aligned 4D SDAM `attention_mask` of shape `(B, 1, L_max, L_max)`.
3. `freeze_vision_tower`:
   Freezes vision encoder parameters (`visual.` / `vision_tower.`) while keeping LLM and projector
   trainable.
4. `RollingCheckpointManager`:
   Enforces the platform disk guardrail: writes intermediate checkpoints to local NVMe `/tmp/forager_ckpts`
   keeping at most `max_keep=2` checkpoints, and exports final weights sequentially to HDFS.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from forager.protocol import ActionType
from forager.sdam_mask import (
    QWEN_VL_IM_END_ID,
    QWEN_VL_IM_START_ID,
    QWEN_VL_IMAGE_PAD_ID,
    QWEN_VL_VISION_END_ID,
    QWEN_VL_VISION_START_ID,
    SDAMMaskConfig,
    build_sdam_mask_from_input_ids,
)


IGNORE_INDEX: int = -100


@dataclass(frozen=True)
class SFTTrainingConfig:
    """Configuration for SDAM-SFT training."""

    model_name_or_path: str = "/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct"
    train_jsonl: str = "data/sft_coldstart_6k.jsonl"
    local_ckpt_dir: str = "/tmp/forager_ckpts/sft/qwen25vl_7b_sft"
    hdfs_export_dir: str = "/mnt/hdfs/lipengyu/DA/checkpoints/forager/sft/qwen25vl_7b_sft"
    max_seq_len: int = 3072
    recent_frames: int = 4
    tokens_per_recent_frame: int = 16
    tokens_per_foraged_frame: int = 32
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    num_epochs: int = 2
    micro_batch_size: int = 2
    grad_accum_steps: int = 2
    max_grad_norm: float = 1.0
    max_ckpts_to_keep: int = 2
    save_every_steps: int = 100


def _encode_text_segment(tokenizer: Any, text: str) -> list[int]:
    if tokenizer is not None and callable(getattr(tokenizer, "encode", None)):
        return [int(x) for x in tokenizer.encode(text, add_special_tokens=False)]
    # Deterministic lightweight fallback when running without an external HF tokenizer
    return [1000 + (ord(ch) % 500) for ch in text[:128]]


class ForagerSFTDataset(Dataset):
    """
    Multi-Modal SFT Dataset for 1-turn and 2-turn Forager trajectories.

    Constructs exact Qwen-VL `input_ids` with `<|im_start|>...<|im_end|>` and
    `<|vision_start|><|image_pad|>...<|vision_end|>`, plus `labels` masked with `-100`
    everywhere except Assistant Turn-1 (`turn1_target`) and Assistant Turn-2 (`turn2_target`).
    """

    def __init__(
        self,
        jsonl_path: Path | str,
        tokenizer: Any = None,
        max_seq_len: int = 3072,
        num_recent_frames: int = 4,
        tokens_per_recent_frame: int = 16,
        tokens_per_foraged_frame: int = 32,
        max_samples: Optional[int] = None,
    ) -> None:
        path = Path(jsonl_path)
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if max_samples is not None and max_samples > 0:
            rows = rows[:max_samples]
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_seq_len = int(max_seq_len)
        self.num_recent_frames = int(num_recent_frames)
        self.tokens_per_recent_frame = int(tokens_per_recent_frame)
        self.tokens_per_foraged_frame = int(tokens_per_foraged_frame)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        act_str = str(row.get("action_type", "Answer"))
        action_type = ActionType(act_str) if act_str in ActionType._value2member_map_ else ActionType.ANSWER
        recall_status = row.get("recall_status")

        input_ids: list[int] = []
        labels: list[int] = []

        def _append_untrained(token_ids: Sequence[int]) -> None:
            input_ids.extend(token_ids)
            labels.extend([IGNORE_INDEX] * len(token_ids))

        def _append_trained_assistant(assistant_text: str) -> None:
            # Header `<|im_start|>assistant\n` is untrained context; body + `<|im_end|>` are trained
            header_ids = [QWEN_VL_IM_START_ID] + _encode_text_segment(self.tokenizer, "assistant\n")
            _append_untrained(header_ids)
            body_ids = _encode_text_segment(self.tokenizer, assistant_text) + [QWEN_VL_IM_END_ID]
            input_ids.extend(body_ids)
            labels.extend(body_ids)

        # 1. System message (untrained)
        sys_ids = (
            [QWEN_VL_IM_START_ID]
            + _encode_text_segment(self.tokenizer, "system\nYou are an online streaming video assistant.")
            + [QWEN_VL_IM_END_ID]
        )
        _append_untrained(sys_ids)

        # 2. Turn-1 User message with ProtectedRecentWindow (w=4 frames) + question (untrained)
        _append_untrained([QWEN_VL_IM_START_ID] + _encode_text_segment(self.tokenizer, "user\n"))
        for _ in range(self.num_recent_frames):
            vis_block = (
                [QWEN_VL_VISION_START_ID]
                + ([QWEN_VL_IMAGE_PAD_ID] * self.tokens_per_recent_frame)
                + [QWEN_VL_VISION_END_ID]
            )
            _append_untrained(vis_block)
        q_ids = _encode_text_segment(self.tokenizer, str(row["prompt"])) + [QWEN_VL_IM_END_ID]
        _append_untrained(q_ids)

        # 3. Turn-1 Assistant decision (<think>...</think><Action>...</Action>) -> TRAINED!
        _append_trained_assistant(str(row["turn1_target"]))

        # 4. Optional Turn-2 Tool/Evidence message (untrained) + Turn-2 Assistant synthesis (TRAINED!)
        turn2_target = row.get("turn2_target")
        if turn2_target:
            _append_untrained([QWEN_VL_IM_START_ID] + _encode_text_segment(self.tokenizer, "user\n"))
            # Inject foraged visual tokens ONLY if ZoomIn or Recall(FOUND)
            # When Recall status == "NONE", strictly 0 visual tokens are injected!
            has_foraged_vis = (action_type == ActionType.ZOOM_IN) or (
                action_type == ActionType.RECALL and recall_status == "FOUND"
            )
            if has_foraged_vis:
                forage_block = (
                    [QWEN_VL_VISION_START_ID]
                    + ([QWEN_VL_IMAGE_PAD_ID] * self.tokens_per_foraged_frame)
                    + [QWEN_VL_VISION_END_ID]
                )
                _append_untrained(forage_block)

            ev_text = str(row.get("turn2_evidence") or "Evidence provided.")
            _append_untrained(_encode_text_segment(self.tokenizer, ev_text) + [QWEN_VL_IM_END_ID])

            # Turn-2 Assistant synthesis (<think>...</think><Answer>...</Answer>) -> TRAINED!
            _append_trained_assistant(str(turn2_target))

        if len(input_ids) > self.max_seq_len:
            input_ids = input_ids[: self.max_seq_len]
            labels = labels[: self.max_seq_len]

        return {
            "sample_id": row.get("sample_id", f"sft_{idx}"),
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "num_recent_frames": self.num_recent_frames,
            "action_type": action_type,
            "recall_status": recall_status,
        }


class ForagerSDAMDataCollator:
    """
    Collates variable-length multi-modal SFT samples into a padded batch with
    exact 4D Stream-Decoupled Attention Masks (`attention_mask` of shape `(B, 1, L_max, L_max)`).
    """

    def __init__(
        self,
        pad_token_id: int = 151643,
        tokenizer_or_config: Any = None,
        mask_dtype: torch.dtype = torch.float32,
    ) -> None:
        self.pad_token_id = int(pad_token_id)
        self.tokenizer_or_config = tokenizer_or_config
        self.mask_dtype = mask_dtype

    def __call__(self, features: Sequence[dict[str, Any]]) -> dict[str, Any]:
        if not features:
            raise ValueError("Cannot collate empty batch")

        batch_size = len(features)
        max_len = max(int(f["input_ids"].numel()) for f in features)

        padded_input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
        padded_labels = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=torch.long)
        sdam_masks_4d = torch.full(
            (batch_size, 1, max_len, max_len),
            float("-inf"),
            dtype=self.mask_dtype,
        )
        sdam_configs: list[SDAMMaskConfig] = []

        for b, feat in enumerate(features):
            ids_1d: torch.Tensor = feat["input_ids"]
            lbl_1d: torch.Tensor = feat["labels"]
            cur_len = int(ids_1d.numel())

            padded_input_ids[b, :cur_len] = ids_1d
            padded_labels[b, :cur_len] = lbl_1d

            mask_b, cfgs_b = build_sdam_mask_from_input_ids(
                input_ids=ids_1d,
                tokenizer_or_config=self.tokenizer_or_config,
                num_recent_frames=int(feat.get("num_recent_frames", 4)),
                action_type=feat.get("action_type", ActionType.ANSWER),
                recall_status=feat.get("recall_status"),
                return_4d=False,
                return_bool=False,
                dtype=self.mask_dtype,
            )
            sdam_masks_4d[b, 0, :cur_len, :cur_len] = mask_b
            # Ensure padded query positions have self-attention = 0.0 to avoid NaN softmax rows
            if cur_len < max_len:
                diag_idx = torch.arange(cur_len, max_len)
                sdam_masks_4d[b, 0, diag_idx, diag_idx] = 0.0
            sdam_configs.append(cfgs_b[0])

        return {
            "input_ids": padded_input_ids,
            "labels": padded_labels,
            "attention_mask": sdam_masks_4d,
            "sdam_configs": sdam_configs,
        }


def freeze_vision_tower(
    model: nn.Module,
    freeze_prefixes: tuple[str, ...] = ("visual.", "model.visual.", "vision_tower."),
    keep_trainable_substrings: tuple[str, ...] = ("merger", "mm_projector", "evidence_gate"),
) -> dict[str, int]:
    """
    Freeze vision encoder backbone parameters while keeping the patch merger/projector
    and LLM parameters trainable.
    """
    frozen_tensors = 0
    frozen_params = 0
    trainable_params = 0

    for name, param in model.named_parameters():
        should_freeze = any(name.startswith(pfx) or f".{pfx}" in name for pfx in freeze_prefixes)
        if should_freeze and not any(sub in name for sub in keep_trainable_substrings):
            param.requires_grad = False
            frozen_tensors += 1
            frozen_params += param.numel()
        else:
            param.requires_grad = True
            trainable_params += param.numel()

    return {
        "frozen_tensors": frozen_tensors,
        "frozen_params": frozen_params,
        "trainable_params": trainable_params,
    }


def compute_sdam_sft_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """
    Compute causal language modeling Cross-Entropy Loss over Assistant Turn-1 and Turn-2 tokens
    (`labels != -100`).
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.shape[-1]
    return F.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        ignore_index=IGNORE_INDEX,
    )


class RollingCheckpointManager:
    """
    Enforces INFRA_LESSONS.md Checkpoint Guardrails:
    1. Saves intermediate checkpoints strictly to worker local NVMe (`/tmp/forager_ckpts/...`).
    2. Retains at most `max_keep=2` intermediate checkpoints (deleting older ones by numeric step).
    3. Sequentially copies final exported HuggingFace weights to `/mnt/hdfs/lipengyu/DA/checkpoints/...`.
    """

    def __init__(
        self,
        local_root: Path | str,
        hdfs_export_root: Optional[Path | str] = None,
        max_keep: int = 2,
    ) -> None:
        if max_keep < 1:
            raise ValueError("max_keep must be >= 1")
        self.local_root = Path(local_root)
        self.hdfs_export_root = Path(hdfs_export_root) if hdfs_export_root is not None else None
        self.max_keep = int(max_keep)

    def save_step_checkpoint(
        self,
        step: int,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        metrics: Optional[dict[str, Any]] = None,
        save_hf_format: bool = False,
        source_model_dir: Optional[Path | str] = None,
    ) -> Path:
        self.local_root.mkdir(parents=True, exist_ok=True)
        step_dir = self.local_root / f"global_step_{int(step)}"
        step_dir.mkdir(parents=True, exist_ok=True)

        if save_hf_format and hasattr(model, "save_pretrained"):
            model.save_pretrained(step_dir, safe_serialization=True)
            if source_model_dir is not None:
                src_p = Path(source_model_dir)
                if src_p.is_dir():
                    for fname in (
                        "chat_template.json",
                        "preprocessor_config.json",
                        "tokenizer.json",
                        "tokenizer_config.json",
                        "vocab.json",
                        "merges.txt",
                        "generation_config.json",
                        "configuration.json",
                    ):
                        f_src = src_p / fname
                        f_dst = step_dir / fname
                        if f_src.is_file() and not f_dst.exists():
                            shutil.copyfile(f_src, f_dst)
        else:
            torch.save(model.state_dict(), step_dir / "model.pt")
        if optimizer is not None:
            torch.save(optimizer.state_dict(), step_dir / "optimizer.pt")
        meta = {"step": int(step), "metrics": metrics or {}}
        (step_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        (self.local_root / "latest_checkpointed_iteration.txt").write_text(f"{int(step)}\n", encoding="utf-8")

        self._prune_old_checkpoints()
        return step_dir

    def _prune_old_checkpoints(self) -> list[int]:
        step_dirs: list[tuple[int, Path]] = []
        for p in self.local_root.glob("global_step_*"):
            if p.is_dir():
                suffix = p.name.removeprefix("global_step_")
                if suffix.isdigit():
                    step_dirs.append((int(suffix), p))
        step_dirs.sort(key=lambda x: x[0])
        removed_steps: list[int] = []
        while len(step_dirs) > self.max_keep:
            old_step, old_path = step_dirs.pop(0)
            shutil.rmtree(old_path)
            removed_steps.append(old_step)
        return removed_steps

    def export_final_to_hdfs(self, step: int) -> Optional[Path]:
        if self.hdfs_export_root is None:
            return None
        src_dir = self.local_root / f"global_step_{int(step)}"
        if not src_dir.exists():
            raise FileNotFoundError(f"Source checkpoint directory missing: {src_dir}")
        self.hdfs_export_root.mkdir(parents=True, exist_ok=True)
        for src_file in sorted(src_dir.rglob("*")):
            if src_file.is_file():
                rel = src_file.relative_to(src_dir)
                dst_file = self.hdfs_export_root / rel
                dst_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src_file, dst_file)
        (self.hdfs_export_root / ".export_complete").write_text(f"global_step_{int(step)}\n", encoding="utf-8")
        return self.hdfs_export_root
