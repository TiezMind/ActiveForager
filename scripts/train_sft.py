#!/usr/bin/env python3
"""
SDAM-SFT Cold-Start Training Entrypoint (`scripts/train_sft.py`).

Supports:
- `--backend mock`: Runs a full end-to-end training & checkpointing loop on CPU with a
  lightweight SDAM-aware Transformer model for instant verification.
- `--backend hf`: Loads `Qwen2.5-VL-7B-Instruct` or `Qwen3-VL-8B-Instruct` on GPUs,
  freezes `visual.` tower, decodes real video frames (`pixel_values`, `image_grid_thw`)
  for multimodal forward passes while injecting 4D Stream-Decoupled Attention Masks
  (`build_sdam_mask_from_input_ids`) into the text decoder, saves rolling checkpoints to
  `/tmp/forager_ckpts/sft` (`max_keep=2`), and exports final weights to HDFS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from forager.protocol import ActionType
from forager.sdam_mask import build_sdam_mask_from_input_ids
from forager.sft_trainer import (
    IGNORE_INDEX,
    ForagerSDAMDataCollator,
    ForagerSFTDataset,
    RollingCheckpointManager,
    compute_sdam_sft_loss,
    freeze_vision_tower,
)
from scripts.run_gpu_probing import _load_real_probe_frames_for_record


class TinySDAMCausalLM(nn.Module):
    """Lightweight SDAM-aware Causal LM for CPU dry-run and unit test verification."""

    def __init__(self, vocab_size: int = 160000, hidden_size: int = 64) -> None:
        super().__init__()
        self.visual = nn.Linear(hidden_size, hidden_size)
        self.merger = nn.Linear(hidden_size, hidden_size)
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Any:
        x = self.embed_tokens(input_ids)  # (B, L, D)
        q = self.q_proj(x).unsqueeze(1)   # (B, 1, L, D)
        k = self.k_proj(x).unsqueeze(1)   # (B, 1, L, D)
        v = self.v_proj(x).unsqueeze(1)   # (B, 1, L, D)
        attn_out = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=attention_mask, is_causal=False
        )
        logits = self.lm_head(attn_out.squeeze(1))
        return type("CausalLMOutput", (), {"logits": logits})()


def _build_hf_multimodal_sft_inputs(
    row: dict[str, Any],
    processor: Any,
    device: torch.device,
    mask_dtype: torch.dtype,
) -> Optional[tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]]:
    """
    Build real multimodal `(model_inputs, labels, sdam_mask_4d)` with real decoded video frames
    (`pixel_values` and `image_grid_thw`) for Qwen2.5-VL / Qwen3-VL SDAM-SFT training.
    """
    real_bundle = _load_real_probe_frames_for_record(row)
    if real_bundle is None:
        return None
    recent_224, zoom_hires, hist_frames = real_bundle

    act_str = str(row.get("action_type", "Answer"))
    action_type = ActionType(act_str) if act_str in ActionType._value2member_map_ else ActionType.ANSWER
    recall_status = row.get("recall_status")

    from forager.agent import SYSTEM_PROMPT_FORAGER

    rec_pils = [Image.fromarray(f.pixels) for f in recent_224]
    all_pils = list(rec_pils)

    q_ts = float(row.get("query_timestamp", 24.0))
    raw_prompt = str(row["prompt"]).replace(
        "Respond with the option letter.",
        "Select the single best option letter (A, B, C, or D) inside <Answer>...</Answer> "
        "(or invoke a foraging tool first if the current 4 low-resolution frames are insufficient).",
    )
    u1_content: list[dict[str, Any]] = [{"type": "image", "image": img} for img in rec_pils]
    u1_content.append(
        {"type": "text", "text": f"[State: stream_time: {q_ts:.1f}s | window_frames: {len(rec_pils)}]\n{raw_prompt}"}
    )

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_FORAGER}]},
        {"role": "user", "content": u1_content},
        {"role": "assistant", "content": [{"type": "text", "text": str(row["turn1_target"])}]},
    ]

    turn2_target = row.get("turn2_target")
    if turn2_target:
        u2_content: list[dict[str, Any]] = []
        if action_type == ActionType.ZOOM_IN and zoom_hires:
            z_img = Image.fromarray(zoom_hires[0].pixels)
            u2_content.append({"type": "image", "image": z_img})
            all_pils.append(z_img)
        elif action_type == ActionType.RECALL and recall_status == "FOUND" and hist_frames:
            h_img = Image.fromarray(hist_frames[0].pixels)
            u2_content.append({"type": "image", "image": h_img})
            all_pils.append(h_img)
        u2_content.append({"type": "text", "text": str(row.get("turn2_evidence") or "Evidence provided.")})
        messages.append({"role": "user", "content": u2_content})
        messages.append({"role": "assistant", "content": [{"type": "text", "text": str(turn2_target)}]})

    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    model_inputs = processor(text=[full_text], images=all_pils, return_tensors="pt").to(device)
    input_ids = model_inputs["input_ids"]

    # Construct labels: mask everything with IGNORE_INDEX (-100) except assistant response spans
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    sdam_mask_4d, cfgs = build_sdam_mask_from_input_ids(
        input_ids=input_ids,
        tokenizer_or_config=processor,
        num_recent_frames=len(rec_pils),
        action_type=action_type,
        recall_status=recall_status,
        return_4d=True,
        return_bool=False,
        dtype=mask_dtype,
        device=device,
    )
    for s, e in cfgs[0].answer_spans:
        labels[0, s:e] = input_ids[0, s:e]
    # Also supervise Turn-1 assistant tool decision (<think>...</think><Action>...</Action><|im_end|>)
    if turn2_target is not None:
        from forager.sdam_mask import QWEN_VL_IM_END_ID, QWEN_VL_IM_START_ID
        ids_1d = input_ids[0].tolist()
        im_starts = [i for i, tok in enumerate(ids_1d) if tok == QWEN_VL_IM_START_ID]
        im_ends = [i for i, tok in enumerate(ids_1d) if tok == QWEN_VL_IM_END_ID]
        # Message order: 0=system, 1=user1, 2=assistant1 (Turn 1), 3=user2, 4=assistant2 (Turn 2)
        if len(im_starts) >= 3:
            t1_start = im_starts[2] + 3  # skip `<|im_start|>assistant\n`
            t1_end = next((idx + 1 for idx in im_ends if idx > t1_start), im_starts[3] if len(im_starts) > 3 else len(ids_1d))
            labels[0, t1_start:t1_end] = input_ids[0, t1_start:t1_end]
    if torch.all(labels == IGNORE_INDEX):
        labels[0, -16:] = input_ids[0, -16:]

    return model_inputs, labels, sdam_mask_4d


def run_sft_training(
    train_jsonl: Path,
    local_ckpt_dir: Path,
    hdfs_export_dir: Optional[Path] = None,
    backend: str = "mock",
    model_path: str = "/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct",
    max_samples: Optional[int] = None,
    max_steps: int = 4,
    batch_size: int = 2,
    lr: float = 1e-4,
    max_ckpts_to_keep: int = 2,
    save_every_steps: int = 1,
) -> dict[str, Any]:
    tokenizer: Any = None
    processor: Any = None
    if backend == "hf":
        from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        cfg._attn_implementation = "sdpa"
        default_mrope = {"type": "mrope", "rope_type": "default", "mrope_section": [16, 24, 24]}
        if hasattr(cfg, "text_config") and cfg.text_config is not None:
            cfg.text_config._attn_implementation = "sdpa"
            if getattr(cfg.text_config, "rope_scaling", None) is None and getattr(cfg, "model_type", "") == "qwen2_5_vl":
                cfg.text_config.rope_scaling = dict(default_mrope)
        if getattr(cfg, "rope_scaling", None) is None and getattr(cfg, "model_type", "") == "qwen2_5_vl":
            cfg.rope_scaling = dict(default_mrope)
        model: nn.Module = AutoModelForImageTextToText.from_pretrained(
            model_path,
            config=cfg,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
            attn_implementation="sdpa",
            trust_remote_code=True,
        )
    else:
        model = TinySDAMCausalLM()

    freeze_stats = freeze_vision_tower(model)
    device = next(model.parameters()).device
    mask_dtype = next(model.parameters()).dtype

    dataset = ForagerSFTDataset(
        jsonl_path=train_jsonl,
        tokenizer=tokenizer,
        max_samples=max_samples or (max_steps * batch_size * 4),
    )
    collator = ForagerSDAMDataCollator(tokenizer_or_config=tokenizer, mask_dtype=mask_dtype)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collator)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=0.01,
        foreach=False if backend == "hf" else None,
    )
    ckpt_mgr = RollingCheckpointManager(
        local_root=local_ckpt_dir,
        hdfs_export_root=hdfs_export_dir,
        max_keep=max_ckpts_to_keep,
    )

    model.train()
    step = 0
    step_losses: list[float] = []
    effective_save_freq = max(1, save_every_steps if backend == "mock" else max(save_every_steps, 50))

    raw_rows_with_frames = [
        r for r in dataset.rows
        if r.get("frame_cache_exists") or "videoholmes" in str(r.get("video_id", ""))
    ]
    if not raw_rows_with_frames:
        raw_rows_with_frames = dataset.rows

    # Balance SFT batches across trajectory types (direct_answer, zoom_in, recall_found, recall_none_hld, wait)
    # so the model does not overfit to 83% direct_answer on Video-Holmes!
    by_traj: dict[str, list[dict[str, Any]]] = {}
    for r in raw_rows_with_frames:
        by_traj.setdefault(str(r.get("trajectory_type", "direct_answer")), []).append(r)
    balanced_rows: list[dict[str, Any]] = []
    traj_Order = ("direct_answer", "zoom_in", "recall_found", "recall_none_hld", "wait")
    for i in range(max(len(v) for v in by_traj.values())):
        for k in traj_Order:
            lst = by_traj.get(k, [])
            if lst:
                balanced_rows.append(lst[i % len(lst)])
    raw_rows_with_frames = balanced_rows

    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            step += 1
            optimizer.zero_grad()

            used_multimodal = False
            if backend == "hf" and processor is not None and raw_rows_with_frames:
                row = raw_rows_with_frames[(step - 1) % len(raw_rows_with_frames)]
                mm_pack = _build_hf_multimodal_sft_inputs(row, processor, device, mask_dtype)
                if mm_pack is not None:
                    mm_inputs, mm_labels, sdam_4d = mm_pack
                    inner = getattr(model, "model", model)
                    lm_mod = getattr(inner, "language_model", None) or inner

                    def _sdam_hook(mod: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
                        kwargs["attention_mask"] = sdam_4d
                        return args, kwargs

                    h = lm_mod.register_forward_pre_hook(_sdam_hook, with_kwargs=True)
                    try:
                        outputs = model(**mm_inputs)
                    finally:
                        h.remove()
                    loss = compute_sdam_sft_loss(outputs.logits, mm_labels)
                    used_multimodal = True

            if not used_multimodal:
                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)
                attention_mask = batch["attention_mask"].to(device=device, dtype=mask_dtype)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = compute_sdam_sft_loss(outputs.logits, labels)

            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=0.2).item())
            optimizer.step()

            loss_val = float(loss.detach().cpu().item())
            step_losses.append(loss_val)
            if backend == "mock" or (step % effective_save_freq == 0) or (step == max_steps):
                ckpt_mgr.save_step_checkpoint(
                    step=step,
                    model=model,
                    optimizer=optimizer if backend == "mock" else None,
                    metrics={"loss": loss_val, "grad_norm": grad_norm, "multimodal": used_multimodal},
                    save_hf_format=(backend == "hf"),
                    source_model_dir=model_path if backend == "hf" else None,
                )

    exported_path = ckpt_mgr.export_final_to_hdfs(step=step) if step > 0 else None
    remaining_ckpts = sorted(p.name for p in local_ckpt_dir.glob("global_step_*") if p.is_dir())

    return {
        "backend": backend,
        "steps_completed": step,
        "step_losses": step_losses,
        "freeze_stats": freeze_stats,
        "remaining_local_ckpts": remaining_ckpts,
        "hdfs_exported_to": str(exported_path) if exported_path else None,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train SDAM-SFT Forager model.")
    ap.add_argument("--backend", choices=["mock", "hf"], default="mock")
    ap.add_argument("--model-path", type=str, default="/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--train-jsonl", type=Path, default=Path("data/sft_coldstart_6k.jsonl"))
    ap.add_argument("--local-ckpt-dir", type=Path, default=Path("/tmp/forager_ckpts/sft/qwen25vl_7b_sft"))
    ap.add_argument(
        "--hdfs-export-dir",
        type=Path,
        default=Path("/mnt/hdfs/lipengyu/DA/checkpoints/forager/sft/qwen25vl_7b_sft"),
    )
    ap.add_argument("--max-steps", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--max-ckpts-to-keep", type=int, default=2)
    args = ap.parse_args(argv)

    summary = run_sft_training(
        train_jsonl=args.train_jsonl,
        local_ckpt_dir=args.local_ckpt_dir,
        hdfs_export_dir=args.hdfs_export_dir,
        backend=args.backend,
        model_path=args.model_path,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        max_ckpts_to_keep=args.max_ckpts_to_keep,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
