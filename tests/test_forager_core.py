"""
Comprehensive CPU unit test suite (`tests/test_forager_core.py`) verifying:
1. Action protocol parsing, Parseability-Anchored Format Reward (R_anchor), and CCV (V_CCV).
2. Causal Streaming Dual-Buffer (ProtectedRecentWindow, HighResRingBuffer, OnlineEventArchive),
   Sub-Agent `NONE` zero-visual-token invariant, and 2D-SDG degradation gating.
3. Stream-Decoupled Attention Masking (SDAM) 2D/4D shapes, causal masking, visual stream
   isolation, recency-override blocking on `<Recall>(FOUND)`, `EvidenceGate`, AND
   real Qwen2.5-VL / Qwen3-VL `input_ids` connected-component scanning under arbitrary
   dynamic resolutions (`extract_qwen_vl_spans_from_input_ids` & `build_sdam_mask_from_input_ids`).
4. Counterfactual Information-Gain GRPO (`IG-GRPO`) reward, Std-Free Group-Mean advantage
   (including degenerate group protection), and Turn-Mean policy gradient loss.
5. Two-turn `StreamingForagerAgent` and end-to-end 12-task `OVO-Bench` evaluation harness
   on both synthetic fixtures and the real `/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from forager.agent import (
    MockVLMEngine,
    StreamingForagerAgent,
)
from forager.dual_buffer import (
    CausalStreamingDualBuffer,
    StructuredEvidenceCard,
    VideoFrame,
)
from forager.ig_grpo import (
    IGGRPOConfig,
    RolloutSample,
    compute_composite_reward,
    compute_counterfactual_ig_reward,
    compute_std_free_group_advantages,
    compute_turn_mean_grpo_loss,
)
from forager.protocol import (
    ActionType,
    AnswerAction,
    RecallAction,
    WaitAction,
    ZoomInAction,
    compute_format_anchor_reward,
    parse_model_response,
    verify_cognitive_consistency,
)
from forager.sdam_mask import (
    QWEN_VL_IM_END_ID,
    QWEN_VL_IM_START_ID,
    QWEN_VL_IMAGE_PAD_ID,
    QWEN_VL_VIDEO_PAD_ID,
    QWEN_VL_VISION_END_ID,
    QWEN_VL_VISION_PAD_ID,
    QWEN_VL_VISION_START_ID,
    EvidenceGate,
    build_sdam_attention_mask,
    build_sdam_mask_from_input_ids,
    extract_qwen_vl_spans_from_input_ids,
    resolve_qwen_vl_special_tokens,
)
from scripts.eval_ovobench import (
    ALL_OVOBENCH_TASKS,
    BACKWARD_TASKS,
    DEFAULT_OVOBENCH_JSON,
    DEFAULT_OVOBENCH_VIDEO_ROOT,
    FORWARD_ACTIVE_TASKS,
    REALTIME_TASKS,
    build_synthetic_ovobench_suite,
    evaluate_ovobench_samples,
    load_ovobench_from_json,
    main as ovobench_cli_main,
)
from forager.sft_trainer import (
    IGNORE_INDEX,
    ForagerSDAMDataCollator,
    ForagerSFTDataset,
)
from forager.verl_adapter import (
    build_verl_ig_grpo_command,
    export_rl_jsonl_to_verl_parquet,
    verl_compute_score,
    verl_compute_std_free_advantage,
)
from scripts.train_ig_grpo import run_mock_ig_grpo_loop
from scripts.train_sft import run_sft_training


def _make_dummy_frame(
    frame_id: int,
    timestamp: float,
    hw: tuple[int, int] = (448, 448),
    color: tuple[int, int, int] = (120, 80, 200),
    scene_tag: str = "living_room",
) -> VideoFrame:
    pixels = np.zeros((hw[0], hw[1], 3), dtype=np.uint8)
    pixels[:, :, 0] = color[0]
    pixels[:, :, 1] = color[1]
    pixels[:, :, 2] = color[2]
    return VideoFrame(frame_id=frame_id, timestamp=timestamp, pixels=pixels, scene_tag=scene_tag)


# ==============================================================================
# 1. Protocol Parsing, R_anchor, and V_CCV Tests
# ==============================================================================


def test_protocol_parse_all_four_orthogonal_actions() -> None:
    # 1. <Answer>
    r_ans = parse_model_response("<think>Object is clearly visible in current frame.</think><Answer>A</Answer>")
    assert r_ans.is_valid
    assert r_ans.action_type == ActionType.ANSWER
    assert isinstance(r_ans.action, AnswerAction)
    assert r_ans.action.text == "A"
    assert compute_format_anchor_reward(r_ans.raw_text) == pytest.approx(1.0)

    # 2. <ZoomIn>
    r_zoom = parse_model_response(
        '<think>The text on the sign is too small and blurry at 224x224; zooming in.</think>'
        '<ZoomIn>{"mode": "spatial_hires", "roi": [0.1, 0.2, 0.8, 0.9], "window_sec": 2.0}</ZoomIn>'
    )
    assert r_zoom.is_valid
    assert r_zoom.action_type == ActionType.ZOOM_IN
    assert isinstance(r_zoom.action, ZoomInAction)
    assert r_zoom.action.mode == "spatial_hires"
    assert r_zoom.action.roi == pytest.approx((0.1, 0.2, 0.8, 0.9))
    assert compute_format_anchor_reward(r_zoom.raw_text) == pytest.approx(1.0)

    # 3. <Recall>
    r_rec = parse_model_response(
        '<think>The red cup was placed earlier in history before the recent 4 frames.</think>'
        '<Recall>{"query": "red cup placed on table", "lookback_sec": 45.0, "top_k": 2}</Recall>'
    )
    assert r_rec.is_valid
    assert r_rec.action_type == ActionType.RECALL
    assert isinstance(r_rec.action, RecallAction)
    assert r_rec.action.query == "red cup placed on table"
    assert r_rec.action.lookback_sec == pytest.approx(45.0)
    assert r_rec.action.top_k == 2

    # 4. <Wait>
    r_wait = parse_model_response(
        '<think>The action step is still unfolding and not yet complete; waiting.</think>'
        '<Wait>{"check_after_sec": 1.5, "trigger": "step_completion"}</Wait>'
    )
    assert r_wait.is_valid
    assert r_wait.action_type == ActionType.WAIT
    assert isinstance(r_wait.action, WaitAction)
    assert r_wait.action.check_after_sec == pytest.approx(1.5)


def test_protocol_pinned_think_prefix_and_partial_format_rewards() -> None:
    pinned = "The fast motion requires 4 fps temporal sampling.</think><ZoomIn>{\"mode\": \"temporal_hifps\", \"roi\": [0, 0, 1, 1], \"window_sec\": 2.0}</ZoomIn>"
    parsed = parse_model_response(pinned, allow_pinned_think_prefix=True)
    assert parsed.is_valid
    assert compute_format_anchor_reward(pinned) == pytest.approx(1.0)

    bad_payload = "<think>Need to zoom in on blurry text.</think><ZoomIn>{\"mode\": \"invalid_mode\", \"roi\": [0.9, 0.1, 0.2, 0.3]}</ZoomIn>"
    assert compute_format_anchor_reward(bad_payload) == pytest.approx(0.7)

    think_only = "<think>Thinking about the video frame without emitting an action tag.</think>"
    assert compute_format_anchor_reward(think_only) == pytest.approx(0.4)

    assert compute_format_anchor_reward("random unstructured text") == pytest.approx(0.0)


def test_verify_cognitive_consistency_ccv() -> None:
    zoom_act = ZoomInAction(mode="spatial_hires", roi=(0.1, 0.1, 0.9, 0.9), window_sec=2.0)
    assert verify_cognitive_consistency("The OCR text is too blurry at low resolution; need to zoom in.", zoom_act)
    assert not verify_cognitive_consistency("The answer is clearly visible right now without tools.", zoom_act)
    assert not verify_cognitive_consistency("This happened 3 minutes ago in history.", zoom_act)

    recall_act = RecallAction(query="blue backpack", lookback_sec=60.0, top_k=2)
    assert verify_cognitive_consistency("The blue backpack appeared earlier in history before current window.", recall_act)
    assert not verify_cognitive_consistency("Zooming into current high-res patch.", recall_act)

    wait_act = WaitAction(check_after_sec=1.0, trigger="clue_reveal")
    assert verify_cognitive_consistency("The clue is not yet visible; we must wait for future frames.", wait_act)
    assert not verify_cognitive_consistency("The object was seen earlier in the past.", wait_act)

    ans_act = AnswerAction(text="B")
    assert verify_cognitive_consistency("The person is holding a guitar in the current frame.", ans_act)
    assert not verify_cognitive_consistency("I cannot answer without calling <Recall> right now.", ans_act)


# ==============================================================================
# 2. Dual-Buffer, Sub-Agent NONE 0-Token Invariant, and 2D-SDG Tests
# ==============================================================================


def test_dual_buffer_sliding_window_and_2d_sdg() -> None:
    env = CausalStreamingDualBuffer()

    for idx in range(6):
        color = (220, 30, 30) if idx < 2 else (30, 200, 50)
        tag = "early_red_object" if idx < 2 else "present_green_scene"
        env.step_frame(_make_dummy_frame(frame_id=idx, timestamp=float(idx), hw=(448, 448), color=color, scene_tag=tag))

    rec_frames, header = env.get_turn1_observation()
    assert len(rec_frames) == 4
    assert [f.timestamp for f in rec_frames] == [2.0, 3.0, 4.0, 5.0]
    assert all(f.resolution == (224, 224) for f in rec_frames)
    assert len(header.split()) <= 64

    zoom_spatial = env.execute_zoom_in(
        ZoomInAction(mode="spatial_hires", roi=(0.25, 0.25, 0.75, 0.75), window_sec=2.0)
    )
    assert len(zoom_spatial) == 1
    assert zoom_spatial[0].resolution == (448, 448)

    zoom_temporal = env.execute_zoom_in(
        ZoomInAction(mode="temporal_hifps", roi=(0.0, 0.0, 1.0, 1.0), window_sec=2.0)
    )
    assert len(zoom_temporal) >= 2
    assert all(f.resolution == (224, 224) for f in zoom_temporal)

    spat_frames = env.apply_2d_degradation("spatial")
    assert len(spat_frames) == 4
    assert all(f.resolution == (112, 112) for f in spat_frames)

    temp_frames = env.apply_2d_degradation("temporal")
    assert len(temp_frames) == 1
    assert temp_frames[0].timestamp == 5.0

    clean_frames = env.apply_2d_degradation("clean")
    assert len(clean_frames) == 4
    assert all(f.resolution == (224, 224) for f in clean_frames)


def test_sub_agent_evidence_card_found_vs_none_zero_visual_tokens() -> None:
    env = CausalStreamingDualBuffer()
    for idx in range(7):
        color = (240, 20, 20) if idx < 3 else (20, 40, 230)
        tag = "chef slicing carrots on cutting board" if idx < 3 else "chef washing hands at sink"
        env.step_frame(_make_dummy_frame(frame_id=idx, timestamp=float(idx), color=color, scene_tag=tag))

    card_found = env.execute_recall(RecallAction(query="chef slicing carrots", lookback_sec=30.0, top_k=2))
    assert card_found.status == "FOUND"
    assert card_found.anchor_frame is not None
    assert card_found.visual_token_count > 0

    card_none = env.execute_recall(RecallAction(query="astronaut riding a horse on mars", lookback_sec=30.0, top_k=2))
    assert card_none.status == "NONE"
    assert card_none.anchor_frame is None
    assert card_none.visual_token_count == 0

    with pytest.raises(ValueError, match="0 visual tokens"):
        StructuredEvidenceCard(
            status="NONE",
            fact_summary="Invalid card",
            anchor_frame=_make_dummy_frame(0, 0.0),
        )


# ==============================================================================
# 3. Stream-Decoupled Attention Masking (SDAM) & Qwen-VL Connected-Component Tests
# ==============================================================================


def test_sdam_attention_mask_shapes_and_blocking_rules() -> None:
    seq_len = 20
    rec_spans = [(0, 6)]
    txt_spans = [(6, 10)]
    forage_spans = [(10, 15)]
    ans_spans = [(15, 20)]

    mask_4d = build_sdam_attention_mask(
        seq_len=seq_len,
        recent_vis_spans=rec_spans,
        foraged_vis_spans=forage_spans,
        text_spans=txt_spans,
        answer_spans=ans_spans,
        action_type=ActionType.RECALL,
        recall_status="FOUND",
        return_4d=True,
        batch_size=2,
        return_bool=False,
    )
    assert mask_4d.shape == (2, 1, seq_len, seq_len)
    m2d = mask_4d[0, 0]

    assert torch.isinf(m2d[3, 4]) and m2d[3, 4] < 0
    assert torch.all(torch.isinf(m2d[10:15, 0:6]))
    assert m2d[12, 10].item() == 0.0
    assert torch.all(torch.isinf(m2d[15:20, 0:6]))
    assert torch.all(m2d[15:20, 6:15] == 0.0)

    mask_zoom_bool = build_sdam_attention_mask(
        seq_len=seq_len,
        recent_vis_spans=rec_spans,
        foraged_vis_spans=forage_spans,
        text_spans=txt_spans,
        answer_spans=ans_spans,
        action_type=ActionType.ZOOM_IN,
        recall_status=None,
        return_4d=False,
        return_bool=True,
    )
    assert mask_zoom_bool.shape == (seq_len, seq_len)
    assert mask_zoom_bool.dtype == torch.bool
    assert not torch.any(mask_zoom_bool[10:15, 0:6])
    assert torch.all(mask_zoom_bool[15:20, 0:6])
    assert torch.all(mask_zoom_bool[15:20, 10:15])


def test_qwen_vl_dynamic_resolution_connected_components_and_sdam_alignment() -> None:
    """
    Simulate a realistic Qwen2.5-VL / Qwen3-VL Turn-2 `input_ids` tensor with unequal
    dynamic-resolution visual token lengths across 4 recent frames + 2 foraged frames:
    - Recent frames (4 components): lengths [12, 18, 12, 24]
    - Turn-1 assistant tool call: `<think>...</think><ZoomIn>...</ZoomIn>`
    - Foraged frames (2 components): lengths [64, 48] (e.g. high-res ROI crops)
    - Turn-2 assistant synthesis: `<|im_start|>assistant\n<think>...</think><Answer>A</Answer><|im_end|>`
    """
    recent_lengths = [12, 18, 12, 24]
    foraged_lengths = [64, 48]

    ids: list[int] = [QWEN_VL_IM_START_ID, 10, 11, QWEN_VL_IM_END_ID]  # system
    ids.append(QWEN_VL_IM_START_ID)  # Turn-1 user start
    expected_recent_spans: list[tuple[int, int]] = []
    for length in recent_lengths:
        ids.append(QWEN_VL_VISION_START_ID)
        s = len(ids)
        ids.extend([QWEN_VL_IMAGE_PAD_ID] * length)
        e = len(ids)
        expected_recent_spans.append((s, e))
        ids.append(QWEN_VL_VISION_END_ID)
    ids.extend([201, 202, 203, QWEN_VL_IM_END_ID])  # question text

    # Turn-1 assistant response
    ids.extend([QWEN_VL_IM_START_ID, 301, 302, 303, 304, QWEN_VL_IM_END_ID])

    # Turn-2 user/tool message with 2 foraged high-res frames
    ids.append(QWEN_VL_IM_START_ID)
    expected_foraged_spans: list[tuple[int, int]] = []
    for length in foraged_lengths:
        ids.append(QWEN_VL_VISION_START_ID)
        s = len(ids)
        ids.extend([QWEN_VL_VIDEO_PAD_ID] * length)
        e = len(ids)
        expected_foraged_spans.append((s, e))
        ids.append(QWEN_VL_VISION_END_ID)
    ids.extend([401, 402, QWEN_VL_IM_END_ID])

    # Turn-2 assistant synthesis
    ans_start = len(ids)
    ids.extend([QWEN_VL_IM_START_ID, 501, 502, 503, 504, 505, QWEN_VL_IM_END_ID])
    seq_len = len(ids)

    input_ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)  # (1, L)

    mask_4d, configs = build_sdam_mask_from_input_ids(
        input_ids=input_ids,
        num_recent_frames=4,
        action_type=ActionType.RECALL,
        recall_status="FOUND",
        return_4d=True,
        return_bool=False,
    )
    cfg = configs[0]

    assert list(cfg.recent_vis_spans) == expected_recent_spans
    assert list(cfg.foraged_vis_spans) == expected_foraged_spans
    assert cfg.num_recent_vis_tokens == sum(recent_lengths)
    assert cfg.num_foraged_vis_tokens == sum(foraged_lengths)
    assert list(cfg.answer_spans) == [(ans_start, seq_len)]

    # Verify total partition covers [0, seq_len) with zero overlap
    covered = set()
    for group in (cfg.recent_vis_spans, cfg.foraged_vis_spans, cfg.text_spans, cfg.answer_spans):
        for s, e in group:
            span_set = set(range(s, e))
            assert not (covered & span_set), "Overlapping spans detected"
            covered |= span_set
    assert covered == set(range(seq_len))

    # Verify exact token-level blocking in 4D mask
    m2d = mask_4d[0, 0]
    for fs, fe in expected_foraged_spans:
        for rs, re_ in expected_recent_spans:
            assert torch.all(torch.isinf(m2d[fs:fe, rs:re_]))
    for rs, re_ in expected_recent_spans:
        assert torch.all(torch.isinf(m2d[ans_start:seq_len, rs:re_]))
    for fs, fe in expected_foraged_spans:
        assert torch.all(m2d[ans_start:seq_len, fs:fe] == 0.0)


def test_qwen_vl_real_tokenizer_integration_on_cpu() -> None:
    """
    Load the real Qwen2.5-VL-7B-Instruct and Qwen3-VL-8B-Instruct tokenizers from HDFS
    on CPU (takes <0.2s, 0 GPU memory) and verify special token resolution and chat template
    connected-component span extraction.
    """
    from transformers import AutoTokenizer

    model_paths = [
        Path("/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct"),
        Path("/mnt/hdfs/lipengyu/DA/models/Qwen3-VL-8B-Instruct"),
    ]
    for path in model_paths:
        if not path.exists():
            continue
        tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
        spec = resolve_qwen_vl_special_tokens(tokenizer)
        assert spec.vision_start_id == QWEN_VL_VISION_START_ID
        assert spec.vision_end_id == QWEN_VL_VISION_END_ID
        assert QWEN_VL_IMAGE_PAD_ID in spec.vision_pad_ids
        assert QWEN_VL_VIDEO_PAD_ID in spec.vision_pad_ids
        assert QWEN_VL_VISION_PAD_ID in spec.vision_pad_ids

        # Construct a real multi-turn chat string with uneven dynamic patch counts
        # Frame 1: 16 pads, Frame 2: 36 pads, Frame 3: 25 pads, Frame 4: 16 pads;
        # Turn-2 Foraged Anchor: 100 pads
        f1 = "<|vision_start|>" + ("<|image_pad|>" * 16) + "<|vision_end|>"
        f2 = "<|vision_start|>" + ("<|image_pad|>" * 36) + "<|vision_end|>"
        f3 = "<|vision_start|>" + ("<|image_pad|>" * 25) + "<|vision_end|>"
        f4 = "<|vision_start|>" + ("<|image_pad|>" * 16) + "<|vision_end|>"
        foraged = "<|vision_start|>" + ("<|image_pad|>" * 100) + "<|vision_end|>"

        chat_text = (
            "<|im_start|>system\nYou are a streaming video assistant.<|im_end|>\n"
            f"<|im_start|>user\n{f1}{f2}{f3}{f4}What did the person put in the drawer earlier?<|im_end|>\n"
            "<|im_start|>assistant\n<think>Occurred earlier in history.</think>"
            "<Recall>{\"query\": \"drawer\", \"lookback_sec\": 60.0, \"top_k\": 2}</Recall><|im_end|>\n"
            f"<|im_start|>user\n{foraged}<EvidenceCard status=\"FOUND\">Key placed in drawer.</EvidenceCard><|im_end|>\n"
            "<|im_start|>assistant\n<think>Evidence confirms key.</think><Answer>C</Answer><|im_end|>"
        )
        encoded = tokenizer(chat_text, return_tensors="pt", add_special_tokens=False)
        cfg = extract_qwen_vl_spans_from_input_ids(
            input_ids=encoded["input_ids"],
            tokenizer_or_config=tokenizer,
            num_recent_frames=4,
            action_type=ActionType.RECALL,
            recall_status="FOUND",
        )
        assert len(cfg.recent_vis_spans) == 4
        assert [e - s for s, e in cfg.recent_vis_spans] == [16, 36, 25, 16]
        assert len(cfg.foraged_vis_spans) == 1
        assert [e - s for s, e in cfg.foraged_vis_spans] == [100]
        assert len(cfg.answer_spans) == 1


def test_evidence_gate_module() -> None:
    gate = EvidenceGate(init_value=0.2)
    assert gate.gate_value.item() == pytest.approx(0.2, rel=1e-4)
    x = torch.ones((2, 10, 8), dtype=torch.float32, requires_grad=True)
    out = gate(x, foraged_vis_spans=[(4, 7)])
    assert out.shape == (2, 10, 8)
    assert torch.allclose(out[:, :4, :], torch.ones_like(out[:, :4, :]))
    assert torch.allclose(out[:, 4:7, :], torch.full_like(out[:, 4:7, :], 0.2), atol=1e-4)
    out.sum().backward()
    assert gate.logit.grad is not None


# ==============================================================================
# 4. IG-GRPO Reward, Std-Free Advantage, and Turn-Mean Loss Tests
# ==============================================================================


def test_counterfactual_ig_reward_and_composite_reward() -> None:
    r_ig_pos = compute_counterfactual_ig_reward(
        logprob_with_evidence=-0.10,
        logprob_without_evidence=-1.40,
        is_correct=True,
        ccv_valid=True,
        answer_token_len=2,
        clip_max=1.0,
    )
    assert r_ig_pos == pytest.approx(0.65)

    assert compute_counterfactual_ig_reward(-0.1, -1.4, is_correct=False, ccv_valid=True) == 0.0
    assert compute_counterfactual_ig_reward(-0.1, -1.4, is_correct=True, ccv_valid=False) == 0.0
    assert compute_counterfactual_ig_reward(-1.5, -0.5, is_correct=True, ccv_valid=True) == 0.0

    sample_zoom = RolloutSample(
        turn1_response='<think>Sign text is too blurry at 224x224; zooming in.</think><ZoomIn>{"mode": "spatial_hires", "roi": [0.1, 0.1, 0.9, 0.9], "window_sec": 2.0}</ZoomIn>',
        turn2_response="<think>High-res crop clearly shows Option A.</think><Answer>A</Answer>",
        final_answer="A",
        gt_answer="A",
        task_type="mcq",
        foraged_visual_tokens=128,
        logprob_with_evidence=-0.05,
        logprob_without_evidence=-1.38,
        answer_token_len=1,
    )
    breakdown = compute_composite_reward(sample_zoom, IGGRPOConfig())
    assert breakdown.is_correct
    assert breakdown.ccv_valid
    assert breakdown.used_foraging_tool
    assert breakdown.r_ig == pytest.approx(1.0)
    assert breakdown.r_anchor == pytest.approx(1.0)
    assert breakdown.total_reward > 1.5


def test_std_free_group_advantages_and_turn_mean_loss() -> None:
    adv_deg, is_deg = compute_std_free_group_advantages([1.2, 1.2, 1.2, 1.2])
    assert is_deg
    assert torch.all(adv_deg == 0.0)

    rewards = [1.5, 0.5, -0.5, 0.5]
    adv, is_deg2 = compute_std_free_group_advantages(rewards)
    assert not is_deg2
    assert torch.allclose(adv, torch.tensor([1.0, 0.0, -1.0, 0.0]))

    t_lp_0_t1 = torch.tensor([-0.3, -0.2], requires_grad=True)
    t_lp_0_t2 = torch.tensor([-0.1, -0.2, -0.1, -0.1], requires_grad=True)
    t_lp_1_t1 = torch.tensor([-0.4, -0.5, -0.3], requires_grad=True)

    turn_logprobs = [[t_lp_0_t1, t_lp_0_t2], [t_lp_1_t1]]
    old_turn_logprobs = [
        [t_lp_0_t1.detach().clone(), t_lp_0_t2.detach().clone()],
        [t_lp_1_t1.detach().clone()],
    ]
    ref_turn_logprobs = [
        [t_lp_0_t1.detach() - 0.05, t_lp_0_t2.detach() - 0.05],
        [t_lp_1_t1.detach() - 0.05],
    ]
    group_adv = torch.tensor([0.8, -0.8])

    loss = compute_turn_mean_grpo_loss(
        turn_logprobs=turn_logprobs,
        old_turn_logprobs=old_turn_logprobs,
        advantages=group_adv,
        ref_turn_logprobs=ref_turn_logprobs,
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert t_lp_0_t1.grad is not None and t_lp_0_t2.grad is not None
    assert t_lp_1_t1.grad is not None


# ==============================================================================
# 5. End-to-End Agent & OVO-Bench 12-Task Evaluation Tests (Synthetic + Real JSON)
# ==============================================================================


def test_ovobench_end_to_end_12_tasks_and_cli(tmp_path: Path) -> None:
    samples = build_synthetic_ovobench_suite(samples_per_task=2)
    assert len(samples) == 24
    assert set(s.task_code for s in samples) == set(ALL_OVOBENCH_TASKS)

    engine = MockVLMEngine()
    base_summary = evaluate_ovobench_samples(samples, engine=engine, mode="simplestream")
    forager_summary = evaluate_ovobench_samples(
        samples,
        engine=engine,
        mode="forager",
        baseline_summary=base_summary,
    )

    assert base_summary.action_distribution["Answer"] == pytest.approx(100.0)
    assert base_summary.avg_foraged_visual_tokens == pytest.approx(0.0)

    for act_name in ("Answer", "ZoomIn", "Recall", "Wait"):
        assert forager_summary.action_distribution[act_name] > 0.0

    assert forager_summary.delta_p_vs_simplestream is not None and forager_summary.delta_p_vs_simplestream > 0.0
    assert forager_summary.delta_m_vs_simplestream is not None and forager_summary.delta_m_vs_simplestream > 0.0
    assert forager_summary.overall_accuracy == pytest.approx(100.0)

    out_json = tmp_path / "ovobench_results.json"
    exit_code = ovobench_cli_main(["--backend", "mock", "--samples-per-task", "1", "--output-json", str(out_json)])
    assert exit_code == 0
    assert out_json.exists()


def test_real_ovobench_json_loading_and_causal_truncation() -> None:
    """
    Verify that the official `/mnt/hdfs/lipengyu/DA/video/raw/OVO-Bench/data/ovo_bench_new.json`
    loads all 12 tasks cleanly, enforces causal timestamp truncation (`timestamp <= query_timestamp`
    for all Backward/Real-Time frames), and runs end-to-end through `evaluate_ovobench_samples`.
    """
    assert DEFAULT_OVOBENCH_JSON.exists(), f"Missing {DEFAULT_OVOBENCH_JSON}"

    samples = load_ovobench_from_json(
        DEFAULT_OVOBENCH_JSON,
        max_samples_per_task=1,
    )
    assert len(samples) == 12
    assert set(s.task_code for s in samples) == set(ALL_OVOBENCH_TASKS)

    for s in samples:
        if s.task_code in REALTIME_TASKS or s.task_code in BACKWARD_TASKS:
            assert 2 <= len(s.options) <= 5
            assert s.gt_answer in ("A", "B", "C", "D", "E")
            # Causal invariant: every pre-loaded frame has timestamp <= query_timestamp
            assert all(f.timestamp <= s.query_timestamp + 1e-6 for f in s.frames)
        elif s.task_code in FORWARD_ACTIVE_TASKS:
            assert s.target_reveal_time is not None
            assert s.target_reveal_time > s.query_timestamp

    engine = MockVLMEngine()
    base_summary = evaluate_ovobench_samples(samples, engine=engine, mode="simplestream")
    forager_summary = evaluate_ovobench_samples(
        samples,
        engine=engine,
        mode="forager",
        baseline_summary=base_summary,
    )
    assert forager_summary.overall_accuracy == pytest.approx(100.0)
    assert forager_summary.delta_p_vs_simplestream is not None and forager_summary.delta_p_vs_simplestream > 0.0
    assert forager_summary.delta_m_vs_simplestream is not None and forager_summary.delta_m_vs_simplestream > 0.0


def test_zero_oracle_leakage_and_full_video_decoding() -> None:
    """
    Anti-cheat verification (`test_zero_oracle_leakage`):
    1. Every sample loaded from `ovo_bench_new.json` has `blind_agent_metadata(include_mock_hints=False)`
       completely free of `gt_answer`, `answer`, `answer_text`, `target_reveal_time`, `distractor_answer`.
    2. Prompts for `REC`, `SSR`, `CRR`, `EPM`, `ASI`, `HLD`, and all RT tasks contain zero `completed count=`
       or `historical event:` Oracle strings, and all decoded video frames have `scene_tag == ""`.
    3. Samples with `query_timestamp > 90.0s` decode 100% real non-uniform video frames via `decord`
       (never falling back to synthetic solid-color frames).
    """
    import json

    raw_records = json.loads(DEFAULT_OVOBENCH_JSON.read_text(encoding="utf-8"))
    long_rec = next(
        r for r in raw_records
        if r.get("task") in ("EPM", "ASI", "HLD") and float(r.get("realtime", 0.0)) > 120.0
    )
    from scripts.eval_ovobench import _parse_single_ovobench_record

    parsed_long = _parse_single_ovobench_record(
        long_rec,
        idx=0,
        video_root=DEFAULT_OVOBENCH_VIDEO_ROOT,
        lazy_decode=False,
        allow_synthetic_fallback=False,
    )
    assert len(parsed_long.frames) >= 8
    assert all(f.scene_tag == "" for f in parsed_long.frames)
    # Verify real video pixels have non-zero spatial standard deviation (not solid color)
    assert float(np.std(parsed_long.frames[-1].pixels)) > 5.0

    # Check blind isolation across all 12 tasks (lazy_decode=True for instant check)
    all_samples = load_ovobench_from_json(
        DEFAULT_OVOBENCH_JSON,
        max_samples_per_task=2,
        lazy_decode=True,
        allow_synthetic_fallback=False,
    )
    forbidden_keys = {"gt_answer", "answer", "answer_text", "target_reveal_time", "distractor_answer", "none_found_answer"}
    for s in all_samples:
        blind_meta = s.blind_agent_metadata(include_mock_hints=False)
        assert not (set(blind_meta.keys()) & forbidden_keys)
        assert "completed count=" not in s.query
        assert "historical event:" not in s.query
        assert "clue_unfolding" not in s.query
        assert "clue_revealed" not in s.query


# ==============================================================================
# 6. Candidate Manifests, SDAM-SFT Trainer, and veRL / IG-GRPO Adapter Tests
# ==============================================================================


def test_sft_6k_and_rl_8k_manifests_disjointness_and_distribution() -> None:
    root = Path(__file__).resolve().parents[1]
    sft_path = root / "data" / "sft_coldstart_6k.jsonl"
    rl_path = root / "data" / "rl_ig_grpo_8k.jsonl"
    summary_path = root / "data" / "manifest_summary.json"

    assert sft_path.is_file() and rl_path.is_file() and summary_path.is_file()

    import json

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["sft_count"] == 6000
    assert summary["rl_count"] == 8000
    assert summary["video_overlap_count"] == 0
    assert summary["sft_tool_bearing_ratio"] >= 0.40


def test_sdam_sft_collator_and_training_step_with_rolling_ckpts(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sft_path = root / "data" / "sft_coldstart_6k.jsonl"

    ds = ForagerSFTDataset(sft_path, max_samples=6)
    collator = ForagerSDAMDataCollator()
    batch = collator([ds[0], ds[1], ds[2]])

    assert batch["input_ids"].shape[0] == 3
    assert batch["labels"].shape == batch["input_ids"].shape
    assert batch["attention_mask"].shape == (
        3,
        1,
        batch["input_ids"].shape[1],
        batch["input_ids"].shape[1],
    )
    # Visual pad tokens must NEVER have active Cross-Entropy loss labels
    for b in range(3):
        vis_positions = batch["input_ids"][b] == QWEN_VL_IMAGE_PAD_ID
        assert torch.all(batch["labels"][b][vis_positions] == IGNORE_INDEX)
        # Assistant tokens must have valid labels != -100
        assert torch.any(batch["labels"][b] != IGNORE_INDEX)

    # Dry-run 4 training steps with max_ckpts_to_keep=2 and HDFS export verification
    local_ckpt = tmp_path / "local_sft_ckpt"
    hdfs_export = tmp_path / "hdfs_sft_export"
    result = run_sft_training(
        train_jsonl=sft_path,
        local_ckpt_dir=local_ckpt,
        hdfs_export_dir=hdfs_export,
        backend="mock",
        max_samples=8,
        max_steps=4,
        batch_size=2,
        max_ckpts_to_keep=2,
    )
    assert result["steps_completed"] == 4
    assert all(np.isfinite(val) for val in result["step_losses"])
    assert result["freeze_stats"]["frozen_tensors"] >= 1
    # Rolling checkpoint manager strictly retains only steps 3 and 4
    assert result["remaining_local_ckpts"] == ["global_step_3", "global_step_4"]
    assert (hdfs_export / ".export_complete").is_file()


def test_verl_ig_grpo_adapter_and_mock_rl_loop(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    rl_path = root / "data" / "rl_ig_grpo_8k.jsonl"

    # 1. Test verl_compute_score on a 2-turn foraging rollout
    sol = (
        '<think>Text is too blurry at 224x224; zooming in on ROI.</think>'
        '<ZoomIn>{"mode": "spatial_hires", "roi": [0.1, 0.1, 0.9, 0.9], "window_sec": 2.0}</ZoomIn>\n'
        "<think>High-res crop confirms B.</think><Answer>B</Answer>"
    )
    res = verl_compute_score(
        data_source="LLaVA-Video-178K-MC",
        solution_str=sol,
        ground_truth="B",
        extra_info={
            "task_type": "mcq",
            "probe_logprob_without_evidence": -1.386,
            "probe_logprob_with_evidence": -0.05,
        },
    )
    assert res["is_correct"] == 1.0
    assert res["ccv_valid"] == 1.0
    assert res["r_ig"] > 0.5
    assert res["score"] > 1.2

    # 2. Test verl_compute_std_free_advantage across 2 prompt groups (one active, one degenerate)
    token_rewards = torch.tensor(
        [
            [0.0, 0.0, 1.5],
            [0.0, 0.0, 0.5],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    resp_mask = torch.ones_like(token_rewards)
    adv, _ = verl_compute_std_free_advantage(token_rewards, resp_mask, index=[0, 0, 1, 1])
    assert torch.allclose(adv[0], torch.tensor([0.5, 0.5, 0.5]))
    assert torch.allclose(adv[1], torch.tensor([-0.5, -0.5, -0.5]))
    assert torch.allclose(adv[2], torch.zeros(3))
    assert torch.allclose(adv[3], torch.zeros(3))

    # 3. Test Parquet export and veRL command builder
    pq_file = tmp_path / "train.parquet"
    pq_info = export_rl_jsonl_to_verl_parquet(rl_path, pq_file, max_samples=16)
    assert pq_info["rows"] == 16
    assert pq_file.is_file()

    cmd = build_verl_ig_grpo_command(
        train_parquet=pq_file,
        model_path="/tmp/da_models/Qwen2.5-VL-7B-Instruct",
        run_name="test_ig_grpo",
    )
    assert "trainer.max_actor_ckpt_to_keep=2" in cmd
    assert "algorithm.norm_adv_by_std_in_grpo=False" in cmd

    # 4. Run 4-step mock IG-GRPO training loop with rolling checkpointing
    rl_ckpt = tmp_path / "local_rl_ckpt"
    rl_hdfs = tmp_path / "hdfs_rl_export"
    rl_summary = run_mock_ig_grpo_loop(
        rl_jsonl=rl_path,
        local_ckpt_dir=rl_ckpt,
        hdfs_export_dir=rl_hdfs,
        max_steps=4,
        group_size=4,
        max_ckpts_to_keep=2,
    )
    assert rl_summary["steps_completed"] == 4
    assert rl_summary["remaining_local_ckpts"] == ["global_step_3", "global_step_4"]
    assert (rl_hdfs / ".export_complete").is_file()


def test_symmetric_3d_sdg_and_lookahead_wait_ig_reward() -> None:
    """
    Verify Symmetric 3D-SDG (`clean`, `spatial`, `temporal`, `premature`) and
    Temporal Lookahead Counterfactual Information Gain Delta I_wait(tau_i) for `<Wait>`.
    """
    import random
    from forager.ig_grpo import compute_lookahead_wait_ig_reward
    from forager.verl_adapter import sample_3d_sdg_mode, verl_compute_score

    rng = random.Random(123)
    modes = {sample_3d_sdg_mode(rng=rng) for _ in range(100)}
    assert modes == {"clean", "spatial", "temporal", "premature"}

    # Positive Lookahead IG when waiting until t_q + Delta_t lifts log pi_ref(y^*)
    r_wait_ig = compute_lookahead_wait_ig_reward(
        logprob_after_wait=-0.05,
        logprob_before_wait=-1.45,
        is_correct=True,
        ccv_valid=True,
        answer_token_len=1,
    )
    assert r_wait_ig == pytest.approx(1.0)

    # Zero Lookahead IG when premature answer is wrong or CCV fails
    assert compute_lookahead_wait_ig_reward(
        logprob_after_wait=-0.05,
        logprob_before_wait=-1.45,
        is_correct=False,
        ccv_valid=True,
    ) == pytest.approx(0.0)

    # Verify verl_compute_score rewards <Wait> trajectory over premature <Recall> on proactive task
    wait_sol = (
        '<think>The clue is still unfolding at current timestamp; emitting Wait.</think>'
        '<Wait>{"check_after_sec": 2.0, "trigger": "clue_reveal"}</Wait>\n'
        "<think>The subsequent frames now reveal the completed clue.</think><Answer>Yes</Answer>"
    )
    premature_recall_sol = (
        '<think>Recalling historical archive prematurely.</think>'
        '<Recall>{"query": "clue", "lookback_sec": 60.0, "top_k": 2}</Recall>\n'
        "<think>Archive does not contain future clue.</think><Answer>No</Answer>"
    )
    score_wait = verl_compute_score(
        data_source="OVO-Bench-FAR",
        solution_str=wait_sol,
        ground_truth="Yes",
        extra_info={
            "task_type": "proactive",
            "target_reveal_time": 12.0,
            "pred_time": 12.0,
            "logprob_before_wait": -1.45,
            "logprob_after_wait": -0.05,
        },
    )
    score_rec = verl_compute_score(
        data_source="OVO-Bench-FAR",
        solution_str=premature_recall_sol,
        ground_truth="Yes",
        extra_info={
            "task_type": "proactive",
            "target_reveal_time": 12.0,
            "pred_time": 8.0,
            "probe_logprob_without_evidence": -1.45,
            "probe_logprob_with_evidence": -1.40,
        },
    )
    assert score_wait["used_foraging_tool"] == 1.0
    assert score_wait["r_ig"] == pytest.approx(1.0)
    assert score_wait["score"] > 1.5
    assert (score_wait["score"] - score_rec["score"]) > 1.2


def test_zero_oracle_leakage_in_streamingbench_loader() -> None:
    """
    Anti-cheat verification (`test_zero_oracle_leakage_in_streamingbench_loader`):
    1. Every sample across all 18 subtasks of `StreamingBench` resolves to a real `.mp4` file
       inside `/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench/videos/` (zero `video_path=None`,
       zero fallback pool).
    2. `blind_agent_metadata(include_mock_hints=False)` is 100% free of `gt_answer`, `answer`,
       `answer_text`, `target_reveal_time`, `distractor_answer`, `none_found_answer`.
    3. Prompts across all 18 subtasks (including `Proactive Output` and `Sequential Question Answering`)
       contain zero `historical event:`, `clue_unfolding`, `clue_revealed`, or `->` answer strings.
    4. Real frame decoding via `decord.VideoReader` produces genuine video frames with `scene_tag == ""`
       and non-zero spatial pixel variance (never solid-color `np.zeros` squares), and raises
       `FileNotFoundError` if a video path is missing.
    """
    from scripts.eval_streamingbench import (
        ALL_STREAMINGBENCH_TASKS,
        FORBIDDEN_METADATA_KEYS,
        StreamingBenchSample,
        load_streamingbench_samples,
    )

    samples = load_streamingbench_samples(max_samples_per_subtask=1, lazy_decode=True)
    assert len(samples) == 18
    assert {s.subtask for s in samples} == set(ALL_STREAMINGBENCH_TASKS)

    for s in samples:
        assert s.video_path is not None and s.video_path.endswith(".mp4")
        assert "/StreamingBench/videos/" in s.video_path
        assert Path(s.video_path).is_file()

        blind_meta = s.blind_agent_metadata(include_mock_hints=False)
        assert not (set(blind_meta.keys()) & set(FORBIDDEN_METADATA_KEYS))

        assert "historical event:" not in s.query
        assert "clue_unfolding" not in s.query
        assert "clue_revealed" not in s.query
        assert "In Turn 1 (when no <EvidenceCard>" not in s.query
        if s.subtask == "Sequential Question Answering":
            assert "->" not in s.query

    # Materialize real frames via decord on both an MCQ sample and a Proactive Output sample
    mcq_sample = next(s for s in samples if s.task_type == "mcq")
    pro_sample = next(s for s in samples if s.subtask == "Proactive Output")

    for target_s in (mcq_sample, pro_sample):
        frames = target_s.materialize_frames(decode_hw=(448, 448), allow_synthetic_fallback=False)
        assert len(frames) >= 4
        assert all(f.scene_tag == "" for f in frames)
        assert float(np.std(frames[-1].pixels)) > 5.0

    # Verify missing video strictly raises FileNotFoundError
    bad_sample = StreamingBenchSample(
        sample_id="bad_01",
        track="rtvu",
        subtask="Object Recognition",
        required_ability="working memory",
        query="Question: test",
        gt_answer="A",
        query_timestamp=10.0,
        video_path="/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench/videos/nonexistent_99999.mp4",
        lazy_decode_kwargs={"cutoff_timestamp": 10.0},
    )
    with pytest.raises(FileNotFoundError):
        bad_sample.materialize_frames(allow_synthetic_fallback=False)



