"""
Forager: Spatio-Temporal On-Demand Foraging Agent for Streaming Video Understanding.

Core modules:
- protocol: Orthogonal 4-action protocol (<Answer>, <ZoomIn>, <Recall>, <Wait>), format anchor reward, CCV.
- dual_buffer: Causal streaming dual-buffer environment (ProtectedRecentWindow, HighResRingBuffer, OnlineEventArchive, 2D-SDG).
- sdam_mask: Stream-Decoupled Attention Masking (SDAM) and Qwen-VL `input_ids` connected-component span alignment.
- ig_grpo: Counterfactual Information-Gain GRPO rewards, Std-Free Group-Mean advantage, Turn-Mean policy loss.
- agent: Unified VLM backends (Mock / HuggingFace Qwen-VL / vLLM) and two-turn streaming controller.
- sft_trainer: Multi-modal SDAM-SFT dataset, 4D mask collator, vision tower freezer, and rolling checkpoint manager.
- verl_adapter: veRL integration for 2D-SDG, custom composite reward, and Std-Free Group-Mean advantage.
"""

from forager.protocol import (
    ActionType,
    AnswerAction,
    ZoomInAction,
    RecallAction,
    WaitAction,
    ParsedAction,
    ParsedResponse,
    parse_model_response,
    compute_format_anchor_reward,
    verify_cognitive_consistency,
)
from forager.dual_buffer import (
    VideoFrame,
    ProtectedRecentWindow,
    HighResRingBuffer,
    EventSegment,
    StructuredEvidenceCard,
    SubAgentEvidenceExtractor,
    OnlineEventArchive,
    CausalStreamingDualBuffer,
)
from forager.sdam_mask import (
    QWEN_VL_IM_START_ID,
    QWEN_VL_IM_END_ID,
    QWEN_VL_VISION_START_ID,
    QWEN_VL_VISION_END_ID,
    QWEN_VL_VISION_PAD_ID,
    QWEN_VL_IMAGE_PAD_ID,
    QWEN_VL_VIDEO_PAD_ID,
    QwenVLSpecialTokenIds,
    TokenSpan,
    SDAMMaskConfig,
    resolve_qwen_vl_special_tokens,
    extract_qwen_vl_spans_from_input_ids,
    build_sdam_attention_mask,
    build_sdam_mask_from_input_ids,
    EvidenceGate,
)
from forager.ig_grpo import (
    IGGRPOConfig,
    RolloutSample,
    compute_task_accuracy_reward,
    compute_counterfactual_ig_reward,
    compute_foraging_cost,
    compute_composite_reward,
    compute_std_free_group_advantages,
    compute_turn_mean_grpo_loss,
)
from forager.agent import (
    BaseVLMEngine,
    MockVLMEngine,
    HFQwenVLEngine,
    VLLMEngine,
    AgentTurnTrace,
    AgentEpisodeResult,
    StreamingForagerAgent,
)
from forager.sft_trainer import (
    IGNORE_INDEX,
    SFTTrainingConfig,
    ForagerSFTDataset,
    ForagerSDAMDataCollator,
    RollingCheckpointManager,
    freeze_vision_tower,
    compute_sdam_sft_loss,
)
from forager.verl_adapter import (
    SDGMode,
    sample_2d_sdg_mode,
    parse_multi_turn_solution_str,
    verl_compute_score,
    verl_compute_std_free_advantage,
    export_rl_jsonl_to_verl_parquet,
    build_verl_ig_grpo_command,
)

__all__ = [
    "ActionType",
    "AnswerAction",
    "ZoomInAction",
    "RecallAction",
    "WaitAction",
    "ParsedAction",
    "ParsedResponse",
    "parse_model_response",
    "compute_format_anchor_reward",
    "verify_cognitive_consistency",
    "VideoFrame",
    "ProtectedRecentWindow",
    "HighResRingBuffer",
    "EventSegment",
    "StructuredEvidenceCard",
    "SubAgentEvidenceExtractor",
    "OnlineEventArchive",
    "CausalStreamingDualBuffer",
    "QWEN_VL_IM_START_ID",
    "QWEN_VL_IM_END_ID",
    "QWEN_VL_VISION_START_ID",
    "QWEN_VL_VISION_END_ID",
    "QWEN_VL_VISION_PAD_ID",
    "QWEN_VL_IMAGE_PAD_ID",
    "QWEN_VL_VIDEO_PAD_ID",
    "QwenVLSpecialTokenIds",
    "TokenSpan",
    "SDAMMaskConfig",
    "resolve_qwen_vl_special_tokens",
    "extract_qwen_vl_spans_from_input_ids",
    "build_sdam_attention_mask",
    "build_sdam_mask_from_input_ids",
    "EvidenceGate",
    "IGGRPOConfig",
    "RolloutSample",
    "compute_task_accuracy_reward",
    "compute_counterfactual_ig_reward",
    "compute_foraging_cost",
    "compute_composite_reward",
    "compute_std_free_group_advantages",
    "compute_turn_mean_grpo_loss",
    "BaseVLMEngine",
    "MockVLMEngine",
    "HFQwenVLEngine",
    "VLLMEngine",
    "AgentTurnTrace",
    "AgentEpisodeResult",
    "StreamingForagerAgent",
    "IGNORE_INDEX",
    "SFTTrainingConfig",
    "ForagerSFTDataset",
    "ForagerSDAMDataCollator",
    "RollingCheckpointManager",
    "freeze_vision_tower",
    "compute_sdam_sft_loss",
    "SDGMode",
    "sample_2d_sdg_mode",
    "parse_multi_turn_solution_str",
    "verl_compute_score",
    "verl_compute_std_free_advantage",
    "export_rl_jsonl_to_verl_parquet",
    "build_verl_ig_grpo_command",
]
