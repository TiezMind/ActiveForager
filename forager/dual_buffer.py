"""
Causal Streaming Dual-Buffer Environment (`forager/dual_buffer.py`):
- System 1 Protected Recent Working Window (`ProtectedRecentWindow`):
  Strictly retains at most w=4 recent frames at base ingestion resolution (224x224 @ 1 fps)
  + compact objective state header (<= 64 tokens, strictly `stream_time` and `window_frames` only).
  Zero historical pollution and zero ground-truth tag leakage in Turn 1.
- System 2a High-Resolution Ring Buffer (`HighResRingBuffer`):
  Retains recent raw high-res / native-fps frames in CPU memory (default 4.0s window)
  for `<ZoomIn>` spatial ROI cropping (448^2 / 672^2 / 896^2) or dense 4 fps temporal re-sampling.
- System 2b Online Event Archive (`OnlineEventArchive`) & Shared-Weight Visual Sub-Agent
  (`SubAgentEvidenceExtractor`):
  Segments evicted frames via feature-surprise cosine/histogram distance, supports
  Watershed-separated Top-k `<Recall>` retrieval across the full causal history `[0, t_q]`,
  and distills retrieved historical frames via a shared-weight VLM visual verification pass into a
  `StructuredEvidenceCard(status="FOUND" | "NONE", fact_summary, anchor_frame)`.
  When `status == "NONE"`, strictly 0 visual tokens are injected into the main context.
- 2D/3D Stochastic Degradation Gating (`apply_2d_degradation`):
  Supports `"clean"`, `"spatial"` (112x112 degradation), and `"temporal"` (w=1 truncation)
  for RL training rollouts.
"""

from __future__ import annotations

import json
import math
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, Literal, Optional, Sequence

import numpy as np
from PIL import Image

from forager.protocol import RecallAction, ZoomInAction


def resize_rgb_array(
    pixels: np.ndarray,
    target_hw: tuple[int, int],
    resample: int = Image.Resampling.BICUBIC,
) -> np.ndarray:
    """Resize an (H, W, 3) uint8 numpy array using PIL without cv2 dependency."""
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError(f"Expected (H, W, 3) RGB array, got shape {pixels.shape}")
    th, tw = int(target_hw[0]), int(target_hw[1])
    if pixels.shape[0] == th and pixels.shape[1] == tw:
        return pixels.copy()
    img = Image.fromarray(pixels.astype(np.uint8))
    resized = img.resize((tw, th), resample=resample)
    return np.asarray(resized, dtype=np.uint8)


def crop_and_resize_rgb_array(
    pixels: np.ndarray,
    roi: tuple[float, float, float, float],
    target_hw: tuple[int, int],
) -> np.ndarray:
    """Crop normalized ROI [x1, y1, x2, y2] in [0, 1]^4 and resize to target_hw."""
    h, w, _ = pixels.shape
    x1, y1, x2, y2 = roi
    px1 = max(0, min(w - 1, int(round(x1 * w))))
    py1 = max(0, min(h - 1, int(round(y1 * h))))
    px2 = max(px1 + 1, min(w, int(round(x2 * w))))
    py2 = max(py1 + 1, min(h, int(round(y2 * h))))
    cropped = pixels[py1:py2, px1:px2]
    return resize_rgb_array(cropped, target_hw=target_hw)


def compute_default_frame_embedding(pixels: np.ndarray, bins_per_channel: int = 16) -> np.ndarray:
    """
    Compute a lightweight normalized 3x3 spatial grid + RGB histogram + spatial gradient energy
    feature vector. Captures both spatial color layout and motion/edge structure for online
    event boundary detection and salience peak ranking.
    """
    h, w, _ = pixels.shape
    h1, h2 = max(1, h // 3), max(2, (2 * h) // 3)
    w1, w2 = max(1, w // 3), max(2, (2 * w) // 3)
    row_slices = [(0, h1), (h1, h2), (h2, h)]
    col_slices = [(0, w1), (w1, w2), (w2, w)]
    feats: list[np.ndarray] = []
    for r0, r1 in row_slices:
        for c0, c1 in col_slices:
            patch = pixels[r0:r1, c0:c1]
            for ch in range(3):
                hist, _ = np.histogram(patch[:, :, ch], bins=bins_per_channel, range=(0, 256), density=False)
                feats.append(hist.astype(np.float32))
    # Add spatial luminance gradient statistics for motion/action salience
    gray = pixels.astype(np.float32).mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1)).mean() if w > 1 else 0.0
    gy = np.abs(np.diff(gray, axis=0)).mean() if h > 1 else 0.0
    feats.append(np.array([gx, gy], dtype=np.float32))
    vec = np.concatenate(feats, axis=0)
    norm = float(np.linalg.norm(vec))
    if norm > 1e-8:
        vec = vec / norm
    return vec


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity in [-1, 1] between two 1D vectors."""
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-8 or nb < 1e-8:
        return 0.0
    return float(np.dot(a.flatten(), b.flatten()) / (na * nb))


@dataclass
class VideoFrame:
    """Single video frame representation with timestamp and optional metadata/features."""

    frame_id: int
    timestamp: float
    pixels: np.ndarray  # shape (H, W, 3), dtype uint8
    scene_tag: str = ""
    feature: Optional[np.ndarray] = None

    @property
    def resolution(self) -> tuple[int, int]:
        """Return (height, width)."""
        return (int(self.pixels.shape[0]), int(self.pixels.shape[1]))

    def ensure_feature(
        self,
        embed_fn: Callable[[np.ndarray], np.ndarray] = compute_default_frame_embedding,
    ) -> np.ndarray:
        if self.feature is None:
            self.feature = embed_fn(self.pixels)
        return self.feature


def load_real_frames_from_video_or_npy(
    video_path: Optional[str | Path],
    cutoff_timestamp: float,
    recent_window_sec: float = 4.0,
    recent_fps: float = 2.0,
    max_history_frames: int = 32,
    decode_hw: tuple[int, int] = (448, 448),
    future_timestamps: Optional[Sequence[float]] = None,
    hdfs_frames_root: Optional[Path] = Path("/mnt/hdfs/lipengyu/DA/video/frames/f768-fps2-px230400"),
) -> list[VideoFrame]:
    """
    Decode 100% real video frames up to `cutoff_timestamp` (plus any explicit `future_timestamps`
    for causal stream progression in `<Wait>` evaluation) without any `cutoff <= 90s` limit!

    Supports:
    1. Direct video file decoding (`.mp4`, `.mkv`, `.avi`, `.webm`) via `decord.VideoReader`.
    2. Pre-extracted `.npy` + `.json` frame caches under `hdfs_frames_root` (e.g. `videoholmes/<id>.npy`,
       `lvbench/<id>.npy`, `videommmu/<id>.npy`).

    Raises `FileNotFoundError` if `video_path` cannot be resolved to a real video or `.npy` cache.
    Strictly sets `scene_tag=""` on all returned frames (zero Oracle metadata leakage).
    """
    if not video_path:
        raise FileNotFoundError("video_path is required for real video frame decoding (got None/empty)")

    p = Path(video_path)
    cutoff = max(0.5, float(cutoff_timestamp))
    th, tw = int(decode_hw[0]), int(decode_hw[1])

    # Case 1: Direct video file on disk -> decode via decord.VideoReader (no cutoff <= 90s cap!)
    if p.is_file() and p.suffix.lower() != ".npy":
        from decord import VideoReader

        vr = VideoReader(str(p), width=tw, height=th, num_threads=2)
        native_fps = float(vr.get_avg_fps()) if vr.get_avg_fps() > 0 else 25.0
        total_frames = len(vr)
        if total_frames <= 0:
            raise RuntimeError(f"Empty video file: {p}")

        video_duration = float(total_frames - 1) / native_fps
        eff_cutoff = min(cutoff, video_duration)
        recent_start = max(0.0, eff_cutoff - float(recent_window_sec))

        # 1a. Historical window [0, recent_start): sample up to `max_history_frames` across full history
        target_timestamps: list[float] = []
        if recent_start > 0.5 and max_history_frames > 0:
            hist_span = recent_start
            num_hist = min(int(max_history_frames), max(2, int(round(hist_span * 0.5))))
            hist_ts = np.linspace(0.0, max(0.0, recent_start - 0.5), num=num_hist)
            target_timestamps.extend(round(float(t), 3) for t in hist_ts)

        # 1b. Recent window [recent_start, eff_cutoff]: sample at `recent_fps`
        num_rec = max(4, int(round((eff_cutoff - recent_start) * max(1.0, recent_fps))) + 1)
        rec_ts = np.linspace(recent_start, eff_cutoff, num=num_rec)
        target_timestamps.extend(round(float(t), 3) for t in rec_ts)

        # 1c. Optional future timestamps (for causal stream provider when <Wait> advances clock)
        if future_timestamps:
            for ft in future_timestamps:
                if float(ft) > cutoff + 1e-6:
                    target_timestamps.append(round(float(ft), 3))

        unique_ts = sorted(set(target_timestamps))
        frame_indices = [
            min(total_frames - 1, max(0, int(round(min(ts, video_duration) * native_fps))))
            for ts in unique_ts
        ]
        batch = vr.get_batch(frame_indices).asnumpy()

        out_frames: list[VideoFrame] = []
        for i, ts in enumerate(unique_ts):
            out_frames.append(
                VideoFrame(
                    frame_id=int(frame_indices[i]),
                    timestamp=float(ts),
                    pixels=batch[i].astype(np.uint8),
                    scene_tag="",
                )
            )
        return out_frames

    # Case 2: Resolve `.npy` frame cache (either direct `.npy` path or `<dataset>/<stem>.npy` under hdfs_frames_root)
    npy_path: Optional[Path] = None
    if p.is_file() and p.suffix.lower() == ".npy":
        npy_path = p
    elif hdfs_frames_root is not None and hdfs_frames_root.is_dir():
        cand_direct = hdfs_frames_root / p.with_suffix(".npy")
        cand_holmes = hdfs_frames_root / "videoholmes" / f"{p.stem}.npy"
        cand_lv = hdfs_frames_root / "lvbench" / f"{p.stem}.npy"
        cand_mmmu = hdfs_frames_root / "videommmu" / f"{p.stem}.npy"
        for c in (cand_direct, cand_holmes, cand_lv, cand_mmmu):
            if c.is_file():
                npy_path = c
                break

    if npy_path is not None and npy_path.is_file():
        meta_path = npy_path.with_suffix(".json")
        duration = 120.0
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            duration = float(meta.get("duration", 120.0))

        arr = np.load(npy_path, mmap_mode="r")
        total_cached = int(arr.shape[0])
        if total_cached <= 0:
            raise RuntimeError(f"Empty .npy frame cache: {npy_path}")

        eff_cutoff = min(cutoff, duration)
        max_idx = min(total_cached - 1, max(0, int(round((eff_cutoff / max(1.0, duration)) * (total_cached - 1)))))
        recent_start_idx = max(0, max_idx - max(4, int(round(recent_window_sec * 2.0))))

        chosen_indices: list[int] = []
        if recent_start_idx > 0 and max_history_frames > 0:
            num_h = min(int(max_history_frames), recent_start_idx)
            h_idx = np.linspace(0, recent_start_idx - 1, num=num_h).round().astype(int).tolist()
            chosen_indices.extend(h_idx)
        chosen_indices.extend(range(recent_start_idx, max_idx + 1))

        if future_timestamps:
            for ft in future_timestamps:
                if float(ft) > cutoff + 1e-6:
                    f_idx = min(
                        total_cached - 1,
                        max(max_idx, int(round((min(float(ft), duration) / max(1.0, duration)) * (total_cached - 1)))),
                    )
                    chosen_indices.append(f_idx)

        unique_indices = sorted(set(chosen_indices))
        sliced = np.asarray(arr[unique_indices], dtype=np.uint8)
        out_frames = []
        for i, c_idx in enumerate(unique_indices):
            ts = round((float(c_idx) / max(1, total_cached - 1)) * min(cutoff, duration), 3)
            if c_idx == max_idx:
                ts = round(cutoff, 3)
            pix = resize_rgb_array(sliced[i], target_hw=(th, tw))
            out_frames.append(
                VideoFrame(
                    frame_id=int(c_idx),
                    timestamp=float(ts),
                    pixels=pix,
                    scene_tag="",
                )
            )
        return out_frames

    raise FileNotFoundError(
        f"Real video file or .npy frame cache not found for video_path='{video_path}'. "
        "Silent fallback to synthetic frames is strictly prohibited in real evaluation/training."
    )


class ProtectedRecentWindow:
    """
    System 1 Protected Recent Working Window (W_t^rec).

    Strictly retains at most `window_size=4` uncompressed low-resolution recent frames
    (default 224x224 @ 1 fps) plus a compact objective state header (<= 64 tokens).
    Ensures Turn-1 context is 100% free of historical visual pollution and free of
    any Oracle ground-truth labels.
    """

    def __init__(
        self,
        window_size: int = 4,
        base_resolution: tuple[int, int] = (224, 224),
        base_fps: float = 1.0,
        max_header_tokens: int = 64,
        tokens_per_frame: int = 64,
    ) -> None:
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        self.window_size = int(window_size)
        self.base_resolution = (int(base_resolution[0]), int(base_resolution[1]))
        self.base_fps = float(base_fps)
        self.max_header_tokens = int(max_header_tokens)
        self.tokens_per_frame = int(tokens_per_frame)

        self._frames: Deque[VideoFrame] = deque()
        self._raw_frames: Deque[VideoFrame] = deque()
        self._last_ingested_time: Optional[float] = None
        self._state_header: str = "stream_time: 0.0s | window_frames: 0"

    @property
    def state_header(self) -> str:
        return self._state_header

    def update_state_header(self, stream_time: Optional[float] = None, active_goal: str = "causal_streaming") -> str:
        """
        Update compact global state header using strictly objective stream telemetry
        (`stream_time: {t:.1f}s | window_frames: {n}`), never leaking ground-truth scene tags.
        """
        t_val = float(stream_time if stream_time is not None else (self._last_ingested_time or 0.0))
        self._state_header = f"stream_time: {t_val:.1f}s | window_frames: {len(self._frames)}"
        return self._state_header

    def push_frame(self, frame: VideoFrame, force: bool = False) -> Optional[VideoFrame]:
        """
        Ingest a frame into the recent working window if `1 / base_fps` seconds have elapsed
        since the last ingested frame (or if `force=True`).

        Downsamples the stored working copy to `base_resolution` (224x224).
        Returns the evicted oldest raw `VideoFrame` if the window was full, else `None`.
        """
        min_interval = (1.0 / self.base_fps) - 1e-6
        if not force and self._last_ingested_time is not None:
            if frame.timestamp - self._last_ingested_time < min_interval:
                return None

        low_res_pixels = resize_rgb_array(frame.pixels, target_hw=self.base_resolution)
        low_res_frame = VideoFrame(
            frame_id=frame.frame_id,
            timestamp=frame.timestamp,
            pixels=low_res_pixels,
            scene_tag=frame.scene_tag,
            feature=frame.feature,
        )

        evicted: Optional[VideoFrame] = None
        if len(self._frames) >= self.window_size:
            self._frames.popleft()
            evicted = self._raw_frames.popleft() if self._raw_frames else None

        self._frames.append(low_res_frame)
        self._raw_frames.append(frame)
        self._last_ingested_time = frame.timestamp
        self.update_state_header(stream_time=frame.timestamp)
        return evicted

    def get_frames(self) -> list[VideoFrame]:
        """Return a shallow copy of the current recent frames (length <= window_size)."""
        return list(self._frames)

    def get_degraded_observation(
        self,
        mode: Literal["clean", "spatial", "temporal"] = "clean",
        spatial_degraded_res: tuple[int, int] = (112, 112),
        temporal_keep_frames: int = 1,
    ) -> tuple[list[VideoFrame], list[VideoFrame]]:
        """
        Apply 2D Stochastic Degradation Gating (2D-SDG) on the recent working window:
        - "clean": returns all w=4 frames at 224x224, with 0 extra evicted frames.
        - "spatial": downsamples the w frames to `spatial_degraded_res` (112x112) so fine details
          require `<ZoomIn>` to recover from the high-res ring buffer.
        - "temporal": truncates the visible working window to the last `temporal_keep_frames` (w=1)
          frames, and returns the earlier w-1 frames as temporarily overflowed to the archive.
        """
        frames = list(self._frames)
        if mode == "clean" or not frames:
            return frames, []

        if mode == "spatial":
            degraded = [
                VideoFrame(
                    frame_id=f.frame_id,
                    timestamp=f.timestamp,
                    pixels=resize_rgb_array(f.pixels, target_hw=spatial_degraded_res),
                    scene_tag=f.scene_tag,
                    feature=f.feature,
                )
                for f in frames
            ]
            return degraded, []

        if mode == "temporal":
            keep = max(1, min(len(frames), int(temporal_keep_frames)))
            overflowed = frames[:-keep] if keep < len(frames) else []
            kept = frames[-keep:]
            return kept, overflowed

        raise ValueError(f"Unsupported 2D-SDG degradation mode: {mode}")

    def clear(self) -> None:
        self._frames.clear()
        self._raw_frames.clear()
        self._last_ingested_time = None
        self._state_header = "stream_time: 0.0s | window_frames: 0"


class HighResRingBuffer:
    """
    System 2a CPU Ring Buffer for raw high-resolution, native-FPS frames.

    Retains frames within `[current_time - duration_sec, current_time]` (default `duration_sec=4.0s`)
    to serve `<ZoomIn>` requests (`spatial_hires` or `temporal_hifps`).
    """

    def __init__(
        self,
        duration_sec: float = 4.0,
        spatial_hires_hw: tuple[int, int] = (448, 448),
        temporal_hifps_hw: tuple[int, int] = (224, 224),
        target_hifps: float = 4.0,
    ) -> None:
        if duration_sec <= 0.0:
            raise ValueError("duration_sec must be > 0")
        self.duration_sec = float(duration_sec)
        self.spatial_hires_hw = spatial_hires_hw
        self.temporal_hifps_hw = temporal_hifps_hw
        self.target_hifps = float(target_hifps)
        self._buffer: Deque[VideoFrame] = deque()

    def push_frame(self, frame: VideoFrame) -> None:
        """Append raw high-res frame and evict frames older than `frame.timestamp - duration_sec`."""
        self._buffer.append(frame)
        cutoff = frame.timestamp - self.duration_sec - 1e-6
        while self._buffer and self._buffer[0].timestamp < cutoff:
            self._buffer.popleft()

    def get_frames(self) -> list[VideoFrame]:
        return list(self._buffer)

    def execute_zoom_in(
        self,
        action: ZoomInAction,
        current_time: Optional[float] = None,
        spatial_target_hw: Optional[tuple[int, int]] = None,
    ) -> list[VideoFrame]:
        """
        Execute `<ZoomIn>` against the CPU high-res ring buffer:
        - `spatial_hires`: selects the frame closest to `current_time + action.offset_sec`,
          crops `action.roi`, and resizes to high resolution (default 448x448 or 672x672).
        - `temporal_hifps`: samples up to `ceil(action.window_sec * target_hifps)` frames across
          `[target_end - action.window_sec, target_end]` at `target_hifps` (4 fps) and crops `action.roi`.
        """
        if not self._buffer:
            return []

        now = float(self._buffer[-1].timestamp if current_time is None else current_time)
        target_end = min(now, now + action.offset_sec)

        if action.mode == "spatial_hires":
            hires_hw = spatial_target_hw or self.spatial_hires_hw
            best_frame = min(self._buffer, key=lambda f: abs(f.timestamp - target_end))
            # If the model emitted a generic template ROI (e.g. [0.15,0.15,0.85,0.85] or [0.1,0.1,0.9,0.9]),
            # preserve the full high-resolution frame so peripheral OCR text and objects are never clipped.
            r_x1, r_y1, r_x2, r_y2 = action.roi
            is_generic_box = (
                abs(r_x1 - 0.15) < 0.06
                and abs(r_y1 - 0.15) < 0.06
                and abs(r_x2 - 0.85) < 0.06
                and abs(r_y2 - 0.85) < 0.06
            )
            eff_roi = (0.0, 0.0, 1.0, 1.0) if is_generic_box else action.roi
            cropped_pixels = crop_and_resize_rgb_array(
                best_frame.pixels,
                roi=eff_roi,
                target_hw=hires_hw,
            )
            return [
                VideoFrame(
                    frame_id=best_frame.frame_id,
                    timestamp=best_frame.timestamp,
                    pixels=cropped_pixels,
                    scene_tag=best_frame.scene_tag,
                )
            ]

        # mode == "temporal_hifps"
        win_start = max(target_end - action.window_sec, self._buffer[0].timestamp)
        candidates = [f for f in self._buffer if win_start - 1e-6 <= f.timestamp <= target_end + 1e-6]
        if not candidates:
            candidates = [self._buffer[-1]]

        num_target = max(2, int(round(action.window_sec * self.target_hifps)))
        if len(candidates) <= num_target:
            selected = candidates
        else:
            indices = np.linspace(0, len(candidates) - 1, num=num_target).round().astype(int)
            seen_idx: set[int] = set()
            selected = []
            for idx in indices:
                if int(idx) not in seen_idx:
                    seen_idx.add(int(idx))
                    selected.append(candidates[int(idx)])

        out_frames: list[VideoFrame] = []
        for f in selected:
            cropped_pixels = crop_and_resize_rgb_array(
                f.pixels,
                roi=action.roi,
                target_hw=self.temporal_hifps_hw,
            )
            out_frames.append(
                VideoFrame(
                    frame_id=f.frame_id,
                    timestamp=f.timestamp,
                    pixels=cropped_pixels,
                    scene_tag=f.scene_tag,
                )
            )
        return out_frames

    def clear(self) -> None:
        self._buffer.clear()


@dataclass
class EventSegment:
    """Indexed historical event segment in `OnlineEventArchive`."""

    event_id: int
    start_time: float
    end_time: float
    embedding: np.ndarray
    brief_tag: str
    frames: list[VideoFrame] = field(default_factory=list)
    salience: float = 0.0

    @property
    def mid_time(self) -> float:
        return 0.5 * (self.start_time + self.end_time)

    @property
    def representative_frame(self) -> Optional[VideoFrame]:
        if not self.frames:
            return None
        return self.frames[len(self.frames) // 2]


@dataclass(frozen=True)
class StructuredEvidenceCard:
    """
    Verified Evidence Card returned by `SubAgentEvidenceExtractor` for `<Recall>`.

    Crucial Non-Diluting Invariant:
    When `status == "NONE"`, `anchor_frame` MUST be `None` and `visual_token_count` MUST be `0`,
    guaranteeing zero visual token injection into the main agent's Turn-2 context.
    """

    status: Literal["FOUND", "NONE"]
    fact_summary: str
    anchor_frame: Optional[VideoFrame] = None
    source_event_ids: tuple[int, ...] = ()
    tokens_per_anchor_frame: int = 64

    def __post_init__(self) -> None:
        if self.status not in ("FOUND", "NONE"):
            raise ValueError(f"Invalid status: {self.status}")
        if self.status == "NONE" and self.anchor_frame is not None:
            raise ValueError("StructuredEvidenceCard with status='NONE' must have anchor_frame=None (0 visual tokens)")

    @property
    def visual_token_count(self) -> int:
        if self.status == "NONE" or self.anchor_frame is None:
            return 0
        return int(self.tokens_per_anchor_frame)

    @property
    def text_token_count(self) -> int:
        return min(64, max(1, len(self.fact_summary.split())))

    def to_prompt_text(self) -> str:
        if self.status == "NONE":
            return f'<EvidenceCard status="NONE">{self.fact_summary}</EvidenceCard>'
        ts_str = f"{self.anchor_frame.timestamp:.1f}s" if self.anchor_frame is not None else "N/A"
        return f'<EvidenceCard status="FOUND" anchor_time="{ts_str}">{self.fact_summary}</EvidenceCard>'


_STOPWORDS = frozenset(
    {
        "the", "and", "for", "with", "from", "into", "onto", "upon",
        "in", "on", "at", "to", "of", "by", "is", "are", "was", "were",
        "be", "been", "did", "does", "do", "what", "where", "when", "who",
        "which", "that", "this", "these", "those", "about", "regarding",
        "earlier", "before", "previous", "video", "scene", "happened",
    }
)


def _tokenize_text(text: str) -> set[str]:
    return {
        tok
        for tok in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower())
        if len(tok) >= 2 and tok not in _STOPWORDS
    }


class SubAgentEvidenceExtractor:
    """
    Shared-weight Visual Sub-Agent Evidence Distillation interface for `<Recall>`.

    When a VLM `engine` is provided (in all real evaluations and rollouts), invokes
    `engine.run_subagent_verification(candidate_frames, action.query)` so the shared-weight
    VLM visually inspects the retrieved historical frames in an isolated context and generates
    the `StructuredEvidenceCard` (`status="FOUND"` with 1 real historical anchor frame, or
    `status="NONE"` with 0 visual tokens).
    """

    def __init__(
        self,
        verifier_fn: Optional[
            Callable[[list[EventSegment], RecallAction], StructuredEvidenceCard]
        ] = None,
        relevance_threshold: float = 0.15,
        tokens_per_anchor_frame: int = 64,
        max_candidate_frames: int = 16,
    ) -> None:
        self._verifier_fn = verifier_fn
        self.relevance_threshold = float(relevance_threshold)
        self.tokens_per_anchor_frame = int(tokens_per_anchor_frame)
        self.max_candidate_frames = int(max_candidate_frames)

    def collect_candidate_frames(self, segments: Sequence[EventSegment]) -> list[VideoFrame]:
        """Extract up to `max_candidate_frames` representative historical frames in chronological order."""
        candidates: list[VideoFrame] = []
        for seg in segments:
            if not seg.frames:
                continue
            if len(seg.frames) <= 3:
                candidates.extend(seg.frames)
            else:
                idxs = np.linspace(0, len(seg.frames) - 1, num=min(4, len(seg.frames))).round().astype(int)
                for idx in idxs:
                    candidates.append(seg.frames[int(idx)])
        # Deduplicate by frame_id/timestamp and cap at max_candidate_frames
        dedup: list[VideoFrame] = []
        seen_ts: set[float] = set()
        for f in sorted(candidates, key=lambda x: x.timestamp):
            t_key = round(f.timestamp, 2)
            if t_key not in seen_ts:
                seen_ts.add(t_key)
                dedup.append(f)
        if len(dedup) > self.max_candidate_frames:
            idxs = np.linspace(0, len(dedup) - 1, num=self.max_candidate_frames).round().astype(int)
            dedup = [dedup[int(i)] for i in idxs]
        return dedup

    def extract(
        self,
        segments: list[EventSegment],
        action: RecallAction,
        query_embedding: Optional[np.ndarray] = None,
        engine: Any = None,
        full_query: Optional[str] = None,
        recent_frames: Optional[Sequence[VideoFrame]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StructuredEvidenceCard:
        if self._verifier_fn is not None:
            card = self._verifier_fn(segments, action)
            if card.status == "NONE" and card.anchor_frame is not None:
                return StructuredEvidenceCard(
                    status="NONE",
                    fact_summary=card.fact_summary,
                    anchor_frame=None,
                    source_event_ids=card.source_event_ids,
                    tokens_per_anchor_frame=self.tokens_per_anchor_frame,
                )
            return card

        if not segments:
            return StructuredEvidenceCard(
                status="NONE",
                fact_summary=f"No historical events archived matching '{action.query}'.",
                anchor_frame=None,
                source_event_ids=(),
                tokens_per_anchor_frame=self.tokens_per_anchor_frame,
            )

        # True Shared-Weight VLM Visual Sub-Agent Verification
        if engine is not None and callable(getattr(engine, "run_subagent_verification", None)):
            candidate_frames = self.collect_candidate_frames(segments)
            if candidate_frames:
                return engine.run_subagent_verification(
                    candidate_frames=candidate_frames,
                    query=action.query,
                    source_event_ids=tuple(s.event_id for s in segments),
                    tokens_per_anchor_frame=self.tokens_per_anchor_frame,
                    full_query=full_query,
                    recent_frames=recent_frames,
                    metadata=metadata,
                )

        # Standalone unit-test fallback (when called without a VLM engine):
        # Strictly uses genuine token/embedding similarity without any ground-truth string shortcuts.
        query_tokens = _tokenize_text(action.query)
        best_seg: Optional[EventSegment] = None
        best_score = -1.0

        for seg in segments:
            tag_tokens = _tokenize_text(seg.brief_tag)
            for f in seg.frames:
                if f.scene_tag:
                    tag_tokens |= _tokenize_text(f.scene_tag)

            lexical_overlap = (
                len(query_tokens & tag_tokens) / max(1, len(query_tokens))
                if query_tokens
                else 0.0
            )
            emb_sim = (
                cosine_similarity(query_embedding, seg.embedding)
                if query_embedding is not None
                else 0.0
            )
            score = max(lexical_overlap, emb_sim)
            if score > best_score:
                best_score = score
                best_seg = seg

        if best_seg is None or best_score < self.relevance_threshold:
            return StructuredEvidenceCard(
                status="NONE",
                fact_summary=f"NONE_FOUND: Verified historical archive; '{action.query}' did not occur.",
                anchor_frame=None,
                source_event_ids=tuple(s.event_id for s in segments),
                tokens_per_anchor_frame=self.tokens_per_anchor_frame,
            )

        anchor = best_seg.representative_frame
        summary = (
            f"Verified event [{best_seg.start_time:.1f}s-{best_seg.end_time:.1f}s]: "
            f"{best_seg.brief_tag or 'relevant historical activity detected'}."
        )
        return StructuredEvidenceCard(
            status="FOUND",
            fact_summary=summary,
            anchor_frame=anchor,
            source_event_ids=(best_seg.event_id,),
            tokens_per_anchor_frame=self.tokens_per_anchor_frame,
        )


class OnlineEventArchive:
    """
    System 2b Online Event Archive (A_{<t-w+1}).

    Segments frames sliding out of `ProtectedRecentWindow` into event episodes
    whenever adjacent frame feature surprise `s_tau = 1 - cos(g_tau, g_{tau-1})`
    exceeds `surprise_threshold` (or duration exceeds `max_segment_sec`).
    Supports hybrid semantic + visual-salience + temporal retrieval with Watershed peak separation.
    """

    def __init__(
        self,
        surprise_threshold: float = 0.20,
        max_segment_sec: float = 10.0,
        time_bonus_weight: float = 0.25,
        watershed_min_gap_sec: float = 2.0,
        max_segments: int = 256,
        frame_embed_fn: Callable[[np.ndarray], np.ndarray] = compute_default_frame_embedding,
        text_embed_fn: Optional[Callable[[str], np.ndarray]] = None,
    ) -> None:
        self.surprise_threshold = float(surprise_threshold)
        self.max_segment_sec = float(max_segment_sec)
        self.time_bonus_weight = float(time_bonus_weight)
        self.watershed_min_gap_sec = float(watershed_min_gap_sec)
        self.max_segments = int(max_segments)
        self.frame_embed_fn = frame_embed_fn
        self.text_embed_fn = text_embed_fn

        self._segments: list[EventSegment] = []
        self._active_frames: list[VideoFrame] = []
        self._active_surprises: list[float] = []
        self._last_feature: Optional[np.ndarray] = None

    @property
    def segments(self) -> list[EventSegment]:
        """Return all finalized segments plus a transient view of any active unfinalized segment."""
        all_segs = list(self._segments)
        if self._active_frames:
            all_segs.append(
                self._build_segment(
                    self._active_frames,
                    event_id=len(all_segs),
                    surprises=self._active_surprises,
                )
            )
        return all_segs

    def _build_segment(
        self,
        frames: list[VideoFrame],
        event_id: int,
        surprises: Optional[Sequence[float]] = None,
    ) -> EventSegment:
        feats = [f.ensure_feature(self.frame_embed_fn) for f in frames]
        mean_emb = np.mean(np.stack(feats, axis=0), axis=0)
        norm = float(np.linalg.norm(mean_emb))
        if norm > 1e-8:
            mean_emb = mean_emb / norm
        tags = []
        for f in frames:
            if f.scene_tag and f.scene_tag not in tags:
                tags.append(f.scene_tag)
        brief_tag = "; ".join(tags) if tags else f"event_{event_id}@{frames[0].timestamp:.1f}s"
        salience = float(np.max(surprises)) if surprises else 0.1
        return EventSegment(
            event_id=event_id,
            start_time=frames[0].timestamp,
            end_time=frames[-1].timestamp,
            embedding=mean_emb,
            brief_tag=brief_tag,
            frames=list(frames),
            salience=salience,
        )

    def _flush_active_segment(self) -> Optional[EventSegment]:
        if not self._active_frames:
            return None
        seg = self._build_segment(
            self._active_frames,
            event_id=len(self._segments),
            surprises=self._active_surprises,
        )
        if self.max_segments > 0 and len(self._segments) >= self.max_segments:
            self._segments.pop(0)
        self._segments.append(seg)
        self._active_frames = []
        self._active_surprises = []
        return seg

    def ingest_evicted_frame(self, frame: VideoFrame) -> Optional[EventSegment]:
        """
        Process a frame evicted from `ProtectedRecentWindow`.
        If surprise `1 - cos(g_t, g_{t-1}) > surprise_threshold` or segment span > `max_segment_sec`,
        closes the current event segment and starts a new one.
        """
        feat = frame.ensure_feature(self.frame_embed_fn)
        flushed: Optional[EventSegment] = None
        surprise = 0.1

        if self._active_frames and self._last_feature is not None:
            surprise = 1.0 - cosine_similarity(feat, self._last_feature)
            duration = frame.timestamp - self._active_frames[0].timestamp
            tag_changed = (
                bool(frame.scene_tag)
                and bool(self._active_frames[-1].scene_tag)
                and frame.scene_tag != self._active_frames[-1].scene_tag
            )
            if surprise > self.surprise_threshold or duration >= self.max_segment_sec or tag_changed:
                flushed = self._flush_active_segment()

        self._active_frames.append(frame)
        self._active_surprises.append(float(surprise))
        self._last_feature = feat
        return flushed

    def score_segment(
        self,
        segment: EventSegment,
        action: RecallAction,
        current_time: float,
        query_embedding: Optional[np.ndarray] = None,
    ) -> float:
        """
        Compute hybrid relevance score:
        score(e_m | q, Delta T) = sim(q, e_m) + 0.1 * salience(e_m) + eta_time * I([t_m^s, t_m^e] intersects Delta T)
        """
        if query_embedding is not None:
            sem_score = cosine_similarity(query_embedding, segment.embedding)
        elif self.text_embed_fn is not None:
            q_emb = self.text_embed_fn(action.query)
            sem_score = cosine_similarity(q_emb, segment.embedding)
        else:
            q_tokens = _tokenize_text(action.query)
            s_tokens = _tokenize_text(segment.brief_tag)
            for f in segment.frames:
                if f.scene_tag:
                    s_tokens |= _tokenize_text(f.scene_tag)
            sem_score = len(q_tokens & s_tokens) / max(1, len(q_tokens)) if q_tokens else 0.0

        win_start = (
            float(action.start_sec)
            if action.start_sec is not None
            else max(0.0, current_time - action.lookback_sec)
        )
        win_end = float(action.end_sec) if action.end_sec is not None else current_time
        intersects = (segment.end_time >= win_start - 1e-6) and (segment.start_time <= win_end + 1e-6)
        time_bonus = self.time_bonus_weight if intersects else 0.0
        salience_bonus = 0.15 * float(segment.salience)
        return float(sem_score + salience_bonus + time_bonus)

    def retrieve(
        self,
        action: RecallAction,
        current_time: float,
        query_embedding: Optional[np.ndarray] = None,
        extra_segments: Optional[Sequence[EventSegment]] = None,
    ) -> list[EventSegment]:
        """
        Retrieve Top-k disjoint event segments using hybrid semantic+salience+temporal scoring
        and Watershed temporal peak separation (preventing adjacent cluster redundancy).
        """
        all_candidates = self.segments + (list(extra_segments) if extra_segments else [])
        if not all_candidates:
            return []

        scored = [
            (
                self.score_segment(
                    seg,
                    action=action,
                    current_time=current_time,
                    query_embedding=query_embedding,
                ),
                seg,
            )
            for seg in all_candidates
        ]
        scored.sort(key=lambda item: (item[0], item[1].end_time), reverse=True)

        selected: list[EventSegment] = []
        for score, cand in scored:
            if len(selected) >= action.top_k:
                break
            too_close = any(
                abs(cand.mid_time - prev.mid_time) < self.watershed_min_gap_sec
                for prev in selected
            )
            if not too_close:
                selected.append(cand)

        # Sort chronologically for coherent sub-agent reading
        selected.sort(key=lambda s: s.start_time)
        return selected

    def clear(self) -> None:
        self._segments.clear()
        self._active_frames.clear()
        self._active_surprises.clear()
        self._last_feature = None


class CausalStreamingDualBuffer:
    """
    Unified Causal Streaming Dual-Buffer Environment orchestrating:
    1. `ProtectedRecentWindow` (w=4 @ 224x224, 1 fps)
    2. `HighResRingBuffer` (last 4.0s raw high-res native-fps frames)
    3. `OnlineEventArchive` + `SubAgentEvidenceExtractor`
    4. `apply_2d_degradation` for RL 2D-SDG training rollouts
    """

    def __init__(
        self,
        recent_window: Optional[ProtectedRecentWindow] = None,
        ring_buffer: Optional[HighResRingBuffer] = None,
        event_archive: Optional[OnlineEventArchive] = None,
        evidence_extractor: Optional[SubAgentEvidenceExtractor] = None,
    ) -> None:
        self.recent_window = recent_window or ProtectedRecentWindow()
        self.ring_buffer = ring_buffer or HighResRingBuffer()
        self.event_archive = event_archive or OnlineEventArchive()
        self.evidence_extractor = evidence_extractor or SubAgentEvidenceExtractor()
        self.current_time: float = 0.0
        self._degradation_mode: Literal["clean", "spatial", "temporal"] = "clean"
        self._temp_overflow_frames: list[VideoFrame] = []

    def step_frame(self, frame: VideoFrame, force_recent_ingest: bool = False) -> None:
        """
        Causal stream ingestion step at timestamp `frame.timestamp`:
        1. Push raw high-res frame into `HighResRingBuffer`.
        2. Push frame into `ProtectedRecentWindow` (at 1 fps base rate).
        3. If an old frame slides out of `ProtectedRecentWindow`, archive it into `OnlineEventArchive`.
        """
        self.current_time = float(frame.timestamp)
        self.ring_buffer.push_frame(frame)
        evicted = self.recent_window.push_frame(frame, force=force_recent_ingest)
        if evicted is not None:
            self.event_archive.ingest_evicted_frame(evicted)
        elif len(self.recent_window._frames) >= self.recent_window.window_size:
            self.event_archive.ingest_evicted_frame(frame)

    def apply_2d_degradation(
        self,
        mode: Literal["clean", "spatial", "temporal"] = "clean",
    ) -> list[VideoFrame]:
        """
        Set and apply 2D Stochastic Degradation Gating (2D-SDG) for RL rollouts.
        Returns the Turn-1 visible recent frames under the chosen degradation mode.
        """
        if mode not in ("clean", "spatial", "temporal"):
            raise ValueError(f"Invalid 2D-SDG mode: {mode}")
        self._degradation_mode = mode
        visible_frames, overflowed = self.recent_window.get_degraded_observation(mode=mode)
        self._temp_overflow_frames = overflowed
        return visible_frames

    def get_turn1_observation(self) -> tuple[list[VideoFrame], str]:
        """
        Return Turn-1 Protected Recent Working Window frames and compact objective state header.
        Strictly free of historical visual or ground-truth caption logs.
        """
        visible_frames, overflowed = self.recent_window.get_degraded_observation(
            mode=self._degradation_mode
        )
        self._temp_overflow_frames = overflowed
        return visible_frames, self.recent_window.state_header

    def execute_zoom_in(
        self,
        action: ZoomInAction,
        spatial_target_hw: Optional[tuple[int, int]] = None,
    ) -> list[VideoFrame]:
        """Execute `<ZoomIn>` on the CPU high-resolution ring buffer."""
        return self.ring_buffer.execute_zoom_in(
            action=action,
            current_time=self.current_time,
            spatial_target_hw=spatial_target_hw,
        )

    def execute_recall(
        self,
        action: RecallAction,
        query_embedding: Optional[np.ndarray] = None,
        engine: Any = None,
        full_query: Optional[str] = None,
        recent_frames: Optional[Sequence[VideoFrame]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> StructuredEvidenceCard:
        """
        Execute `<Recall>` on the online event archive (including any frames pushed out
        by `temporal` 2D-SDG degradation) and distill via `SubAgentEvidenceExtractor`
        (using shared-weight VLM visual verification when `engine` is provided).
        """
        extra_segs: list[EventSegment] = []
        if self._temp_overflow_frames:
            extra_segs.append(
                self.event_archive._build_segment(
                    self._temp_overflow_frames,
                    event_id=len(self.event_archive.segments) + 1000,
                )
            )
        eff_action = (
            RecallAction(
                query=action.query,
                lookback_sec=max(action.lookback_sec, self.current_time),
                top_k=max(action.top_k, 12),
                start_sec=action.start_sec,
                end_sec=action.end_sec,
            )
            if engine is not None
            else action
        )
        retrieved = self.event_archive.retrieve(
            action=eff_action,
            current_time=self.current_time,
            query_embedding=query_embedding,
            extra_segments=extra_segs,
        )
        return self.evidence_extractor.extract(
            segments=retrieved,
            action=action,
            query_embedding=query_embedding,
            engine=engine,
            full_query=full_query,
            recent_frames=recent_frames if recent_frames is not None else self.recent_window.get_frames(),
            metadata=metadata,
        )

    def reset(self) -> None:
        self.recent_window.clear()
        self.ring_buffer.clear()
        self.event_archive.clear()
        self.current_time = 0.0
        self._degradation_mode = "clean"
        self._temp_overflow_frames = []
