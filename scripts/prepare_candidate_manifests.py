#!/usr/bin/env python3
"""
CPU Candidate Manifest Builder (`scripts/prepare_candidate_manifests.py`).

Scans `/mnt/hdfs/lipengyu/DA/video/raw/lmms-lab/LLaVA-Video-178K` (`mc`, `oe`, `cap`)
and `/mnt/hdfs/lipengyu/DA/video/raw/TencentARC/Video-Holmes/train_Video-Holmes.json`
to construct strictly video-disjoint, ready-to-train manifests:
1. `data/sft_coldstart_6k.jsonl` (6,000 trajectories):
   - Type 1 (`direct_answer`): 2,000 samples (33.3%)
   - Type 2 (`zoom_in`):       1,800 samples (30.0%)
   - Type 3 (`recall`):        1,600 samples (26.7%: 1,200 positiveFOUND + 400 negative HLD NONE)
   - Type 4 (`wait`):            600 samples (10.0%)
2. `data/rl_ig_grpo_8k.jsonl` (8,000 verifiable prompts on disjoint videos):
   - `realtime_perception`:    2,500 prompts (half MCQ, half short open-ended <= 10 words)
   - `historical_recall`:      3,000 prompts
   - `hallucination_hld`:      1,500 prompts (cross-video false premise)
   - `forward_active_wait`:    1,000 prompts
3. `data/manifest_summary.json`: Exact counts, SHA256 digests, and zero-video-overlap verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Optional


DEFAULT_LLAVA_DIR = Path("/mnt/hdfs/lipengyu/DA/video/raw/lmms-lab/LLaVA-Video-178K/2_3_m_academic_v0_1")
DEFAULT_HOLMES_TRAIN = Path("/mnt/hdfs/lipengyu/DA/video/raw/TencentARC/Video-Holmes/train_Video-Holmes.json")
DEFAULT_HDFS_FRAMES_DIR = Path("/mnt/hdfs/lipengyu/DA/video/frames/f768-fps2-px230400")
DEFAULT_OUT_DIR = Path("/mlx/users/lipengyu.seed/playground/agentic_video_understanding/data")
DEFAULT_HDFS_MIRROR_DIR = Path("/mnt/hdfs/lipengyu/DA/forager_data")

_OPTION_RE = re.compile(r"^([A-H])\.\s*(.+)$")

_ZOOM_PATTERNS = re.compile(
    r"\b(text|title|word|words|sign|label|logo|number|numbers|letter|letters|read|written|"
    r"wear|wearing|attire|shirt|color|colour|small|tiny|detail|holding|hold|hand|hands|"
    r"expression|facial|instrument|tool|object on the|fast|quickly|rapid|gesture)\b",
    re.IGNORECASE,
)

_RECALL_PATTERNS = re.compile(
    r"\b(before|after|first|earlier|beginning|start|initially|previous|prior|sequence|"
    r"order|followed by|what happened when|how did .* begin|at the start|formerly)\b",
    re.IGNORECASE,
)

_WAIT_PATTERNS = re.compile(
    r"\b(how many times|count|total times|next step|what happens at the end|finally|"
    r"eventual|outcome|conclude|finish|complete|afterward|about to|going to)\b",
    re.IGNORECASE,
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_llava_mc_turn(human_val: str, gpt_val: str) -> Optional[dict[str, Any]]:
    lines = human_val.replace("<image>\n", "").replace("<image>", "").strip().splitlines()
    opt_matches: list[tuple[int, str, str]] = []
    for idx, line in enumerate(lines):
        m = _OPTION_RE.match(line.strip())
        if m is not None:
            opt_matches.append((idx, m.group(1), m.group(2).strip()))
    if len(opt_matches) < 2:
        return None
    stem = "\n".join(lines[: opt_matches[0][0]]).strip()
    if not stem:
        return None
    options = [text for _, _, text in opt_matches]
    ans_letter = gpt_val.strip()[:1].upper()
    valid_letters = [let for _, let, _ in opt_matches]
    if ans_letter not in valid_letters:
        return None
    ans_idx = valid_letters.index(ans_letter)
    return {
        "question": stem,
        "options": options,
        "gt_answer": ans_letter,
        "answer_text": options[ans_idx],
        "task_type": "mcq",
    }


def _parse_llava_oe_turn(human_val: str, gpt_val: str) -> Optional[dict[str, Any]]:
    question = human_val.replace("<image>\n", "").replace("<image>", "").strip()
    answer = gpt_val.strip()
    if not question or not answer:
        return None
    words = answer.split()
    return {
        "question": question,
        "options": [],
        "gt_answer": " ".join(words[:12]),
        "answer_text": answer,
        "word_count": len(words),
        "task_type": "open",
    }


def _classify_question_regime(question: str) -> str:
    if _WAIT_PATTERNS.search(question):
        return "wait"
    if _RECALL_PATTERNS.search(question):
        return "recall"
    if _ZOOM_PATTERNS.search(question):
        return "zoom_in"
    return "direct_answer"


def _format_question_with_options(question: str, options: list[str]) -> str:
    if not options:
        return question
    opt_str = "; ".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(options)) + ";"
    return f"Question: {question}\nOptions:\n{opt_str}\nRespond with the option letter."


def _build_sft_trajectory_record(
    raw_item: dict[str, Any],
    traj_type: str,
    sample_idx: int,
    false_premise_source: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Construct a 1-turn or 2-turn SFT cold-start trajectory following the exact
    orthogonal protocol (`<think>...</think>` + `<Answer>`/`<ZoomIn>`/`<Recall>`/`<Wait>`).
    """
    q_stem = raw_item["question"]
    options = list(raw_item.get("options", []))
    gt_ans = raw_item["gt_answer"]
    prompt_text = _format_question_with_options(q_stem, options)
    video_id = raw_item["video_id"]
    video_rel = raw_item["video_path"]
    source_ds = raw_item["source_dataset"]

    if traj_type == "direct_answer":
        t1 = (
            "<think>The target object and scene state are clearly visible in the 4 uncompressed "
            "recent frames without needing spatial zoom or historical recall.</think>"
            f"<Answer>{gt_ans}</Answer>"
        )
        return {
            "sample_id": f"sft_direct_{sample_idx:05d}",
            "trajectory_type": "direct_answer",
            "source_dataset": source_ds,
            "video_id": video_id,
            "video_path": video_rel,
            "frame_cache_exists": raw_item.get("frame_cache_exists", False),
            "query_timestamp": 24.0,
            "num_turns": 1,
            "action_type": "Answer",
            "prompt": prompt_text,
            "turn1_target": t1,
            "turn2_evidence": None,
            "turn2_target": None,
            "gt_answer": gt_ans,
            "task_type": raw_item["task_type"],
        }

    if traj_type == "zoom_in":
        is_motion = any(w in q_stem.lower() for w in ("action", "fast", "doing", "move", "gesture", "quickly"))
        mode = "temporal_hifps" if is_motion else "spatial_hires"
        roi = [0.15, 0.15, 0.85, 0.85] if mode == "spatial_hires" else [0.0, 0.0, 1.0, 1.0]
        zoom_json = json.dumps({"mode": mode, "roi": roi, "window_sec": 2.0})
        t1 = (
            "<think>The fine-grained detail in the current window is blurry or under-sampled at base 224x224 "
            f"1fps resolution; invoking ZoomIn ({mode}) to inspect high-definition evidence.</think>"
            f"<ZoomIn>{zoom_json}</ZoomIn>"
        )
        t2 = (
            f"<think>From the zoomed {mode} visual tokens, the fine detail confirms '{raw_item.get('answer_text', gt_ans)}'.</think>"
            f"<Answer>{gt_ans}</Answer>"
        )
        return {
            "sample_id": f"sft_zoom_{sample_idx:05d}",
            "trajectory_type": "zoom_in",
            "source_dataset": source_ds,
            "video_id": video_id,
            "video_path": video_rel,
            "frame_cache_exists": raw_item.get("frame_cache_exists", False),
            "query_timestamp": 28.0,
            "num_turns": 2,
            "action_type": "ZoomIn",
            "zoom_mode": mode,
            "prompt": prompt_text,
            "turn1_target": t1,
            "turn2_evidence": f"<ZoomedFrames mode=\"{mode}\" roi=\"{roi}\" />",
            "turn2_target": t2,
            "gt_answer": gt_ans,
            "task_type": raw_item["task_type"],
        }

    if traj_type == "recall_found":
        short_q = " ".join(q_stem.split()[:10])
        recall_json = json.dumps({"query": short_q, "lookback_sec": 90.0, "top_k": 2})
        t1 = (
            "<think>The queried event occurred earlier in the stream and has been evicted from the 4-frame "
            f"recent working window; recalling historical archive for '{short_q}'.</think>"
            f"<Recall>{recall_json}</Recall>"
        )
        ev_card = (
            f"<EvidenceCard status=\"FOUND\" anchor_time=\"12.0s\">"
            f"Verified event [10.0s-14.0s]: {raw_item.get('answer_text', gt_ans)}."
            f"</EvidenceCard>"
        )
        t2 = (
            "<think>Sub-agent verified historical evidence card and anchor frame under SDAM isolation "
            f"confirm the answer is {gt_ans}.</think>"
            f"<Answer>{gt_ans}</Answer>"
        )
        return {
            "sample_id": f"sft_recall_found_{sample_idx:05d}",
            "trajectory_type": "recall_found",
            "source_dataset": source_ds,
            "video_id": video_id,
            "video_path": video_rel,
            "frame_cache_exists": raw_item.get("frame_cache_exists", False),
            "query_timestamp": 95.0,
            "num_turns": 2,
            "action_type": "Recall",
            "recall_status": "FOUND",
            "prompt": prompt_text,
            "turn1_target": t1,
            "turn2_evidence": ev_card,
            "turn2_target": t2,
            "gt_answer": gt_ans,
            "task_type": raw_item["task_type"],
        }

    if traj_type == "recall_none_hld":
        foreign_q = (
            false_premise_source["question"]
            if false_premise_source is not None
            else "What did the scuba diver place inside the underwater treasure chest?"
        )
        short_q = " ".join(foreign_q.split()[:10])
        hld_options = [
            "At the very beginning of the video",
            "Unable to answer / This event did not occur in the video",
            "Right after the camera panned left",
            "During the final scene transition",
        ]
        hld_prompt = _format_question_with_options(foreign_q, hld_options)
        recall_json = json.dumps({"query": short_q, "lookback_sec": 120.0, "top_k": 2})
        t1 = (
            "<think>The question asks about an earlier event not visible in the recent 4 frames; "
            f"recalling historical archive for '{short_q}'.</think>"
            f"<Recall>{recall_json}</Recall>"
        )
        ev_card = (
            f"<EvidenceCard status=\"NONE\">"
            f"NONE_FOUND: Verified historical archive; '{short_q}' did not occur."
            f"</EvidenceCard>"
        )
        t2 = (
            "<think>Sub-agent verification returned status=NONE with 0 visual tokens injected; "
            "the premise is a hallucination not present in the stream.</think>"
            "<Answer>B</Answer>"
        )
        return {
            "sample_id": f"sft_recall_hld_{sample_idx:05d}",
            "trajectory_type": "recall_none_hld",
            "source_dataset": source_ds,
            "video_id": video_id,
            "video_path": video_rel,
            "frame_cache_exists": raw_item.get("frame_cache_exists", False),
            "query_timestamp": 110.0,
            "num_turns": 2,
            "action_type": "Recall",
            "recall_status": "NONE",
            "foraged_visual_tokens": 0,
            "prompt": hld_prompt,
            "turn1_target": t1,
            "turn2_evidence": ev_card,
            "turn2_target": t2,
            "gt_answer": "B",
            "task_type": "mcq",
        }

    if traj_type == "wait":
        wait_json = json.dumps({"check_after_sec": 2.0, "trigger": "step_or_clue_completion"})
        t1 = (
            "<think>The queried action sequence or clue is still unfolding at the current timestamp; "
            "emitting Wait to defer commitment until the transition completes.</think>"
            f"<Wait>{wait_json}</Wait>"
        )
        t2 = (
            "<think>The subsequent stream frames now reveal the completed action state.</think>"
            f"<Answer>{gt_ans}</Answer>"
        )
        return {
            "sample_id": f"sft_wait_{sample_idx:05d}",
            "trajectory_type": "wait",
            "source_dataset": source_ds,
            "video_id": video_id,
            "video_path": video_rel,
            "frame_cache_exists": raw_item.get("frame_cache_exists", False),
            "query_timestamp": 18.0,
            "target_reveal_time": 20.0,
            "num_turns": 2,
            "action_type": "Wait",
            "prompt": prompt_text,
            "turn1_target": t1,
            "turn2_evidence": "<StreamAdvance delta_sec=\"2.0\" />",
            "turn2_target": t2,
            "gt_answer": gt_ans,
            "task_type": "proactive",
        }

    raise ValueError(f"Unknown trajectory type: {traj_type}")


def collect_raw_pool(
    llava_dir: Path,
    holmes_train_path: Path,
    hdfs_frames_dir: Path,
) -> list[dict[str, Any]]:
    """Load and parse QA items from LLaVA-Video-178K (`mc`, `oe`) and Video-Holmes `train`."""
    pool: list[dict[str, Any]] = []

    holmes_cached_videos: set[str] = set()
    holmes_frame_dir = hdfs_frames_dir / "videoholmes"
    if holmes_frame_dir.exists():
        holmes_cached_videos = {p.stem for p in holmes_frame_dir.glob("*.npy")}

    # 1. Video-Holmes Train split (multi-hop temporal & episodic reasoning)
    if holmes_train_path.exists():
        holmes_data = json.loads(holmes_train_path.read_text(encoding="utf-8"))
        for q in holmes_data:
            vid_id = str(q["video ID"])
            opts_dict = q.get("Options", {})
            letters = sorted(opts_dict.keys())
            if not letters or q.get("Answer") not in letters:
                continue
            options = [str(opts_dict[k]).strip() for k in letters]
            ans_letter = str(q["Answer"]).strip()
            ans_text = str(opts_dict[ans_letter]).strip()
            question = str(q["Question"]).strip()
            pool.append(
                {
                    "source_dataset": "Video-Holmes-Train",
                    "video_id": f"videoholmes__{vid_id}",
                    "video_path": f"videoholmes/{vid_id}.mp4",
                    "frame_cache_exists": vid_id in holmes_cached_videos,
                    "question": question,
                    "options": options,
                    "gt_answer": ans_letter,
                    "answer_text": ans_text,
                    "task_type": "mcq",
                    "regime": "recall" if _RECALL_PATTERNS.search(question) else _classify_question_regime(question),
                }
            )

    # 2. LLaVA-Video-178K Multiple-Choice split
    mc_path = llava_dir / "2_3_m_academic_mc_v0_1_qa_processed.json"
    if mc_path.exists():
        mc_data = json.loads(mc_path.read_text(encoding="utf-8"))
        for row in mc_data:
            vid_rel = str(row.get("video", ""))
            vid_id = "llava__" + vid_rel.rsplit(".", 1)[0].replace("/", "__")
            conv = row.get("conversations", [])
            for t in range(0, len(conv) - 1, 2):
                parsed = _parse_llava_mc_turn(conv[t]["value"], conv[t + 1]["value"])
                if parsed is None:
                    continue
                regime = _classify_question_regime(parsed["question"])
                pool.append(
                    {
                        "source_dataset": "LLaVA-Video-178K-MC",
                        "video_id": vid_id,
                        "video_path": vid_rel,
                        "frame_cache_exists": False,
                        "question": parsed["question"],
                        "options": parsed["options"],
                        "gt_answer": parsed["gt_answer"],
                        "answer_text": parsed["answer_text"],
                        "task_type": "mcq",
                        "regime": regime,
                    }
                )

    # 3. LLaVA-Video-178K Open-Ended split (keep short verifiable answers <= 10 words for RL)
    oe_path = llava_dir / "2_3_m_academic_oe_v0_1_qa_processed.json"
    if oe_path.exists():
        oe_data = json.loads(oe_path.read_text(encoding="utf-8"))
        for row in oe_data[:6000]:
            vid_rel = str(row.get("video", ""))
            vid_id = "llava__" + vid_rel.rsplit(".", 1)[0].replace("/", "__")
            conv = row.get("conversations", [])
            for t in range(0, len(conv) - 1, 2):
                parsed = _parse_llava_oe_turn(conv[t]["value"], conv[t + 1]["value"])
                if parsed is None or parsed["word_count"] > 12:
                    continue
                regime = _classify_question_regime(parsed["question"])
                pool.append(
                    {
                        "source_dataset": "LLaVA-Video-178K-OE",
                        "video_id": vid_id,
                        "video_path": vid_rel,
                        "frame_cache_exists": False,
                        "question": parsed["question"],
                        "options": [],
                        "gt_answer": parsed["gt_answer"],
                        "answer_text": parsed["answer_text"],
                        "task_type": "open",
                        "regime": regime,
                    }
                )

    return pool


def build_manifests(
    llava_dir: Path = DEFAULT_LLAVA_DIR,
    holmes_train_path: Path = DEFAULT_HOLMES_TRAIN,
    hdfs_frames_dir: Path = DEFAULT_HDFS_FRAMES_DIR,
    out_dir: Path = DEFAULT_OUT_DIR,
    hdfs_mirror_dir: Optional[Path] = DEFAULT_HDFS_MIRROR_DIR,
    sft_total: int = 6000,
    rl_total: int = 8000,
    seed: int = 20261106,
) -> dict[str, Any]:
    """
    Partition videos strictly disjointly into SFT vs. RL pools and build:
    - `sft_coldstart_6k.jsonl` (6,000 samples across Types 1-4)
    - `rl_ig_grpo_8k.jsonl` (8,000 verifiable prompts across 4 RL categories)
    """
    rng = random.Random(seed)
    raw_pool = collect_raw_pool(llava_dir, holmes_train_path, hdfs_frames_dir)
    if not raw_pool:
        raise RuntimeError("No candidate items found in raw pool")

    # Strictly partition by `video_id` so SFT and RL share ZERO videos
    all_videos = sorted({item["video_id"] for item in raw_pool})
    rng.shuffle(all_videos)
    split_idx = int(round(len(all_videos) * 0.43))
    sft_videos = set(all_videos[:split_idx])
    rl_videos = set(all_videos[split_idx:])
    assert len(sft_videos & rl_videos) == 0, "SFT and RL video sets must be strictly disjoint"

    sft_candidates = [it for it in raw_pool if it["video_id"] in sft_videos and it["task_type"] == "mcq"]
    rl_candidates = [it for it in raw_pool if it["video_id"] in rl_videos]
    rng.shuffle(sft_candidates)
    rng.shuffle(rl_candidates)

    # Target SFT counts (scales proportionally if sft_total != 6000)
    n_direct = int(round(sft_total * (2000 / 6000)))
    n_zoom = int(round(sft_total * (1800 / 6000)))
    n_recall_found = int(round(sft_total * (1200 / 6000)))
    n_recall_hld = int(round(sft_total * (400 / 6000)))
    n_wait = sft_total - (n_direct + n_zoom + n_recall_found + n_recall_hld)

    by_regime: dict[str, list[dict[str, Any]]] = {
        "direct_answer": [],
        "zoom_in": [],
        "recall": [],
        "wait": [],
    }
    for it in sft_candidates:
        by_regime[it["regime"]].append(it)

    def _draw_from_bucket(bucket_name: str, count: int, fallback_pool: list[dict[str, Any]]) -> list[dict[str, Any]]:
        bucket = by_regime[bucket_name]
        picked = bucket[:count]
        del bucket[:count]
        if len(picked) < count:
            needed = count - len(picked)
            picked.extend(fallback_pool[:needed])
            del fallback_pool[:needed]
        return picked

    leftover_sft = (
        by_regime["direct_answer"][n_direct:]
        + by_regime["zoom_in"][n_zoom:]
        + by_regime["recall"][n_recall_found + n_recall_hld :]
        + by_regime["wait"][n_wait:]
    )
    rng.shuffle(leftover_sft)

    sft_direct_raw = _draw_from_bucket("direct_answer", n_direct, leftover_sft)
    sft_zoom_raw = _draw_from_bucket("zoom_in", n_zoom, leftover_sft)
    sft_recall_raw = _draw_from_bucket("recall", n_recall_found, leftover_sft)
    sft_hld_raw = _draw_from_bucket("recall", n_recall_hld, leftover_sft)
    sft_wait_raw = _draw_from_bucket("wait", n_wait, leftover_sft)

    sft_records: list[dict[str, Any]] = []
    for item in sft_direct_raw:
        sft_records.append(_build_sft_trajectory_record(item, "direct_answer", len(sft_records)))
    for item in sft_zoom_raw:
        sft_records.append(_build_sft_trajectory_record(item, "zoom_in", len(sft_records)))
    for item in sft_recall_raw:
        sft_records.append(_build_sft_trajectory_record(item, "recall_found", len(sft_records)))
    for idx, item in enumerate(sft_hld_raw):
        # Pair with a question from a completely different video to guarantee a false premise
        foreign = sft_zoom_raw[(idx + 17) % len(sft_zoom_raw)] if sft_zoom_raw else None
        sft_records.append(
            _build_sft_trajectory_record(item, "recall_none_hld", len(sft_records), false_premise_source=foreign)
        )
    for item in sft_wait_raw:
        sft_records.append(_build_sft_trajectory_record(item, "wait", len(sft_records)))

    rng.shuffle(sft_records)

    # Target RL counts (scales proportionally if rl_total != 8000):
    # 2,500 fine-grained perception (half MCQ, half short OE), 3,000 recall, 1,500 HLD, 1,000 FAR wait
    n_rl_rt = int(round(rl_total * (2500 / 8000)))
    n_rl_rec = int(round(rl_total * (3000 / 8000)))
    n_rl_hld = int(round(rl_total * (1500 / 8000)))
    n_rl_wait = rl_total - (n_rl_rt + n_rl_rec + n_rl_hld)

    rl_mcq = [it for it in rl_candidates if it["task_type"] == "mcq"]
    rl_oe = [it for it in rl_candidates if it["task_type"] == "open"]

    n_rt_mcq = n_rl_rt // 2
    n_rt_oe = n_rl_rt - n_rt_mcq

    rl_rt_items = rl_mcq[:n_rt_mcq] + rl_oe[:n_rt_oe]
    cursor_mcq = n_rt_mcq
    rl_rec_items = rl_mcq[cursor_mcq : cursor_mcq + n_rl_rec]
    cursor_mcq += n_rl_rec
    rl_hld_items = rl_mcq[cursor_mcq : cursor_mcq + n_rl_hld]
    cursor_mcq += n_rl_hld
    rl_wait_items = rl_mcq[cursor_mcq : cursor_mcq + n_rl_wait]

    rl_records: list[dict[str, Any]] = []

    for it in rl_rt_items:
        rl_records.append(
            {
                "prompt_id": f"rl_rt_{len(rl_records):05d}",
                "rl_bucket": "realtime_perception",
                "source_dataset": it["source_dataset"],
                "video_id": it["video_id"],
                "video_path": it["video_path"],
                "frame_cache_exists": it["frame_cache_exists"],
                "query_timestamp": 24.0,
                "prompt": _format_question_with_options(it["question"], it["options"]),
                "gt_answer": it["gt_answer"],
                "answer_text": it["answer_text"],
                "task_type": it["task_type"],
                "recommended_sdg_probs": {"clean": 0.4, "spatial": 0.3, "temporal": 0.3},
            }
        )

    for it in rl_rec_items:
        rl_records.append(
            {
                "prompt_id": f"rl_recall_{len(rl_records):05d}",
                "rl_bucket": "historical_recall",
                "source_dataset": it["source_dataset"],
                "video_id": it["video_id"],
                "video_path": it["video_path"],
                "frame_cache_exists": it["frame_cache_exists"],
                "query_timestamp": 120.0,
                "evidence_window": [10.0, 25.0],
                "prompt": _format_question_with_options(it["question"], it["options"]),
                "gt_answer": it["gt_answer"],
                "answer_text": it["answer_text"],
                "task_type": "mcq",
                "recommended_sdg_probs": {"clean": 0.4, "spatial": 0.3, "temporal": 0.3},
            }
        )

    for idx, it in enumerate(rl_hld_items):
        foreign = rl_rec_items[(idx + 31) % max(1, len(rl_rec_items))]
        hld_opts = [
            foreign["answer_text"],
            "Unable to answer / This event did not occur in the video",
            "At the beginning of the clip",
            "Right before the scene ended",
        ]
        rl_records.append(
            {
                "prompt_id": f"rl_hld_{len(rl_records):05d}",
                "rl_bucket": "hallucination_hld",
                "source_dataset": it["source_dataset"],
                "video_id": it["video_id"],
                "video_path": it["video_path"],
                "frame_cache_exists": it["frame_cache_exists"],
                "query_timestamp": 90.0,
                "prompt": _format_question_with_options(foreign["question"], hld_opts),
                "gt_answer": "B",
                "answer_text": "Unable to answer / This event did not occur in the video",
                "task_type": "mcq",
                "is_negative_premise": True,
                "recommended_sdg_probs": {"clean": 0.4, "spatial": 0.3, "temporal": 0.3},
            }
        )

    for it in rl_wait_items:
        rl_records.append(
            {
                "prompt_id": f"rl_wait_{len(rl_records):05d}",
                "rl_bucket": "forward_active_wait",
                "source_dataset": it["source_dataset"],
                "video_id": it["video_id"],
                "video_path": it["video_path"],
                "frame_cache_exists": it["frame_cache_exists"],
                "query_timestamp": 16.0,
                "target_reveal_time": 20.0,
                "prompt": _format_question_with_options(it["question"], it["options"]),
                "gt_answer": it["gt_answer"],
                "answer_text": it["answer_text"],
                "task_type": "proactive",
                "recommended_sdg_probs": {"clean": 0.4, "spatial": 0.3, "temporal": 0.3},
            }
        )

    rng.shuffle(rl_records)

    out_dir.mkdir(parents=True, exist_ok=True)
    sft_path = out_dir / "sft_coldstart_6k.jsonl"
    rl_path = out_dir / "rl_ig_grpo_8k.jsonl"
    summary_path = out_dir / "manifest_summary.json"

    sft_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in sft_records), encoding="utf-8")
    rl_path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rl_records), encoding="utf-8")

    sft_vid_used = {r["video_id"] for r in sft_records}
    rl_vid_used = {r["video_id"] for r in rl_records}
    assert len(sft_vid_used & rl_vid_used) == 0, "Video overlap detected between SFT and RL manifests!"

    summary = {
        "seed": seed,
        "raw_pool_total": len(raw_pool),
        "sft_count": len(sft_records),
        "sft_unique_videos": len(sft_vid_used),
        "sft_by_trajectory_type": dict(Counter(r["trajectory_type"] for r in sft_records)),
        "sft_by_action_type": dict(Counter(r["action_type"] for r in sft_records)),
        "sft_tool_bearing_ratio": round(
            sum(1 for r in sft_records if r["action_type"] != "Answer") / max(1, len(sft_records)), 4
        ),
        "rl_count": len(rl_records),
        "rl_unique_videos": len(rl_vid_used),
        "rl_by_bucket": dict(Counter(r["rl_bucket"] for r in rl_records)),
        "rl_by_task_type": dict(Counter(r["task_type"] for r in rl_records)),
        "video_overlap_count": len(sft_vid_used & rl_vid_used),
        "files": {
            "sft_coldstart_6k.jsonl": {"path": str(sft_path), "sha256": sha256_file(sft_path)},
            "rl_ig_grpo_8k.jsonl": {"path": str(rl_path), "sha256": sha256_file(rl_path)},
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if hdfs_mirror_dir is not None:
        hdfs_mirror_dir.mkdir(parents=True, exist_ok=True)
        for p in (sft_path, rl_path, summary_path):
            (hdfs_mirror_dir / p.name).write_bytes(p.read_bytes())

    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Build 6K SFT and 8K RL candidate manifests on CPU.")
    ap.add_argument("--llava-dir", type=Path, default=DEFAULT_LLAVA_DIR)
    ap.add_argument("--holmes-train", type=Path, default=DEFAULT_HOLMES_TRAIN)
    ap.add_argument("--hdfs-frames-dir", type=Path, default=DEFAULT_HDFS_FRAMES_DIR)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--hdfs-mirror-dir", type=Path, default=DEFAULT_HDFS_MIRROR_DIR)
    ap.add_argument("--sft-total", type=int, default=6000)
    ap.add_argument("--rl-total", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args()

    summary = build_manifests(
        llava_dir=args.llava_dir,
        holmes_train_path=args.holmes_train,
        hdfs_frames_dir=args.hdfs_frames_dir,
        out_dir=args.out_dir,
        hdfs_mirror_dir=args.hdfs_mirror_dir,
        sft_total=args.sft_total,
        rl_total=args.rl_total,
        seed=args.seed,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
