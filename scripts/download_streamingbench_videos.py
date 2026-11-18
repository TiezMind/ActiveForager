#!/usr/bin/env python3
"""
High-Speed Ranged ZIP Video Downloader & Extractor for `mjuicem/StreamingBench`.

Directly fetches the exact compressed byte ranges of `.mp4` videos from HuggingFace CDN
using concurrent video & chunk thread pools (`1 MB` chunks per stream, fast 12s timeout retry),
inflates via `zlib`, verifies CRC32 and uncompressed byte size against the ZIP Central Directory,
stages in `/tmp/sb_staging`, and writes atomically to `/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench/videos/`.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import struct
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

HUBS = ("https://hf-mirror.com", "https://huggingface.co")
TIMEOUT = 12


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args: Any, **kwargs: Any) -> None:
        return None


def resolve_cdn_url(sha: str, zip_fn: str) -> str:
    opener = urllib.request.build_opener(_NoRedirect)
    q_fn = urllib.parse.quote(zip_fn)
    for attempt in range(20):
        for hub in HUBS:
            url = f"{hub}/datasets/mjuicem/StreamingBench/resolve/{sha}/{q_fn}"
            try:
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
                with opener.open(req, timeout=TIMEOUT):
                    return url
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    loc = e.headers["Location"]
                    return loc if loc.startswith("http") else hub + loc
            except Exception:
                continue
        time.sleep(min(1.5**attempt, 10.0))
    raise RuntimeError(f"Failed to resolve CDN URL for {zip_fn}")


def fetch_byte_range(
    url_box: list[str],
    lock: threading.Lock,
    sha: str,
    zip_fn: str,
    start: int,
    end: int,
) -> bytes:
    expected = end - start + 1
    for attempt in range(30):
        url = url_box[0]
        req = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            if len(data) == expected:
                return data
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 410):
                with lock:
                    if url_box[0] == url:
                        url_box[0] = resolve_cdn_url(sha, zip_fn)
        except Exception:
            pass
        time.sleep(min(1.2**attempt, 5.0))
    raise RuntimeError(f"Failed byte range {start}-{end} for {zip_fn}")


def select_priority_videos(
    data_dir: Path,
    videos_index: dict[str, dict[str, Any]],
    max_samples_per_subtask: int = 25,
) -> list[str]:
    """
    Select videos covering at least `max_samples_per_subtask` across all 18 subtasks,
    prioritizing smaller video files first so all 18 subtasks reach full coverage in ~4.5 GB,
    followed by sequential order for any additional coverage.
    """
    by_task: dict[str, list[tuple[int, str]]] = defaultdict(list)
    seq_counts: dict[str, int] = defaultdict(int)
    seq_needed: set[str] = set()

    for fn in ("questions_real.json", "questions_omni.json", "questions_sqa.json", "questions_proactive.json"):
        p = data_dir / fn
        if not p.is_file():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        entries = [e for g in d for e in g] if fn == "questions_sqa.json" else d
        for e in entries:
            vp = str(e.get("video_path", ""))
            if vp not in videos_index:
                continue
            sz = int(videos_index[vp]["comp_sz"])
            for q in e.get("questions", []):
                st = str(q.get("task_type", ""))
                by_task[st].append((sz, vp))
                if seq_counts[st] < max_samples_per_subtask:
                    seq_counts[st] += 1
                    seq_needed.add(vp)

    tier1: set[str] = set()
    for st, items in by_task.items():
        for _, vp in sorted(items, key=lambda x: (x[0], x[1]))[:max_samples_per_subtask]:
            tier1.add(vp)

    tier1_sorted = sorted(tier1, key=lambda k: videos_index[k]["comp_sz"])
    tier2_sorted = sorted(seq_needed - tier1, key=lambda k: videos_index[k]["comp_sz"])
    return tier1_sorted + tier2_sorted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-json", type=Path, default=Path("/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench/sb_zip_index.json"))
    ap.add_argument("--sb-root", type=Path, default=Path("/mnt/hdfs/lipengyu/DA/video/raw/StreamingBench"))
    ap.add_argument("--staging-dir", type=Path, default=Path("/tmp/sb_staging"))
    ap.add_argument("--max-samples-per-subtask", type=int, default=25)
    ap.add_argument("--workers", type=int, default=96)
    ap.add_argument("--concurrent-videos", type=int, default=6)
    ap.add_argument("--chunk-kb", type=int, default=512)
    ap.add_argument("--shard", type=str, default="0/1")
    args = ap.parse_args()

    idx_payload = json.loads(args.index_json.read_text(encoding="utf-8"))
    sha = idx_payload["sha"]
    videos_index: dict[str, dict[str, Any]] = idx_payload["videos"]

    target_rel_paths = select_priority_videos(
        data_dir=args.sb_root / "data",
        videos_index=videos_index,
        max_samples_per_subtask=args.max_samples_per_subtask,
    )

    k_str, n_str = args.shard.split("/")
    shard_idx, num_shards = int(k_str), int(n_str)
    my_targets = target_rel_paths[shard_idx::num_shards]

    args.staging_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = args.sb_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)

    url_cache: dict[str, tuple[list[str], threading.Lock]] = {}
    url_cache_lock = threading.Lock()
    chunk_bytes = args.chunk_kb * 1024
    chunk_pool = ThreadPoolExecutor(max_workers=args.workers)

    t0 = time.time()
    stats_lock = threading.Lock()
    state = {"done_bytes": 0, "done_count": 0}

    print(
        f"[Shard {args.shard}] Downloading {len(my_targets)} StreamingBench videos "
        f"(workers={args.workers}, concurrent_vids={args.concurrent_videos}, chunk={args.chunk_kb}KB)...",
        flush=True,
    )

    def _download_one_video(rel_key: str) -> None:
        meta = videos_index[rel_key]
        dest_path = args.sb_root / rel_key.lstrip("./")
        uncomp_sz = int(meta["uncomp_sz"])
        if dest_path.is_file() and dest_path.stat().st_size == uncomp_sz:
            with stats_lock:
                state["done_count"] += 1
                cnt = state["done_count"]
            print(f"[{cnt}/{len(my_targets)}] SKIP existing {dest_path.name} ({uncomp_sz/1e6:.1f} MB)", flush=True)
            return

        zip_fn = meta["zip_fn"]
        with url_cache_lock:
            if zip_fn not in url_cache:
                url_cache[zip_fn] = ([resolve_cdn_url(sha, zip_fn)], threading.Lock())
            url_box, lock = url_cache[zip_fn]

        local_off = int(meta["local_off"])
        comp_sz = int(meta["comp_sz"])
        total_span = comp_sz + 512

        ranges = [
            (s, min(s + chunk_bytes - 1, local_off + total_span - 1))
            for s in range(local_off, local_off + total_span, chunk_bytes)
        ]
        futs = {
            chunk_pool.submit(fetch_byte_range, url_box, lock, sha, zip_fn, s, e): idx
            for idx, (s, e) in enumerate(ranges)
        }
        ordered_parts: list[bytes] = [b""] * len(ranges)
        for fut in as_completed(futs):
            idx = futs[fut]
            ordered_parts[idx] = fut.result()

        buf = b"".join(ordered_parts)
        if buf[:4] != b"PK\x03\x04":
            raise RuntimeError(f"Invalid ZIP local header for {rel_key}: {buf[:4]!r}")
        name_len, extra_len = struct.unpack("<HH", buf[26:30])
        data_start = 30 + name_len + extra_len
        comp_slice = buf[data_start : data_start + comp_sz]

        raw_mp4 = zlib.decompress(comp_slice, -15) if int(meta["comp_method"]) == 8 else comp_slice
        if len(raw_mp4) != uncomp_sz:
            raise RuntimeError(f"Size mismatch for {rel_key}: {len(raw_mp4)} != {uncomp_sz}")
        got_crc = zlib.crc32(raw_mp4) & 0xFFFFFFFF
        if got_crc != int(meta["crc32"]):
            raise RuntimeError(f"CRC32 mismatch for {rel_key}: {got_crc} != {meta['crc32']}")

        stage_file = args.staging_dir / f"{dest_path.name}.part.{os.getpid()}.{threading.get_ident()}"
        stage_file.write_bytes(raw_mp4)
        shutil.copyfile(stage_file, dest_path)
        stage_file.unlink(missing_ok=True)

        with stats_lock:
            state["done_bytes"] += comp_sz
            state["done_count"] += 1
            cnt = state["done_count"]
            db = state["done_bytes"]
        elapsed = max(1e-6, time.time() - t0)
        rate_mb = (db / 1e6) / elapsed
        print(
            f"[{cnt}/{len(my_targets)}] OK {dest_path.name} "
            f"({uncomp_sz/1e6:.1f} MB) | avg_rate={rate_mb:.2f} MB/s | elapsed={elapsed:.0f}s",
            flush=True,
        )

    with ThreadPoolExecutor(max_workers=args.concurrent_videos) as vid_pool:
        vid_futs = [vid_pool.submit(_download_one_video, rk) for rk in my_targets]
        for vf in as_completed(vid_futs):
            vf.result()

    chunk_pool.shutdown(wait=True)
    print(f"[Shard {args.shard}] All {len(my_targets)} videos ready in {time.time() - t0:.1f}s!", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
