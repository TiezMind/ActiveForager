#!/usr/bin/env python3
"""Parallel ranged downloader for Hugging Face repos (stdlib only) with hf-mirror fallback.

Why: behind the devbox proxy each TCP stream is capped at ~1 MB/s, hf_transfer stalls and
hf_xet gets throttled after ~1 GB per file. Plain HTTP range requests against the resolved
CDN URL scale linearly with the number of streams, so we fetch fixed-size ranges from a
thread pool, retry stalled ranges, and verify every LFS file against the sha256 published
by the Hub API.

Usage: scripts/hf_parallel_download.py <repo_id> <local_dir> [--workers 48] [--revision main]
"""
import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

HUBS = ("https://huggingface.co", "https://hf-mirror.com")
TIMEOUT = 60


def api_files(repo, revision, repo_type):
    last_err = None
    for hub in HUBS:
        url = f"{hub}/api/{repo_type}s/{repo}/revision/{revision}?blobs=true"
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                info = json.load(r)
            return info["sha"], [
                (s["rfilename"], s.get("size"), (s.get("lfs") or {}).get("sha256"))
                for s in info["siblings"]
            ]
        except Exception as e:
            last_err = e
    raise last_err


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        return None


def resolve(repo, sha, fname):
    """Return the final CDN URL (one resolver call per file). `repo` carries the
    'datasets/' prefix for dataset repos. Falls back across HUBS on 429 rate-limits."""
    opener = urllib.request.build_opener(_NoRedirect, urllib.request.ProxyHandler())
    for attempt in range(30):
        for hub in HUBS:
            url = f"{hub}/{repo}/resolve/{sha}/{fname}"
            try:
                req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "Mozilla/5.0"})
                with opener.open(req, timeout=TIMEOUT) as r:
                    return url  # served directly (small, non-LFS file)
            except urllib.error.HTTPError as e:
                if e.code in (301, 302, 303, 307, 308):
                    loc = e.headers["Location"]
                    return loc if loc.startswith("http") else hub + loc
                if e.code == 429:
                    continue
        wait = min(120, 5 * (attempt + 1))
        print(f"[{time.strftime('%H:%M:%S')}] {fname}: HTTP 429 on all hubs, retrying in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"{fname}: still rate limited after 30 attempts")


def fetch_range(url_box, lock, repo, sha, fname, path, start, end):
    for attempt in range(20):
        url = url_box[0]
        req = urllib.request.Request(
            url,
            headers={"Range": f"bytes={start}-{end}", "User-Agent": "Mozilla/5.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            if len(data) != end - start + 1:
                raise IOError(f"short read {len(data)} != {end - start + 1}")
            fd = os.open(path, os.O_WRONLY)
            try:
                os.pwrite(fd, data, start)
            finally:
                os.close(fd)
            return len(data)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403, 410):  # signed CDN URL expired
                with lock:
                    if url_box[0] == url:
                        url_box[0] = resolve(repo, sha, fname)
            time.sleep(min(2**attempt, 30))
        except Exception:
            time.sleep(min(2**attempt, 30))
    raise RuntimeError(f"range {start}-{end} of {fname} failed")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(64 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def download(repo, local_dir, workers, revision, repo_type="model", include=None, chunk_mb=32, staging_dir=None):
    chunk = chunk_mb * 1024 * 1024
    sha, files = api_files(repo, revision, repo_type)
    if include:
        files = [f for f in files if any(fnmatch.fnmatch(f[0], p) for p in include)]
    if repo_type == "dataset":
        repo = f"datasets/{repo}"
    os.makedirs(local_dir, exist_ok=True)
    pool = ThreadPoolExecutor(max_workers=workers)
    t0, done_bytes = time.time(), 0
    for fname, size, lfs_sha in sorted(files, key=lambda x: x[1] or 0):
        dest = os.path.join(local_dir, fname)
        if os.path.exists(dest) and (size is None or os.path.getsize(dest) == size):
            if lfs_sha is None or os.path.exists(dest + ".sha256ok"):
                print(f"[{time.strftime('%H:%M:%S')}] {fname} already complete, skipping", flush=True)
                continue
        os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
        if staging_dir:
            os.makedirs(staging_dir, exist_ok=True)
            part = os.path.join(staging_dir, fname.replace("/", "__") + ".part")
        else:
            part = dest + ".part"
        url_box, lock = [resolve(repo, sha, fname)], threading.Lock()
        print(f"[{time.strftime('%H:%M:%S')}] starting {fname} ({(size or 0) / 1e9:.2f} GB)...", flush=True)
        if not size:
            with urllib.request.urlopen(url_box[0], timeout=TIMEOUT) as r, open(part, "wb") as f:
                f.write(r.read())
        else:
            with open(part, "wb") as f:
                f.truncate(size)
            futs = [
                pool.submit(fetch_range, url_box, lock, repo, sha, fname, part, s, min(s + chunk, size) - 1)
                for s in range(0, size, chunk)
            ]
            completed_chunks = 0
            total_chunks = len(futs)
            for fut in as_completed(futs):
                done_bytes += fut.result()
                completed_chunks += 1
                if completed_chunks % 32 == 0 or completed_chunks == total_chunks:
                    rate = done_bytes / max(time.time() - t0, 1e-6) / 1e6
                    print(
                        f"[{time.strftime('%H:%M:%S')}] {fname}: {completed_chunks}/{total_chunks} chunks "
                        f"(avg {rate:.1f} MB/s)",
                        flush=True,
                    )
        if lfs_sha is not None:
            got = sha256_file(part)
            if got != lfs_sha:
                raise RuntimeError(f"sha256 mismatch for {fname}: {got} != {lfs_sha}")
        if staging_dir:
            shutil.copyfile(part, dest)
            os.remove(part)
        else:
            os.replace(part, dest)
        if lfs_sha is not None:
            open(dest + ".sha256ok", "w").write(lfs_sha + "\n")
        rate = done_bytes / max(time.time() - t0, 1e-6) / 1e6
        print(f"[{time.strftime('%H:%M:%S')}] {fname} ok ({(size or 0) / 1e9:.2f} GB, avg {rate:.1f} MB/s)", flush=True)
    open(os.path.join(local_dir, ".download_complete"), "w").write(f"{repo}@{sha}\n")
    print(f"done {repo}@{sha} in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("local_dir")
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--revision", default="main")
    ap.add_argument("--repo-type", choices=["model", "dataset"], default="model")
    ap.add_argument("--include", nargs="*", help="fnmatch patterns of repo files to fetch")
    ap.add_argument("--chunk-mb", type=int, default=32)
    ap.add_argument("--staging-dir", help="download each file here first, then copy it to local_dir (for HDFS mounts)")
    a = ap.parse_args()
    download(a.repo, a.local_dir, a.workers, a.revision, a.repo_type, a.include, a.chunk_mb, a.staging_dir)
    sys.exit(0)
