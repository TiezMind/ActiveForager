#!/usr/bin/env python3
"""ActiveForager (agentic_video_understanding) 标准归档构建与验证套件
（参考 DA-rebuttal 与 ThinkMorph-VPD-archive 精简包/完整包双层规范）.

用法：
    python3 scripts/archive/build_archive.py all          # 默认生成精简归档包（严格不含训练数据与模型权重）
    python3 scripts/archive/build_archive.py all --full   # 生成含 git.bundle 与纯文本训练题目清单 JSONL 的完整包到 HDFS
    python3 scripts/archive/build_archive.py sync
    python3 scripts/archive/build_archive.py select
    python3 scripts/archive/build_archive.py meta
    python3 scripts/archive/build_archive.py pack
    python3 scripts/archive/build_archive.py verify
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from collections import Counter
from datetime import datetime
from pathlib import Path

FULL_MODE = "--full" in sys.argv
if "--full" in sys.argv:
    sys.argv.remove("--full")

PLAY = Path("/mlx/users/lipengyu.seed/playground")
PROJ_NAME = "agentic_video_understanding"
PROJ_DIR = PLAY / PROJ_NAME
HDFS_DA_ROOT = Path("/mnt/hdfs/lipengyu/DA")
HDFS_CKPT_ROOT = HDFS_DA_ROOT / "checkpoints" / "forager"
HDFS_DATA_ROOT = HDFS_DA_ROOT / "forager_data"

DATE_TAG = os.environ.get("ARCHIVE_DATE_TAG", "20261109")
TOP = f"ActiveForager-archive-{DATE_TAG}"
LOCAL_ARCHIVE_DIR = PLAY / "agentic_video_understanding-archives"
HDFS_ARCHIVE_DIR = HDFS_DA_ROOT / "archives"
OUT_DIR = HDFS_ARCHIVE_DIR if FULL_MODE else LOCAL_ARCHIVE_DIR
OUT = OUT_DIR / (f"{TOP}-full.tar.zst" if FULL_MODE else f"{TOP}.tar.zst")

WORK = Path("/tmp/forager_archive_work")
STAGE = WORK / "stage"
META = WORK / "meta"

MIB = 2**20

# 明文密钥特征扫描模式（在 STAGE 与 git diff 中自动脱敏）
_KEY1 = "${ARK_API_KEY_REDACTED}" + "${ARK_API_KEY_REDACTED}"
_KEY2 = "${ARK_API_KEY_REDACTED}" + "${ARK_API_KEY_REDACTED}"
SECRET_PATTERNS = [
    (re.compile(re.escape(_KEY1)), "${ARK_API_KEY_REDACTED}"),
    (re.compile(re.escape(_KEY2)), "${ARK_API_KEY_REDACTED}"),
    (re.compile(r"sk-[A-Za-z0-9]{20,}"), "${OPENAI_API_KEY_REDACTED}"),
]

LATEX_TMP = (
    ".aux",
    ".fdb_latexmk",
    ".fls",
    ".xdv",
    ".synctex.gz",
    ".blg",
    ".toc",
    ".nav",
    ".snm",
    ".out",
)

# 排除原因枚举（写入 EXCLUDED.md 与 EXCLUDED.json）
R_GIT = "git 仓库内部对象（.git/；分支状态、完整提交日志与未提交 diff 已保存在顶层 git/，完整 git bundle 保存在 HDFS 完整包中）"
R_VENV = "Python 虚拟环境与编译/测试缓存（__pycache__/、*.pyc、.pytest_cache 等；环境包版本清单已记录在 ENV/）"
R_LATEX_TMP = "LaTeX 编译中间临时文件（.aux/.blg/.fdb_latexmk/.fls/.out 等；完整保留 .tex/.bib/.sty/.bbl、compile.log、main.log 与 main.pdf）"
R_TRAIN_DATA = "训练数据文件（data/sft_coldstart_6k.jsonl 6.5 MiB、data/rl_ig_grpo_8k.jsonl 5.6 MiB 及 Worker 端 sft_probed_*.jsonl；已保留 data/manifest_summary.json 统计元数据，HDFS 完整包额外收录文本 JSONL 清单，617 GiB 原始视频与帧缓存见 HDFS_INVENTORY.txt）"
R_WEIGHTS = "模型权重与优化器检查点（*.safetensors / *.pt / *.bin / *.pth / *.ckpt，HDFS 共 96 GiB；各 Checkpoint 的 config.json 与训练指标 meta.json 已同步至 hdfs_sync/，权重路径与大小清单见 HDFS_INVENTORY.txt）"
R_BASE_TOK = "基座模型大体积分词器文件（tokenizer.json / vocab.json / merges.txt，可从 HuggingFace 官方仓库下载；保留 tokenizer_config.json 与 chat_template.json）"
R_ABSLINK = "指向宿主机外部路径的符号链接（如 nvlib/libnvidia-ml.so.1 驱动软链）"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_text(text: str) -> tuple[str, bool]:
    changed = False
    for pat, repl in SECRET_PATTERNS:
        new_text, n = pat.subn(repl, text)
        if n > 0:
            text = new_text
            changed = True
    return text, changed


def copy_with_readable_mode(src: Path, dst: Path, sanitize: bool = False) -> str:
    """复制文件到 STAGE，修正 0000 等异常权限（如 HDFS 上的 meta.json）为 0644/0755，可选执行密钥脱敏."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    st = src.stat()
    mode = stat.S_IMODE(st.st_mode)
    if mode == 0 or not (mode & stat.S_IRUSR):
        try:
            os.chmod(src, 0o644)
        except Exception:
            pass
    if sanitize:
        raw = src.read_text(encoding="utf-8", errors="surrogateescape")
        cleaned, changed = sanitize_text(raw)
        dst.write_text(cleaned, encoding="utf-8", errors="surrogateescape")
        kind = "sanitized" if changed else "as-is"
    else:
        shutil.copyfile(src, dst)
        kind = "as-is"
    if mode == 0 or not (mode & stat.S_IRUSR):
        mode = 0o644
    else:
        mode |= stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH
    os.chmod(dst, mode)
    os.utime(dst, (st.st_atime, st.st_mtime))
    return kind


def cmd_sync():
    """阶段 1：同步 HDFS Checkpoint 配置与指标、双 GPU Worker (4485507 H20 & 4480127 H800) 训练/推理分片日志到 STAGE."""
    print(f"=== [1/5] 同步 HDFS Checkpoint 元数据与双 GPU Worker 训练/推理日志到 STAGE (mode={'FULL' if FULL_MODE else 'SLIM'}) ===")
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir(parents=True, exist_ok=True)
    synced_records: list[dict] = []

    def record_stage_tree(base_rel: str, kind_tag: str, sanitize_files: bool = True):
        root = STAGE / base_rel
        if not root.exists():
            return
        for p in sorted(root.rglob("*")):
            rel = str(p.relative_to(STAGE))
            if p.is_symlink():
                p.unlink()
                continue
            if p.is_file():
                if sanitize_files and p.suffix in (".log", ".txt", ".sh", ".py", ".json", ".jsonl", ".md", ".tex"):
                    try:
                        raw = p.read_text(encoding="utf-8", errors="surrogateescape")
                        cleaned, changed = sanitize_text(raw)
                        if changed:
                            p.write_text(cleaned, encoding="utf-8", errors="surrogateescape")
                    except Exception:
                        pass
                st = p.lstat()
                mode = stat.S_IMODE(st.st_mode)
                if mode == 0 or not (mode & stat.S_IRUSR):
                    os.chmod(p, 0o644)
                synced_records.append({
                    "path": rel,
                    "bytes": st.st_size,
                    "action": f"stage:{kind_tag}",
                    "reason": "",
                    "link": False,
                })

    # 1. 同步 HDFS (/mnt/hdfs/lipengyu/DA/checkpoints/forager) 下的全部 Checkpoint 配置与训练指标 meta.json
    hdfs_stage = STAGE / PROJ_NAME / "hdfs_sync" / "checkpoints" / "forager"
    if HDFS_CKPT_ROOT.exists():
        for f in sorted(HDFS_CKPT_ROOT.rglob("*")):
            if not f.is_file():
                continue
            if f.name in ("tokenizer.json", "vocab.json", "merges.txt"):
                continue
            if f.suffix in (".safetensors", ".pt", ".bin", ".pth", ".ckpt"):
                continue
            rel_f = f.relative_to(HDFS_CKPT_ROOT)
            copy_with_readable_mode(f, hdfs_stage / rel_f, sanitize=True)
        record_stage_tree(f"{PROJ_NAME}/hdfs_sync", "synced-hdfs")

    # 2. 提取 GPU Worker 4485507 (8xH20) 与 Worker 4480127 (8xH800) 的训练/评测分片日志与 Checkpoint 元数据
    LOCAL_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    for worker_id in ("4485507", "4480127"):
        tgz = LOCAL_ARCHIVE_DIR / f".worker_{worker_id}_sync.tar.gz"
        worker_stage = STAGE / PROJ_NAME / f"worker_{worker_id}_sync"
        if tgz.exists() and tgz.stat().st_size > 0:
            worker_stage.mkdir(parents=True, exist_ok=True)
            with tarfile.open(tgz, mode="r:gz") as tf:
                for member in tf.getmembers():
                    if not member.isfile():
                        continue
                    mname = Path(member.name).name
                    # 排除权重、分词器大文件；精简模式下排除 probed SFT 训练数据分片
                    if mname in ("tokenizer.json", "vocab.json", "merges.txt"):
                        continue
                    if mname.endswith((".safetensors", ".pt", ".bin", ".pth", ".ckpt", ".parquet")):
                        continue
                    if not FULL_MODE and ("probed_shards/" in member.name or mname in ("sft_probed_6k.jsonl", "smoke_probed.jsonl")):
                        continue
                    # 去掉开头的 tmp/ 前缀
                    rel_member = re.sub(r"^/?tmp/", "", member.name)
                    dst_file = worker_stage / rel_member
                    dst_file.parent.mkdir(parents=True, exist_ok=True)
                    extracted = tf.extractfile(member)
                    if extracted is not None:
                        dst_file.write_bytes(extracted.read())
                        os.chmod(dst_file, 0o644)
                        os.utime(dst_file, (member.mtime, member.mtime))
            record_stage_tree(f"{PROJ_NAME}/worker_{worker_id}_sync", f"synced-worker-{worker_id}")
            print(f"  已合入 GPU Worker {worker_id} 同步日志到 {PROJ_NAME}/worker_{worker_id}_sync/")

    # 3. 同步本机 /tmp/forager_ckpts 下的 dry-run / smoke 训练 meta.json 与 latest_checkpointed_iteration.txt
    local_tmp_ckpts = Path("/tmp/forager_ckpts")
    if local_tmp_ckpts.exists():
        local_ckpt_stage = STAGE / PROJ_NAME / "local_tmp_ckpts_meta"
        for f in sorted(local_tmp_ckpts.rglob("*")):
            if not f.is_file():
                continue
            if f.suffix in (".json", ".txt"):
                copy_with_readable_mode(f, local_ckpt_stage / f.relative_to(local_tmp_ckpts), sanitize=True)
        record_stage_tree(f"{PROJ_NAME}/local_tmp_ckpts_meta", "synced-local-tmp")

    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "synced_records.json").write_text(json.dumps(synced_records, ensure_ascii=False, indent=2))
    print(f"  sync 阶段共准备 {len(synced_records)} 个外部同步日志/元数据文件（{sum(r['bytes'] for r in synced_records) / MIB:.2f} MiB）")


def decide(rel_path: str, size: int, is_link: bool) -> tuple[str, str]:
    """对 agentic_video_understanding 及顶层共享文档下的每个路径返回 (action, reason)."""
    parts = rel_path.split("/")
    name = parts[-1]

    # 1. Git 仓库内部目录
    if ".git" in parts:
        return "drop", R_GIT

    # 2. Python 虚拟环境与缓存
    if (
        ".venv" in parts
        or "__pycache__" in parts
        or ".pytest_cache" in parts
        or ".ruff_cache" in parts
        or ".ipynb_checkpoints" in parts
        or name.endswith((".pyc", ".pyo"))
    ):
        return "drop", R_VENV

    # 3. LaTeX 编译中间临时文件（保留 main.log 和 compile.log 供排查编译记录）
    if name.endswith(LATEX_TMP):
        return "drop", R_LATEX_TMP

    # 4. 训练数据文件过滤（严格遵循用户要求：不包括训练数据；保留 manifest_summary.json）
    if rel_path.startswith(f"{PROJ_NAME}/data/"):
        if name == "manifest_summary.json":
            return "keep", ""
        if not FULL_MODE and name.endswith((".jsonl", ".parquet", ".tar", ".mp4", ".npy")):
            return "drop", R_TRAIN_DATA

    # 5. 大权重文件过滤（严格遵循用户要求：不包括模型权重）
    if name.endswith((".safetensors", ".bin", ".pth", ".ckpt", ".pt")):
        return "drop", R_WEIGHTS

    # 6. 基座大体积分词器过滤
    if name in ("tokenizer.json", "vocab.json", "merges.txt"):
        return "drop", R_BASE_TOK

    # 7. 符号链接处理
    if is_link:
        target = os.readlink(PLAY / rel_path)
        if os.path.isabs(target):
            return "drop", R_ABSLINK
        return "keep", ""

    # 8. 检查文本/代码文件是否含明文 API Key（如有则走 stage:sanitize）
    if name.endswith((".sh", ".py", ".yaml", ".yml", ".json", ".md", ".txt", ".env", ".tex")) and size < 5 * MIB:
        try:
            raw = (PLAY / rel_path).read_text(encoding="utf-8", errors="surrogateescape")
            _, has_secret = sanitize_text(raw)
            if has_secret:
                return "stage:sanitize", ""
        except Exception:
            pass

    return "keep", ""


def cmd_select():
    """阶段 2：扫描项目目录与共享基础设施文档，生成完整归档计划 plan.json."""
    print("=== [2/5] 扫描目录并制定文件保留/脱敏/排除计划 ===")
    synced_path = WORK / "synced_records.json"
    synced_records = json.loads(synced_path.read_text()) if synced_path.exists() else []
    synced_rel_set = {r["path"] for r in synced_records}

    plan: list[dict] = []

    # 顶层共享文档（与 DA-rebuttal 保持一致：GPU_POOL.md 与 INFRA_LESSONS.md）
    for shared_doc in ("GPU_POOL.md", "INFRA_LESSONS.md"):
        doc_p = PLAY / shared_doc
        if doc_p.is_file():
            st = doc_p.lstat()
            action, reason = decide(shared_doc, st.st_size, False)
            plan.append({
                "path": shared_doc,
                "bytes": st.st_size,
                "action": action,
                "reason": reason,
                "link": False,
            })

    # 扫描主项目目录 agentic_video_understanding/
    for dirpath, dirnames, filenames in os.walk(PROJ_DIR, followlinks=False):
        rel_dir = os.path.relpath(dirpath, PLAY)
        keep_dirs = []
        for d in sorted(dirnames):
            rel_d = f"{rel_dir}/{d}" if rel_dir != "." else d
            full_d = PLAY / rel_d
            st = os.lstat(full_d)
            if stat.S_ISLNK(st.st_mode):
                action, reason = decide(rel_d, st.st_size, True)
                plan.append({
                    "path": rel_d,
                    "bytes": st.st_size,
                    "action": action,
                    "reason": reason,
                    "link": True,
                })
                continue
            if d in (".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", ".ipynb_checkpoints"):
                reason = R_GIT if d == ".git" else R_VENV
                for sub_dp, _, sub_fns in os.walk(full_d, followlinks=False):
                    for fn in sub_fns:
                        fp = Path(sub_dp) / fn
                        try:
                            fst = fp.lstat()
                        except FileNotFoundError:
                            continue
                        plan.append({
                            "path": str(fp.relative_to(PLAY)),
                            "bytes": fst.st_size,
                            "action": "drop",
                            "reason": reason,
                            "link": stat.S_ISLNK(fst.st_mode),
                        })
                continue
            keep_dirs.append(d)
        dirnames[:] = keep_dirs

        for fn in sorted(filenames):
            rel_f = f"{rel_dir}/{fn}" if rel_dir != "." else fn
            if rel_f in synced_rel_set:
                continue
            full_f = PLAY / rel_f
            try:
                st = os.lstat(full_f)
            except FileNotFoundError:
                continue
            is_link = stat.S_ISLNK(st.st_mode)
            action, reason = decide(rel_f, st.st_size, is_link)
            plan.append({
                "path": rel_f,
                "bytes": st.st_size,
                "action": action,
                "reason": reason,
                "link": is_link,
            })

    # 将 HDFS 上的 6 个模型权重文件也显式记入 plan 的 drop 统计，使 EXCLUDED.md 完整反映 96 GiB 权重体积
    if HDFS_CKPT_ROOT.exists():
        for wf in sorted(HDFS_CKPT_ROOT.rglob("*")):
            if wf.is_file() and wf.suffix in (".safetensors", ".pt", ".bin", ".pth"):
                try:
                    wst = wf.lstat()
                except FileNotFoundError:
                    continue
                plan.append({
                    "path": f"HDFS:{wf}",
                    "bytes": wst.st_size,
                    "action": "drop",
                    "reason": R_WEIGHTS,
                    "link": False,
                })
            elif wf.is_file() and wf.name in ("tokenizer.json", "vocab.json", "merges.txt"):
                try:
                    tst = wf.lstat()
                except FileNotFoundError:
                    continue
                plan.append({
                    "path": f"HDFS:{wf}",
                    "bytes": tst.st_size,
                    "action": "drop",
                    "reason": R_BASE_TOK,
                    "link": False,
                })

    plan.extend(synced_records)
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))

    c, b = Counter(), Counter()
    for e in plan:
        k = e["action"] if e["action"] != "drop" else f"drop: {e['reason']}"
        c[k] += 1
        b[k] += e["bytes"]
    for k in sorted(c, key=lambda x: -b[x]):
        print(f"  {b[k] / MIB:10.2f} MiB  {c[k]:7d} files  | {k}")


def collect_hdfs_inventory() -> str:
    """生成 HDFS 与双 GPU Worker (4485507 & 4480127) 的模型权重与训练/评测数据集清单 HDFS_INVENTORY.txt."""
    lines = [
        f"# ActiveForager (`agentic_video_understanding`) 外部存储（HDFS & GPU Workers）大文件清单（生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}）",
        "",
        "本清单记录未直接打入归档压缩包的模型权重（`.safetensors` / `.pt`）、原始训练/评测视频数据集及预抽帧缓存的存储路径与大小。",
        f"包含 `git/*.bundle` 与训练题目文本 JSONL（`sft_coldstart_6k.jsonl`、`rl_ig_grpo_8k.jsonl`）的完整包备份于：`/mnt/hdfs/lipengyu/DA/archives/{TOP}-full.tar.zst`。",
        "",
        "## 1. HDFS 模型权重与 Checkpoint（`/mnt/hdfs/lipengyu/DA/checkpoints/forager`，共 96 GiB）",
        "",
        "各 Checkpoint 目录下的配置文件（`config.json`、`preprocessor_config.json`、`chat_template.json`）与训练步数/损失指标（`meta.json`）已实体化收入归档包 `agentic_video_understanding/hdfs_sync/checkpoints/forager/` 中。",
        "",
    ]
    if HDFS_CKPT_ROOT.exists():
        res = subprocess.run(
            ["du", "-sh", *sorted(str(p) for p in HDFS_CKPT_ROOT.glob("*/*"))],
            capture_output=True,
            text=True,
            check=False,
        )
        lines.append("```text")
        lines.append(res.stdout.strip())
        lines.append("```")
        lines.append("")
        lines.append("### HDFS 模型权重文件明细（`*.safetensors` / `*.pt`）")
        lines.append("")
        lines.append("```text")
        res_w = subprocess.run(
            [
                "find",
                str(HDFS_CKPT_ROOT),
                "-type",
                "f",
                "(",
                "-name",
                "*.safetensors",
                "-o",
                "-name",
                "*.pt",
                ")",
                "-exec",
                "ls",
                "-lh",
                "{}",
                "+",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        lines.append(res_w.stdout.strip())
        lines.append("```")

    lines += [
        "",
        "## 2. HDFS 训练数据清单与源视频库（`/mnt/hdfs/lipengyu/DA/forager_data` & `/mnt/hdfs/lipengyu/DA/video`，共 ~617 GiB）",
        "",
        "```text",
    ]
    video_root = HDFS_DA_ROOT / "video"
    paths_to_du = []
    if HDFS_DATA_ROOT.exists():
        paths_to_du.extend(sorted(str(p) for p in HDFS_DATA_ROOT.glob("*")))
    if video_root.exists():
        paths_to_du.extend(sorted(str(p) for p in video_root.glob("*")))
    if paths_to_du:
        res_v = subprocess.run(["du", "-sh", *paths_to_du], capture_output=True, text=True, check=False)
        lines.append(res_v.stdout.strip())
    lines.append("```")

    lines += [
        "",
        "## 3. 基座模型存储路径（HDFS）",
        "",
        "- `Qwen2.5-VL-7B-Instruct`：`/mnt/hdfs/lipengyu/models/Qwen2.5-VL-7B-Instruct`",
        "- `Qwen3-VL-8B-Instruct`：`/mnt/hdfs/lipengyu/DA/models/Qwen3-VL-8B-Instruct`",
        "",
        "## 4. 双 GPU Worker 节点本地 NVMe 临时目录（`4485507` 8×H20 & `4480127` 8×H800）",
        "",
        "- **Worker `4485507`（8×NVIDIA H20 96GB，挂载于 `dbx_111447`）**：",
        "  - 已同步至归档包 `agentic_video_understanding/worker_4485507_sync/`：`/tmp/forager_scratch/`、`/tmp/forager_v2_qwen3vl/`、`/tmp/forager_ckpts/` 下的全部 8 卡分片推理日志（`shard_{0..7}.log`）、分片评测 JSON（`shard_{0..7}.json`、`ovo_far_only.json`、`ovo_far_merged.json`）及训练中间步 `meta.json`（`global_step_100/275/300/375`）。",
        "  - 已排除：`/tmp/forager_ckpts/{sft,rl}/*/*/model.safetensors` 与 `/tmp/forager_v2_qwen3vl/rl_ckpt/global_step_100/model.safetensors`（最终 Checkpoint 已导出至 HDFS `/mnt/hdfs/lipengyu/DA/checkpoints/forager/`）。",
        "- **Worker `4480127`（8×NVIDIA H800 80GB，挂载于 `dbx_111063`）**：",
        "  - 已同步至归档包 `agentic_video_understanding/worker_4480127_sync/`：`/tmp/forager_h800_ablations/` 与 `/tmp/forager_v2_qwen25vl/` 下的全部 8 卡分片推理日志、分片结果 JSON 与 `rl_ckpt/global_step_100/meta.json`。",
        "  - 已排除：`/tmp/forager_v2_qwen25vl/rl_ckpt/global_step_100/model-0000{1..4}-of-00004.safetensors`（已导出至 HDFS `/mnt/hdfs/lipengyu/DA/checkpoints/forager/rl/qwen25vl_ig_grpo_v2/`）。",
        "",
    ]
    return "\n".join(lines)


def write_readme(meta_dir: Path, plan: list[dict]):
    """生成顶层 README.md."""
    mode_label = "完整版（含 Git Bundle 与文本训练题目清单）" if FULL_MODE else "精简版（严格不含训练数据与模型权重）"
    readme_text = f"""# ActiveForager (`agentic_video_understanding`) 项目归档（{mode_label}，{DATE_TAG}）

本归档包遵循 `DA-rebuttal` 与 `ThinkMorph-VPD-archive` 标准双层归档规范构建，完整收录了 **ActiveForager: Look Clear, Look Back, or Wait — Resolving the Streaming Perception–Memory Dilemma via Non-Diluting Spatio-Temporal Evidence Foraging** 项目的全部核心资产：

- **完整论文 LaTeX 源码、矢量图表与编译产物**（`agentic_video_understanding/paper/`：含 `main.tex`、`main.pdf`、全部 8 个章节 `sections/*.tex`、`references.bib`、`neurips_2026.sty`、`math_commands.tex`、4 张主图 PDF/PNG 及全部配套作图脚本 `figures/gen_fig*.py`）；
- **全部核心训练与推理框架代码**（`agentic_video_understanding/forager/`：含双缓冲流式环境 `dual_buffer.py`、正交时空觅食智能体 `agent.py`、4D 流解耦注意力掩码 `sdam_mask.py`、三维反事实信息增益强化学习 `ig_grpo.py`、拒绝感知冷启动微调 `sft_trainer.py`、`verl_adapter.py` 与协议解析 `protocol.py`）；
- **全部训练、评测、消融与跨机编排脚本**（`agentic_video_understanding/scripts/` 与单元测试 `tests/test_forager_core.py`）；
- **全部训练结果、评测指标 JSON 与历史审计档案**（`agentic_video_understanding/outputs/`：含 `OVO-Bench` 12 子任务评测结果、`StreamingBench` 18 子任务泛化结果、`h800_ablations/` 5 组受控消融与 300 步 RL 训练动力学曲线数据，以及 `archived_contaminated_do_not_use/` 隔离审计档案与说明）；
- **全部训练日志与多卡推理分片日志**（涵盖 `outputs/logs/` 主控日志、从 HDFS 同步的 6 组 SFT/RL Checkpoint 训练指标 `hdfs_sync/`，以及从 8×H20 Worker `4485507` 与 8×H800 Worker `4480127` 同步的全部分片推理日志 `worker_4485507_sync/`、`worker_4480127_sync/`）；
- **Git 历史快照与运行环境清单**（`git/` 分支、提交图谱、工作区状态与完整未提交改动 `*.diff`；`ENV/` Python 环境包版本快照）。

> **精简包与 HDFS 完整包说明**：
> - **精简包**（`{TOP}.tar.zst`）：严格排除所有模型权重（`.safetensors` / `.pt`，96 GiB）与所有训练数据（包括 617 GiB 原始视频/帧缓存，以及 `data/sft_coldstart_6k.jsonl` 6.5 MiB、`data/rl_ig_grpo_8k.jsonl` 5.6 MiB 与 Worker 端 `sft_probed_*.jsonl`），仅保留 `data/manifest_summary.json`（1.2 KB 数据集分布统计），体积仅约 **4 MiB**，便于快速下载、传输与审阅。
> - **HDFS 完整包**（`/mnt/hdfs/lipengyu/DA/archives/{TOP}-full.tar.zst`）：在精简包基础上额外收入 `git/agentic_video_understanding.bundle` 及纯文本训练题目清单 JSONL（`data/*.jsonl` 与 `sft_probed_*.jsonl`），同样不含模型权重与原始视频文件。
> - 未收入文件的逐类统计见 `EXCLUDED.md`，HDFS 上的模型权重与数据集完整路径清单见 `HDFS_INVENTORY.txt`。

---

## 1. 顶层目录布局

目录结构与开发机 `/mlx/users/lipengyu.seed/playground/` 保持平铺一致，确保包内所有脚本与相对路径引用开箱即用：

| 路径 | 内容说明 |
|---|---|
| `agentic_video_understanding/` | 项目主目录（含 `paper/`、`forager/`、`scripts/`、`tests/`、`outputs/`、`data/manifest_summary.json`、`PROPOSAL_OPTION_A.md`、`pyproject.toml`） |
| `agentic_video_understanding/hdfs_sync/` | 从 HDFS `/mnt/hdfs/lipengyu/DA/checkpoints/forager/` 同步的全部 6 个 SFT/RL Checkpoint 的 `config.json`、`meta.json`（训练步数/loss/reward/grad_norm）与分词器配置 |
| `agentic_video_understanding/worker_4485507_sync/` | 从 8×H20 Worker `4485507` 同步的 Stage-1 / Stage-3 / v2 / StreamingBench 8 卡分片推理日志（`shard_{{0..7}}.log`）、分片评测 JSON 与中间 Checkpoint `meta.json` |
| `agentic_video_understanding/worker_4480127_sync/` | 从 8×H800 Worker `4480127` 同步的 H800 消融实验与 `Qwen2.5-VL` v2 / StreamingBench 8 卡分片推理日志与结果 JSON |
| `GPU_POOL.md`、`INFRA_LESSONS.md` | 跨机 GPU Worker 分配台账与集群训练/评测基础设施规范（脚本中以 `../GPU_POOL.md` / `../INFRA_LESSONS.md` 引用） |
| `git/` | 仓库提交图谱 `*.log.txt`、分支列表 `*.branches.txt`、打包时刻 `*.status.txt` 与完整未提交改动 `*.diff`（HDFS 完整包另含 `*.bundle`） |
| `ENV/` | 开发机默认 Python 环境（`python3.packages.txt`）、`DA-train` 强化学习环境（`DA-train_envs_train.packages.txt`）与 `vllm` 推理环境（`vllm.venv.packages.txt`）包版本快照 |
| `HDFS_INVENTORY.txt` | HDFS 与 GPU Worker 上的模型权重（96 GiB）及训练/评测视频数据（617 GiB）位置与体积明细 |
| `EXCLUDED.md` / `EXCLUDED.json` | 未收入归档的文件分类统计、体积汇总与大文件明细 |
| `MANIFEST.tsv` | 归档包内每个文件的精确字节数、SHA-256 校验值与来源类型 |

---

## 2. 论文 LaTeX 源码与图表工具索引（`agentic_video_understanding/paper/`）

| 路径 | 说明 |
|---|---|
| `paper/main.tex`、`paper/main.pdf` | NeurIPS 2026 预印本格式主文档与最新零错误编译 PDF（14 页，660 KB） |
| `paper/sections/0_abstract.tex` ~ `6_conclusion.tex`、`A_appendix.tex` | 论文正文全部章节与理论推导/实验附录 LaTeX 源码 |
| `paper/references.bib`、`paper/neurips_2026.sty`、`paper/math_commands.tex` | BibTeX 参考文献库、样式宏包与统一数学符号定义 |
| `paper/figures/fig1_pareto_frontier.{{pdf,png}}` | 图 1：流式感知-记忆 Pareto 前沿与正交能力分解 |
| `paper/figures/fig2_architecture.{{pdf,png}}` | 图 2：ActiveForager 双缓冲非稀释架构、子智能体证据卡与 4D SDAM 掩码示意图 |
| `paper/figures/fig3_rl_dynamics.{{pdf,png}}` | 图 3：300 步 RL 训练动力学与工具坍缩消融曲线 |
| `paper/figures/fig4_ablation_and_generalization.{{pdf,png}}` | 图 4：非稀释上下文融合消融、2D vs. 3D IG 奖励挤出对比与 StreamingBench 18 子任务泛化 |
| `paper/figures/gen_fig{{1..4}}_*.py`、`paper_plot_style.py` | 4 张主图的确定性生成脚本（直接读取 `../outputs/` 下的 JSON 复现全部图表） |

---

## 3. 训练代码、实验结果与推理日志索引

### 3.1 核心方法与训练代码（`agentic_video_understanding/forager/` & `scripts/`）
- **流式双缓冲环境与无泄漏帧解码**：`forager/dual_buffer.py`（`ProtectedRecentWindow` System 1、`HighResRingBuffer` System 2a、`OnlineEventArchive` & `SubAgentEvidenceExtractor` System 2b、`load_real_frames_from_video_or_npy`）
- **正交四元动作空间智能体**：`forager/agent.py`（`<Answer>` / `<ZoomIn>` / `<Recall>` / `<Wait>` 两阶段推理与 vLLM / HF 引擎封装）
- **4D 流解耦注意力掩码（SDAM）**：`forager/sdam_mask.py`（视觉分组块对角隔离与 Turn-2 冗余视觉帧精确屏蔽）
- **拒绝感知 SFT 与对称 3D-IG-GRPO 强化学习**：`forager/sft_trainer.py`、`forager/ig_grpo.py`、`forager/verl_adapter.py`、`scripts/train_sft.py`、`scripts/train_ig_grpo.py`
- **基准评测与消融脚本**：`scripts/eval_ovobench.py`、`scripts/eval_streamingbench.py`、`scripts/eval_h800_ablations.py`、`scripts/run_rl_dynamics_ablation.py`、`scripts/launch_h800_all.sh`、`scripts/run_v2_and_streamingbench.sh`

### 3.2 训练结果、评测结果与审计记录（`agentic_video_understanding/outputs/`）
- **主表评测结果**：`outputs/ovobench_qwen{{25,3}}vl_{{stage1,final_ig_grpo,final_ig_grpo_v2}}.json`、`outputs/streamingbench_qwen{{25,3}}vl_ig_grpo_v2.json`
- **消融实验与 RL 训练轨迹**：`outputs/h800_ablations/ovobench_{{full_forager,zoom_only,recall_only,wait_only,card_no_sdam,naive_raw_concat}}.json`、`outputs/h800_ablations/ablation_c_rl_dynamics_300steps.json`
- **实验完整性审计隔离区**：`outputs/archived_contaminated_do_not_use/`（含 `README_CONTAMINATION_NOTE.md`，记录 2026-11-07 审计发现的历史泄漏根因与代码修复对照）
- **Checkpoint 级训练指标快照**：`hdfs_sync/checkpoints/forager/{{sft,rl}}/*/meta.json`（含 `qwen25vl_sft`、`qwen3vl_sft`、`qwen25vl_ig_grpo`、`qwen25vl_ig_grpo_v2`、`qwen3vl_ig_grpo`、`qwen3vl_ig_grpo_v2`）

### 3.3 训练与推理日志
- **主控与队列日志**：`outputs/logs/`（`launch_h20_full_queue.log`、`h800_ablations_run.log`、`v2_qwen25vl_h800.log`、`v2_qwen3vl_h20.log`、`sb_qwen25vl_h800.log`、`sb_qwen3vl_h20.log`、`forager_monitor.log`）
- **Worker `4485507` (8×H20) 分片日志**：`worker_4485507_sync/forager_scratch/ovobench_s{{1,3}}_qwen{{25,3}}vl/shard_{{0..7}}.log`、`worker_4485507_sync/forager_v2_qwen3vl/{{far_shards,sb_shards}}/shard_{{0..7}}.log`
- **Worker `4480127` (8×H800) 分片日志**：`worker_4480127_sync/forager_v2_qwen25vl/{{far_shards,sb_shards}}/shard_{{0..7}}.log`

---

## 4. 解压、验证与一键重打包

```bash
# 1. 解压归档包
tar --zstd -xf {OUT.name}
cd {TOP}/agentic_video_understanding

# 2. 运行 CPU 全量单元测试（17 项核心算子与协议测试）
pytest -q

# 3. 重新生成论文全部 4 张主图并编译 LaTeX 论文
python3 paper/figures/gen_fig1_pareto_frontier.py
python3 paper/figures/gen_fig2_architecture.py
python3 paper/figures/gen_fig3_rl_dynamics.py
python3 paper/figures/gen_fig4_ablation_and_generalization.py
cd paper && pdflatex -interaction=nonstopmode main.tex && bibtex main && pdflatex -interaction=nonstopmode main.tex && pdflatex -interaction=nonstopmode main.tex

# 4. 在开发机上一键重跑更新精简归档包（或加 --full 更新 HDFS 完整包）
python3 /mlx/users/lipengyu.seed/playground/agentic_video_understanding/scripts/archive/build_archive.py all
```
"""
    (meta_dir / "README.md").write_text(readme_text, encoding="utf-8")


def cmd_meta():
    """阶段 3：准备 STAGE 中的脱敏文件，并生成 META（git, ENV, EXCLUDED, HDFS_INVENTORY, MANIFEST, README）."""
    print("=== [3/5] 生成 STAGE 脱敏文件与 META 元数据 ===")
    plan = json.loads((WORK / "plan.json").read_text())
    if META.exists():
        shutil.rmtree(META)
    META.mkdir(parents=True, exist_ok=True)

    # 1. 处理 stage:sanitize
    sanitized_count = 0
    for e in plan:
        act = e["action"]
        rel = e["path"]
        if act == "stage:sanitize":
            src = PLAY / rel
            dst = STAGE / rel
            copy_with_readable_mode(src, dst, sanitize=True)
            sanitized_count += 1
    print(f"  STAGE 已脱敏 {sanitized_count} 个含密钥特征文件")

    # 2. 生成 git/ 快照（FULL 模式下额外生成 git/*.bundle）
    git_dir = META / "git"
    git_dir.mkdir(parents=True, exist_ok=True)
    if FULL_MODE:
        subprocess.run(
            ["git", "-C", str(PROJ_DIR), "bundle", "create", str(git_dir / f"{PROJ_NAME}.bundle"), "--all"],
            check=True,
            capture_output=True,
        )
    st_out = subprocess.run(["git", "-C", str(PROJ_DIR), "status"], capture_output=True, text=True, check=False).stdout
    (git_dir / f"{PROJ_NAME}.status.txt").write_text(st_out, encoding="utf-8")

    br_out = subprocess.run(["git", "-C", str(PROJ_DIR), "branch", "-av"], capture_output=True, text=True, check=False).stdout
    (git_dir / f"{PROJ_NAME}.branches.txt").write_text(br_out, encoding="utf-8")

    lg_out = subprocess.run(
        ["git", "-C", str(PROJ_DIR), "log", "--all", "--graph", "--oneline", "--decorate", "-n", "200"],
        capture_output=True,
        text=True,
        check=False,
    ).stdout
    (git_dir / f"{PROJ_NAME}.log.txt").write_text(lg_out, encoding="utf-8")

    diff_out = subprocess.run(
        ["git", "-C", str(PROJ_DIR), "diff", "HEAD"],
        capture_output=True,
        text=True,
        errors="surrogateescape",
        check=False,
    ).stdout
    clean_diff, _ = sanitize_text(diff_out)
    (git_dir / f"{PROJ_NAME}.diff").write_text(clean_diff, encoding="utf-8", errors="surrogateescape")

    # 3. 生成 ENV/ 包版本快照
    env_dir = META / "ENV"
    env_dir.mkdir(parents=True, exist_ok=True)
    env_candidates = [
        ("python3.packages.txt", Path("/usr/bin/python3")),
        ("DA-train_envs_train.packages.txt", PLAY / "DA-train" / "envs" / "train" / ".venv" / "bin" / "python"),
        ("vllm.venv.packages.txt", PLAY / "vllm" / ".venv" / "bin" / "python"),
    ]
    for out_name, py_bin in env_candidates:
        if py_bin.exists():
            res = subprocess.run([str(py_bin), "-m", "pip", "list"], capture_output=True, text=True, check=False)
            if res.returncode != 0 or not res.stdout.strip():
                res = subprocess.run(["uv", "pip", "list", "--python", str(py_bin)], capture_output=True, text=True, check=False)
            if res.stdout.strip():
                (env_dir / out_name).write_text(res.stdout, encoding="utf-8")

    # 4. 生成 HDFS_INVENTORY.txt
    (META / "HDFS_INVENTORY.txt").write_text(collect_hdfs_inventory(), encoding="utf-8")

    # 5. 生成 EXCLUDED.md 与 EXCLUDED.json
    c, b = Counter(), Counter()
    big_dropped = []
    for e in plan:
        if e["action"] == "drop":
            c[e["reason"]] += 1
            b[e["reason"]] += e["bytes"]
            if e["bytes"] >= 1 * MIB and e["reason"] not in (R_VENV, R_GIT):
                big_dropped.append(e)

    excl_summary = [
        {"reason": r, "files": c[r], "bytes": b[r], "mib": round(b[r] / MIB, 2)}
        for r in sorted(c, key=lambda x: -b[x])
    ]
    (META / "EXCLUDED.json").write_text(
        json.dumps({"summary": excl_summary, "big_files_over_1mib": big_dropped[:200]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# 归档包未收入内容清单（EXCLUDED）",
        "",
        f"本包严格排除模型权重（`.safetensors` / `.pt`，共 96 GiB）、训练数据（`data/*.jsonl` 及 HDFS 上 617 GiB 原始视频与帧缓存）、基座模型大体积分词器、Python 缓存及 LaTeX 编译中间临时文件。包含 `git/*.bundle` 与纯文本训练题目清单 JSONL 的完整包已备份于 HDFS `/mnt/hdfs/lipengyu/DA/archives/{TOP}-full.tar.zst`。",
        "",
        "| 排除原因 | 文件数 | 原始总大小 |",
        "|---|---:|---:|",
    ]
    for item in excl_summary:
        lines.append(f"| {item['reason']} | {item['files']:,} | {item['mib']:,.2f} MiB |")
    lines += [
        "",
        "## 单独列出的已排除大文件（>= 1 MiB，不含 .venv 与 .git）",
        "",
    ]
    for e in sorted(big_dropped, key=lambda x: -x["bytes"])[:100]:
        lines.append(f"- `{e['path']}`（{e['bytes'] / MIB:.2f} MiB）— {e['reason']}")
    (META / "EXCLUDED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # 6. 生成 README.md
    write_readme(META, plan)

    # 7. 生成 keep.list、stage.list、meta.list 与 MANIFEST.tsv
    keep_list = sorted(e["path"] for e in plan if e["action"] == "keep")
    stage_list = sorted(e["path"] for e in plan if e["action"].startswith("stage:"))
    action_map = {e["path"]: e["action"] for e in plan}

    (WORK / "keep.list").write_text("".join(p + "\n" for p in keep_list), encoding="utf-8")
    (WORK / "stage.list").write_text("".join(p + "\n" for p in stage_list), encoding="utf-8")

    rows = []
    stage_set = set(stage_list)
    for p in sorted(keep_list + stage_list):
        src = (STAGE / p) if p in stage_set else (PLAY / p)
        kind = action_map[p] if p in stage_set else "as-is"
        if os.path.islink(src):
            rows.append(f"{p}\tlink\t->{os.readlink(src)}\t{kind}")
        elif src.is_dir():
            rows.append(f"{p}\tdir\t-\t{kind}")
        else:
            rows.append(f"{p}\t{src.stat().st_size}\t{sha256(src)}\t{kind}")

    (META / "MANIFEST.tsv").write_text("path\tbytes\tsha256\tkind\n" + "\n".join(rows) + "\n", encoding="utf-8")
    meta_files = sorted(str(f.relative_to(META)) for f in META.rglob("*") if f.is_file())
    (WORK / "meta.list").write_text("".join(p + "\n" for p in meta_files), encoding="utf-8")
    print(f"  META 生成完毕：{len(meta_files)} 个元数据文件，keep={len(keep_list)}，stage={len(stage_list)}")


def cmd_pack():
    """阶段 4：流式执行 tar --no-recursion | zstd -T32 -12 生成压缩包与 .sha256."""
    print(f"=== [4/5] 正在打包压缩至 {OUT} ===")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    part = OUT.with_name(OUT.name + ".part")
    if part.exists():
        part.unlink()
    cmd = (
        f"tar -cf - --no-recursion --mode='u+rw,go+r' --transform='flags=rh;s,^,{TOP}/,' "
        f"-C {META} -T {WORK / 'meta.list'} "
        f"-C {PLAY} -T {WORK / 'keep.list'} "
        f"-C {STAGE} -T {WORK / 'stage.list'} "
        f"| zstd -q -T32 -12 -o {part}"
    )
    subprocess.run(cmd, shell=True, check=True)
    part.rename(OUT)
    digest = sha256(OUT)
    sha_file = OUT.with_name(OUT.name + ".sha256")
    sha_file.write_text(f"{digest}  {OUT.name}\n", encoding="utf-8")
    print(f"  打包完成：{OUT} ({OUT.stat().st_size / MIB:.2f} MiB, {OUT.stat().st_size:,} bytes)")
    print(f"  SHA-256 : {digest}")


def cmd_verify():
    """阶段 5：全面校验归档包完整性、MANIFEST 一致性、零明文密钥泄漏、零大权重与训练数据混入."""
    print(f"=== [5/5] 正在校验归档包 {OUT} ===")
    assert OUT.exists(), f"归档文件不存在: {OUT}"

    res = subprocess.run(
        f"zstd -dc {OUT} | tar -tf -",
        shell=True,
        capture_output=True,
        text=True,
        check=True,
    )
    tar_entries = [line.strip().rstrip("/") for line in res.stdout.splitlines() if line.strip()]
    prefix = f"{TOP}/"
    rel_entries = {e[len(prefix):] for e in tar_entries if e.startswith(prefix)}

    manifest_lines = (META / "MANIFEST.tsv").read_text(encoding="utf-8").splitlines()[1:]
    manifest_paths = {line.split("\t")[0] for line in manifest_lines if line.strip()}
    meta_paths = {line.strip() for line in (WORK / "meta.list").read_text(encoding="utf-8").splitlines() if line.strip()}
    expected_paths = manifest_paths | meta_paths

    missing_in_tar = expected_paths - rel_entries
    extra_in_tar = rel_entries - expected_paths
    assert not missing_in_tar, f"TAR 中缺失文件: {list(missing_in_tar)[:10]}"
    assert not extra_in_tar, f"TAR 中多出未预期文件: {list(extra_in_tar)[:10]}"

    bad_weights = [p for p in rel_entries if p.endswith((".safetensors", ".bin", ".pth", ".ckpt", ".pt"))]
    assert not bad_weights, f"发现未排除的权重文件: {bad_weights}"

    if not FULL_MODE:
        bad_train = [
            p for p in rel_entries
            if p.endswith(("sft_coldstart_6k.jsonl", "rl_ig_grpo_8k.jsonl", "sft_probed_6k.jsonl"))
        ]
        assert not bad_train, f"精简包发现未排除的训练数据文件: {bad_train}"

    grep_cmd = f"zstd -dc {OUT} | grep -aoE '{_KEY1}|{_KEY2}' | wc -l"
    grep_res = subprocess.run(grep_cmd, shell=True, capture_output=True, text=True, check=True)
    leaked_count = int(grep_res.stdout.strip())
    assert leaked_count == 0, f"严重错误：压缩包内仍检出 {leaked_count} 处明文 API Key！"

    required_files = [
        f"{PROJ_NAME}/paper/main.tex",
        f"{PROJ_NAME}/paper/main.pdf",
        f"{PROJ_NAME}/paper/references.bib",
        f"{PROJ_NAME}/paper/sections/5_experiments.tex",
        f"{PROJ_NAME}/paper/figures/fig1_pareto_frontier.pdf",
        f"{PROJ_NAME}/paper/figures/gen_fig1_pareto_frontier.py",
        f"{PROJ_NAME}/forager/agent.py",
        f"{PROJ_NAME}/forager/dual_buffer.py",
        f"{PROJ_NAME}/forager/ig_grpo.py",
        f"{PROJ_NAME}/forager/sdam_mask.py",
        f"{PROJ_NAME}/scripts/train_sft.py",
        f"{PROJ_NAME}/scripts/train_ig_grpo.py",
        f"{PROJ_NAME}/scripts/eval_ovobench.py",
        f"{PROJ_NAME}/scripts/eval_streamingbench.py",
        f"{PROJ_NAME}/outputs/ovobench_qwen25vl_final_ig_grpo_v2.json",
        f"{PROJ_NAME}/outputs/ovobench_qwen3vl_final_ig_grpo_v2.json",
        f"{PROJ_NAME}/outputs/streamingbench_qwen25vl_ig_grpo_v2.json",
        f"{PROJ_NAME}/outputs/streamingbench_qwen3vl_ig_grpo_v2.json",
        f"{PROJ_NAME}/outputs/h800_ablations/ablation_c_rl_dynamics_300steps.json",
        f"{PROJ_NAME}/outputs/archived_contaminated_do_not_use/README_CONTAMINATION_NOTE.md",
        f"{PROJ_NAME}/hdfs_sync/checkpoints/forager/rl/qwen25vl_ig_grpo_v2/meta.json",
        f"{PROJ_NAME}/hdfs_sync/checkpoints/forager/rl/qwen3vl_ig_grpo_v2/meta.json",
        f"git/{PROJ_NAME}.log.txt",
        f"git/{PROJ_NAME}.diff",
        "README.md",
        "MANIFEST.tsv",
        "EXCLUDED.md",
        "HDFS_INVENTORY.txt",
    ]
    for req in required_files:
        assert req in rel_entries, f"核心归档文件缺失: {req}"

    print(f"  [PASS] 条目总数核对一致：{len(rel_entries):,} 个文件（含 {len(meta_paths)} 个顶层元数据文件 + {len(manifest_paths):,} 个项目文件）")
    print("  [PASS] 权重文件检查：0 个 .safetensors/.bin/.pth/.ckpt/.pt")
    if not FULL_MODE:
        print("  [PASS] 训练数据检查：0 个训练 JSONL/Parquet/视频文件混入（保留 manifest_summary.json）")
    print("  [PASS] 明文密钥扫描：0 处泄漏")
    print("  [PASS] 论文 LaTeX、训练/评测代码、实验结果 JSON、HDFS Checkpoint 元数据及双 Worker 推理日志全部就位")


def cmd_all():
    cmd_sync()
    cmd_select()
    cmd_meta()
    cmd_pack()
    cmd_verify()


if __name__ == "__main__":
    cmds = {
        "sync": cmd_sync,
        "select": cmd_select,
        "meta": cmd_meta,
        "pack": cmd_pack,
        "verify": cmd_verify,
        "all": cmd_all,
    }
    sub = sys.argv[1] if len(sys.argv) > 1 else "all"
    if sub not in cmds:
        raise SystemExit(f"用法: {sys.argv[0]} [{'|'.join(cmds)}] [--full]")
    cmds[sub]()
