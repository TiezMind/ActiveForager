#!/usr/bin/env bash
# ==============================================================================
# Cross-Devbox Synchronization Tool for `agentic_video_understanding`
#
# Topology:
#   - H800 Devbox (candy-hl)   : dbx_111063 (ssh alias: devbox-hl)
#   - H20  Devbox (candy-wlby) : dbx_111447 (ssh alias: devbox-wlby)
#   - Peer Alias               : devbox-peer (automatically points to the other devbox)
#   - Shared HDFS Git Hub      : /mnt/hdfs/lipengyu/DA/repos/agentic_video_understanding.git
#
# Usage:
#   sync-forager [bi|status|push|pull] [--force]
# ==============================================================================
set -euo pipefail

MODE="${1:-bi}"
FORCE=0
for arg in "$@"; do
    if [[ "$arg" == "--force" ]]; then
        FORCE=1
    fi
done

PROJECT_DIR="/mlx/users/lipengyu.seed/playground/agentic_video_understanding"
HDFS_BARE_REPO="/mnt/hdfs/lipengyu/DA/repos/agentic_video_understanding.git"
PEER_HOST="devbox-peer"

cd "$PROJECT_DIR"

RSYNC_EXCLUDES=(
    --exclude='__pycache__/'
    --exclude='*.pyc'
    --exclude='.pytest_cache/'
    --exclude='.git/'
    --exclude='*.safetensors'
    --exclude='*.pt'
)

case "$MODE" in
    status)
        echo "=== [1/3] Local Machine ($(hostname)) Git Status ==="
        git status -sb
        echo ""
        echo "=== [2/3] Peer Machine ($PEER_HOST) Git Status ==="
        ssh "$PEER_HOST" "cd '$PROJECT_DIR' && echo \"Host: \$(hostname)\" && git status -sb"
        echo ""
        echo "=== [3/3] File Differences (Local -> Peer Dry-Run) ==="
        rsync -avzni "${RSYNC_EXCLUDES[@]}" "$PROJECT_DIR/" "$PEER_HOST:$PROJECT_DIR/"
        ;;

    push)
        echo ">>> [Push] Syncing Git commits to HDFS origin & $PEER_HOST..."
        if [[ -d "$HDFS_BARE_REPO" ]]; then
            git push origin main 2>/dev/null || true
        fi
        git push peer main 2>/dev/null || true

        echo ">>> [Push] Syncing working tree, data/, and outputs/ -> $PEER_HOST..."
        if [[ "$FORCE" -eq 1 ]]; then
            rsync -avz --delete "${RSYNC_EXCLUDES[@]}" "$PROJECT_DIR/" "$PEER_HOST:$PROJECT_DIR/"
        else
            rsync -avzu "${RSYNC_EXCLUDES[@]}" "$PROJECT_DIR/" "$PEER_HOST:$PROJECT_DIR/"
        fi
        echo ">>> [Push] Done."
        ;;

    pull)
        echo ">>> [Pull] Fetching Git commits from HDFS origin & $PEER_HOST..."
        if [[ -d "$HDFS_BARE_REPO" ]]; then
            git pull --rebase origin main 2>/dev/null || true
        fi
        git pull --rebase peer main 2>/dev/null || true

        echo ">>> [Pull] Syncing working tree, data/, and outputs/ <- $PEER_HOST..."
        if [[ "$FORCE" -eq 1 ]]; then
            rsync -avz --delete "${RSYNC_EXCLUDES[@]}" "$PEER_HOST:$PROJECT_DIR/" "$PROJECT_DIR/"
        else
            rsync -avzu "${RSYNC_EXCLUDES[@]}" "$PEER_HOST:$PROJECT_DIR/" "$PROJECT_DIR/"
        fi
        echo ">>> [Pull] Done."
        ;;

    bi)
        echo ">>> [Bi-Sync] 1. Syncing Git commits via HDFS origin & $PEER_HOST..."
        if [[ -d "$HDFS_BARE_REPO" ]]; then
            git pull --rebase origin main 2>/dev/null || true
        fi
        git pull --rebase peer main 2>/dev/null || true
        if [[ -d "$HDFS_BARE_REPO" ]]; then
            git push origin main 2>/dev/null || true
        fi
        git push peer main 2>/dev/null || true

        echo ">>> [Bi-Sync] 2. Merging newer working tree / data / outputs (Local <-> $PEER_HOST)..."
        rsync -avzu "${RSYNC_EXCLUDES[@]}" "$PEER_HOST:$PROJECT_DIR/" "$PROJECT_DIR/"
        rsync -avzu "${RSYNC_EXCLUDES[@]}" "$PROJECT_DIR/" "$PEER_HOST:$PROJECT_DIR/"
        echo ">>> [Bi-Sync] Completed. Both devboxes are in sync."
        ;;

    *)
        echo "Usage: sync-forager [bi|status|push|pull] [--force]" >&2
        exit 2
        ;;
esac
