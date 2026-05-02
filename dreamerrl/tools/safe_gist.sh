#!/usr/bin/env bash
set -euo pipefail

# --- 0. Usage check ---
if [ $# -ne 1 ]; then
  echo "Usage: $0 <relative-path-to-directory>"
  echo "Example: $0 dreamerrl/models"
  exit 1
fi

TARGET_DIR="$1"

# --- 1. Ensure we are inside a git repo ---
if ! git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  echo "❌ Not inside a git repository."
  exit 1
fi

# --- 2. Show the branch ---
BRANCH=$(git branch --show-current)
echo "📌 Current branch: $BRANCH"

# --- 3. Ensure working tree is clean ---
if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "❌ Working tree is not clean. Commit or stash changes first."
  git status --short
  exit 1
fi

# --- 4. Resolve repo root ---
REPO_ROOT=$(git rev-parse --show-toplevel)
ABS_TARGET="$REPO_ROOT/$TARGET_DIR"

# --- 5. Validate directory exists ---
if [ ! -d "$ABS_TARGET" ]; then
  echo "❌ Directory does not exist: $ABS_TARGET"
  exit 1
fi

echo "📁 Target directory: $ABS_TARGET"

# --- 6. Collect non-empty Python files ---
FILES=$(find "$ABS_TARGET" \
  -type f -name "*.py" \
  -size +0c \
  ! -path "*/__pycache__/*" \
  | sort)

if [ -z "$FILES" ]; then
  echo "❌ No non-empty Python files found in $ABS_TARGET"
  exit 1
fi

echo "📄 Files to gist:"
echo "$FILES"

# --- 7. Confirm ---
read -p "Proceed with gist creation? (y/N): " yn
case $yn in
    [Yy]* ) ;;
    * ) echo "Cancelled."; exit 0;;
esac

# --- 8. Create gist ---
gh gist create $FILES --public --desc "Snapshot of $TARGET_DIR from branch $BRANCH"

echo "✅ Gist created from $TARGET_DIR on branch $BRANCH"
