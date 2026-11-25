#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s [--dry-run]\n' "${0##*/}"
}

rsync_options=(--archive --human-readable --info=stats1)
case "${1:-}" in
  "") ;;
  --dry-run) rsync_options+=(--dry-run --itemize-changes) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
destination="$HOME/OneDrive/Documents/Woodbadge/TTCP_CAGE2"

if [[ ! -d "$destination" ]]; then
  printf 'Destination directory is unavailable: %s\n' "$destination" >&2
  exit 1
fi

exec rsync "${rsync_options[@]}" -- "$project_dir/runs" "$destination/"
