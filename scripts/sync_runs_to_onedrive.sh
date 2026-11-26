#!/usr/bin/env bash
set -euo pipefail

usage() {
  printf 'Usage: %s [--dry-run]\n' "${0##*/}"
}

rclone_options=(--progress --exclude 'pytest/**')
case "${1:-}" in
  "") ;;
  --dry-run) rclone_options+=(--dry-run) ;;
  -h|--help) usage; exit 0 ;;
  *) usage >&2; exit 2 ;;
esac
if (( $# > 1 )); then
  usage >&2
  exit 2
fi

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
destination="onedrive:Documents/Woodbadge/TTCP_CAGE2/runs"

exec rclone copy "${rclone_options[@]}" -- "$project_dir/runs" "$destination"
