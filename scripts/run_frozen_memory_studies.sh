#!/usr/bin/env bash
set -euo pipefail
source_tree="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
git_dir="$(git -C "$source_tree" rev-parse --path-format=absolute --git-common-dir)"
root="$(dirname "$git_dir")"
session=dreamer-frozen-memory-studies
target_study="${2:-all}"
if [[ "$target_study" == "--pane" ]]; then target_study="${3:-all}"; fi
if [[ "$target_study" == "cage2-final" ]]; then session=dreamer-cage2-final-history; fi
out="$(realpath -m "${1:?Specify an output directory under the main runs/}")"
case "$out" in "$root"/runs/*) ;; *) exit 2 ;; esac

if [[ "${2:-}" == "--pane" ]]; then
    study="${3:-all}"
    export DIAGNOSTIC_READY_STUDY="$study" DIAGNOSTIC_TMUX_SESSION="$session"
    if [[ "$study" == all ]]; then export DIAGNOSTIC_READY_STUDY=tmaze; fi
    export PYTHONPATH="$source_tree:$source_tree/environments/cage2:$root/environments/cage2/vendor/cage-challenge-2/CybORG"
    export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MPLBACKEND=Agg MALLOC_ARENA_MAX=2
    export UV_PROJECT_ENVIRONMENT="$root/.venv"
    command=(uv run --project "$root" --locked --offline --no-sync python -u
             "$source_tree/scripts/frozen_memory_studies.py" "$root" --out "$out" --study "$study")
    printf '%q ' "${command[@]}" > "$out/launch_command"
    printf '\n' >> "$out/launch_command"
    set +e
    "${command[@]}" >> "$out/console.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$out/exit_status"
    exit "$status"
fi

mkdir -p "$out"
printf -v pane 'bash %q %q --pane %q' "$source_tree/scripts/run_frozen_memory_studies.sh" "$out" "${2:-all}"
if tmux has-session -t "=$session" 2>/dev/null; then
    [[ "$(tmux display-message -p -t "$session:0.0" '#{pane_dead}')" == 1 ]] || exit 3
    tmux respawn-pane -t "$session:0.0" "$pane"
else
    tmux new-session -d -s "$session" -c "$source_tree" 'exec sleep 60'
    tmux set-option -w -t "$session:0" remain-on-exit on
    tmux respawn-pane -k -t "$session:0.0" "$pane"
fi
