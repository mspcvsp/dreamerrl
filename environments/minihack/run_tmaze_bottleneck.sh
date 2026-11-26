#!/usr/bin/env bash
# Starts or resumes this CPU-only evaluation in the dedicated tmux session.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
session=dreamer-memory-diagnostics
out="$(realpath -m -- "${1:?Pass a dedicated path under the main runs directory}")"
case "$out" in "$root"/runs/*) ;; *) exit 2 ;; esac

if [[ "${2:-}" == "--pane" ]]; then
    cd "$root/.experiments/minihack-tmaze-memory"
    export DIAGNOSTIC_TMUX="$session" CUDA_VISIBLE_DEVICES=""
    export DIAGNOSTIC_READY_CHANNEL="$session-ready"
    export UV_PROJECT_ENVIRONMENT=.venv-legacy MPLBACKEND=Agg OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2
    command=(uv run --locked --offline --no-default-groups --group legacy -- python -u
             "$root/environments/minihack/tmaze_bottleneck.py" "$out" --supervise)
    printf '%q ' "${command[@]}" > "$out/launch_command"
    printf '\n' >> "$out/launch_command"
    set +e
    "${command[@]}" >> "$out/console.log" 2>&1 &
    worker=$!
    printf '%s\n' "$worker" > "$out/worker_pid"
    wait "$worker"
    status=$?
    printf '%s\n' "$status" > "$out/exit_status"
    printf '%s\t%s\t%s\t%s\n' "$(date -u +%FT%TZ)" "$session" "$worker" "$status" >> "$out/attempts.tsv"
    exit "$status"
fi

mkdir -p "$out"
printf -v pane 'bash %q %q --pane' "$root/environments/minihack/run_tmaze_bottleneck.sh" "$out"
if tmux has-session -t "=$session" 2>/dev/null; then
    if [[ "$(tmux display-message -p -t "$session:0.0" '#{pane_dead}')" != 1 ]]; then
        printf 'Session %s already has a running pane\n' "$session" >&2
        exit 3
    fi
    tmux respawn-pane -t "$session:0.0" "$pane"
else
    tmux new-session -d -s "$session" -c "$root"
    tmux set-option -w -t "$session:0" remain-on-exit on
    tmux respawn-pane -k -t "$session:0.0" "$pane"
fi
