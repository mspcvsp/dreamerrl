#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "$0")" && pwd)"
source "$script_dir/uv_shell.sh"
cd "$script_dir/.."
output="$(realpath "$1")"
case "$output" in "$PWD"/runs/*) ;; *) exit 2 ;; esac
test -d "$output"
test ! -e "$output/final_report.json"
finish() {
  status=$?
  result=fail
  if [[ "$status" == 0 ]]; then result=pass; fi
  printf '{"result":"%s","exit_status":%s,"tests":"pytest.log","console":"console.log"}\n' \
    "$result" "$status" > "$output/final_report.json.tmp"
  mv "$output/final_report.json.tmp" "$output/final_report.json"
  printf '%s\n' "$status" > "$output/exit_status"
}
trap finish EXIT
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=2 PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
echo "START hierarchy-v2 readiness"
uv sync --locked --check
bash -n scripts/run_hierarchy_v2.sh
uv run --locked --no-sync python -B -m pytest -q --assert=plain --rootdir=. --confcutdir=. \
  --import-mode=importlib --noconftest -c /dev/null -o "pythonpath=$PWD/environments/cage2" \
  -p no:cacheprovider --basetemp "$output/pytest" \
  environments/cage2/test_launch_hierarchy_v2.py \
  environments/cage2/test_experiment_supervisor.py \
  environments/cage2/test_concurrency_workflow.py \
  -k 'test_launch_hierarchy_v2 or three_workers_share_watchdog or concurrency_throughput_and_isolation or single_arm_cli' \
  | tee "$output/pytest.log"
