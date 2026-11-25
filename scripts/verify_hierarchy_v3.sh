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
echo "START hierarchy-v3 readiness"
uv sync --locked --check
bash scripts/run_hierarchy_v2.sh --manifest "$2" --output-root "$output/unused" --check-only \
  | tee "$output/manifest_gate.json"
uv run --locked --no-sync python -B -m pytest -q --assert=plain --rootdir=. --confcutdir=. \
  --import-mode=importlib --noconftest -c /dev/null -o "pythonpath=$PWD/environments/cage2" \
  -p no:cacheprovider --basetemp "$output/pytest" \
  dreamerrl/tests/functional/test_hierarchical_actor.py::test_noncyber_collect_update_restart_evaluate \
  | tee "$output/pytest.log"
