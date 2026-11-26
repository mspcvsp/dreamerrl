"""Supervise the bounded frozen-state studies and report their exit status."""
import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
import resource
import signal
from pathlib import Path
import subprocess
import sys

from dreamerrl.eval.memory.dreamer import file_sha256
from dreamerrl.eval.memory.reporting import write_json
from experiment_supervisor import CHECKPOINT_EXIT, Job, supervise
from thermal_stop import REQUEST, STATUS, read_record

SOURCE = Path(__file__).resolve().parents[1]
STUDIES = {
    "tmaze": ("minihack", "tmaze_intervention", "legacy"),
    "cage2": ("cage2", "cage2_memory_pilot", "dev"),
    "history": ("cage2", "archived_history_baselines", "dev"),
    "cage2-final": ("cage2", "history_common_support", "dev"),
}
SESSION = os.environ.get("DIAGNOSTIC_TMUX_SESSION", "dreamer-frozen-memory-studies")
FINAL_STUDY = "cage2-final"
APPROVED_PLAN = "0d04596"


def revision(path):
    return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()


def configuration(study):
    folder, module, _ = STUDIES[study]
    filename = "history_common_support_plan.json" if study == FINAL_STUDY else module + ".json"
    return SOURCE / "environments" / folder / filename


def load_spec(study):
    config = configuration(study)
    if study == FINAL_STUDY:
        relative = config.relative_to(SOURCE)
        original = subprocess.check_output(["git", "-C", SOURCE, "show", f"{APPROVED_PLAN}:{relative}"])
        if config.read_bytes() != original:
            raise RuntimeError("Final CAGE2 study configuration differs from the approved plan")
    return json.loads(config.read_text())


def worker(root, run_dir, study):
    import torch

    spec = load_spec(study)
    if study == FINAL_STUDY:
        maximum = spec["maximum"]
        resource.setrlimit(resource.RLIMIT_AS, (maximum["memory_gib"] * 2**30,) * 2)
        signal.alarm(maximum["wall_minutes"] * 60)
    torch.set_num_threads(spec.get("maximum", {}).get("cpu_threads", 2))
    torch.set_num_interop_threads(1)
    folder, module, _ = STUDIES[study]
    config = configuration(study)
    adapter = importlib.import_module(f"environments.{folder}.{module}")
    out = run_dir.parents[3] / study
    out.mkdir(exist_ok=True)
    metadata = dict(pid=os.getpid(), tmux_session=SESSION, argv=sys.argv, source_revision=revision(SOURCE),
        config=str(config), config_sha256=file_sha256(config), lock_sha256=file_sha256(root / "uv.lock"),
        packages={key: importlib.metadata.version(key) for key in ("torch", "numpy", "gymnasium", "matplotlib")},
        python=sys.version, log=str(run_dir.parent.parent / "console.log"), output=str(out))
    if study in ("cage2", FINAL_STUDY):
        vendor = root / "environments/cage2/vendor/cage-challenge-2"
        import CybORG
        metadata["cyborg_revision"] = revision(vendor)
        if metadata["cyborg_revision"] != "26ce1c1253fa9e2e73f25e6a7f2da32860c11257" or not Path(CybORG.__file__).resolve().is_relative_to(vendor):
            raise RuntimeError("Unpinned CAGE2 simulator")
    elif study == "tmaze":
        metadata["packages"].update({key: importlib.metadata.version(key) for key in ("nle", "minihack")})
    old = out / "execution_manifest.json"
    if old.exists():
        previous = json.loads(old.read_text())
        for key in ("source_revision", "config_sha256", "lock_sha256", "packages"):
            if previous[key] != metadata[key]:
                raise ValueError(f"Resumption mismatch on {key}; choose a fresh output")
    write_json(old, metadata)
    if study == os.environ.get("DIAGNOSTIC_READY_STUDY", "tmaze"):
        write_json(run_dir.parents[3] / "launch_ready.json", metadata)
        if os.environ.get("TMUX"):
            subprocess.run(["tmux", "wait-for", "-S", SESSION + "-ready"], check=True)

    def should_stop():
        request = read_record(run_dir / REQUEST)
        if request is None:
            return False
        write_json(run_dir / STATUS, dict(request_id=request["request_id"], checkpoint_status="saved", checkpoint=str(out)))
        return True

    completed = adapter.run(root, spec, out, should_stop)
    if not completed:
        return CHECKPOINT_EXIT
    write_json(run_dir / "complete.json", dict(checkpoint=str(out / "results.json")))
    return 0


def _worker_command(root, study, run_dir):
    _, _, group = STUDIES[study]
    venv = root / (".venv-legacy" if group == "legacy" else ".venv")
    return ["env", f"UV_PROJECT_ENVIRONMENT={venv}", "uv", "run", "--project", str(root),
            "--locked", "--offline", "--no-sync", "--no-default-groups", "--group", group,
            "python", "-u", str(Path(__file__).resolve()), str(root), "--worker", study, str(run_dir)]


def orchestrate(root, out, study):
    studies = ("tmaze", "cage2") if study == "all" else (study,)
    is_final = study == FINAL_STUDY
    maximum = load_spec(study)["maximum"] if is_final else {}
    if is_final and (out / "launch_ready.json").exists():
        raise RuntimeError("The approved final study has already launched into this directory")
    if not out.is_relative_to(root / "runs") or out == root / "runs":
        raise ValueError("Use a dedicated directory inside the main runs directory")
    if subprocess.check_output(["git", "-C", SOURCE, "status", "--porcelain"], text=True).strip():
        raise RuntimeError("Commit the reviewed diagnostic sources before executing")
    if file_sha256(SOURCE / "uv.lock") != file_sha256(root / "uv.lock"):
        raise RuntimeError("The execution and source lockfiles differ")
    out.mkdir(parents=True, exist_ok=True)
    attempt = out / datetime.now(timezone.utc).strftime("supervisor_%Y%m%dT%H%M%S%fZ")
    jobs = []
    for index, study in enumerate(studies, 1):
        run_dir = attempt / f"job_{index:03d}" / "results" / study
        run_dir.mkdir(parents=True)
        spec = configuration(study)
        jobs.append((index, Job(run_dir, _worker_command(root, study, run_dir), spec, study,
                                load_spec(study).get("rng_seed", 0))))
    watchdog = [sys.executable, "-u", str(SOURCE / "environments/cage2/thermal_watchdog.py"),
                "--supervisor-control", str(attempt), "--log", str(attempt / "thermal.jsonl"),
                "--gpu", "0", "--warn-c", "80", "--stop-c", "87", "--sustain-seconds", "30"]
    status = 1
    try:
        result, status = supervise(attempt, jobs, watchdog, concurrency=1, readiness_timeout=60,
                                   checkpoint_timeout=300, wall_seconds=maximum.get("wall_minutes", 60)*60)
    except Exception as exc:
        result = {"jobs": [], "error": repr(exc)}
    names = {j["arm"]: j for j in result["jobs"]}
    successful = status == 0 and all(names.get(name, {}).get("state") == "completed" for name in studies)
    failed = result.get("error") or any(j["state"] in ("failed", "launch_failed") for j in names.values())
    state = "successful" if successful else "failed" if failed else "incomplete"
    lines = ["# Frozen memory studies", "", f"Execution state **{state}**.", ""]
    for name in studies:
        row = names.get(name, {})
        report = out / name / "report.md"
        suffix = f" [Study report]({name}/report.md)." if report.is_file() else ""
        lines.append(f"- {name} worker state {row.get('state', 'unavailable')}, exit code {row.get('exit_code')}.{suffix}")
    if is_final:
        lines += ["", "This was the final approved study. Any replication or broader evaluation requires separate approval.", ""]
    else:
        lines += ["", "Broad CAGE2 evaluation remains NO-GO pending the labeled replay review.", ""]
    (out / "report.md").write_text("\n".join(lines))
    write_json(out / "completion_manifest.json", dict(state=state, exit_code=status, session=SESSION,
        supervisor_pid=os.getpid(), attempt=str(attempt), source_revision=revision(SOURCE), supervisor=result,
        report=str(out / "report.md"), artifacts=[str(p) for p in sorted(out.rglob("*")) if p.is_file()]))
    return status


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    command = parser.add_mutually_exclusive_group(required=True)
    command.add_argument("--out", type=Path)
    command.add_argument("--worker", nargs=2, metavar=("STUDY", "STATUS_DIR"))
    parser.add_argument("--study", choices=("all", *STUDIES), default="all")
    args = parser.parse_args()
    if args.worker:
        study, run = args.worker
        return worker(args.root.resolve(), Path(run), study)
    return orchestrate(args.root.resolve(), args.out.resolve(), args.study)


if __name__ == "__main__":
    sys.exit(main())
