"""Read-only verification of the original baseline manifest; no uv, simulator or learner calls."""

import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
ROOT = DIRECTORY.parents[2]


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    manifest = json.loads((DIRECTORY / "original_baseline.json").read_text())
    baseline = manifest["baseline"]
    for suffix, expected in [("", baseline["tag_object"]), ("^{}", baseline["commit"])]:
        actual = subprocess.check_output(["git", "rev-parse", baseline["tag"] + suffix], cwd=ROOT, text=True).strip()
        assert actual == expected, "Baseline tag changed"
    for path, expected in manifest["historical_artifact_inventory"].items():
        actual = ROOT / path
        assert actual.stat().st_size == expected["bytes"], path
        assert sha256(actual) == expected["sha256"], path
    for filename, expected in baseline["script_sha256"].items():
        content = subprocess.check_output(["git", "show", f"{baseline['tag']}:environments/cage2/{filename}"], cwd=ROOT)
        assert hashlib.sha256(content).hexdigest() == expected, filename
    config = manifest["configuration"]
    assert sha256(DIRECTORY / config["file"]) == config["sha256"]
    assert json.loads((DIRECTORY / config["file"]).read_text()) == json.loads(
        (ROOT / config["historical_source"]).read_text()
    )
    deps = manifest["dependencies"]
    assert sha256(ROOT / deps["uv_lock"]) == deps["uv_lock_sha256"]
    gitlink = subprocess.check_output(
        ["git", "ls-tree", baseline["tag"], "environments/cage2/vendor/cage-challenge-2"], cwd=ROOT, text=True
    )
    assert deps["cyborg_revision"] in gitlink
    native = deps["cusparselt_repair"]
    wheel = ROOT / "environments/cage2/wheels" / native["repaired_filename"]
    assert sha256(wheel) == native["repaired_sha256"]
    with zipfile.ZipFile(wheel) as archive:
        for name, expected in native["native_library_sha256"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == expected, name
    run = ROOT / "runs" / manifest["experiment_id"]
    budget = json.loads((run / "budget.json").read_text())
    assert budget == manifest["execution"]["original_budget"]
    scores = json.loads((run / "scorecard.json").read_text())
    schedule = manifest["execution"]["fixed_original_schedule"]
    assert schedule["evaluation_checkpoints"] == [
        {"update": row["update"], "environment_steps": row["env_steps"]} for row in scores[1:]
    ]
    complete = json.loads((run / "complete.json").read_text())
    assert complete["final_update"] == schedule["final_update"]
    assert complete["final_env_steps"] == schedule["final_environment_steps"]
    assert (
        sha256(ROOT / manifest["source_checkpoint"]["repository_relative_path"])
        == manifest["source_checkpoint"]["sha256"]
    )
    print(
        json.dumps(
            dict(
                passed=True,
                tag_unchanged=True,
                historical_files_verified=len(manifest["historical_artifact_inventory"]),
                script_hashes_match_tag=True,
                resolved_config_matches=True,
                lockfile_and_cyborg_match=True,
                repaired_wheel_and_native_library_match=True,
                schedule_matches_original=True,
                source_checkpoint_hash_matches=True,
                scope=(
                    "Read-only hashes, Git objects and JSON comparisons; "
                    "no training, dependency installation or reporting rerun."
                ),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
