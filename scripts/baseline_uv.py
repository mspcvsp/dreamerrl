"""Provision/verify the immutable CAGE2 tag with its unchanged historical uv lock."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "environments/cage2/reproducibility/original_baseline.json"
CHECKOUT = ROOT / ".baseline/cage2_baseline_v1"


def command(*args, **kwargs):
    return subprocess.check_output(list(args), text=True, **kwargs).strip()


def run(*args, **kwargs):
    subprocess.run(list(args), check=True, **kwargs)


def verify_source(manifest):
    expected = manifest["baseline"]["commit"]
    assert command("git", "-C", str(ROOT), "rev-parse", manifest["baseline"]["tag"] + "^{}") == expected
    assert command("git", "-C", str(CHECKOUT), "rev-parse", "HEAD") == expected
    assert not command("git", "-C", str(CHECKOUT), "status", "--porcelain"), "Baseline checkout must stay clean"
    project = CHECKOUT / "environments/cage2"
    assert (
        command("git", "-C", str(project / "vendor/cage-challenge-2"), "rev-parse", "HEAD")
        == manifest["dependencies"]["cyborg_revision"]
    )
    assert hashlib.sha256((project / "uv.lock").read_bytes()).hexdigest() == manifest["dependencies"]["uv_lock_sha256"]
    return project


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["setup", "verify"])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    for key in ("VIRTUAL_ENV", "UV_PROJECT_ENVIRONMENT", "UV_PYTHON"):
        os.environ.pop(key, None)
    if args.action == "setup":
        if not CHECKOUT.exists():
            CHECKOUT.parent.mkdir(parents=True, exist_ok=True)
            run("git", "-C", str(ROOT), "worktree", "add", "--detach", str(CHECKOUT), manifest["baseline"]["commit"])
        run(
            "git",
            "-C",
            str(CHECKOUT),
            "submodule",
            "update",
            "--init",
            "--",
            "environments/cage2/vendor/cage-challenge-2",
        )
        project = verify_source(manifest)
        run("uv", "python", "install", "3.10.19")
        # Reuse the one existing hash-pinned repair implementation, never duplicate it.
        run(
            "uv",
            "run",
            "--no-project",
            "--managed-python",
            "--python",
            "3.10.19",
            "python",
            str(ROOT / "environments/cage2/prepare_native_wheel.py"),
        )
        native = manifest["dependencies"]["cusparselt_repair"]
        source = ROOT / "environments/cage2/wheels" / native["repaired_filename"]
        assert hashlib.sha256(source.read_bytes()).hexdigest() == native["repaired_sha256"]
        destination = project / "wheels" / native["repaired_filename"]
        destination.parent.mkdir(exist_ok=True)
        shutil.copyfile(source, destination)
        run("uv", "sync", "--locked", "--project", str(project))
    else:
        project = verify_source(manifest)
        env = os.environ.copy()
        for key in (
            "CONDA_PREFIX",
            "CONDA_DEFAULT_ENV",
            "CONDA_SHLVL",
            "VIRTUAL_ENV",
            "PYTHONPATH",
            "PYTHONHOME",
            "LD_LIBRARY_PATH",
            "LD_PRELOAD",
            "UV_PROJECT_ENVIRONMENT",
            "UV_PYTHON",
        ):
            env.pop(key, None)
        run("uv", "sync", "--locked", "--project", str(project), env=env)
        arguments = ["uv", "run", "--locked", "--project", str(project), "python", str(project / "smoke.py")]
        if args.output:
            arguments.extend(["--output", str(args.output.resolve())])
        run(*arguments, env=env)
        verify_source(manifest)


if __name__ == "__main__":
    main()
