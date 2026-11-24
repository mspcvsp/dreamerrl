"""Lightweight checks for locked profiles and activation-independent execution."""

import json
import subprocess
from pathlib import Path

import tomli

ROOT = Path(__file__).resolve().parents[2]


def test_runtime_conflicts_are_explicit_and_legacy_only():
    data = tomli.loads((ROOT / "pyproject.toml").read_text())
    groups = data["dependency-groups"]
    assert data["tool"]["uv"]["default-groups"] == ["dev"]
    assert [{"group": "dev"}, {"group": "legacy"}] in data["tool"]["uv"]["conflicts"]
    assert "gymnasium==1.4.0" in groups["dev"]
    assert "setuptools==84.2.1" in groups["dev"]
    assert "gymnasium==1.2.0" in groups["legacy"]
    assert "setuptools==80.9.0" in groups["legacy"]
    assert not any("nle" in x or "minihack" in x for x in groups["dev"] if isinstance(x, str))


def test_combining_profiles_is_rejected_without_installation():
    result = subprocess.run(
        ["uv", "sync", "--locked", "--group", "legacy", "--dry-run"], cwd=ROOT, capture_output=True, text=True
    )
    assert result.returncode != 0
    assert "dev" in result.stderr and "legacy" in result.stderr and "incompatible" in result.stderr


def test_shell_cleanup_does_not_call_conda_or_source_external_activation():
    script = ROOT / "scripts/uv_shell.sh"
    result = subprocess.run(
        [
            "bash",
            "--noprofile",
            "--norc",
            "-c",
            "export CONDA_PREFIX=/not-an-environment VIRTUAL_ENV=/not-an-environment; "
            'source "$1"; test -z "${CONDA_PREFIX:-}"; test -z "${VIRTUAL_ENV:-}"; '
            "command -v uv",
            "test",
            str(script),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "uv" in result.stdout
    for path in [script, *(ROOT / "environments/cage2").glob("*.sh")]:
        text = path.read_text()
        assert "conda.sh" not in text and "conda activate" not in text and "conda deactivate" not in text


def test_editor_paths_have_no_external_environment_dependency():
    data = json.loads((ROOT / "pyrightconfig.json").read_text())
    assert all("conda" not in path.lower() for path in data["extraPaths"])
    assert ".venv/lib/python3.10/site-packages" in data["extraPaths"]
    assert ".venv-legacy/lib/python3.10/site-packages" in data["extraPaths"]


def test_frozen_baseline_lock_and_git_tag_are_preserved():
    import hashlib

    manifest = json.loads((ROOT / "environments/cage2/reproducibility/original_baseline.json").read_text())
    lock = ROOT / manifest["dependencies"]["uv_lock"]
    assert hashlib.sha256(lock.read_bytes()).hexdigest() == manifest["dependencies"]["uv_lock_sha256"]
    original = subprocess.check_output(["git", "show", "cage2_baseline_v1:environments/cage2/uv.lock"], cwd=ROOT)
    assert lock.read_bytes() == original
