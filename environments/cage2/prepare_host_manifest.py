"""Prepare a reviewable execution manifest after committing and annotating a NEW tag."""

import argparse
import importlib.metadata
import json
from pathlib import Path

from host_experiment_support import CONFIG, ROOT, git, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if git("status", "--porcelain"):
        raise ValueError("Review and commit all source changes first")
    if git("cat-file", "-t", args.tag) != "tag" or git("rev-parse", args.tag + "^{}") != git("rev-parse", "HEAD"):
        raise ValueError("An annotated experiment tag must identify this exact source commit")
    # The output is reviewed/committed separately; execution uses a clean detached
    # source checkout of this tag and reads the reviewed manifest from the separate manifest checkout.
    files = git("ls-files").splitlines()
    fingerprints = {name: sha256(ROOT / name) for name in files if (ROOT / name).is_file()}
    baseline = json.loads((ROOT / "environments/cage2/reproducibility/original_baseline.json").read_text())
    native = baseline["dependencies"]["cusparselt_repair"]
    versions = {d.metadata["Name"].lower().replace("_", "-"): d.version for d in importlib.metadata.distributions()}
    manifest = dict(
        schema_version=1,
        source_tag=args.tag,
        tag_object=git("rev-parse", args.tag),
        source_commit=git("rev-parse", "HEAD"),
        config_sha256=sha256(CONFIG),
        source_fingerprints=fingerprints,
        launch_approval=None,
        execution=json.loads(CONFIG.read_text()),
        checkpoint_provenance="All six arms fresh; no historical checkpoint used",
        dependencies=dict(
            cyborg_revision=baseline["dependencies"]["cyborg_revision"],
            package_versions=versions,
            native_wheel_path="environments/cage2/wheels/" + native["repaired_filename"],
            native_wheel_sha256=native["repaired_sha256"],
        ),
    )
    if args.output.exists():
        raise ValueError("Refuse to overwrite an existing experiment manifest")
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print("Prepared for review and separate commit:", args.output)
    print("Formal launch remains blocked until explicit user approval is recorded and committed.")


if __name__ == "__main__":
    main()
