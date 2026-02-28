#!/usr/bin/env python3
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]  # repo root

SEED_PATTERNS = [
    r"torch\.manual_seed",
    r"torch\.cuda\.manual_seed",
    r"torch\.cuda\.manual_seed_all",
    r"np\.random\.seed",
    r"random\.seed",
    r"Generator\(",
    r"\.seed\(",
    r"reset\(.*seed=",
]

EXCLUDE_DIRS = {"venv", ".venv", "env", "build", "dist", "__pycache__", ".git", "site-packages", "tests"}


def should_skip(path: Path):
    return any(part in EXCLUDE_DIRS for part in path.parts)


def scan_file(path: Path):
    findings = []
    try:
        text = path.read_text()
    except Exception:
        return findings

    for pattern in SEED_PATTERNS:
        for match in re.finditer(pattern, text):
            line_no = text.count("\n", 0, match.start()) + 1
            findings.append((pattern, line_no, match.group()))
    return findings


def main():
    print("=== Dreamer Seed Audit ===\n")
    violations = []

    for path in ROOT.rglob("*.py"):
        if should_skip(path):
            continue

        findings = scan_file(path)
        if findings:
            print(f"\nFile: {path}")
            for pattern, line, snippet in findings:
                print(f"  Line {line:4d}:  {snippet}")
                violations.append((path, line, snippet))

    print("\n=== Summary ===")
    if not violations:
        print("No unexpected seeding detected. RNG usage is clean.")
    else:
        print(f"Found {len(violations)} potential RNG issues.")
        print("Review each to ensure it matches the Dreamer seeding contract.")


if __name__ == "__main__":
    main()
