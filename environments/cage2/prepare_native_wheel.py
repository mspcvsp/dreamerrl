"""Reproducibly repair NVIDIA's 0.8.0 ARM wheel metadata, never its native code.

The filename says aarch64 but the upstream WHEEL file incorrectly says sbsa.
A public-version-compatible local suffix makes this repair visible in uv.lock.
ZIP_STORED and fixed timestamps make the wheel hash independent of zlib versions.
"""

from __future__ import annotations

import base64
import csv
import hashlib
import io
import json
import urllib.request
import zipfile
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
UPSTREAM_URL = (
    "https://files.pythonhosted.org/packages/46/10/"
    "8dcd1175260706a2fc92a16a52e306b71d4c1ea0b0cc4a9484183399818a/"
    "nvidia_cusparselt_cu13-0.8.0-py3-none-manylinux2014_aarch64.whl"
)
UPSTREAM_SHA256 = "400c6ed1cf6780fc6efedd64ec9f1345871767e6a1a0a552a1ea0578117ea77c"
FILENAME = "nvidia_cusparselt_cu13-0.8.0+spark1-py3-none-manylinux2014_aarch64.whl"
EXPECTED_SHA256 = "3974cc125c2ad53997dd8384e60ff048f79b36422aa3fdde517afc3aa64d48c0"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    wheels = PROJECT / "wheels"
    wheels.mkdir(exist_ok=True)
    target = wheels / FILENAME
    if target.exists() and EXPECTED_SHA256 and sha256(target) == EXPECTED_SHA256:
        print("Verified cached compatibility wheel:", target)
        return
    upstream = wheels / "upstream.whl"
    if not upstream.exists() or sha256(upstream) != UPSTREAM_SHA256:
        urllib.request.urlretrieve(UPSTREAM_URL, upstream)
    assert sha256(upstream) == UPSTREAM_SHA256, "Upstream wheel checksum mismatch"
    old = "nvidia_cusparselt_cu13-0.8.0.dist-info/"
    new = "nvidia_cusparselt_cu13-0.8.0+spark1.dist-info/"
    records, native_hashes = [], {}
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(upstream) as source, zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_STORED) as output:
        for item in sorted(source.infolist(), key=lambda row: row.filename):
            if item.is_dir() or item.filename == old + "RECORD":
                continue
            data = source.read(item)
            name = item.filename.replace(old, new, 1) if item.filename.startswith(old) else item.filename
            if name == new + "WHEEL":
                assert b"Tag: py3-none-manylinux2014_sbsa" in data
                data = data.replace(b"Tag: py3-none-manylinux2014_sbsa", b"Tag: py3-none-manylinux2014_aarch64")
            elif name == new + "METADATA":
                assert b"\nVersion: 0.8.0\n" in data
                data = data.replace(b"\nVersion: 0.8.0\n", b"\nVersion: 0.8.0+spark1\n", 1)
            else:
                assert data == source.read(item), name
                if ".so" in name:
                    native_hashes[name] = hashlib.sha256(data).hexdigest()
            info = zipfile.ZipInfo(name, date_time=(2020, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = item.external_attr
            output.writestr(info, data)
            digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode()
            records.append((name, "sha256=" + digest, str(len(data))))
        buffer = io.StringIO(newline="")
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerows(sorted(records))
        writer.writerow((new + "RECORD", "", ""))
        info = zipfile.ZipInfo(new + "RECORD", date_time=(2020, 1, 1, 0, 0, 0))
        info.create_system = 3
        info.external_attr = 0o644 << 16
        output.writestr(info, buffer.getvalue().encode())
    temporary.replace(target)
    actual = sha256(target)
    if EXPECTED_SHA256:
        assert actual == EXPECTED_SHA256, "Repaired wheel differs from the locked artifact"
    with zipfile.ZipFile(target) as built:
        assert all(hashlib.sha256(built.read(name)).hexdigest() == digest for name, digest in native_hashes.items())
    report = dict(upstream_url=UPSTREAM_URL, upstream_sha256=UPSTREAM_SHA256, repaired_filename=FILENAME,
                  repaired_sha256=actual, native_library_sha256=native_hashes,
                  changes=["WHEEL platform tag sbsa -> aarch64", "visible local version +spark1", "RECORD hashes"],
                  native_library_bytes_unchanged=True)
    (wheels / "repair_provenance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
