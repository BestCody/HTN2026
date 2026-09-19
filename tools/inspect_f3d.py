"""Inspect a Fusion 360 F3D archive without modifying the source file."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import zipfile
from pathlib import Path

PRINTABLE = re.compile(rb"[\x20-\x7e]{4,}")


def validate_members(archive: zipfile.ZipFile, destination: Path) -> None:
    root = destination.resolve()
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if target != root and root not in target.parents:
            raise ValueError(f"Unsafe archive member: {member.filename}")


def safe_extract(source: Path, archive: zipfile.ZipFile, destination: Path) -> None:
    validate_members(archive, destination)
    try:
        archive.extractall(destination)
    except NotImplementedError:
        # Recent F3D files use ZIP method 93 (Zstandard), which the Python 3.12
        # standard library cannot decode. Windows' bsdtar supports it.
        subprocess.run(
            ["tar", "-xf", str(source.resolve()), "-C", str(destination.resolve())],
            check=True,
        )


def inspect(source: Path, destination: Path) -> dict[str, object]:
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(source) as archive:
        safe_extract(source, archive, destination)
        members = [
            {
                "path": entry.filename,
                "compressed_bytes": entry.compress_size,
                "bytes": entry.file_size,
            }
            for entry in archive.infolist()
            if not entry.is_dir()
        ]

    strings: dict[str, list[str]] = {}
    for path in destination.rglob("*"):
        if not path.is_file():
            continue
        found = []
        for value in PRINTABLE.findall(path.read_bytes()):
            text = value.decode("ascii", errors="ignore").strip()
            if text and text not in found:
                found.append(text)
        if found:
            strings[str(path.relative_to(destination))] = found

    report = {
        "source": str(source.resolve()),
        "source_bytes": source.stat().st_size,
        "members": members,
        "ascii_strings": strings,
    }
    (destination / "inspection.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    report = inspect(args.source, args.destination)
    print(f"Extracted {len(report['members'])} files")
    print(f"Inspection report: {(args.destination / 'inspection.json').resolve()}")


if __name__ == "__main__":
    main()
