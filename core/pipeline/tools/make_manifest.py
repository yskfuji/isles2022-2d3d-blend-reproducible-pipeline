#!/usr/bin/env python3
"""Print a list of every tracked source file in this bundle (no data, no weights)."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    exts = {".py", ".yaml", ".yml", ".md", ".cff", ".json", ".txt"}
    files = []
    for p in sorted(ROOT.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(ROOT)
        if any(part.startswith(".") for part in rel.parts):
            continue
        if rel.parts and rel.parts[0] in {"runs", "results", "Datasets", "logs"}:
            continue
        if p.suffix not in exts:
            continue
        files.append(str(rel))
    print(json.dumps({"root": str(ROOT), "n_files": len(files), "files": files}, indent=2))


if __name__ == "__main__":
    main()
