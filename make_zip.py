"""Create graph-rag-analyzer.zip containing only the project's essential files.

Usage:  python make_zip.py

The archive is built from an explicit allowlist, so .venv, generated HTML
(graph.html), __pycache__, .pyc, .env and Git metadata can never end up in it.
"""

from __future__ import annotations

import sys
import zipfile
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ARCHIVE = PROJECT_DIR / "graph-rag-analyzer.zip"
INCLUDE = [
    "graph_engine.py",
    "extractor.py",
    "multi_hop.py",
    "app.py",
    "requirements.txt",
    "README.md",
]


def main() -> int:
    missing = [name for name in INCLUDE if not (PROJECT_DIR / name).is_file()]
    if missing:
        print(f"ERROR: missing required file(s): {', '.join(missing)}", file=sys.stderr)
        return 1

    ARCHIVE.unlink(missing_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zf:
        for name in INCLUDE:
            zf.write(PROJECT_DIR / name, arcname=name)

    with zipfile.ZipFile(ARCHIVE) as zf:
        if (bad := zf.testzip()) is not None:
            print(f"ERROR: corrupt entry in archive: {bad}", file=sys.stderr)
            return 1
        entries = zf.infolist()

    for info in entries:
        print(f"  added {info.filename:<18} {info.file_size:>8,} bytes")
    size = ARCHIVE.stat().st_size
    print(f"\nCreated {ARCHIVE.name}: {len(entries)} files, {size:,} bytes ({size / 1024:.1f} KB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
