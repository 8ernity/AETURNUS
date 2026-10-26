"""Build a source package from tracked files only.

``git archive`` is not used because the package must be reproducible from the
working tree while still excluding runtime state.  ``git ls-files -z`` gives the
exact tracked inventory, and a post-write check rejects any SQLite state file.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.common.utils import get_root_dir

DB_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal")
DEFAULT_OUTPUT = "AETURNUS_source.zip"


def tracked_files(root: Path) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=str(root),
        capture_output=True,
        check=True,
    )
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    return [root / name for name in names]


def build_package(output: Path, root: Path | None = None) -> dict:
    root_dir = root or get_root_dir()
    output_path = output if output.is_absolute() else root_dir / output
    files = tracked_files(root_dir)
    db_files = [str(path.relative_to(root_dir)) for path in files if path.name.endswith(DB_SUFFIXES)]
    if db_files:
        raise RuntimeError(
            "Refusing to package tracked SQLite state. Untrack or remove: " + ", ".join(db_files)
        )

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if not path.is_file():
                continue
            archive.write(path, arcname=f"AETURNUS/{path.relative_to(root_dir).as_posix()}")

    with zipfile.ZipFile(output_path, "r") as archive:
        packaged = archive.namelist()
    bad = [name for name in packaged if name.endswith(DB_SUFFIXES)]
    if bad:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Package contains SQLite state: {bad}")

    return {
        "output": str(output_path),
        "tracked_files": len(files),
        "packaged_files": len(packaged),
        "db_files": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output zip path")
    args = parser.parse_args()
    result = build_package(Path(args.output))
    print(
        f"Package OK: {result['packaged_files']} tracked files, "
        f"no DB state -> {result['output']}"
    )


if __name__ == "__main__":
    main()
