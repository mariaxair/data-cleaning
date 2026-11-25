"""Copy one original document from each raw corpus category.

This is a sharing utility, separate from the cleaning and generation pipeline.
Categories are the first directory below each batch (for example,
``batch1/Misinformation``). Files are searched recursively, but the selected
file is copied byte-for-byte and its original relative path is recorded.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from collections import Counter
from pathlib import Path


# Prefer files someone can open and read when a category contains several types.
EXTENSION_PRIORITY = {
    ".txt": 0,
    ".md": 1,
    ".html": 2,
    ".htm": 2,
    ".json": 3,
    ".csv": 4,
    ".docx": 5,
    ".xlsx": 6,
    ".pdf": 7,
}
EXCLUDED_EXTENSIONS = {".rar", ".zip", ".7z", ".part"}
EXCLUDED_PARTS = {"_debug", ".git", ".venv", "__pycache__"}


def io_path(path: Path) -> Path:
    """Support deeply nested Windows paths without changing stored paths."""
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def safe_name(value: str) -> str:
    # Category names normally are already safe. This only protects the shared
    # output folder from separators and Windows-invalid control characters.
    value = value.replace("/", "_").replace("\\", "_").replace(":", "_")
    return "".join("_" if ord(char) < 32 or char in '<>"|?*' else char for char in value).strip(" .") or "category"


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be greater than zero")
    return number


def category_key(path: Path, root: Path) -> str:
    relative = path.relative_to(root)
    if len(relative.parts) < 2:
        return relative.parts[0] if relative.parts else "root"
    return "/".join(relative.parts[:2])


def eligible(path: Path, root: Path) -> bool:
    if not path.is_file() or path.suffix.casefold() in EXCLUDED_EXTENSIONS:
        return False
    if path.name.casefold() in {"index.csv", "index.json"}:
        return False
    relative = path.relative_to(root)
    if any(part.casefold() in EXCLUDED_PARTS for part in relative.parts):
        return False
    return path.suffix.casefold() in EXTENSION_PRIORITY


def choose(candidates: list[Path], root: Path) -> Path:
    return min(candidates, key=lambda path: (
        EXTENSION_PRIORITY[path.suffix.casefold()],
        path.stat().st_size == 0,
        path.relative_to(root).as_posix().casefold(),
    ))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=Path(r"C:\Users\acer\Desktop\zte"))
    parser.add_argument("--output", type=Path, default=Path("data/raw-category-examples"))
    parser.add_argument("--limit-categories", type=positive_int,
                        help="Copy only the first N categories alphabetically")
    args = parser.parse_args()
    source_root = args.source_root.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    source_io = io_path(source_root)
    output_io = io_path(output_root)
    if not source_io.is_dir():
        parser.error(f"Source folder does not exist: {source_root}")
    if output_io.exists() and any(output_io.iterdir()):
        parser.error("Output folder is not empty; choose a new --output folder")
    if output_io.is_relative_to(source_io):
        parser.error("Output folder must be outside the raw source folder")

    grouped: dict[str, list[Path]] = {}
    for path in sorted(source_io.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if eligible(path, source_io):
            grouped.setdefault(category_key(path, source_io), []).append(path)
    categories = sorted(grouped, key=str.casefold)
    if args.limit_categories is not None:
        categories = categories[:args.limit_categories]
    if not categories:
        parser.error("No supported documents found")

    output_io.mkdir(parents=True, exist_ok=True)
    records = []
    extension_counts = Counter()
    for category in categories:
        candidates = grouped[category]
        selected = choose(candidates, source_io)
        relative = selected.relative_to(source_io)
        category_folder = safe_name(category.replace("/", "__")) + "-" + hashlib.sha256(category.encode("utf-8")).hexdigest()[:8]
        destination = output_io / category_folder / selected.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(io_path(selected), destination)
        data = destination.read_bytes()
        record = {
            "category": category,
            "source_relative_path": relative.as_posix(),
            "example_filename": selected.name,
            "extension": selected.suffix.casefold(),
            "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "candidate_count": len(candidates),
            "selection": "readable extension priority, then nonempty file, then relative path",
            "example_path": str(destination.relative_to(output_io)).replace("\\", "/"),
        }
        records.append(record)
        extension_counts[selected.suffix.casefold()] += 1
        print(f"{category}: {relative.as_posix()}", flush=True)

    manifest = output_io / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n",
                        encoding="utf-8", newline="\n")
    summary = {
        "source_root": "not stored; see source_relative_path relative to the supplied source root",
        "output": str(output_root),
        "categories_found_with_supported_files": len(grouped),
        "categories_exported": len(records),
        "categories_without_supported_files": [],
        "extension_counts": dict(extension_counts),
        "selection_policy": "One byte-preserved original per batch/category; text formats preferred over binary formats",
        "pipeline": "Standalone sharing utility; no cleaning, OCR, metadata extraction or API calls",
    }
    (output_io / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                                             encoding="utf-8")
    (output_io / "README.md").write_text(
        "# Raw category examples\n\n"
        "One original file was copied from each top-level category under batch1 or batch2.\n"
        "The copied bytes are unchanged. `manifest.jsonl` stores relative source paths,\n"
        "not local absolute paths. This folder is a sharing sample, not a pipeline stage.\n",
        encoding="utf-8",
    )
    print(json.dumps({"categories_exported": len(records), "extension_counts": dict(extension_counts)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
