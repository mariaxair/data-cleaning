"""Export selected cleaned Markdown languages into a separate final folder.

The source manifest is the authority for language and cleaning status. By
default, only records with status ``pass`` and claimed language ``ar``, ``fr``,
or ``en`` are copied. The Markdown body is copied byte-for-byte; no OCR,
translation, rewriting, or additional cleaning is performed.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path


LANGUAGES = {"ar", "fr", "en"}


def io_path(path: Path) -> Path:
    """Use the Windows extended path form for deep corpus paths."""
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/week4-md"),
                        help="Markdown export containing manifest.jsonl")
    parser.add_argument("--output", type=Path, default=Path("data/week4-md-final-ar-fr-en"),
                        help="Separate destination folder")
    parser.add_argument("--include-review", action="store_true",
                        help="Also copy records marked review; they remain flagged in the manifest")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()
    if input_root == output_root or input_root.is_relative_to(output_root):
        raise SystemExit("Output must be a separate folder outside the input folder")

    input_manifest = io_path(input_root / "manifest.jsonl")
    if not input_manifest.is_file():
        raise SystemExit(f"Missing manifest: {input_manifest}")
    io_path(output_root).mkdir(parents=True, exist_ok=True)

    allowed_statuses = {"pass", "review"} if args.include_review else {"pass"}
    counts: Counter[str] = Counter()
    language_counts: Counter[str] = Counter()
    records: list[dict] = []
    copied_paths: set[str] = set()

    with input_manifest.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            source_record = json.loads(line)
            language = source_record.get("claimed_language")
            status = source_record.get("status")
            if language not in LANGUAGES:
                counts["excluded_language"] += 1
                continue
            if status not in allowed_statuses:
                counts[f"excluded_status_{status or 'missing'}"] += 1
                continue

            source_value = source_record.get("markdown_path")
            if not source_value:
                counts["missing_markdown_path"] += 1
                continue
            source = io_path(Path(source_value))
            if not source.is_file():
                counts["missing_markdown_file"] += 1
                continue

            try:
                relative = Path(source).relative_to(io_path(input_root / "pass"))
            except ValueError:
                try:
                    relative = Path(source).relative_to(io_path(input_root / "review"))
                except ValueError:
                    counts["path_outside_input"] += 1
                    continue
            # Review and pass records can share the same source-relative name.
            # Keep review records in their own subtree when explicitly included.
            destination_relative = Path("review") / relative if status == "review" else relative
            destination = io_path(output_root / language / destination_relative)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination_key = str(destination).casefold()
            if destination_key in copied_paths:
                raise SystemExit(f"Destination collision at manifest line {line_number}: {destination}")
            copied_paths.add(destination_key)
            if destination.exists():
                if destination.read_bytes() != source.read_bytes():
                    raise SystemExit(f"Existing destination differs; use a new output folder: {destination}")
                counts["unchanged"] += 1
            else:
                shutil.copyfile(source, destination)
                counts["copied"] += 1
            result = dict(source_record)
            result["final_markdown_path"] = str(destination)
            result["selected_language"] = language
            result["selection_status"] = status
            records.append(result)
            language_counts[language] += 1

    manifest_path = io_path(output_root / "manifest.jsonl")
    temporary_manifest = io_path(output_root / "manifest.jsonl.tmp")
    with temporary_manifest.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    temporary_manifest.replace(manifest_path)

    summary = {
        "input": str(input_root),
        "output": str(output_root),
        "languages": sorted(LANGUAGES),
        "included_statuses": sorted(allowed_statuses),
        "records_exported": len(records),
        "language_counts": dict(language_counts),
        "selection_counts": dict(counts),
        "body_policy": "Markdown copied byte-for-byte from the cleaned Markdown export",
        "review_policy": "Review records are excluded by default; use --include-review to include them",
    }
    io_path(output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
