"""Command-line entry point."""

import argparse
import json
from pathlib import Path

from config import INPUT_DIR, LOG_LEVEL, PipelineConfig
from corpus_pipeline.logging_utils import configure_logging
from corpus_pipeline.ocr.registry import create_adapter
from corpus_pipeline.processor import process_batch


def main() -> int:
    """Run the configured pipeline over files and/or an input folder."""
    parser = argparse.ArgumentParser(description="Extract and clean corpus documents.")
    parser.add_argument("paths", nargs="*", help="PDF, TXT, MD, or DOCX files")
    parser.add_argument("--folder", type=Path, help="Process supported files in a folder")
    parser.add_argument("--force", action="store_true", help="Reprocess files even if complete")
    parser.add_argument("--test-connection", action="store_true", help="Check the configured OCR endpoint")
    args = parser.parse_args()
    logger = configure_logging(LOG_LEVEL)
    config = PipelineConfig()
    paths = [Path(path) for path in args.paths]
    folder = args.folder or (INPUT_DIR if not paths else None)
    if folder:
        paths.extend(sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}))
    if args.test_connection:
        result = create_adapter(config).test_connection()
        logger.info(f"ocr connection test succeeded provider={result.provider}")
        if not args.paths and not args.folder:
            return 0
    if not paths:
        parser.error("Pass at least one file or use --folder.")
    summary = process_batch(paths, config, force=args.force)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    (config.output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(
        f"batch complete processed={summary['files_processed']} skipped={summary['files_skipped']} "
        f"failed={summary['files_failed']} ocr_pages={summary['pages_routed_to_ocr']} "
        f"provider={summary['provider']} characters={summary['total_characters']}"
    )
    return 1 if summary["files_failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
