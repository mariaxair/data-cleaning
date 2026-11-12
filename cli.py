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
    parser.add_argument("--clean", action="store_true", help="Enable Stage 2.2 semantic corrections")
    parser.add_argument("--rules-report", type=Path, help="Write aggregated cleaning hit counters")
    parser.add_argument("--synthesize", action="store_true", help="Stage 3: synthesize QA from cleaned TXT/MD")
    parser.add_argument("--screening-mode", choices=["legal", "automatic"],
                        help="Legacy legal screening or model-discovered themes in any domain")
    parser.add_argument("--discover-topics", action="store_true",
                        help="Discover and group document themes without generating QA")
    parser.add_argument("--topic-max-calls", type=int, help="Maximum topic discovery/matching model attempts")
    parser.add_argument("--process-corpus", action="store_true", help="Stage 4: govern QA JSONL")
    parser.add_argument("--review", action="store_true", help="Stage 5: review QA JSONL offline by default")
    parser.add_argument("--llm-review", action="store_true", help="Enable configured model review/repair")
    parser.add_argument("--dry-run", action="store_true", help="Plan synthesis without model calls")
    parser.add_argument("--max-samples", type=int, help="Maximum synthesis units scheduled")
    parser.add_argument("--stage-output", type=Path, help="Optional stages' output root")
    parser.add_argument("--manual-overrides", type=Path, help="Document tier/metadata overrides JSON")
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction, default=None,
                        help="Write per-document metrics and history (default: on)")
    parser.add_argument("--benchmark-dir", type=Path, help="Metrics history and comparison folder")
    parser.add_argument("--reference-dir", type=Path, help="Folder of <source-stem>.txt references")
    parser.add_argument("--benchmark-report", type=Path, help="Write the comparison report here")
    parser.add_argument("--compare-benchmarks", action="store_true",
                        help="Compare saved benchmark history and exit")
    args = parser.parse_args()
    logger = configure_logging(LOG_LEVEL)
    config = PipelineConfig()
    if args.screening_mode:
        config.synthesis_screening_mode = args.screening_mode
    if args.discover_topics:
        config.synthesis_screening_mode = "automatic"
    if args.topic_max_calls is not None:
        config.topic_max_calls = args.topic_max_calls
    if args.clean:
        config.semantic_rules_enabled = True
    if args.stage_output:
        config.stage_output_dir = args.stage_output
    if args.manual_overrides:
        config.manual_overrides_path = args.manual_overrides
    if args.llm_review:
        config.review_llm_enabled = True
    if args.benchmark is not None:
        config.benchmark_enabled = args.benchmark
    if args.benchmark_dir:
        config.benchmark_dir = args.benchmark_dir
    if args.reference_dir:
        config.benchmark_reference_dir = args.reference_dir
    if args.compare_benchmarks:
        from corpus_pipeline.artifacts import write_json
        from corpus_pipeline.benchmark import load_comparison, write_comparison
        comparison = load_comparison(config)
        if args.benchmark_report:
            write_json(args.benchmark_report, comparison)
        else:
            write_comparison(config, comparison)
        logger.info(
            f"benchmark comparison documents={comparison['documents']} "
            f"groups={len(comparison['groups'])}")
        return 0
    requested = [name for name, enabled in (
        ("synthesize", args.synthesize or args.discover_topics),
        ("process-corpus", args.process_corpus), ("review", args.review)) if enabled]
    if not requested:
        requested = [name for name, enabled in (
            ("synthesize", config.synthesis_enabled), ("process-corpus", config.corpus_processing_enabled),
            ("review", config.review_enabled)) if enabled]
    if len(requested) > 1:
        parser.error("Select one optional stage per invocation.")
    stage = requested[0] if requested else "extract"
    if args.dry_run and stage != "synthesize":
        parser.error("--dry-run requires --synthesize.")
    if args.llm_review and stage != "review":
        parser.error("--llm-review requires --review.")
    paths = [Path(path) for path in args.paths]
    folder = args.folder or (INPUT_DIR if not paths else None)
    if folder:
        extensions = {".jsonl"} if stage in {"process-corpus", "review"} else (
            {".txt", ".md"} if stage == "synthesize" else {".pdf", ".txt", ".md", ".docx"})
        paths.extend(sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions))
    if args.test_connection:
        result = create_adapter(config).test_connection()
        logger.info(f"ocr connection test succeeded provider={result.provider}")
        if not args.paths and not args.folder:
            return 0
    if not paths:
        parser.error("Pass at least one file or use --folder.")
    if stage == "review":
        from corpus_pipeline.artifacts import read_jsonl
        from corpus_pipeline.review.runner import review_corpus
        try:
            result = review_corpus([row for path in paths for row in read_jsonl(path)], config)
        except (ValueError, RuntimeError, OSError) as exc:
            parser.error(str(exc))
        logger.info("corpus review " + json.dumps(result))
        return 0
    if stage == "process-corpus":
        from corpus_pipeline.artifacts import read_jsonl
        from corpus_pipeline.governance.runner import process_corpus
        try:
            result = process_corpus([row for path in paths for row in read_jsonl(path)], config)
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        logger.info("corpus governance " + json.dumps(result))
        return 0
    if stage == "synthesize":
        from corpus_pipeline.synthesis.runner import synthesize
        try:
            result = synthesize(paths, config, dry_run=args.dry_run, max_samples=args.max_samples,
                                topics_only=args.discover_topics)
        except (ValueError, RuntimeError, OSError) as exc:
            # Configuration errors have safe messages; never expose arbitrary provider response bodies.
            parser.error(str(exc))
        logger.info(
            f"synthesis dry_run={args.dry_run} selected={result.get('selected_units', result.get('selected', 0))} "
            f"failed={result.get('failed', 0)}")
        return 1 if result.get("failed") else 0
    summary = process_batch(paths, config, force=args.force)
    if args.benchmark_report:
        from corpus_pipeline.artifacts import write_json
        from corpus_pipeline.benchmark import load_comparison
        write_json(args.benchmark_report, load_comparison(config))
    if args.rules_report:
        from collections import Counter
        hits = Counter()
        for result in summary["results"]:
            pages = result.get("pages")
            if pages is None and result.get("report_path"):
                pages = json.loads(Path(result["report_path"]).read_text(encoding="utf-8")).get("pages", [])
            for page in pages or []:
                hits.update(page.get("cleaning_hits", {}))
        args.rules_report.parent.mkdir(parents=True, exist_ok=True)
        args.rules_report.write_text(json.dumps(dict(hits), indent=2), encoding="utf-8")
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
