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
    parser.add_argument("--metadata", action="store_true", help="Discover evidence-backed metadata from cleaned TXT/MD")
    parser.add_argument("--metadata-llm", action="store_true", help="Enable configured metadata model calls")
    parser.add_argument("--metadata-model", help="Explicit text model for metadata extraction")
    parser.add_argument("--metadata-max-calls", type=int, help="Metadata model-call budget")
    parser.add_argument("--with-metadata", action="store_true", help="Attach existing metadata references to QA provenance")
    parser.add_argument("--metadata-evaluate", type=Path, help="Evaluate metadata against annotated reference JSONL")
    parser.add_argument("--chunk-evaluate", type=Path, help="Evaluate saved chunk JSONL with deterministic checks and an optional LLM judge")
    parser.add_argument("--chunk-judge-model", "--quality-judge-model", dest="chunk_judge_model",
                        help="Explicit text model for quality judging (defaults to SYNTHESIS_MODEL)")
    parser.add_argument("--chunk-judge-max-calls", "--quality-judge-max-calls", dest="chunk_judge_max_calls",
                        type=int, default=100, help="Maximum LLM judge calls per evaluation (0 disables judging)")
    parser.add_argument("--metadata-quality-evaluate", type=Path,
                        help="Judge saved metadata observations and evidence in a documents.jsonl")
    parser.add_argument("--qa-quality-evaluate", type=Path,
                        help="Judge generated question/answer records against their source")
    parser.add_argument("--review-quality-evaluate", type=Path,
                        help="Independently judge review audit records against their source and QA")
    parser.add_argument("--screening-mode", choices=["legal", "automatic"],
                        help="Legacy legal screening or model-discovered themes in any domain")
    parser.add_argument("--discover-topics", action="store_true",
                        help="Discover and group document themes without generating QA")
    parser.add_argument("--topic-max-calls", type=int, help="Maximum topic discovery/matching model attempts")
    parser.add_argument("--process-corpus", action="store_true", help="Stage 4: govern QA JSONL")
    parser.add_argument("--review", action="store_true", help="Stage 5: review QA JSONL offline by default")
    parser.add_argument("--llm-review", action="store_true", help="Enable configured model review/repair")
    parser.add_argument("--dry-run", action="store_true", help="Plan synthesis/metadata without model calls or embeddings")
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
    evaluation_targets = [args.chunk_evaluate, args.metadata_evaluate, args.metadata_quality_evaluate,
                          args.qa_quality_evaluate, args.review_quality_evaluate]
    if sum(value is not None for value in evaluation_targets) > 1:
        parser.error("Select one evaluation target per invocation; each target writes its own report.")
    logger = configure_logging(LOG_LEVEL)
    config = PipelineConfig()
    if args.metadata_llm:
        config.metadata_llm_enabled = True
    if args.metadata_model is not None:
        config.metadata_model = args.metadata_model
    if args.metadata_max_calls is not None:
        config.metadata_max_calls = args.metadata_max_calls
    if args.with_metadata:
        config.synthesis_metadata_enabled = True
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
    if args.metadata_evaluate:
        from corpus_pipeline.benchmark import evaluate_metadata
        try:
            result = evaluate_metadata(config.stage_output_dir / "metadata" / "documents.jsonl",
                                       args.metadata_evaluate, config.stage_output_dir / "metadata" / "evaluation.json",
                                       min_precision=config.metadata_eval_min_precision,
                                       min_recall=config.metadata_eval_min_recall,
                                       min_examples=config.metadata_eval_min_examples)
        except (ValueError, OSError, KeyError, TypeError):
            parser.error("Metadata evaluation requires valid documents and annotated reference JSONL.")
        logger.info("metadata evaluation " + json.dumps(result))
        return 0
    if args.chunk_evaluate:
        from corpus_pipeline.benchmark import evaluate_chunk_quality
        judge_model = args.chunk_judge_model if args.chunk_judge_model is not None else config.synthesis_model
        try:
            result = evaluate_chunk_quality(
                args.chunk_evaluate, config.stage_output_dir / "synthesis" / "chunking" / "evaluation.json",
                config=config, judge_model=judge_model, max_calls=args.chunk_judge_max_calls)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError):
            parser.error("Chunk evaluation failed; provide a valid chunk JSONL, compatible provider credentials, and an explicit judge model.")
        logger.info("chunk evaluation " + json.dumps({key: value for key, value in result.items() if key != "results"}))
        return 0
    quality_requests = [("metadata", args.metadata_quality_evaluate), ("qa", args.qa_quality_evaluate),
                        ("review", args.review_quality_evaluate)]
    selected_quality = [(target, path) for target, path in quality_requests if path]
    if selected_quality:
        if len(selected_quality) != 1:
            parser.error("Select one quality evaluation target per invocation.")
        from corpus_pipeline import benchmark
        target, input_path = selected_quality[0]
        evaluator = {"metadata": benchmark.evaluate_metadata_quality,
                     "qa": benchmark.evaluate_qa_quality,
                     "review": benchmark.evaluate_review_quality}[target]
        output = config.stage_output_dir / "evaluation" / f"{target}-quality.json"
        judge_model = args.chunk_judge_model if args.chunk_judge_model is not None else config.synthesis_model
        try:
            result = evaluator(input_path, output, config=config, judge_model=judge_model,
                               max_calls=args.chunk_judge_max_calls)
        except (ValueError, RuntimeError, OSError, KeyError, TypeError):
            parser.error(f"{target.title()} quality evaluation failed; provide valid input and configured provider credentials/model.")
        logger.info(f"{target} quality evaluation " + json.dumps(
            {key: value for key, value in result.items() if key != "results"}))
        return 0
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
        ("metadata", args.metadata),
        ("process-corpus", args.process_corpus), ("review", args.review)) if enabled]
    if not requested:
        requested = [name for name, enabled in (
            ("synthesize", config.synthesis_enabled), ("process-corpus", config.corpus_processing_enabled),
            ("metadata", config.metadata_enabled),
            ("review", config.review_enabled)) if enabled]
    if len(requested) > 1:
        parser.error("Select one optional stage per invocation.")
    stage = requested[0] if requested else "extract"
    if args.dry_run and stage not in {"synthesize", "metadata"}:
        parser.error("--dry-run requires --synthesize or --metadata.")
    if (args.metadata_llm or args.metadata_model or args.metadata_max_calls is not None) and stage != "metadata":
        parser.error("Metadata model settings require --metadata.")
    if args.with_metadata and stage != "synthesize":
        parser.error("--with-metadata requires --synthesize.")
    if args.test_connection and stage == "metadata":
        parser.error("Metadata does not use the OCR connection test; use --metadata --dry-run to inspect its plan.")
    if args.llm_review and stage != "review":
        parser.error("--llm-review requires --review.")
    if (args.chunk_judge_model is not None or args.chunk_judge_max_calls != 100) and not (
            args.chunk_evaluate or selected_quality):
        parser.error("Judge settings require a chunk or quality evaluation target.")
    paths = [Path(path) for path in args.paths]
    folder = args.folder or (INPUT_DIR if not paths else None)
    if folder:
        extensions = {".jsonl"} if stage in {"process-corpus", "review"} else (
            {".txt", ".md"} if stage in {"synthesize", "metadata"} else {".pdf", ".txt", ".md", ".docx"})
        paths.extend(sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in extensions))
    if args.test_connection:
        result = create_adapter(config).test_connection()
        logger.info(f"ocr connection test succeeded provider={result.provider}")
        if not args.paths and not args.folder:
            return 0
    if not paths:
        parser.error("Pass at least one file or use --folder.")
    if stage == "metadata":
        from corpus_pipeline.metadata.runner import discover_metadata
        try:
            result = discover_metadata(paths, config, dry_run=args.dry_run)
        except (ValueError, RuntimeError, OSError):
            parser.error("Metadata could not start; check METADATA_MODEL, provider credentials, configuration, and output permissions.")
        logger.info(f"metadata processed={result.get('processed', len(result.get('documents', [])))} "
                    f"model_calls={result.get('model_calls', 0)} dry_run={args.dry_run}")
        return 1 if result.get("failed") else 0
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
