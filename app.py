"""Gradio GUI for extraction, optional training stages, and benchmarks."""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator

import gradio as gr

from config import DEFAULT_PROVIDER, LOG_LEVEL, PipelineConfig, provider_choices, provider_profile
from corpus_pipeline.logging_utils import configure_logging
from corpus_pipeline.ocr.registry import create_adapter
from corpus_pipeline.processor import process_batch

configure_logging(LOG_LEVEL)

_DEFAULTS = PipelineConfig()
STAGE_OUTPUT_DEFAULT = str(_DEFAULTS.stage_output_dir)
BENCHMARK_DIR_DEFAULT = str(_DEFAULTS.benchmark_dir)
REFERENCE_DIR_DEFAULT = str(_DEFAULTS.benchmark_reference_dir)
MANUAL_OVERRIDES_DEFAULT = str(_DEFAULTS.manual_overrides_path or "data/overrides.json")
LIVE_LOG_LINES = 400

# Malformed JSONL records raise KeyError/TypeError/AttributeError from stage internals;
# report those as stage failures instead of letting the GUI event crash.
STAGE_ERRORS = (ValueError, RuntimeError, OSError, KeyError, TypeError, AttributeError)

# Only these providers expose an OpenAI-compatible text endpoint for stages 3 and 5.
GENERATION_PROVIDER_IDS = {"openai", "heiyu", "mistral", "openai_compatible"}
GENERATION_PROVIDERS = [choice for choice in provider_choices() if choice[1] in GENERATION_PROVIDER_IDS]
GENERATION_DEFAULT = (
    DEFAULT_PROVIDER if DEFAULT_PROVIDER in GENERATION_PROVIDER_IDS
    else GENERATION_PROVIDERS[0][1]
)


def _provider_defaults(provider: str) -> tuple[str, str, str, str]:
    """Return base URL, model, API key, and a status line for one provider."""
    try:
        profile = provider_profile(provider)
    except ValueError as exc:
        return "", "", "", f"**{exc}**"
    key = os.environ.get(profile.api_key_env, "").strip()
    if key:
        status = (
            f"Key loaded from `{profile.api_key_env}` (masked, session-only, never logged). "
            "Edit it here to override for this session."
        )
    else:
        status = (
            f"No key found. Add `{profile.api_key_env}=...` to `.env`, "
            "or type a key here for this session only."
        )
    if profile.notes:
        status = f"{status}\n\n{profile.notes}"
    return profile.base_url, profile.model, key, status


def _generation_defaults(provider: str) -> tuple[str, str, str, str]:
    """Base URL, text model, key, and status for synthesis/LLM review."""
    try:
        profile = provider_profile(provider)
    except ValueError as exc:
        return "", "", "", f"**{exc}**"
    key = os.environ.get(profile.api_key_env, "").strip()
    model = os.environ.get("SYNTHESIS_MODEL", "").strip()
    status = (
        "Set a text-capable model explicitly. This is separate from the OCR vision model; "
        "the same provider key is used."
    )
    if not model:
        status += " `SYNTHESIS_MODEL` is empty, so live synthesis and LLM review will refuse to start."
    return profile.base_url, model, key, status


def _runtime_config(
    provider: str,
    key: str,
    base_url: str,
    model: str,
    *,
    synthesis_model: str = "",
    stage_output: str = "",
    manual_overrides: str | None = None,
) -> PipelineConfig:
    key = (key or "").strip()
    config = PipelineConfig(
        ocr_provider=provider,
        ocr_api_key=key,
        session_api_key_override=bool(key),
        ocr_base_url=base_url.strip(),
        ocr_model=model.strip(),
    )
    if synthesis_model and synthesis_model.strip():
        config.synthesis_model = synthesis_model.strip()
    if stage_output and stage_output.strip():
        config.stage_output_dir = Path(stage_output.strip()).expanduser()
    if manual_overrides:
        config.manual_overrides_path = Path(manual_overrides).expanduser()
    return config


def _test_connection(provider: str, key: str, base_url: str, model: str) -> str:
    try:
        result = create_adapter(_runtime_config(provider, key, base_url, model)).test_connection()
        return f"Connection successful. Provider: {result.provider}"
    except Exception as exc:
        return f"Connection failed: {exc}"


def _collect_paths(files: Any, folder: str, extensions: set[str]) -> list[Path]:
    """Combine uploads and a folder path, keeping supported files in a stable order."""
    if files is None:
        items: list[Any] = []
    elif isinstance(files, (str, Path)):
        items = [files]
    else:
        items = list(files)
    paths: list[Path] = [Path(item) for item in items if item]
    folder = (folder or "").strip()
    if folder:
        folder_path = Path(folder).expanduser()
        if not folder_path.is_dir():
            raise ValueError(f"Folder does not exist or is not a directory: {folder_path}")
        paths.extend(sorted(
            path for path in folder_path.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions))
    deduplicated = list(dict.fromkeys(path.resolve() for path in paths))
    unsupported = [path for path in deduplicated if path.suffix.lower() not in extensions]
    if unsupported:
        raise ValueError(f"Unsupported file type: {unsupported[0].name}")
    return deduplicated


def _single_path(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        value = value[0] if value else None
    return str(value) if value else ""


def _stage_output_path(value: str) -> Path:
    return Path((value or STAGE_OUTPUT_DEFAULT).strip()).expanduser()


def _concurrency_value(value: Any, default: int, name: str) -> int:
    """Interpret a GUI concurrency field; 0 means no limit."""
    if value in (None, ""):
        return default
    try:
        number = int(value)
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be an integer; use 0 for no limit.") from None
    if number < 0:
        raise ValueError(f"{name} cannot be negative; use 0 for no limit.")
    return number


def _stage_files(output: Path, names: tuple[str, ...]) -> list[str]:
    return [str(output / name) for name in names if (output / name).is_file()]


def _run(
    files: list[str] | None,
    folder: str,
    provider: str,
    key: str,
    base_url: str,
    model: str,
    file_concurrency: float | None = None,
    page_concurrency: float | None = None,
    ocr_concurrency: float | None = None,
    report: Callable[[str], None] | None = None,
) -> tuple[Any, ...]:
    try:
        paths = _collect_paths(files, folder, {".pdf", ".txt", ".md", ".docx"})
        file_workers = _concurrency_value(
            file_concurrency, _DEFAULTS.file_concurrency, "Documents at once")
        page_workers = _concurrency_value(
            page_concurrency, _DEFAULTS.page_concurrency, "Pages per document")
        ocr_workers = _concurrency_value(
            ocr_concurrency, _DEFAULTS.ocr_concurrency, "OCR calls in total")
    except (ValueError, OSError) as exc:
        return [], [], "", "", str(exc), []
    if not paths:
        return [], [], "", "", "Upload files or enter a folder path.", []
    config = _runtime_config(provider, key, base_url, model)
    config.file_concurrency = file_workers
    config.page_concurrency = page_workers
    config.ocr_concurrency = ocr_workers

    summary = process_batch(paths, config, progress=report)
    status_rows, page_rows, original, cleaned, outputs = [], [], [], [], []
    for result in summary["results"]:
        status_rows.append([Path(result["source"]).name, result.get("status", "failed"),
                            result.get("detected_type", ""), result.get("route_summary", ""),
                            "; ".join(result.get("warnings", [])) or result.get("error", "")])
        if result.get("output_path"):
            outputs.append(result["output_path"])
        if "original_text" in result:
            original.append(f"### {Path(result['source']).name}\n{result['original_text']}")
            cleaned.append(f"### {Path(result['source']).name}\n{result['cleaned_text']}")
        for page in result.get("pages", []):
            signals = page["signals"]
            page_rows.append([Path(result["source"]).name, page["page_number"], page["route"],
                              signals["extracted_characters"], signals["valid_character_ratio"],
                              signals["full_page_image_ratio"], "; ".join(page["warnings"]),
                              page["ocr_failed"]])
    summary_text = (f"{summary['files_processed']} processed, {summary['files_skipped']} skipped, "
                    f"{summary['files_failed']} failed; {summary['pages_routed_to_ocr']} pages routed to OCR; "
                    f"{summary['total_characters']} output characters. Provider: {summary['provider']}.")
    if summary.get("benchmark"):
        summary_text += f" Metrics history: {summary['benchmark']['history_path']}."
    return status_rows, page_rows, "\n\n".join(original), "\n\n".join(cleaned), summary_text, outputs


def _run_stream(
    files: list[str] | None,
    folder: str,
    provider: str,
    key: str,
    base_url: str,
    model: str,
    file_concurrency: float | None = None,
    page_concurrency: float | None = None,
    ocr_concurrency: float | None = None,
) -> Iterator[tuple[Any, ...]]:
    """Stream live extraction events, then yield the final tables and outputs."""
    updates: queue.Queue[tuple[str, Any]] = queue.Queue()
    outcome: dict[str, Any] = {}

    def notify(message: str) -> None:
        updates.put(("log", message))

    def work() -> None:
        try:
            result = _run(
                files, folder, provider, key, base_url, model,
                file_concurrency, page_concurrency, ocr_concurrency, notify)
            outcome["result"] = result
            updates.put(("result", result))
        except BaseException as exc:  # Keep the generator alive to report the failure.
            outcome["error"] = str(exc) or exc.__class__.__name__
        finally:
            updates.put(("done", None))

    worker = threading.Thread(target=work, daemon=True, name="gui-extraction")
    worker.start()

    lines: list[str] = []
    last_message = "Starting extraction"
    last_update = 0.0
    result: tuple[Any, ...] | None = None
    while result is None:
        batch: list[tuple[str, Any]] = []
        try:
            batch.append(updates.get(timeout=0.25))
            while True:
                try:
                    batch.append(updates.get_nowait())
                except queue.Empty:
                    break
        except queue.Empty:
            pass

        got_message = False
        for kind, payload in batch:
            if kind == "log":
                lines.append(f"[{time.strftime('%H:%M:%S')}] {payload}")
                last_message = payload
                got_message = True
            elif kind == "result":
                result = payload
            elif kind == "done" and "result" not in outcome:
                outcome.setdefault("error", "worker stopped without a result")
        if len(lines) > LIVE_LOG_LINES:
            lines = lines[-LIVE_LOG_LINES:]

        now = time.monotonic()
        if result is None and (got_message or now - last_update >= 1.0):
            yield ([], [], "", "", f"Running — {last_message}", [], "\n".join(lines))
            last_update = now
        if result is None and not worker.is_alive():
            break

    if result is None:
        error = outcome.get("error", "unknown error")
        result = ([], [], "", "", f"Extraction failed: {error}", [])
    yield (*result, "\n".join(lines))


def _run_synthesis(
    files: Any,
    folder: str,
    overrides_file: Any,
    overrides_path: str,
    max_samples: float | None,
    dry_run: bool,
    stage_output: str,
    provider: str,
    key: str,
    base_url: str,
    synthesis_model: str,
    screening_mode: str = "legal",
    topics_only: bool = False,
    topic_max_calls: float | None = None,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[Any, str, list[str]]:
    try:
        uploads = _collect_paths(files, "", {".txt", ".md"})
        paths = uploads or _collect_paths(None, folder, {".txt", ".md"})
        if not paths:
            return None, "Upload cleaned TXT/MD files or use the data/out folder.", []
        override = _single_path(overrides_file) or (overrides_path or "").strip() or None
        config = _runtime_config(
            provider, key, base_url, synthesis_model,
            synthesis_model=synthesis_model,
            stage_output=stage_output,
            manual_overrides=override,
        )
        config.synthesis_screening_mode = screening_mode
        if topic_max_calls is not None:
            config.topic_max_calls = int(topic_max_calls)
        from corpus_pipeline.synthesis.runner import synthesize
        limit = int(max_samples) if max_samples not in (None, "") else None
        result = synthesize(paths, config, dry_run=bool(dry_run), max_samples=limit,
                            topics_only=bool(topics_only),
                            progress=lambda message: progress((0, None), desc=message))
        output = _stage_output_path(stage_output) / "synthesis"
        artifacts = sorted(str(path) for path in output.glob("*") if path.is_file())
        if dry_run:
            status = (f"Dry run: {result.get('selected_units', 0)} of "
                      f"{result.get('eligible_units', 0)} eligible units selected; no model calls.")
        elif topics_only:
            status = (f"Discovered {result.get('topic_groups', 0)} topic groups; "
                      f"{result.get('topic_model_calls', 0)} model calls; no QA generation.")
        else:
            status = (f"Generated {result.get('generated', 0)} QA records; "
                      f"{result.get('failed', 0)} failed.")
        if screening_mode == "automatic":
            status += f" Theme details and group members: {output / 'topics.json'}."
            if result.get("topic_pending_documents", 0) or result.get("topic_ungrouped_documents", 0):
                status += (f" {result.get('topic_pending_documents', 0)} themes pending; "
                           f"{result.get('topic_ungrouped_documents', 0)} documents not yet grouped. "
                           "See topic warnings (dry run, model error, or call budget).")
        return result, status, artifacts
    except STAGE_ERRORS as exc:
        return None, f"Synthesis failed: {exc}", []


def _read_stage_inputs(files: Any, input_path: str) -> list[Path]:
    # Explicit uploads win over the prefilled path so a file is never processed twice.
    paths = _collect_paths(files, "", {".jsonl"})
    path_text = (input_path or "").strip()
    if not paths and path_text:
        explicit = Path(path_text).expanduser()
        if not explicit.is_file():
            raise ValueError(f"Input file does not exist: {explicit}")
        if explicit.suffix.lower() != ".jsonl":
            raise ValueError("Governance and review consume JSONL files.")
        paths.append(explicit.resolve())
    return list(dict.fromkeys(paths))


def _topic_group_rows(stage_output: str, mode: str) -> list[list[Any]]:
    """Show discovered group membership without further model requests."""
    if mode != "automatic":
        return []
    from corpus_pipeline.settings import load_json
    try:
        result = load_json(_stage_output_path(stage_output) / "synthesis" / "topics.json")
        return [[group["topic"], len(group["members"]),
                 "\n".join(member["source_file"] for member in group["members"])]
                for group in result.get("groups", [])]
    except STAGE_ERRORS:
        return []


def _run_governance(
    files: Any,
    input_path: str,
    stage_output: str,
) -> tuple[Any, str, list[str]]:
    try:
        paths = _read_stage_inputs(files, input_path)
        if not paths:
            return None, "Provide at least one QA JSONL file.", []
        from corpus_pipeline.artifacts import read_jsonl
        from corpus_pipeline.governance.runner import process_corpus
        records = [row for path in paths for row in read_jsonl(path)]
        config = PipelineConfig()
        config.stage_output_dir = _stage_output_path(stage_output)
        result = process_corpus(records, config)
        output = config.stage_output_dir / "governance"
        status = (f"{len(records)} records: {result.get('pass', 0)} pass, "
                  f"{result.get('manual', 0)} manual, {result.get('intercept', 0)} intercept, "
                  f"{result.get('duplicates', 0)} duplicates.")
        return result, status, _stage_files(
            output, ("summary.json", "pass.jsonl", "manual.jsonl", "intercept.jsonl", "duplicates.jsonl"))
    except STAGE_ERRORS as exc:
        return None, f"Governance failed: {exc}", []


def _run_review(
    files: Any,
    input_path: str,
    llm_review: bool,
    stage_output: str,
    provider: str,
    key: str,
    base_url: str,
    synthesis_model: str,
) -> tuple[Any, str, list[str]]:
    try:
        paths = _read_stage_inputs(files, input_path)
        if not paths:
            return None, "Provide at least one governed JSONL file.", []
        if llm_review and not (synthesis_model or "").strip():
            return None, "Set the text model in the generation settings before enabling LLM review.", []
        from corpus_pipeline.artifacts import read_jsonl
        from corpus_pipeline.review.runner import review_corpus
        records = [row for path in paths for row in read_jsonl(path)]
        config = _runtime_config(
            provider, key, base_url, synthesis_model,
            synthesis_model=synthesis_model,
            stage_output=stage_output,
        )
        config.review_llm_enabled = bool(llm_review)
        result = review_corpus(records, config)
        output = config.stage_output_dir / "review"
        mode = "model review/repair" if llm_review else "offline deterministic review"
        status = (f"{len(records)} records via {mode}: {result.get('accept', 0)} accepted, "
                  f"{result.get('reject', 0)} rejected, {result.get('manual', 0)} manual; "
                  f"{result.get('repair_attempts', 0)} repair attempts.")
        return result, status, _stage_files(
            output, ("summary.json", "accepted.jsonl", "rejected.jsonl", "manual.jsonl", "audit.jsonl"))
    except STAGE_ERRORS as exc:
        return None, f"Review failed: {exc}", []


def _load_benchmarks(
    benchmark_dir: str,
    reference_dir: str,
) -> tuple[list[list[Any]], str, str, Any, list[str]]:
    try:
        from corpus_pipeline.benchmark import load_comparison
        config = PipelineConfig(
            benchmark_dir=Path((benchmark_dir or "").strip() or BENCHMARK_DIR_DEFAULT),
            benchmark_reference_dir=Path((reference_dir or "").strip() or REFERENCE_DIR_DEFAULT),
        )
        comparison = load_comparison(config)
        rows = [[
            group["provider"],
            group["model"],
            group["documents"],
            group["documents_with_reference"],
            group["ocr_pages"],
            group["ocr_attempted"],
            group["ocr_failure_rate"],
            group["ocr_pages_per_minute"],
            group["mean_document_wall_s"],
            group["mean_valid_character_ratio"],
            group["mean_cer"],
            group["mean_cer_precision"],
            group["mean_cer_recall"],
            group["mean_cer_f1"],
            group["mean_wer"],
            group["mean_raw_cer"],
            group["mean_cer_improvement"],
        ] for group in comparison["groups"]]
        leader = comparison.get("lowest_mean_cer")
        if leader:
            leader_text = (
                f"**Lowest mean CER:** `{leader['provider']} / {leader['model']}` — "
                f"{leader['mean_cer']:.4f}"
            )
        else:
            leader_text = (
                "No reference-based CER yet. Add `data/reference/<source-stem>.txt` "
                "and rerun extraction with `--force`."
            )
        history = Path(comparison["history_path"])
        status = f"{comparison['documents']} saved document rows in {history}."
        if not rows:
            status += " No benchmark rows yet; run extraction first."
        downloads = [str(history)] if history.is_file() else []
        downloads += [str(config.benchmark_dir / name)
                      for name in ("comparison.json", "comparison.csv")
                      if (config.benchmark_dir / name).is_file()]
        return rows, status, leader_text, comparison, downloads
    except (OSError, ValueError) as exc:
        return [], f"Benchmark read failed: {exc}", "", None, []


with gr.Blocks(title="Corpus Processing Pipeline") as demo:
    gr.Markdown(
        "# Corpus processing pipeline\n"
        "Extract, clean, synthesize, govern, review, and compare OCR quality. "
        "Each stage has its own tab; use the text generation settings for Stages 3 and 5."
    )

    with gr.Accordion("Text generation settings for Synthesis and LLM Review", open=False):
        with gr.Row():
            gen_provider = gr.Dropdown(choices=GENERATION_PROVIDERS, value=GENERATION_DEFAULT,
                                       label="Provider", scale=1)
            gen_api_key = gr.Textbox(label="API key", type="password",
                                     placeholder="Loaded from .env", scale=1)
            gen_base_url = gr.Textbox(label="Base URL", scale=1)
            gen_model = gr.Textbox(label="Text model (SYNTHESIS_MODEL)", scale=1)
        gen_status = gr.Markdown("Set a text-capable model; the OCR vision model is not reused.")

    with gr.Tabs():
        with gr.Tab("1 · Extract & clean"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### OCR settings (session only)")
                    provider = gr.Dropdown(choices=provider_choices(), value=DEFAULT_PROVIDER,
                                           label="Provider")
                    api_key = gr.Textbox(label="API key", type="password",
                                         placeholder="Loaded from .env")
                    base_url = gr.Textbox(label="Base URL")
                    model = gr.Textbox(label="Model")
                    key_status = gr.Markdown("Select a provider to load its defaults.")
                    test_button = gr.Button("Test connection")
                    connection_status = gr.Textbox(label="Connection status", interactive=False)
                with gr.Column(scale=2):
                    uploads = gr.File(label="Upload files", file_count="multiple", type="filepath",
                                      file_types=[".pdf", ".txt", ".md", ".docx"])
                    folder = gr.Textbox(label="Or enter a folder path")
                    with gr.Accordion("Concurrency (0 = no limit)", open=False):
                        with gr.Row():
                            file_workers = gr.Number(
                                label="Documents at once", value=_DEFAULTS.file_concurrency,
                                precision=0, minimum=0)
                            page_workers = gr.Number(
                                label="Pages per document", value=_DEFAULTS.page_concurrency,
                                precision=0, minimum=0)
                            ocr_workers = gr.Number(
                                label="OCR calls in total", value=_DEFAULTS.ocr_concurrency,
                                precision=0, minimum=0)
                        gr.Markdown(
                            "Set **Pages per document** and **OCR calls in total** to `0` to send "
                            "every scanned page at once; set **Documents at once** to `0` to start "
                            "every file immediately. Large bursts can exhaust memory or trigger "
                            "provider rate limits."
                        )
                    run_button = gr.Button("Run extraction & cleaning", variant="primary")
                    overall_status = gr.Textbox(label="Progress and summary", interactive=False)
                    live_log = gr.Textbox(
                        label="Live activity", lines=16, max_lines=24,
                        interactive=False, autoscroll=True,
                        placeholder="Per-file and per-page events will appear here while the run is active.")
                    gr.Markdown(
                        "Benchmark metrics are written automatically. "
                        "Use the **Benchmarks** tab to compare providers and models."
                    )
            file_table = gr.Dataframe(
                headers=["File", "Status", "Detected type", "Route summary", "Warnings"],
                label="Per-file status", interactive=False, wrap=True)
            page_table = gr.Dataframe(
                headers=["File", "Page", "Route", "Characters", "Valid ratio",
                         "Image ratio", "Warnings", "OCR failed"],
                label="PDF page report", interactive=False, wrap=True)
            with gr.Row():
                original_preview = gr.Textbox(label="Extracted text", lines=20, max_lines=32)
                cleaned_preview = gr.Textbox(label="Cleaned output", lines=20, max_lines=32)
            downloads = gr.File(label="Download resulting text files", file_count="multiple")

        with gr.Tab("2 · Synthesize"):
            gr.Markdown(
                "Generate QA from any subject. Automatic mode discovers themes and groups related "
                "documents with the configured text model; no manual theme list is needed. "
                "Dry run makes no model calls; uncached themes remain pending. "
                "Extraction outputs are in `data/out` by default."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    synth_screening_mode = gr.Dropdown(
                        choices=[("Automatic themes · all domains", "automatic"),
                                 ("Legacy legal screening", "legal")],
                        value="automatic", label="Document screening")
                    synth_topics_only = gr.Checkbox(
                        label="Discover/group themes only (no QA generation)", value=False)
                    synth_topic_max_calls = gr.Number(
                        label="Maximum model calls for theme discovery and grouping",
                        value=_DEFAULTS.topic_max_calls, precision=0, minimum=1)
                    synth_uploads = gr.File(
                        label="Upload cleaned TXT/MD", file_count="multiple", type="filepath",
                        file_types=[".txt", ".md"])
                    synth_folder = gr.Textbox(
                        label="Or folder of cleaned files",
                        placeholder="e.g. data/out (leave blank when uploading)")
                    synth_overrides_file = gr.File(
                        label="Manual overrides JSON (optional upload)",
                        file_count="single", type="filepath", file_types=[".json"])
                    synth_overrides_path = gr.Textbox(
                        label="Or overrides path", value=MANUAL_OVERRIDES_DEFAULT)
                with gr.Column(scale=1):
                    synth_max_samples = gr.Number(label="Maximum units", value=10,
                                                  precision=0, minimum=1)
                    synth_dry_run = gr.Checkbox(label="Dry run (no model calls)", value=True)
                    synth_stage_output = gr.Textbox(label="Stage output folder",
                                                    value=STAGE_OUTPUT_DEFAULT)
                    synth_run = gr.Button("Run synthesis", variant="primary")
            synth_status = gr.Textbox(label="Synthesis summary", interactive=False)
            synth_summary = gr.JSON(label="Run details")
            synth_topic_groups = gr.Dataframe(
                headers=["Discovered theme", "Documents", "Member files"],
                label="Automatic theme groups", interactive=False, wrap=True)
            synth_downloads = gr.File(label="Synthesis artifacts", file_count="multiple")

        with gr.Tab("3 · Govern"):
            gr.Markdown(
                "Screen, deduplicate, and route QA candidates into pass, manual, intercept, "
                "and duplicate buckets. This stage is offline and makes no model calls."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    gov_uploads = gr.File(
                        label="Upload QA JSONL", file_count="multiple", type="filepath",
                        file_types=[".jsonl"])
                    gov_path = gr.Textbox(
                        label="Or JSONL path",
                        value=str(Path(STAGE_OUTPUT_DEFAULT) / "synthesis" / "qa.jsonl"))
                with gr.Column(scale=1):
                    gov_stage_output = gr.Textbox(label="Stage output folder",
                                                  value=STAGE_OUTPUT_DEFAULT)
                    gov_run = gr.Button("Run governance", variant="primary")
            gov_status = gr.Textbox(label="Governance summary", interactive=False)
            gov_summary = gr.JSON(label="Bucket counts")
            gov_downloads = gr.File(label="Governance artifacts", file_count="multiple")

        with gr.Tab("4 · Review"):
            gr.Markdown(
                "Score and route governed records. Offline review is deterministic; enable LLM "
                "review to use the text generation settings above."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    rev_uploads = gr.File(
                        label="Upload governed JSONL", file_count="multiple", type="filepath",
                        file_types=[".jsonl"])
                    rev_path = gr.Textbox(
                        label="Or JSONL path",
                        value=str(Path(STAGE_OUTPUT_DEFAULT) / "governance" / "pass.jsonl"))
                with gr.Column(scale=1):
                    rev_llm = gr.Checkbox(
                        label="Enable model review and repair", value=False,
                        info="Requires a text model in the generation settings.")
                    rev_stage_output = gr.Textbox(label="Stage output folder",
                                                  value=STAGE_OUTPUT_DEFAULT)
                    rev_run = gr.Button("Run review", variant="primary")
            rev_status = gr.Textbox(label="Review summary", interactive=False)
            rev_summary = gr.JSON(label="Decision counts")
            rev_downloads = gr.File(label="Review artifacts", file_count="multiple")

        with gr.Tab("5 · Benchmarks"):
            gr.Markdown(
                "Every extraction writes per-document metrics and appends to the history. "
                "This tab compares runs grouped by provider and model. "
                "Reference-based CER/WER requires a corrected `<source-stem>.txt` in the reference folder."
            )
            with gr.Row():
                bench_history_dir = gr.Textbox(label="Benchmark history folder",
                                               value=BENCHMARK_DIR_DEFAULT, scale=2)
                bench_reference_dir = gr.Textbox(label="Reference folder",
                                                 value=REFERENCE_DIR_DEFAULT, scale=2)
                bench_refresh = gr.Button("Refresh comparison", variant="primary", scale=1)
            bench_status = gr.Textbox(label="History status", interactive=False)
            bench_leader = gr.Markdown("Refresh to compare saved runs.")
            bench_table = gr.Dataframe(
                headers=["Provider", "Model", "Docs", "With ref", "OCR pages", "OCR attempts",
                         "Fail rate", "Pages/min", "Mean wall s", "Valid ratio",
                         "CER", "CER precision", "CER recall", "CER F1", "WER",
                         "Raw CER", "CER gain"],
                label="Provider/model comparison", interactive=False, wrap=True)
            bench_comparison = gr.JSON(label="Comparison details")
            bench_downloads = gr.File(label="Benchmark artifacts", file_count="multiple")

    provider_defaults = [base_url, model, api_key, key_status]
    provider.change(_provider_defaults, inputs=[provider], outputs=provider_defaults)
    demo.load(_provider_defaults, inputs=[provider], outputs=provider_defaults)
    test_button.click(_test_connection, inputs=[provider, api_key, base_url, model],
                      outputs=[connection_status])
    run_button.click(_run_stream, inputs=[uploads, folder, provider, api_key, base_url, model,
                                          file_workers, page_workers, ocr_workers],
                     outputs=[file_table, page_table, original_preview, cleaned_preview,
                              overall_status, downloads, live_log])

    generation_defaults = [gen_base_url, gen_model, gen_api_key, gen_status]
    gen_provider.change(_generation_defaults, inputs=[gen_provider], outputs=generation_defaults)
    demo.load(_generation_defaults, inputs=[gen_provider], outputs=generation_defaults)

    synth_run.click(
        _run_synthesis,
        inputs=[synth_uploads, synth_folder, synth_overrides_file, synth_overrides_path,
                synth_max_samples, synth_dry_run, synth_stage_output,
                gen_provider, gen_api_key, gen_base_url, gen_model,
                synth_screening_mode, synth_topics_only, synth_topic_max_calls],
        outputs=[synth_summary, synth_status, synth_downloads]).then(
            _topic_group_rows, inputs=[synth_stage_output, synth_screening_mode],
            outputs=[synth_topic_groups])

    gov_run.click(
        _run_governance,
        inputs=[gov_uploads, gov_path, gov_stage_output],
        outputs=[gov_summary, gov_status, gov_downloads])

    rev_run.click(
        _run_review,
        inputs=[rev_uploads, rev_path, rev_llm, rev_stage_output,
                gen_provider, gen_api_key, gen_base_url, gen_model],
        outputs=[rev_summary, rev_status, rev_downloads])

    bench_outputs = [bench_table, bench_status, bench_leader, bench_comparison, bench_downloads]
    bench_refresh.click(_load_benchmarks, inputs=[bench_history_dir, bench_reference_dir],
                        outputs=bench_outputs)
    demo.load(_load_benchmarks, inputs=[bench_history_dir, bench_reference_dir],
              outputs=bench_outputs)


if __name__ == "__main__":
    demo.queue().launch()
