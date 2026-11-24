"""Gradio GUI for extraction, optional training stages, and benchmarks."""

import os
import math
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
STAGE_OUTPUT_DEFAULT = "data"  # GUI root; legacy CLI defaults are unchanged.
BENCHMARK_DIR_DEFAULT = str(_DEFAULTS.benchmark_dir)
REFERENCE_DIR_DEFAULT = str(_DEFAULTS.benchmark_reference_dir)
MANUAL_OVERRIDES_DEFAULT = str(_DEFAULTS.manual_overrides_path or "")
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

# Advanced settings are grouped by the stage that consumes them. The defaults
# come from PipelineConfig (and therefore still honor .env); GUI edits only
# override the current run.
STAGE_SETTINGS = {
    "generation": [
        ("ocr_timeout_s", "int", "Request timeout (seconds)"),
        ("ocr_max_retries", "nonnegative_int", "HTTP retry attempts"),
        ("ocr_retry_base_s", "float", "Initial retry delay (seconds)"),
    ],
    "benchmarks": [
        ("benchmark_dir", "path", "Benchmark history folder"),
        ("benchmark_reference_dir", "path", "Corrected reference text folder"),
    ],
    "extract": [
        ("output_dir", "path", "Extraction output folder"),
        ("cache_db", "path", "OCR cache database"),
        ("include_page_markers", "bool", "Include page markers in extracted text"),
        ("page_marker_template", "text", "Page marker template (use {page} for the page number)"),
        ("languages", "languages", "OCR language hints"),
        ("semantic_rules_enabled", "bool", "Apply deterministic text corrections"),
        ("rules_data_path", "path", "Correction rules file"),
        ("wrong_words_path", "path", "Word corrections file"),
        ("render_dpi", "int", "PDF render DPI"),
        ("ocr_timeout_s", "int", "OCR request timeout (seconds)"),
        ("ocr_max_retries", "nonnegative_int", "OCR retry attempts"),
        ("ocr_retry_base_s", "float", "Retry delay (seconds)"),
        ("min_extracted_chars", "nonnegative_int", "Minimum extracted characters before OCR"),
        ("min_valid_char_ratio", "float01", "Minimum valid-character ratio"),
        ("full_page_image_ratio", "float01", "Full-page image threshold"),
        ("benchmark_enabled", "bool", "Save extraction benchmark metrics"),
        ("benchmark_strip_diacritics", "bool", "Ignore Arabic diacritics in comparison"),
        ("benchmark_unify_alef", "bool", "Unify Alef variants in comparison"),
        ("benchmark_unify_ya", "bool", "Unify Ya variants in comparison"),
        ("benchmark_max_eval_chars", "int", "Maximum benchmark characters per document"),
    ],
    "synthesis": [
        ("synthesis_language", "text", "Generated QA language"),
        ("synthesis_concurrency", "int", "QA units generated concurrently"),
        ("synthesis_max_attempts", "int", "Attempts per QA unit"),
        ("synthesis_max_tokens", "int", "Maximum generation tokens"),
        ("topic_sample_chars", "int", "Characters sampled per document"),
        ("topic_group_batch_size", "int", "Existing theme groups compared per request"),
        ("topic_similarity_threshold", "float01", "Theme similarity threshold"),
        ("general_chunk_chars", "int", "Text chunk size (characters)"),
        ("article_group_size", "choice_int:2|3|4", "Articles per QA unit"),
        ("min_article_chars", "int", "Minimum article length"),
        ("screening_path", "path", "Screening rules file"),
        ("synthesis_metadata_enabled", "bool", "Attach metadata references to QA provenance"),
    ],
    "metadata": [
        ("metadata_semantic_threshold", "float01", "Semantic boundary threshold"),
        ("metadata_chunk_tokens", "int", "Maximum tokens per semantic chunk"),
        ("metadata_chunk_chars", "int", "Maximum characters per chunk"),
        ("metadata_similarity_window", "int", "Semantic similarity window"),
        ("metadata_delimiters", "delimiters", "Sentence delimiters (one per line; use \\n for newline)"),
        ("metadata_include_delim", "choice:prev|next", "Keep delimiter with previous or next chunk"),
        ("metadata_max_attempts", "int", "Metadata extraction attempts"),
        ("metadata_max_tokens", "int", "Maximum metadata response tokens"),
        ("metadata_match_threshold", "float01", "Field match threshold"),
        ("metadata_match_margin", "float01", "Minimum field-match margin"),
        ("metadata_promotion_documents", "int", "Documents required to promote a field"),
        ("metadata_min_confidence", "float01", "Minimum confidence for field promotion"),
    ],
    "metadata_chunking": [
        ("metadata_semantic_threshold", "float01", "Semantic boundary threshold"),
        ("metadata_chunk_tokens", "int", "Maximum tokens per chunk"),
        ("metadata_chunk_chars", "int", "Maximum characters per chunk"),
        ("metadata_similarity_window", "int", "Semantic similarity window"),
        ("metadata_delimiters", "delimiters", "Sentence delimiters (one per line; use \\n for newline)"),
        ("metadata_include_delim", "choice:prev|next", "Keep delimiter with previous or next chunk"),
    ],
    "metadata_extraction": [
        ("metadata_max_attempts", "int", "Metadata extraction attempts"),
        ("metadata_max_tokens", "int", "Maximum metadata response tokens"),
        ("metadata_match_threshold", "float01", "Field match threshold"),
        ("metadata_match_margin", "float01", "Minimum field-match margin"),
        ("metadata_promotion_documents", "int", "Documents required to promote a field"),
        ("metadata_min_confidence", "float01", "Minimum confidence for field promotion"),
    ],
    "governance": [
        ("language_mixing_threshold", "float01", "Language mixing threshold"),
        ("dialect_backend", "choice:lexicon|marbert", "Dialect detection method"),
        ("dialect_model_path", "optional_path", "Dialect model path (for model method)"),
        ("dialect_lexicon_path", "path", "Dialect lexicon file"),
        ("dialect_pass_threshold", "float01", "Dialect pass threshold"),
        ("dialect_intercept_threshold", "float01", "Dialect interception threshold"),
        ("dialect_degraded_intercept_threshold", "float01", "Degraded-model interception threshold"),
        ("dialect_validated", "bool", "Mark dialect model as validated"),
    ],
    "review": [
        ("review_max_attempts", "nonnegative_int", "Maximum targeted repair attempts"),
        ("synthesis_max_tokens", "int", "Maximum review / repair response tokens"),
    ],
    "evaluation": [
        ("metadata_eval_min_precision", "float01", "Evaluation minimum precision"),
        ("metadata_eval_min_recall", "float01", "Evaluation minimum recall"),
        ("metadata_eval_min_examples", "int", "Evaluation minimum examples"),
    ],
}

# The OCR tab exposes only settings used before Cleaning. The legacy extract
# schema remains available to existing callback callers.
STAGE_SETTINGS["ocr"] = [item for item in STAGE_SETTINGS["extract"] if item[0] not in {
    "output_dir", "semantic_rules_enabled", "rules_data_path", "wrong_words_path"}]
STAGE_SETTINGS["cleaning"] = [item for item in STAGE_SETTINGS["extract"] if item[0] in {
    "languages", "include_page_markers", "page_marker_template", "semantic_rules_enabled",
    "rules_data_path", "wrong_words_path"}]
STAGE_SETTINGS["agents"] = [item for item in STAGE_SETTINGS["synthesis"] if item[0] in {
    "synthesis_language", "synthesis_concurrency", "synthesis_max_attempts", "synthesis_max_tokens",
    "min_article_chars"}]


def _setting_default(config: PipelineConfig, name: str, kind: str) -> Any:
    value = getattr(config, name)
    if kind == "delimiters":
        return "\n".join(item.replace("\n", "\\n") for item in value)
    if kind == "languages":
        return value
    if kind == "optional_path":
        return str(value) if value is not None else ""
    return str(value) if kind == "path" and value is not None else value


def _setting_controls(stage: str, *, title: str = "Advanced pipeline settings") -> list[Any]:
    """Create an advanced-settings panel and return its controls in schema order."""
    controls = []
    with gr.Accordion(title, open=False):
        for name, kind, label in STAGE_SETTINGS[stage]:
            value = _setting_default(_DEFAULTS, name, kind)
            if kind == "bool":
                control = gr.Checkbox(label=label, value=value)
            elif kind in {"int", "nonnegative_int"}:
                control = gr.Number(label=label, value=value, precision=0, minimum=1 if kind == "int" else 0)
            elif kind in {"float", "float01"}:
                control = gr.Number(label=label, value=value, minimum=0, maximum=1 if kind == "float01" else None)
            elif kind.startswith(("choice:", "choice_int:")):
                choices = kind.split(":", 1)[1].split("|")
                if kind.startswith("choice_int:"):
                    choices = [int(item) for item in choices]
                control = gr.Dropdown(choices=choices, value=value, label=label)
            elif kind == "languages":
                control = gr.Dropdown(multiselect=True, allow_custom_value=True,
                    choices=[("Arabic", "ar"), ("French", "fr"), ("English", "en")],
                    value=value, label=label)
            else:
                control = gr.Textbox(label=label, value=value)
            controls.append(control)
    return controls


def _apply_stage_settings(config: PipelineConfig, stage: str, values: tuple[Any, ...]) -> None:
    """Apply GUI settings to one run without modifying process environment."""
    if not values:
        return
    schema = STAGE_SETTINGS[stage]
    shared = (STAGE_SETTINGS["benchmarks"] if stage == "extract" else
              STAGE_SETTINGS["governance"] + STAGE_SETTINGS["generation"] if stage == "review" else
              STAGE_SETTINGS["generation"] if stage in {"synthesis", "metadata", "agents"} else [])
    if shared and len(values) == len(schema) + len(shared):
        schema = schema + shared
    if len(values) != len(schema):
        raise ValueError(f"Invalid {stage} settings: expected {len(schema)} values, got {len(values)}.")
    converted = {}
    for (name, kind, label), value in zip(schema, values):
        if kind == "bool":
            value = bool(value)
        elif kind in {"int", "nonnegative_int"}:
            number = float(value)
            minimum = 1 if kind == "int" else 0
            if not math.isfinite(number) or not number.is_integer() or number < minimum:
                raise ValueError(f"{label}: enter an integer >= {minimum}.")
            value = int(number)
        elif kind in {"float", "float01"}:
            value = float(value)
            if not math.isfinite(value) or value < 0 or (kind == "float01" and value > 1):
                raise ValueError(f"{label}: enter a finite number " + ("between 0 and 1." if kind == "float01" else ">= 0."))
        elif kind.startswith(("choice:", "choice_int:")):
            choices = kind.split(":", 1)[1].split("|")
            if str(value) not in choices:
                raise ValueError(f"{label}: choose one of {', '.join(choices)}.")
            if kind.startswith("choice_int:"):
                value = int(value)
        elif kind == "path":
            if not str(value or "").strip():
                raise ValueError(f"{label}: enter a path.")
            value = Path(str(value).strip()).expanduser()
        elif kind == "optional_path":
            value = Path(str(value).strip()).expanduser() if str(value or "").strip() else None
        elif kind == "delimiters":
            value = [item.replace("\\n", "\n") for item in str(value).splitlines() if item]
        elif kind == "languages":
            value = list(value or [])
            if not value:
                raise ValueError("Select at least one language hint.")
        if name == "page_marker_template":
            from string import Formatter
            if any(field not in {None, "page"} for _, field, _, _ in Formatter().parse(value)):
                raise ValueError("The page marker template supports only {page}.")
            value.format(page=1)
        converted[name] = value
    for name, value in converted.items():
        setattr(config, name, value)


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
    synthesis_model: str | None = None,
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
    if synthesis_model is not None:
        config.synthesis_model = synthesis_model.strip()
    if stage_output and stage_output.strip():
        config.stage_output_dir = Path(stage_output.strip()).expanduser()
    if manual_overrides is not None:
        config.manual_overrides_path = Path(manual_overrides).expanduser() if manual_overrides.strip() else None
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
    force: bool = False,
    test_connection: bool = False,
    rules_report: bool = False,
    *advanced_settings: Any,
    ocr_only: bool = False,
) -> tuple[Any, ...]:
    try:
        paths = _collect_paths(files, folder, {".pdf", ".txt", ".md", ".docx"})
        file_workers = _concurrency_value(
            file_concurrency, _DEFAULTS.file_concurrency, "Documents at once")
        page_workers = _concurrency_value(
            page_concurrency, _DEFAULTS.page_concurrency, "Pages per document")
        ocr_workers = _concurrency_value(
            ocr_concurrency, _DEFAULTS.ocr_concurrency, "OCR calls in total")
        config = _runtime_config(provider, key, base_url, model)
        _apply_stage_settings(config, "extract", advanced_settings)
    except (ValueError, OSError, TypeError) as exc:
        return [], [], "", "", str(exc), []
    if not paths:
        return [], [], "", "", "Upload files or enter a folder path.", []
    config.file_concurrency = file_workers
    config.page_concurrency = page_workers
    config.ocr_concurrency = ocr_workers

    if test_connection:
        try:
            create_adapter(config).test_connection()
        except Exception:
            return [], [], "", "", "OCR connection test failed; check the provider, key, URL and model.", []
    options = {"apply_cleaning": False} if ocr_only else {}
    summary = (process_batch(paths, config, force=True, progress=report, **options) if force else
               process_batch(paths, config, progress=report, **options))
    status_rows, page_rows, original, cleaned, outputs = [], [], [], [], []
    for result in summary["results"]:
        status_rows.append([Path(result["source"]).name, result.get("status", "failed"),
                            result.get("detected_type", ""), result.get("route_summary", ""),
                            "; ".join(result.get("warnings", [])) or result.get("error", "")])
        if result.get("output_path"):
            outputs.append(result["output_path"])
        if result.get("report_path") and Path(result["report_path"]).is_file():
            outputs.append(result["report_path"])
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
    if rules_report:
        from collections import Counter
        from corpus_pipeline.artifacts import write_json
        from corpus_pipeline.settings import load_json
        counts = Counter()
        for result in summary["results"]:
            pages = result.get("pages")
            if pages is None and result.get("report_path"):
                pages = load_json(Path(result["report_path"])).get("pages", [])
            for page in pages or []:
                counts.update(page.get("cleaning_hits", {}))
        report_path = config.output_dir / "rules-report.json"
        write_json(report_path, dict(counts))
        outputs.append(str(report_path))
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
    force: bool = False,
    test_connection: bool = False,
    rules_report: bool = False,
    *advanced_settings: Any,
    ocr_only: bool = False,
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
                file_concurrency, page_concurrency, ocr_concurrency, notify,
                force, test_connection, rules_report,
                *advanced_settings, ocr_only=ocr_only)
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
    *advanced_settings: Any,
    progress=gr.Progress(track_tqdm=False),
) -> tuple[Any, str, list[str]]:
    try:
        uploads = _collect_paths(files, "", {".txt", ".md"})
        paths = uploads or _collect_paths(None, folder, {".txt", ".md"})
        if not paths:
            return None, "Upload cleaned TXT/MD or select the Cleaning output folder.", []
        override = (overrides_path or "").strip() or None
        config = _runtime_config(
            provider, key, base_url, synthesis_model,
            synthesis_model=synthesis_model,
            stage_output=stage_output,
            manual_overrides=override or "",
        )
        config.synthesis_screening_mode = screening_mode
        _apply_stage_settings(config, "synthesis", advanced_settings)
        if topic_max_calls is not None:
            config.topic_max_calls = int(topic_max_calls)
        from corpus_pipeline.synthesis.runner import synthesize
        limit = int(max_samples) if max_samples not in (None, "") else None
        result = synthesize(paths, config, dry_run=bool(dry_run), max_samples=limit,
                            topics_only=bool(topics_only),
                            progress=lambda message: progress((0, None), desc=message))
        output = _stage_output_path(stage_output) / "synthesis"
        artifacts = sorted(str(path) for path in output.rglob("*") if path.is_file() and
                           "checkpoints" not in path.parts and "cache" not in path.parts)
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
    *advanced_settings: Any,
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
        _apply_stage_settings(config, "governance", advanced_settings)
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
    *advanced_settings: Any,
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
        _apply_stage_settings(config, "review", advanced_settings)
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


def _run_metadata(files: Any, folder: str, dry_run: bool, use_model: bool,
                  max_calls: float, stage_output: str, provider: str, key: str,
                  base_url: str, metadata_model: str,
                  semantic_enabled: bool | None = None,
                  embedding_model: str | None = None,
                  *advanced_settings: Any) -> tuple[Any, str, list, list, list[str]]:
    """Session settings stay in memory; metadata never rewrites earlier stages."""
    try:
        from corpus_pipeline.artifacts import read_jsonl
        from corpus_pipeline.metadata.runner import discover_metadata
        from corpus_pipeline.settings import load_json
        paths = _collect_paths(files, folder, {".txt", ".md"})
        if not paths:
            return None, "Upload cleaned TXT/MD or select a folder.", [], [], []
        config = _runtime_config(provider, key, base_url, "", stage_output=stage_output)
        config.metadata_model = (metadata_model or "").strip()
        config.metadata_llm_enabled = bool(use_model)
        config.metadata_max_calls = int(max_calls)
        if semantic_enabled is not None:
            config.metadata_semantic_enabled = bool(semantic_enabled)
        if embedding_model is not None or semantic_enabled is not None:
            config.metadata_embedding_model = (embedding_model or "").strip()
        config.metadata_skip_window = 0  # Fixed source-order invariant, not an editable toggle.
        try:
            _apply_stage_settings(config, "metadata", advanced_settings)
            config.validate_metadata()
        except (ValueError, TypeError):
            return None, "Invalid metadata settings: check sizes, thresholds, delimiter policy and request limits.", [], [], []
        result = discover_metadata(paths, config, dry_run=bool(dry_run))
        output = config.stage_output_dir / "metadata"
        registry = load_json(output / "field-registry.json")
        fields = [[row["canonical_label"], row["value_type"], row["scope"], row["occurrence_count"],
                   row["status"], row["min_confidence"], row["field_id"]] for row in registry.get("fields", [])]
        queue_path = output / "review-queue.jsonl"
        queue = read_jsonl(queue_path) if queue_path.is_file() else []
        reviews = [[row.get("source", ""), row.get("observation", {}).get("label", ""),
                    row.get("observation", {}).get("status", row.get("status")),
                    row.get("observation", {}).get("reason", row.get("reason", ""))] for row in queue]
        chunkers = result.get("chunkers", {})
        semantic_chunks = chunkers.get("chonkie_semantic_1.7.0", 0)
        fallback_chunks = sum(count for name, count in chunkers.items()
                              if name != "chonkie_semantic_1.7.0")
        if dry_run:
            status = "Dry run: no provider calls or embeddings initialized. Semantic chunker is not run."
        else:
            status = (f"Metadata: {result['processed']} documents; {result['model_calls']} model calls; "
                      f"{result['review_cases']} review cases. Chonkie: {semantic_chunks} documents; "
                      f"other chunkers: {fallback_chunks}. Shadow mode: training text is unaffected.")
        names = (("plan.json", "dry-run-summary.json") if dry_run else (
            "documents.jsonl", "observations.jsonl", "field-registry.json", "registry-history.jsonl",
            "summary.json", "review-queue.jsonl", "plan.json"))
        return result, status, fields, reviews, _stage_files(output, names)
    except STAGE_ERRORS:
        return None, ("Metadata failed. Check the input files/output permissions; live extraction requires "
                      "an explicit metadata model and a valid provider key."), [], [], [],


def _run_metadata_evaluation(reference_file: Any, stage_output: str,
                             min_precision: float, min_recall: float,
                             min_examples: float) -> tuple[Any, str]:
    """Evaluate saved metadata against an annotated JSONL file."""
    try:
        reference = _single_path(reference_file)
        if not reference:
            return None, "Upload annotated reference JSONL first."
        config = PipelineConfig()
        config.stage_output_dir = _stage_output_path(stage_output)
        _apply_stage_settings(config, "evaluation", (min_precision, min_recall, min_examples))
        from corpus_pipeline.benchmark import evaluate_metadata
        result = evaluate_metadata(
            config.stage_output_dir / "metadata" / "documents.jsonl", Path(reference),
            config.stage_output_dir / "metadata" / "evaluation.json",
            min_precision=config.metadata_eval_min_precision, min_recall=config.metadata_eval_min_recall,
            min_examples=config.metadata_eval_min_examples)
        precision, recall = result.get("precision"), result.get("recall")
        display = lambda value: "unavailable" if value is None else f"{value:.3f}"
        return result, f"Evaluation complete: precision {display(precision)}, recall {display(recall)}."
    except STAGE_ERRORS as exc:
        return None, f"Metadata evaluation failed: {exc}"


def _run_llm_quality_evaluation(target: str, input_path: str, stage_output: str,
                                provider: str, key: str, base_url: str, model: str,
                                max_calls: float | None) -> tuple[Any, str, list[str]]:
    """Run an opt-in bounded LLM judge and keep its report outside production JSONL."""
    try:
        source = Path((input_path or "").strip()).expanduser()
        if not source.is_file():
            return None, f"Select an existing {target} JSONL file to evaluate.", []
        if max_calls in (None, "") or int(max_calls) < 0:
            return None, "Judge call budget must be a non-negative integer.", []
        config = _runtime_config(provider, key, base_url, model, synthesis_model=model,
                                 stage_output=stage_output)
        evaluators = {
            "chunks": "evaluate_chunk_quality",
            "metadata": "evaluate_metadata_quality",
            "qa": "evaluate_qa_quality",
            "review": "evaluate_review_quality",
        }
        if target not in evaluators:
            return None, "Unknown quality evaluation target.", []
        from corpus_pipeline import benchmark
        evaluator = getattr(benchmark, evaluators[target])
        output_root = config.stage_output_dir
        output = (output_root / "synthesis" / "chunking" / "evaluation.json" if target == "chunks"
                  else output_root / "evaluation" / f"{target}-quality.json")
        kwargs = {"config": config, "judge_model": model, "max_calls": int(max_calls)}
        if target == "chunks":
            result = evaluator(source, output, **kwargs)
        else:
            result = evaluator(source, output, **kwargs)
        status = (f"Evaluation complete: {result.get('judged_records', result.get('judged_chunks', 0))} judged, "
                  f"{result.get('flagged_records', result.get('flagged_chunks', 0))} flagged, "
                  f"{result.get('judge_calls', 0)} judge calls. LLM scores are triage signals.")
        return result, status, [str(output)]
    except STAGE_ERRORS as exc:
        return None, f"{target.title()} quality evaluation failed: {exc}", []


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


def _run_ocr_stream(files, folder, provider, key, base_url, model, file_workers, page_workers,
                    ocr_workers, force, preflight, root, *settings):
    """Use the existing streaming worker while exporting raw extraction only."""
    config = PipelineConfig()
    count = len(STAGE_SETTINGS["ocr"])
    try:
        _apply_stage_settings(config, "ocr", settings[:count])
        _apply_stage_settings(config, "benchmarks", settings[count:])
    except (ValueError, TypeError) as error:
        yield [], [], "", "", str(error), [], ""
        return
    config.output_dir = _stage_output_path(root) / "ocr"
    config.semantic_rules_enabled = False
    legacy = tuple(_setting_default(config, name, kind) for name, kind, _ in
                   STAGE_SETTINGS["extract"] + STAGE_SETTINGS["benchmarks"])
    for update in _run_stream(files, folder, provider, key, base_url, model, file_workers, page_workers,
                              ocr_workers, force, preflight, False, *legacy, ocr_only=True):
        yield (*update[:3], *update[4:])


def _run_cleaning(files, folder, root, format_enabled, force, *settings,
                  progress=gr.Progress(track_tqdm=False)):
    try:
        from corpus_pipeline.cleaning import clean_files
        config = PipelineConfig(stage_output_dir=_stage_output_path(root))
        _apply_stage_settings(config, "cleaning", settings)
        paths = _collect_paths(files, "", {".txt", ".md"}) or _collect_paths(None, folder, {".txt", ".md"})
        if not paths:
            return None, "Select OCR text files or an OCR folder.", "", "", []
        result = clean_files(paths, config, format_cleaning=bool(format_enabled), force=bool(force),
                             progress=lambda message: progress((0, None), desc=message))
        before, after, downloads = [], [], []
        for record in result.pop("results"):
            for key in ("output_path", "report_path"):
                if record.get(key):
                    downloads.append(record[key])
            if record.get("original_text") is not None:
                before.append(record["original_text"])
                after.append(record["cleaned_text"])
        downloads += _stage_files(config.stage_output_dir / "cleaning", ("summary.json", "rules-report.json"))
        return result, f"{result['files_processed']} cleaned, {result['files_skipped']} skipped, {result['files_failed']} failed.", "\n\n".join(before), "\n\n".join(after), downloads
    except STAGE_ERRORS:
        return None, "Cleaning failed; check the input, output folder and correction rules.", "", "", []


def _run_synthesis_agents(files, path, max_samples, dry_run, root, provider, key, base_url, model,
                          questions_only, *settings, progress=gr.Progress(track_tqdm=False)):
    try:
        from corpus_pipeline.synthesis.runner import synthesize_chunks
        from corpus_pipeline.artifacts import read_jsonl
        config = _runtime_config(provider, key, base_url, model, synthesis_model=model, stage_output=root)
        _apply_stage_settings(config, "agents", settings)
        paths = _read_stage_inputs(files, path)
        if not paths:
            return None, "Select saved chunks JSONL from the chunking step.", []
        result = synthesize_chunks([row for source in paths for row in read_jsonl(source)], config,
                                   max_samples=int(max_samples), dry_run=bool(dry_run),
                                   questions_only=bool(questions_only),
                                   progress=lambda message: progress((0, None), desc=message))
        label = "questions" if questions_only else "QA records"
        status = (f"Dry run: {result['selected_units']} chunks selected; no model calls." if dry_run else
                  f"Generated {result['generated']} {label}; {result['failed']} failed.")
        output = config.stage_output_dir / "synthesis"
        artifacts = sorted(str(p) for p in output.rglob("*") if p.is_file() and
                           "checkpoints" not in p.parts and "cache" not in p.parts)
        return result, status, artifacts
    except STAGE_ERRORS:
        return None, "Synthesis failed; check chunks and settings. Live generation requires a text model and provider key.", []


def _run_synthesis_metadata(files, folder, dry_run, use_model, max_calls, root, provider, key,
                            base_url, model, semantic, embedding, *settings):
    """Keep metadata and the chunks consumed by the agents inside Synthesis."""
    metadata_root = _stage_output_path(root) / "synthesis"
    result = _run_metadata(files, folder, dry_run, use_model, max_calls, str(metadata_root), provider,
                           key, base_url, model, semantic, embedding, *settings)
    if result[0] is not None and not dry_run:
        try:
            from corpus_pipeline.metadata.runner import export_synthesis_chunks
            paths = _collect_paths(files, "", {".txt", ".md"}) or _collect_paths(None, folder, {".txt", ".md"})
            chunks = export_synthesis_chunks(PipelineConfig(stage_output_dir=_stage_output_path(root)), paths,
                                               metadata_root=metadata_root)
            result[0]["synthesis_chunks"] = chunks
            return result[0], result[1] + f" {chunks['chunks']} chunks ready for the agents.", result[2], result[3], [*result[4], chunks["output"]]
        except STAGE_ERRORS:
            return result[0], result[1] + " Chunk export failed; rerun before generating questions.", result[2], result[3], result[4]
    return result


def _run_corpus(files, input_path, root):
    try:
        from corpus_pipeline.artifacts import export_corpus, read_jsonl
        paths = _read_stage_inputs(files, input_path)
        if not paths:
            return None, "Select accepted review JSONL first.", []
        result = export_corpus([row for path in paths for row in read_jsonl(path)], _stage_output_path(root) / "corpus")
        return result, f"{result['exported']} accepted QA exported; {result['excluded']} excluded.", _stage_files(
            _stage_output_path(root) / "corpus", ("corpus.jsonl", "provenance.jsonl", "summary.json"))
    except STAGE_ERRORS:
        return None, "Corpus export failed; check the accepted review files and output folder.", []


def _run_synthesis_chunking(files, folder, dry_run, max_calls, root, provider, key,
                            base_url, model, semantic, embedding, *settings):
    """Share the metadata preparer, with all metadata API calls explicitly disabled."""
    return _run_synthesis_metadata(files, folder, dry_run, False, max_calls, root, provider, key,
                                   base_url, model, semantic, embedding, *settings)


with gr.Blocks(title="Corpus Processing Pipeline") as demo:
    gr.Markdown("# Corpus processing pipeline\nOCR → Cleaning → Synthesis → Governance → Review → Corpus. Settings apply to the session only.")
    stage_output = gr.Textbox(label="Output root for all pipeline stages", value=STAGE_OUTPUT_DEFAULT,
                              info="Each step writes to a named subfolder. Existing outputs are not moved.")
    with gr.Tabs():
        with gr.Tab("1 - OCR"):
            gr.Markdown("Raw text extraction and per-page OCR → `data/ocr/`. Cleaning runs in the next tab.")
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
                    folder = gr.Textbox(label="Or enter a folder path", value="data/in")
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
                    with gr.Row():
                        extract_force = gr.Checkbox(label="Force reprocess files", value=False)
                        extract_preflight = gr.Checkbox(label="Test OCR connection before running", value=False)
                    extract_advanced = _setting_controls("ocr")
                    run_button = gr.Button("Run OCR", variant="primary")
                    overall_status = gr.Textbox(label="Progress and summary", interactive=False)
                    live_log = gr.Textbox(
                        label="Live activity", lines=16, max_lines=24,
                        interactive=False, autoscroll=True,
                        placeholder="Per-file and per-page events will appear here while the run is active.")
                    gr.Markdown(
                        "Benchmark metrics follow the option in advanced settings. "
                        "Use the **OCR benchmarks** panel below to compare providers and models."
                    )
            benchmark_controls = _setting_controls("benchmarks", title="Benchmark folders")
            file_table = gr.Dataframe(
                headers=["File", "Status", "Detected type", "Route summary", "Warnings"],
                label="Per-file status", interactive=False, wrap=True)
            page_table = gr.Dataframe(
                headers=["File", "Page", "Route", "Characters", "Valid ratio",
                         "Image ratio", "Warnings", "OCR failed"],
                label="PDF page report", interactive=False, wrap=True)
            original_preview = gr.Textbox(label="OCR / directly extracted text", lines=20, max_lines=32)
            downloads = gr.File(label="Download OCR text and routing reports", file_count="multiple")

            with gr.Accordion("OCR benchmarks", open=False):
                gr.Markdown(
                    "Every extraction writes per-document metrics and appends to the history. "
                    "This panel compares runs grouped by provider and model. "
                    "Reference-based CER/WER requires a corrected `<source-stem>.txt` in the reference folder."
                )
                with gr.Row():
                    bench_refresh = gr.Button("Refresh comparison", variant="primary", scale=1)
                gr.Markdown("Reads the shared benchmark history folder selected above.")
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

        with gr.Tab("2 - Cleaning"):
            gr.Markdown("Format cleaning + deterministic semantic correction → `data/cleaning/`. No model calls.")
            clean_uploads = gr.File(label="OCR text files", file_count="multiple", type="filepath", file_types=[".txt", ".md"])
            clean_folder = gr.Textbox(label="Or OCR text folder", value="data/ocr")
            with gr.Row():
                clean_format = gr.Checkbox(label="Enable format cleaning", value=True)
                clean_force = gr.Checkbox(label="Force cleaning again", value=False)
            clean_controls = _setting_controls("cleaning", title="Semantic correction, languages and page markers")
            clean_run = gr.Button("Run Cleaning", variant="primary")
            clean_status = gr.Textbox(label="Cleaning status", interactive=False)
            clean_summary = gr.JSON(label="Cleaning counters and summary")
            with gr.Row():
                clean_before = gr.Textbox(label="Before Cleaning", lines=18)
                clean_after = gr.Textbox(label="After Cleaning", lines=18)
            clean_downloads = gr.File(label="Cleaned text and reports", file_count="multiple")

        with gr.Tab("3 - Synthesis"):
            gr.Markdown("Chunking → Metadata → Question agent → Answer agent. Exports follow the JORADP field names.")
            with gr.Accordion("Text generation provider for Synthesis, Metadata and LLM Review", open=False):
                with gr.Row():
                    gen_provider = gr.Dropdown(choices=GENERATION_PROVIDERS, value=GENERATION_DEFAULT,
                                               label="Provider", scale=1)
                    gen_api_key = gr.Textbox(label="API key", type="password",
                                             placeholder="Loaded from .env", scale=1)
                    gen_base_url = gr.Textbox(label="Base URL", scale=1)
                    gen_model = gr.Textbox(label="Text model (SYNTHESIS_MODEL)", scale=1)
                gen_status = gr.Markdown("Set a text-capable model; the OCR vision model is not reused.")
                generation_controls = _setting_controls("generation", title="Generation request timeout and retries")

            with gr.Accordion("1. Chunking", open=True):
                gr.Markdown("Create ordered source chunks. This step makes no metadata model calls.")
                with gr.Row():
                    with gr.Column():
                        meta_uploads = gr.File(label="Cleaned TXT/MD", file_count="multiple", type="filepath",
                                               file_types=[".txt", ".md"])
                        meta_folder = gr.Textbox(label="Or folder of cleaned files", value="data/cleaning")
                        meta_semantic = gr.Checkbox(label="Use Chonkie semantic chunking", value=_DEFAULTS.metadata_semantic_enabled,
                                                    info="Requires optional semantic dependencies and an embedding model.")
                        meta_embedding_model = gr.Dropdown(
                            choices=list(dict.fromkeys(["minishlab/potion-base-8M", "minishlab/potion-multilingual-128M"] +
                                                        ([_DEFAULTS.metadata_embedding_model] if _DEFAULTS.metadata_embedding_model else []))),
                            value=_DEFAULTS.metadata_embedding_model or None, label="Embedding model",
                            allow_custom_value=True, interactive=_DEFAULTS.metadata_semantic_enabled,
                            info="Choose a preset or enter a Model2Vec/Hugging Face ID or local path.")
                        meta_semantic.change(lambda enabled: gr.update(interactive=bool(enabled)),
                                             inputs=[meta_semantic], outputs=[meta_embedding_model])
                    with gr.Column():
                        meta_dry_run = gr.Checkbox(label="Dry run (no provider/embedding calls)", value=True)
                        meta_chunking_controls = _setting_controls("metadata_chunking", title="Chunking parameters")
                        gr.Markdown("Chunk order is preserved; noncontiguous merging stays disabled.")
                        chunk_run = gr.Button("1. Create / refresh chunks", variant="primary")
                chunk_eval_path = gr.Textbox(label="Chunks JSONL to evaluate",
                                             value="data/synthesis/chunking/chunks.jsonl")
                with gr.Row():
                    chunk_judge_budget = gr.Number(label="Judge call budget (0 = deterministic only)", value=100,
                                                   minimum=0, precision=0)
                    chunk_evaluate = gr.Button("Evaluate chunk quality")
                chunk_eval_status = gr.Textbox(label="Chunk evaluation status", interactive=False)
                chunk_eval_summary = gr.JSON(label="Chunk quality report")
                chunk_status = gr.Textbox(label="Chunking status", interactive=False)
                chunk_summary = gr.JSON(label="Chunking plan / summary")
                chunk_downloads = gr.File(label="Chunking artifacts", file_count="multiple")

            with gr.Accordion("2. Metadata extraction", open=True):
                gr.Markdown("Discover evidence-backed, open-ended metadata. Legal syntax preserves structure; content metadata comes from the enabled model.")
                with gr.Row():
                    meta_model = gr.Textbox(label="Metadata text model (METADATA_MODEL)", value=_DEFAULTS.metadata_model)
                    meta_live = gr.Checkbox(label="Enable metadata model calls", value=_DEFAULTS.metadata_llm_enabled)
                    meta_max_calls = gr.Number(label="Maximum metadata model calls", value=_DEFAULTS.metadata_max_calls,
                                               minimum=1, precision=0)
                meta_extraction_controls = _setting_controls("metadata_extraction", title="Metadata extraction parameters")
                meta_advanced = [*meta_chunking_controls, *meta_extraction_controls]
                meta_run = gr.Button("2. Extract metadata and refresh chunks", variant="primary")
                meta_status = gr.Textbox(label="Metadata status", interactive=False)
                meta_summary = gr.JSON(label="Metadata plan / summary")
                meta_fields = gr.Dataframe(headers=["Field", "Type", "Scope", "Documents", "Status", "Min confidence", "ID"],
                                            label="Discovered field registry", interactive=False, wrap=True)
                meta_queue = gr.Dataframe(headers=["Source", "Label", "Status", "Reason"], label="Optional review queue",
                                           interactive=False, wrap=True)
                meta_downloads = gr.File(label="Metadata artifacts", file_count="multiple")
                with gr.Row():
                    meta_quality_path = gr.Textbox(label="Metadata documents JSONL",
                                                   value="data/synthesis/metadata/documents.jsonl")
                    meta_judge_budget = gr.Number(label="Judge call budget (0 = deterministic only)", value=100,
                                                  minimum=0, precision=0)
                    meta_quality_evaluate = gr.Button("Evaluate metadata quality")
                meta_quality_status = gr.Textbox(label="Metadata quality status", interactive=False)
                meta_quality_summary = gr.JSON(label="Metadata quality report")
                with gr.Accordion("Compare metadata against annotations", open=False):
                    meta_reference = gr.File(label="Annotated reference JSONL", file_count="single",
                                             type="filepath", file_types=[".jsonl"])
                    meta_evaluation_controls = _setting_controls("evaluation", title="Evaluation thresholds")
                    meta_evaluate = gr.Button("Run annotated metadata evaluation")
                    meta_eval_status = gr.Textbox(label="Evaluation status", interactive=False)
                    meta_eval_summary = gr.JSON(label="Annotation comparison results")

            with gr.Accordion("3–4. Question and answer generation", open=True):
                gr.Markdown("The agents consume saved chunks. Answers reuse validated question checkpoints and include a public explanation.")
                with gr.Row():
                    with gr.Column(scale=1):
                        synth_uploads = gr.File(label="Upload saved chunks JSONL", file_count="multiple", type="filepath",
                                                file_types=[".jsonl"])
                        synth_folder = gr.Textbox(label="Or saved chunks JSONL path",
                                                  value="data/synthesis/chunking/chunks.jsonl")
                    with gr.Column(scale=1):
                        synth_max_samples = gr.Number(label="Maximum units", value=_DEFAULTS.synthesis_max_samples,
                                                      precision=0, minimum=1)
                        synth_dry_run = gr.Checkbox(label="Dry run (no model calls)", value=True)
                        agent_controls = _setting_controls("agents", title="Question / answer parameters")
                        synth_questions_only = gr.State(False)
                        questions_only_state = gr.State(True)
                        question_run = gr.Button("3. Generate questions")
                        synth_run = gr.Button("4. Generate answers (reuse saved questions)", variant="primary")
                synth_status = gr.Textbox(label="Generation status", interactive=False)
                synth_summary = gr.JSON(label="Generation run details")
                synth_downloads = gr.File(label="Generation artifacts", file_count="multiple")
                with gr.Row():
                    qa_quality_path = gr.Textbox(label="Generated QA JSONL",
                                                 value="data/synthesis/qa.jsonl")
                    qa_judge_budget = gr.Number(label="Judge call budget (0 = deterministic only)", value=100,
                                                minimum=0, precision=0)
                    qa_quality_evaluate = gr.Button("Evaluate question / answer quality")
                qa_quality_status = gr.Textbox(label="QA quality status", interactive=False)
                qa_quality_summary = gr.JSON(label="Question / answer quality report")
            with gr.Accordion("Optional document theme discovery", open=False):
                gr.Markdown("Discover and group themes from the cleaned files selected above; this never replaces generated QA.")
                synth_screening_mode = gr.State("automatic")
                synth_topics_only = gr.State(True)
                synth_overrides_path = gr.Textbox(label="Manual overrides JSON (optional)", value=MANUAL_OVERRIDES_DEFAULT)
                synth_topic_max_calls = gr.Number(label="Maximum theme model calls", value=_DEFAULTS.topic_max_calls, precision=0, minimum=1)
                synth_advanced = _setting_controls("synthesis", title="Theme discovery parameters")
                topic_run = gr.Button("Discover document themes")
                topic_status = gr.Textbox(label="Theme status", interactive=False)
                topic_summary = gr.JSON(label="Theme plan / summary")
                topic_downloads = gr.File(label="Theme artifacts", file_count="multiple")
                synth_topic_groups = gr.Dataframe(headers=["Discovered theme", "Documents", "Member files"],
                                                  label="Automatic theme groups", interactive=False, wrap=True)

        with gr.Tab("4 - Governance"):
            gr.Markdown(
                "Screen, deduplicate, and route QA candidates into pass, manual, intercept, "
                "and duplicate buckets. This stage is offline and makes no model calls."
            )
            dialect_controls = _setting_controls("governance", title="Language and dialect settings (shared with Review)")
            with gr.Row():
                with gr.Column(scale=1):
                    gov_uploads = gr.File(
                        label="Upload QA JSONL", file_count="multiple", type="filepath",
                        file_types=[".jsonl"])
                    gov_path = gr.Textbox(
                        label="Or JSONL path",
                        value=str(Path(STAGE_OUTPUT_DEFAULT) / "synthesis" / "qa.jsonl"))
                with gr.Column(scale=1):
                    gr.Markdown("Uses the shared language/dialect settings in this tab.")
                    gov_run = gr.Button("Run governance", variant="primary")
            gov_status = gr.Textbox(label="Governance summary", interactive=False)
            gov_summary = gr.JSON(label="Bucket counts")
            gov_downloads = gr.File(label="Governance artifacts", file_count="multiple")

        with gr.Tab("5 - Review"):
            gr.Markdown(
                "Score and route governed records. Offline review is deterministic; enable LLM "
                "review to use the text generation settings in Synthesis."
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
                        label="Enable model review and repair", value=_DEFAULTS.review_llm_enabled,
                        info="Requires a text model in the generation settings.")
                    gr.Markdown("Uses the shared language/dialect settings in Governance.")
                    rev_advanced = _setting_controls("review")
                    rev_run = gr.Button("Run review", variant="primary")
            rev_status = gr.Textbox(label="Review summary", interactive=False)
            rev_summary = gr.JSON(label="Decision counts")
            rev_downloads = gr.File(label="Review artifacts", file_count="multiple")
            with gr.Accordion("Independent quality evaluation", open=True):
                gr.Markdown("A second judge assesses whether the saved review decisions and rationales fit the QA and source. This does not change review outputs.")
                with gr.Row():
                    review_quality_path = gr.Textbox(label="Review audit JSONL",
                                                     value="data/review/audit.jsonl")
                    review_judge_budget = gr.Number(label="Judge call budget (0 = no model calls)", value=100,
                                                    minimum=0, precision=0)
                    review_quality_evaluate = gr.Button("Evaluate review quality")
                review_quality_status = gr.Textbox(label="Review evaluation status", interactive=False)
                review_quality_summary = gr.JSON(label="Independent review quality report")

        with gr.Tab("6 - Corpus"):
            gr.Markdown("Export accepted records to `data/corpus/corpus.jsonl` (instruction / input / output). Manual and rejected records are excluded.")
            corpus_uploads = gr.File(label="Accepted review JSONL", file_count="multiple", type="filepath", file_types=[".jsonl"])
            corpus_path = gr.Textbox(label="Or accepted review JSONL path", value="data/review/accepted.jsonl")
            corpus_run = gr.Button("Export Corpus", variant="primary")
            corpus_status = gr.Textbox(label="Corpus status", interactive=False)
            corpus_summary = gr.JSON(label="Corpus summary")
            corpus_downloads = gr.File(label="Training corpus, provenance and summary", file_count="multiple")

    provider_defaults = [base_url, model, api_key, key_status]
    provider.change(_provider_defaults, inputs=[provider], outputs=provider_defaults)
    demo.load(_provider_defaults, inputs=[provider], outputs=provider_defaults)
    test_button.click(_test_connection, inputs=[provider, api_key, base_url, model],
                      outputs=[connection_status])
    run_button.click(_run_ocr_stream, inputs=[uploads, folder, provider, api_key, base_url, model,
                                          file_workers, page_workers, ocr_workers,
                                          extract_force, extract_preflight, stage_output,
                                          *extract_advanced, *benchmark_controls],
                     outputs=[file_table, page_table, original_preview,
                              overall_status, downloads, live_log])

    clean_run.click(_run_cleaning,
                    inputs=[clean_uploads, clean_folder, stage_output, clean_format, clean_force, *clean_controls],
                    outputs=[clean_summary, clean_status, clean_before, clean_after, clean_downloads])
    synth_run.click(_run_synthesis_agents,
                    inputs=[synth_uploads, synth_folder, synth_max_samples, synth_dry_run, stage_output,
                            gen_provider, gen_api_key, gen_base_url, gen_model, synth_questions_only,
                            *agent_controls, *generation_controls],
                    outputs=[synth_summary, synth_status, synth_downloads])
    question_run.click(_run_synthesis_agents,
                    inputs=[synth_uploads, synth_folder, synth_max_samples, synth_dry_run, stage_output,
                            gen_provider, gen_api_key, gen_base_url, gen_model, questions_only_state,
                            *agent_controls, *generation_controls],
                    outputs=[synth_summary, synth_status, synth_downloads])
    corpus_run.click(_run_corpus, inputs=[corpus_uploads, corpus_path, stage_output],
                     outputs=[corpus_summary, corpus_status, corpus_downloads])

    generation_defaults = [gen_base_url, gen_model, gen_api_key, gen_status]
    gen_provider.change(_generation_defaults, inputs=[gen_provider], outputs=generation_defaults)
    demo.load(_generation_defaults, inputs=[gen_provider], outputs=generation_defaults)

    topic_run.click(
        _run_synthesis,
        inputs=[meta_uploads, meta_folder, synth_overrides_path,
                synth_max_samples, synth_dry_run, stage_output,
                gen_provider, gen_api_key, gen_base_url, gen_model,
                synth_screening_mode, synth_topics_only, synth_topic_max_calls, *synth_advanced, *generation_controls],
        outputs=[topic_summary, topic_status, topic_downloads]).then(
            _topic_group_rows, inputs=[stage_output, synth_screening_mode],
            outputs=[synth_topic_groups])

    chunk_run.click(
        _run_synthesis_chunking,
        inputs=[meta_uploads, meta_folder, meta_dry_run, meta_max_calls, stage_output,
                gen_provider, gen_api_key, gen_base_url, meta_model, meta_semantic,
                meta_embedding_model, *meta_advanced, *generation_controls],
        outputs=[meta_summary, meta_status, meta_fields, meta_queue, meta_downloads])
    chunk_evaluate.click(
        lambda path, root, provider, key, base_url, model, budget:
            _run_llm_quality_evaluation("chunks", path, root, provider, key, base_url, model, budget)[:2],
        inputs=[chunk_eval_path, stage_output, gen_provider, gen_api_key, gen_base_url, gen_model, chunk_judge_budget],
        outputs=[chunk_eval_summary, chunk_eval_status])
    meta_run.click(
        _run_synthesis_metadata,
        inputs=[meta_uploads, meta_folder, meta_dry_run, meta_live, meta_max_calls, stage_output,
                gen_provider, gen_api_key, gen_base_url, meta_model, meta_semantic,
                meta_embedding_model, *meta_advanced, *generation_controls],
        outputs=[meta_summary, meta_status, meta_fields, meta_queue, meta_downloads])
    meta_evaluate.click(
        lambda reference, root, *values: _run_metadata_evaluation(reference, str(_stage_output_path(root) / "synthesis"), *values),
        inputs=[meta_reference, stage_output, *meta_evaluation_controls],
        outputs=[meta_eval_summary, meta_eval_status])
    meta_quality_evaluate.click(
        lambda path, root, provider, key, base_url, model, budget:
            _run_llm_quality_evaluation("metadata", path, root, provider, key, base_url, model, budget)[:2],
        inputs=[meta_quality_path, stage_output, gen_provider, gen_api_key, gen_base_url, gen_model, meta_judge_budget],
        outputs=[meta_quality_summary, meta_quality_status])
    qa_quality_evaluate.click(
        lambda path, root, provider, key, base_url, model, budget:
            _run_llm_quality_evaluation("qa", path, root, provider, key, base_url, model, budget)[:2],
        inputs=[qa_quality_path, stage_output, gen_provider, gen_api_key, gen_base_url, gen_model, qa_judge_budget],
        outputs=[qa_quality_summary, qa_quality_status])

    gov_run.click(
        _run_governance,
        inputs=[gov_uploads, gov_path, stage_output, *dialect_controls],
        outputs=[gov_summary, gov_status, gov_downloads])

    rev_run.click(
        _run_review,
        inputs=[rev_uploads, rev_path, rev_llm, stage_output,
                gen_provider, gen_api_key, gen_base_url, gen_model, *rev_advanced, *dialect_controls, *generation_controls],
        outputs=[rev_summary, rev_status, rev_downloads])
    review_quality_evaluate.click(
        lambda path, root, provider, key, base_url, model, budget:
            _run_llm_quality_evaluation("review", path, root, provider, key, base_url, model, budget)[:2],
        inputs=[review_quality_path, stage_output, gen_provider, gen_api_key, gen_base_url, gen_model,
                review_judge_budget],
        outputs=[review_quality_summary, review_quality_status])

    bench_outputs = [bench_table, bench_status, bench_leader, bench_comparison, bench_downloads]
    bench_refresh.click(_load_benchmarks, inputs=benchmark_controls,
                        outputs=bench_outputs)
    demo.load(_load_benchmarks, inputs=benchmark_controls,
              outputs=bench_outputs)

    stage_output.change(
        lambda root: (str(_stage_output_path(root) / "ocr"), str(_stage_output_path(root) / "cleaning"),
                      str(_stage_output_path(root) / "synthesis" / "chunking" / "chunks.jsonl"),
                      str(_stage_output_path(root) / "synthesis" / "qa.jsonl"),
                      str(_stage_output_path(root) / "governance" / "pass.jsonl"),
                      str(_stage_output_path(root) / "review" / "accepted.jsonl"),
                      str(_stage_output_path(root) / "synthesis" / "metadata" / "documents.jsonl"),
                      str(_stage_output_path(root) / "review" / "audit.jsonl"),
                      str(_stage_output_path(root) / "synthesis" / "chunking" / "chunks.jsonl"),
                      str(_stage_output_path(root) / "synthesis" / "qa.jsonl")),
        inputs=[stage_output], outputs=[clean_folder, meta_folder, synth_folder, gov_path, rev_path, corpus_path,
                                        meta_quality_path, review_quality_path, chunk_eval_path, qa_quality_path])


if __name__ == "__main__":
    demo.queue().launch()
