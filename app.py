"""Gradio test GUI for session-only OCR settings and corpus processing."""

import os
from pathlib import Path
from typing import Any

import gradio as gr

from config import DEFAULT_PROVIDER, LOG_LEVEL, PipelineConfig, provider_choices, provider_profile
from corpus_pipeline.logging_utils import configure_logging
from corpus_pipeline.ocr.registry import create_adapter
from corpus_pipeline.processor import process_batch

configure_logging(LOG_LEVEL)


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


def _runtime_config(provider: str, key: str, base_url: str, model: str) -> PipelineConfig:
    key = (key or "").strip()
    return PipelineConfig(
        ocr_provider=provider,
        ocr_api_key=key,
        session_api_key_override=bool(key),
        ocr_base_url=base_url.strip(),
        ocr_model=model.strip(),
    )


def _test_connection(provider: str, key: str, base_url: str, model: str) -> str:
    try:
        result = create_adapter(_runtime_config(provider, key, base_url, model)).test_connection()
        return f"Connection successful. Provider: {result.provider}"
    except Exception as exc:
        return f"Connection failed: {exc}"


def _run(files: list[str] | None, folder: str, provider: str, key: str, base_url: str, model: str,
         progress=gr.Progress(track_tqdm=False)) -> tuple[Any, ...]:
    paths = [Path(item) for item in (files or []) if item]
    if folder and folder.strip():
        folder_path = Path(folder.strip()).expanduser()
        if not folder_path.is_dir():
            return [], [], "", "", "Folder path does not exist or is not a directory.", []
        paths.extend(sorted(p for p in folder_path.rglob("*") if p.is_file() and p.suffix.lower() in {".pdf", ".txt", ".md", ".docx"}))
    paths = list(dict.fromkeys(p.resolve() for p in paths))
    if not paths:
        return [], [], "", "", "Upload files or enter a folder path.", []
    config = _runtime_config(provider, key, base_url, model)

    steps = {"count": 0}

    def report(message: str) -> None:
        # Gradio 6: progress(value, desc=...) with an unknown total for the spinner.
        steps["count"] += 1
        progress((steps["count"], None), desc=message)

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
    return status_rows, page_rows, "\n\n".join(original), "\n\n".join(cleaned), summary_text, outputs


with gr.Blocks(title="Corpus Processing Pipeline") as demo:
    gr.Markdown("# Corpus processing pipeline\nPer-page PDF routing, OCR, and deterministic text cleaning.")
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### OCR settings (session only)")
            provider = gr.Dropdown(choices=provider_choices(), value=DEFAULT_PROVIDER, label="Provider")
            api_key = gr.Textbox(label="API key", type="password", placeholder="Loaded from .env")
            base_url = gr.Textbox(label="Base URL")
            model = gr.Textbox(label="Model")
            key_status = gr.Markdown("Select a provider to load its defaults.")
            test_button = gr.Button("Test connection")
            connection_status = gr.Textbox(label="Connection status", interactive=False)
        with gr.Column(scale=2):
            uploads = gr.File(label="Upload files", file_count="multiple", type="filepath",
                              file_types=[".pdf", ".txt", ".md", ".docx"])
            folder = gr.Textbox(label="Or enter a folder path")
            run_button = gr.Button("Run full pipeline", variant="primary")
            overall_status = gr.Textbox(label="Progress and summary", interactive=False)
    file_table = gr.Dataframe(headers=["File", "Status", "Detected type", "Route summary", "Warnings"],
                              label="Per-file status", interactive=False, wrap=True)
    page_table = gr.Dataframe(headers=["File", "Page", "Route", "Characters", "Valid ratio", "Image ratio", "Warnings", "OCR failed"],
                              label="PDF page report", interactive=False, wrap=True)
    with gr.Row():
        original_preview = gr.Textbox(label="Extracted text", lines=24, max_lines=32)
        cleaned_preview = gr.Textbox(label="Cleaned output", lines=24, max_lines=32)
    downloads = gr.File(label="Download resulting text files", file_count="multiple")

    provider_defaults = [base_url, model, api_key, key_status]
    provider.change(_provider_defaults, inputs=[provider], outputs=provider_defaults)
    demo.load(_provider_defaults, inputs=[provider], outputs=provider_defaults)
    test_button.click(_test_connection, inputs=[provider, api_key, base_url, model], outputs=[connection_status])
    run_button.click(_run, inputs=[uploads, folder, provider, api_key, base_url, model],
                     outputs=[file_table, page_table, original_preview, cleaned_preview, overall_status, downloads])


if __name__ == "__main__":
    demo.queue().launch()
