# Corpus Processing Pipeline

This project extracts and cleans PDF, TXT, Markdown, and DOCX files for downstream model training. PDF routing is page by page: good text layers are kept, low text or image dominated pages go to OCR, and corrupted text layers are marked SUSPECT and sent to OCR. OCR results are cached by rendered page hash, provider, model, and endpoint. File completion records make successful runs resumable.

The GUI uses Gradio because it provides a compact local interface for masked settings, file uploads, progress, tables, previews, and downloads without a separate frontend build.

## Setup

Python 3.10 or newer is required. On Windows, use the working Python executable on PATH (this avoids stale registrations in the py launcher):

    python -m venv .venv
    .\.venv\Scripts\Activate.ps1
    python --version
    python -m pip install -r requirements.txt

Copy `.env.example` to `.env` and fill in the keys for the providers you use. There is one variable per provider, and the pipeline reads only the variable belonging to the selected provider:

| Provider id | GUI label | Key variable | Default base URL | Default model |
| --- | --- | --- | --- | --- |
| `heiyu` | HeiyuCode (OpenAI-compatible) | `HEIYU_API_KEY` | `https://www.heiyucode.com/v1` | `gpt-6-sol` |
| `openai` | OpenAI | `OPENAI_API_KEY` | `https://api.openai.com/v1` | `gpt-4o-mini` |
| `anthropic` | Anthropic | `ANTHROPIC_API_KEY` | `https://api.anthropic.com` | `claude-sonnet-4-5` |
| `google` | Google Gemini | `GOOGLE_API_KEY` | Gemini v1beta | `gemini-2.5-flash` |
| `mistral` | Mistral | `MISTRAL_API_KEY` | `https://api.mistral.ai/v1` | `pixtral-12b-2409` |
| `azure` | Azure OpenAI | `AZURE_OPENAI_API_KEY` | required | deployment name |
| `openai_compatible` | Other OpenAI-compatible | `OCR_API_KEY` | required | required |

`.env` is git-ignored. The shell environment takes priority over `.env`, and a non-empty key typed in the GUI takes priority for that session only. No key is ever written to logs or reports.

## Run

    python cli.py .\data\raw\law.pdf .\data\raw\memo.docx
    python cli.py
    python cli.py --folder .\incoming
    python cli.py --force .\data\raw\law.pdf
    python cli.py --test-connection
    python app.py

The GUI listens on the local Gradio address printed at startup. Selecting a provider automatically loads that provider's base URL, model, and API key (from `.env`), so switching providers only requires choosing it from the dropdown. Typed values override `.env` for the current session only. The GUI supports multiple uploads or a folder path, per-file and per-page reports, side-by-side previews, and TXT downloads.

## OCR providers

`config.py` holds one `ProviderProfile` per provider: its label, key variable, default base URL, default model, and notes. `PipelineConfig.__post_init__` fills an empty base URL or model from the active profile, so switching provider is just:

    OCR_PROVIDER = "anthropic"   # or select it in the GUI

No pipeline code changes are involved. A base URL entered without a path (for example `https://www.heiyucode.com`) is normalized to `/v1/chat/completions` by the OpenAI-compatible adapter.

`--test-connection` sends a small text-bearing PNG (not a 1x1 image) because some gateways reject degenerate images. Authentication failures name the expected environment variable, and provider error messages are redacted before they reach logs or the GUI.

To add a provider:

1. Add a `ProviderProfile` entry to `PROVIDER_PROFILES` in `config.py` (id, label, key variable, defaults).
2. Add an adapter module under `corpus_pipeline/ocr/` implementing `OCRAdapter.ocr_page(image: bytes, language: list[str]) -> OCRResult`.
3. Register the adapter class in `corpus_pipeline/ocr/registry.py`.

An OpenAI-compatible service usually needs only steps 1 and 3, reusing `OpenAICompatibleAdapter`. Never log request objects or credentials.

## Outputs and tests

Each input produces a UTF-8 TXT file and a JSON route report under OUTPUT_DIR; summary.json reports batch totals. Page reports include route, extracted character count, valid-character ratio, estimated image coverage, warnings, cleaning counters, and OCR failure status. OCR failures are recorded per page and do not abort the rest of the PDF. Failed or partial files are retried on the next run; successful page OCR is reused from the cache.

    pytest

## Scope and limitations

The implementation covers document extraction and OCR post-processing (the source specification's Stages 1 and 2), plus deterministic export for later training preparation. QA synthesis, legal-document tier screening, dialect classification, and the eight-dimension QA review require separately configured generation/classification models and validated domain rules; they are not silently simulated here.

ftfy repairs common mojibake in extracted Unicode text, but it does not reconstruct arbitrary Arabic glyph mappings from embedded PDF font programs. Pages that fail the configured text-quality checks are sent to OCR. OCR uses the configured vision provider; the local PaddleOCR engine described in the separate README is not included.

Cleaning is intentionally conservative: metadata headers, HTML residue, long repeated-line loops, and adjacent duplicated words are counted. Semantic substitutions, date reordering, article-number repairs, and document-specific wrong-word corrections need a reviewed correction table and are not guessed by default.
