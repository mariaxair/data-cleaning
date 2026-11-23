# Corpus Processing Pipeline

## Metadata discovery (opt-in, shadow mode)

Use **Synthesis → Chunking / Metadata** in the GUI or the legacy CLI commands below after extraction.
Legal syntax is recognized deterministically only to preserve document/article boundaries and
route downstream synthesis. Content metadata for all readable document types, including legal
identifiers and types, comes from open-ended, evidence-backed model observations when explicitly
enabled; there is no fixed per-domain metadata inventory. Fields accumulate in a versioned
registry, and ambiguous mappings remain in a review queue. Nothing changes existing training/OCR
exports by default.

```powershell
# Plan only: no provider calls, model initialization, or registry mutation.
.\.venv\Scripts\python.exe cli.py --metadata --dry-run --folder data/ocr-out
# Metadata/chunk processing without model extraction; discovered content metadata remains pending.
.\.venv\Scripts\python.exe cli.py --metadata --folder data/ocr-out
# Optional generic extraction: set METADATA_MODEL and the provider key in .env first.
.\.venv\Scripts\python.exe cli.py --metadata --metadata-llm --metadata-max-calls 100 --folder data/ocr-out
# Evaluate operator-authored annotations (format documented in docs/PIPELINE.md).
.\.venv\Scripts\python.exe cli.py --metadata-evaluate data/metadata-reference.jsonl
# Optional references only; QA generation does not depend on metadata.
.\.venv\Scripts\python.exe cli.py --synthesize --screening-mode automatic --with-metadata --folder data/ocr-out
```

CLI artifacts live under `data/training/metadata/` (GUI: `data/synthesis/metadata/`): `documents.jsonl`, `observations.jsonl`,
`field-registry.json`, append-only `registry-history.jsonl`, `review-queue.jsonl`, and
`summary.json`. Dry runs write only `plan.json`/`dry-run-summary.json`. Reruns reuse
validated request caches and count each content-identical document once per registry field.

Semantic chunking is optional: install `requirements-semantic.txt` only if needed, set
`METADATA_EMBEDDING_MODEL` explicitly, then enable `METADATA_SEMANTIC_ENABLED=true`.
Chonkie **1.7.0** is pinned and its wheel's public API was checked. On this Windows/Python 3.15
environment installation is currently blocked: `chonkie-core 0.10.2` has no compatible binary wheel,
and its source build uses PyO3 0.27.2, which supports Python only through 3.14. Updating Chonkie alone
does not resolve that native dependency. Keep the existing venv; the deterministic fallback works.
The optional real `SemanticChunker` import/offset tests run when a compatible installation exists;
they skip if Chonkie is absent and fail if an installed package cannot import. Offline fake adapters
cover failure handling and exact reconstruction. No model weights are downloaded by tests.
Real embeddings and Arabic/French boundary quality have **not** been benchmarked here.
The default paragraph fallback preserves exact text and reports degraded mode explicitly.
Legal boundaries always take priority. Skip-window merging is rejected.

See [the metadata engineering guide](docs/PIPELINE.md#metadata-discovery-shadow-stage) and
[implementation status / remaining rollout work](docs/METADATA_DISCOVERY_PLAN.md#implementation-status).

### Chunk quality evaluation

After creating chunks, run the optional LLM-judge benchmark from this directory:

```powershell
.\.venv\Scripts\python.exe cli.py --chunk-evaluate data/synthesis/chunking/chunks.jsonl `
  --chunk-judge-model YOUR_TEXT_MODEL --chunk-judge-max-calls 100
```

The judge uses the configured OpenAI-compatible provider/key and scores coherence, completeness,
and whether each chunk can stand alone. The report is written to
`data/synthesis/chunking/evaluation.json`; low scores and deterministic size/source-offset problems
are flagged. Increase the call budget to cover more chunks, or set it to `0` for integrity/size
checks only. This is a triage signal, not ground truth; the source file must still be available at
its saved provenance path for exact offset checks.

## Automatic themes across all domains

The Synthesis tab now defaults to **Automatic themes · all domains**. The configured text model
discovers each document's theme and a short thematic summary, then groups documents by meaning.
You do not supply a list of themes, keyword rules, or a manual override for every file.
The legal screening mode remains available for compatibility.

1. Select a folder of cleaned TXT/MD documents in Synthesis and the text model/provider.
2. To inspect themes before creating QA, select **Discover/group themes only**.
3. Turn **Dry run** off to permit model discovery. Dry run uses cached themes if available;
   otherwise it shows pending themes and plans generic passages without any model calls.
4. Set **Maximum model calls for theme discovery and grouping** to your intended budget.
   This limit is separate from **Maximum units**, which caps QA generation only.
5. Run the stage. The theme table displays the groups and member files; topics.json contains
   per-document themes, confidence, warnings, and group membership. No original files are moved.

The shipped 100-call discovery cap is intentionally small. For 4,000 documents, discovery needs
roughly one call per uncached document plus semantic comparison calls and possible retries.
Raise the limit deliberately or rerun incrementally: validated descriptions and comparisons are
cached by content/model/prompt and reused even after an upload is renamed.
Only provider calls initiated by the operator are made; no weights or heavy packages are downloaded.

    .\.venv\Scripts\python.exe cli.py --discover-topics --dry-run --folder data/out
    .\.venv\Scripts\python.exe cli.py --discover-topics --topic-max-calls 100 --folder data/out
    .\.venv\Scripts\python.exe cli.py --synthesize --screening-mode automatic --max-samples 10 --folder data/out

General documents use bounded passages and subject-appropriate QA prompts, rather than being forced
into legal article groups and consultation questions. Recognized laws/rulings can retain their
specialized segmentation. Unknown or failed topic assignments are reported explicitly and do not
make usable general text ineligible. Partial extraction/corrupt-text safeguards still apply.

See [the multidomain engineering notes](docs/PIPELINE.md#automatic-theme-discovery-and-multidomain-synthesis)
for costs, cache behavior, new settings, and limitations of model-based grouping.

This project extracts and cleans PDF, TXT, Markdown, and DOCX files for downstream model training. PDF routing is page by page: good text layers are kept, low text or image dominated pages go to OCR, and corrupted text layers are marked SUSPECT and sent to OCR. OCR results are cached by rendered page hash, provider, model, and endpoint. File completion records make successful runs resumable.

Files and pages are processed concurrently. A batch runs up to `FILE_CONCURRENCY` files at once, each file OCRs up to `PAGE_CONCURRENCY` pages at once, and `OCR_CONCURRENCY` caps simultaneous provider calls across the whole run. Defaults are 4/4/4; lower them if the provider rate-limits the account, or raise them for more throughput. Setting a limit to `0` removes that cap: `PAGE_CONCURRENCY=0` submits every scanned page of a document at once, and `OCR_CONCURRENCY=0` removes the batch-wide provider cap, so it must also be `0` if all pages should run together. `FILE_CONCURRENCY=0` starts every document immediately. These are exposed in the GUI under **Concurrency** in the OCR tab. Unlimited bursts can exhaust memory or trigger provider throttling; raise limits gradually. Page order in the output is always the original PDF order, and a failed page never aborts the rest of the document.

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
| `heiyu` | HeiyuCode (OpenAI-compatible) | `HEIYU_API_KEY` | `https://www.heiyucode.com/v1` | `gpt-6.1-sol` |
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
    python cli.py --no-benchmark .\data\raw\law.pdf
    python cli.py --reference-dir data/reference .\data\raw\law.pdf
    python cli.py --compare-benchmarks
    python app.py

The GUI listens on the local Gradio address printed at startup. Selecting a provider automatically loads that provider's base URL, model, and API key (from `.env`), so switching providers only requires choosing it from the dropdown. Typed values override `.env` for the current session only. The GUI has one tab per pipeline stage (all new GUI outputs default under `data/`):

| Tab | Purpose |
| --- | --- |
| **1 - OCR** | Per-page routing, extraction/OCR, raw text and page reports in `data/ocr/`. Benchmarks are a collapsed panel here. |
| **2 - Cleaning** | Independent format-cleaning and deterministic-correction switches; previews, TXT and rule counters in `data/cleaning/`. |
| **3 - Synthesis** | Chunking → metadata extraction → question agent → answer agent. Optional Chonkie/embedding controls are here. |
| **4 - Governance** | Screen/deduplicate `data/synthesis/qa.jsonl`; write pass/manual/intercept/duplicates in `data/governance/`. |
| **5 - Review** | Score/repair governed QA; accepted/rejected/manual and audit files in `data/review/`. |
| **6 - Corpus** | Export accepted QA to `data/corpus/corpus.jsonl`, with separate provenance and summary. |

Synthesis writes `data/synthesis/chunking/chunks.jsonl`, `metadata/`,
`questions/questions.jsonl`, `answers/answers.jsonl`, and `qa.jsonl`. Chunk exports carry source text,
language, source identity, and provenance; open-ended, evidence-backed discoveries are stored in
the `metadata` array rather than forced into legal-only columns. Legal structure is used to preserve
boundaries and route synthesis, not to fabricate `law_number` or `law_type` metadata. Questions add
`uuid`, `instruction`, `prompt_Q`; answers add `output`, `prompt_A`. Technical provenance and quality flags are additive. Review retains its structured dimension scores instead of copying
the example's misspelled `socre` free-text field. The training export contains only
`instruction`, `input` (empty), and `output`; source text/prompts remain in the earlier audit artifacts.

### Automated quality judging

The benchmark module can run deterministic chunk/source-integrity checks and use the configured
OpenAI-compatible text-generation provider as an opt-in LLM judge. Metadata judging checks cited
evidence against source text; QA judging considers the question/answer and source passage; review
judging independently assesses the saved review decision and scores. Reports are separate from
production JSONL and all judge calls are bounded (default 100 records per run; set the budget to 0
for deterministic checks only). LLM scores are triage signals, not ground truth.

```powershell
# Chunk coherence plus source/offset integrity; uses SYNTHESIS_MODEL unless overridden.
.\.venv\Scripts\python.exe cli.py --chunk-evaluate data/synthesis/chunking/chunks.jsonl
# Evaluate one other saved artifact at a time. Reports go under <stage-output>/evaluation/.
.\.venv\Scripts\python.exe cli.py --metadata-quality-evaluate data/training/metadata/documents.jsonl
.\.venv\Scripts\python.exe cli.py --qa-quality-evaluate data/training/synthesis/qa.jsonl
.\.venv\Scripts\python.exe cli.py --review-quality-evaluate data/training/review/audit.jsonl
# Deterministic evidence/source checks without model calls.
.\.venv\Scripts\python.exe cli.py --qa-quality-evaluate data/training/synthesis/qa.jsonl --quality-judge-max-calls 0
```

Run `.\.venv\Scripts\python.exe app.py`, then follow the six tabs in order. In Synthesis,
disable its metadata dry run to save actual chunks; **1. Chunk text** makes no metadata API calls.
Use **2. Extract metadata** with the model checkbox enabled for generic extraction. Then disable
the agent dry run, choose a text model/key, run **3. Generate questions**, and **4. Generate answers**.
The answer button reuses question checkpoints; it also generates the existing public explanation.
Dry runs create plans only, not usable new chunks/QA. Generic metadata and dialect checks remain
explicitly degraded/pending when no model is enabled. Offline Review can route records to manual;
Corpus never promotes them automatically.

The OCR tab keeps uploads/folder selection, per-file and per-page reports, raw text previews, and
TXT/JSON downloads. Operational settings can be changed directly in the GUI: no PowerShell
environment commands or configuration-file edits are needed. Advanced controls stay collapsed.
Edits override only the current session and are never written to `.env` or `config.py`.

- **Metadata:** enable/disable **Use Chonkie semantic chunking**, then select an embedding preset
  or type a custom model ID/local folder. Model choice is editable only while semantic chunking is
  enabled. Disable **Dry run** to run it. Model API extraction has its own checkbox and model field.
- **OCR:** page markers/template, languages (including custom hints), routing
  thresholds, DPI, concurrency, retries, benchmark options, force reprocessing, optional OCR connection
  test are available. **Cleaning** contains the independent format and semantic switches,
  correction tables and aggregated rule-counter downloads.
- **Synthesis:** screening mode, discovery-only, dry run, budgets, generation limits, segmentation,
  screening/override files and metadata provenance references are available. Overrides default to
  empty, so a missing optional example file cannot block a run.
- **Governance/review:** shared language/dialect controls select lexicon or local MARBERT and their
  paths/thresholds. Review adds a model toggle, repair limit and response token limit.
- **Metadata evaluation:** upload annotations and set precision/recall/example thresholds in its
  own panel. Benchmark comparison uses the shared history/reference folders.

The GUI output root defaults to `data` and is selected once above the tabs; changing it updates
the suggested inputs for Cleaning, Synthesis, Governance, Review and Corpus. Existing files are
not moved. Legacy CLI commands/default paths and `process_file`'s combined extraction/cleaning
behavior remain compatible. Benchmark folders, dialect settings, and text-provider
timeouts/retries also have one shared control group rather than copies in every tab. OCR retains
its own transport settings. Text-provider/key/URL settings apply to Synthesis, Metadata and Review;
the synthesis/review model and metadata model remain explicit separate choices. Emptying a model or
override field really overrides its environment default. Explicit synthesis/governance/review uploads
take precedence over prefilled input paths.

Stage **Run** buttons replace the CLI stage-selection flags (`SYNTHESIS_ENABLED`, `METADATA_ENABLED`,
`CORPUS_PROCESSING_ENABLED`, `REVIEW_ENABLED`); they need no extra checkboxes. Noncontiguous merging
(`METADATA_SKIP_WINDOW`) stays fixed at zero to preserve source order, rather than offering an invalid
setting. CLI output paths become GUI downloads; no duplicate filename options are required. All other
runtime `PipelineConfig` fields have a control or direct file/model/provider selector.

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

Each input produces a UTF-8 TXT file and a JSON route report under OUTPUT_DIR; summary.json reports batch totals. Page reports include route, extracted character count, valid-character ratio, estimated image coverage, warnings, cleaning counters, OCR failure status, and OCR latency/cache/confidence fields. OCR failures are recorded per page and do not abort the rest of the PDF. Failed or partial files are retried on the next run; successful page OCR is reused from the cache.

    pytest

## Benchmarking

Benchmarking is on by default. Each processed document writes `<stem>.metrics.json` next to its TXT
and report. It records reference-free quality proxies, OCR operations (pages routed/succeeded/failed,
cache hits, latency percentiles, pages per minute, provider confidence), cleaning before/after sizes
and per-rule hit counts, and a snapshot of the latest optional-stage summaries. These proxies flag
suspicious output; they are not accuracy measurements.

For real accuracy, add a corrected transcription at `data/reference/<source-stem>.txt`, or use
`--reference-dir`. The record then includes character CER, word WER, precision, recall, F1, raw OCR
accuracy, per-page CER when both texts keep `--- Page N ---` markers, and a raw-vs-cleaned CER delta
so you can see whether cleaning helped. A `<stem>.ref.txt` beside the source is also accepted.
Missing references are not an error. To benchmark documents processed before this feature existed,
rerun them with `--force`; cached OCR pages are reused, so no provider calls repeat.

Every completed batch appends one row per document to `data/benchmarks/history.jsonl` and rebuilds
`comparison.json`/`comparison.csv`, grouped by provider and model. Compare models over time with:

    python cli.py --compare-benchmarks --benchmark-report data/benchmarks/report.json

Provider and model are part of the OCR cache identity, so switching either forces fresh OCR. Keep
DPI, languages, concurrency, and cleaning settings fixed when comparing. Comparisons are capped at
`BENCHMARK_MAX_EVAL_CHARS` (100,000 by default) and set `truncated=true` when a document exceeds it.
See [docs/PIPELINE.md](docs/PIPELINE.md#7-benchmarking-ocr-and-pipeline-stages) for definitions,
normalisation, and the full schema.

## Scope and limitations

Stages 2.2–5 are now opt-in. Existing extraction invocations, concurrency, lam-alef repair, TXT bytes, and report fields retain their defaults. Stages 3–5 and benchmarking are available in the Gradio tabs, and the same stages remain available from the CLI for automation. See [docs/PIPELINE.md](docs/PIPELINE.md) for the engineering walkthrough, configuration reference, nine Mermaid diagrams with ASCII fallbacks, and implemented/degraded/specification-only matrix.

ftfy repairs common mojibake in extracted Unicode text, but it does not reconstruct arbitrary Arabic glyph mappings from embedded PDF font programs. Pages that fail the configured text-quality checks are sent to OCR. OCR uses the configured vision provider; the local PaddleOCR engine described in the separate README is not included.

Text-layer extraction repairs one specific PDF quirk: PyMuPDF reverses each Arabic span, so a lam-alef ligature glyph that expands to its two letters in logical order ends up swapped ("أولا" is read as "أوال", "الإدماج" as "اإلدماج"). The extractor uses MuPDF's glyph trace to identify exactly those synthesised ligature halves and swaps only them back; ordinary "ال" sequences are left untouched.

Semantic rules repair article-digit confusions and apply explicit reviewed glyph, date, footer, and word mappings. The shipped domain mappings and screening/dialect lexicons are empty; missing files are no-ops. No legal corrections or classifier quality are invented.

## Optional training stages

Run from the repository root, using the venv. Every optional stage is disabled in config by default; a CLI flag selects it for that invocation, and the equivalent GUI tab runs it interactively. Multiple stage selectors in one CLI invocation are rejected. In the GUI, stages run one at a time from their own tabs.

    .\.venv\Scripts\python.exe cli.py --clean --rules-report data/rules-hits.json data/in/law.pdf
    .\.venv\Scripts\python.exe cli.py --synthesize --dry-run --max-samples 10 --manual-overrides data/overrides.json data/out/law-HASH.txt
    .\.venv\Scripts\python.exe cli.py --synthesize --max-samples 10 --manual-overrides data/overrides.json data/out/law-HASH.txt
    .\.venv\Scripts\python.exe cli.py --process-corpus data/training/synthesis/qa.jsonl
    .\.venv\Scripts\python.exe cli.py --review data/training/governance/pass.jsonl
    .\.venv\Scripts\python.exe cli.py --review --llm-review data/training/governance/pass.jsonl

Live synthesis and LLM review require an explicitly configured SYNTHESIS_MODEL and the selected provider's environment key; the built-in text client supports OpenAI-compatible providers. Dry runs and default governance/review make no provider calls. An example manual override is provided under synthesis/data/overrides.example.json.

Synthesis writes candidates, source evidence, provenance, checkpoints, and failures. Governance writes pass/manual/intercept/duplicate JSONL. Review writes accepted/rejected/manual JSONL, a full audit artifact, and summary counts. Skipped semantic checks and unvalidated dialect classifiers route to manual review, never automatic acceptance.

No model weights or heavy ML packages are installed. MARBERT uses a local fine-tuned classification checkpoint with explicit labels, loaded only when selected. It falls back to lexicon/manual mode if unavailable. Configure validated thresholds only after the specification's annotation acceptance gates.

The specification requests hidden native chain of thought; this implementation stores a public explanatory rationale and sets native_chain_of_thought to null. Semantic grounding and personal names still need model/human review. The specification names eight review dimensions but lists nine; all nine listed dimensions are implemented.

Run the complete offline demonstration (synthetic text and fake scores, not training data), or the tests:

    .\.venv\Scripts\python.exe -m examples.offline_workflow
    .\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp data/pytest-run
