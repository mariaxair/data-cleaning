# Corpus pipeline engineering guide

## Automatic theme discovery and multidomain synthesis

This extension supersedes the legal-only screening/segmentation and GUI restrictions described
in the original guide below when **automatic** mode is selected. Legacy **legal** mode preserves
the keyword/manual policy. There is no predefined theme list in automatic mode.

### Workflow

~~~mermaid
flowchart TD
    Text["Cleaned documents from any domain"] --> Cache{"Topic description cached?"}
    Cache -->|yes| Descriptions["Model-discovered theme + summary"]
    Cache -->|no, live run| Model["Text model: distributed document excerpts"]
    Model --> Descriptions
    Descriptions --> Match["Model compares semantic meaning to learned groups"]
    Match --> Existing["Join closest group above similarity threshold"]
    Match --> New["Create new theme group"]
    Existing --> Catalog["topics.json + UI group table"]
    New --> Catalog
    Catalog --> Only{"Topics only?"}
    Only -->|yes| Stop["Stop before QA generation"]
    Only -->|no| Segment["General passages or recognized legal structure"]
    Segment --> QA["Subject-appropriate question, answer, public explanation"]
    Cache -->|dry run without cache| Pending["Theme pending; plan general passages without calls"]
~~~

ASCII: documents -> cached/model theme descriptions -> semantic comparison with learned groups
-> topics.json; stop in discovery-only mode OR segment -> generate subject-appropriate QA.

The model supplies a free topic title and summary in French, regardless of input language, plus
a structural route (general/statutory_law/case_law). That limited structure list is not a list of
themes: it only chooses a splitter. The model is instructed not to assume the document is legal.
Long documents are sampled at beginning, middle, and end; themes are approximate and may miss
secondary subjects. No text is discarded from the cleaned files on the basis of an unknown theme.

The semantic matcher compares each discovered description with every existing group, in batches
bounded by TOPIC_GROUP_BATCH_SIZE. It selects the highest model-scored match above the configured
threshold, or creates a group using the document's own discovered title/summary. This works even
when titles have different words; it is a model judgment rather than exact keyword matching.
Scores are not calibrated probabilities. Membership is incremental in input order and may change
if input order or model/prompt changes. It is not an embedding-based global clustering optimum.
Related broad subjects can still split or merge incorrectly; inspect the representative groups.
Multi-topic documents currently receive one primary group.

Descriptions run with SYNTHESIS_CONCURRENCY workers in bounded batches. Grouping is sequential
against the current catalogue. All discovery and matching attempts share TOPIC_MAX_CALLS, separate
from the QA max-samples limit. Descriptions and comparisons are cached by the full content
fingerprint/prompt/provider/endpoint/model/token cap, excluding filenames and credentials.
Identical files share description calls and membership. Changing file content invalidates its
description even when changed text is outside sampled excerpts.

If an output is invalid or a call fails, the configured validation-attempt limit applies within
the shared budget. Exceptions are replaced with safe warnings. An exhausted budget leaves the
remaining themes or matches explicitly pending; it never fabricates successful groups. Rerunning
reuses cache hits without consuming the call budget, so discovery can progress in controlled
increments. Deleting topics.json does not remove cached responses in synthesis/topic-cache/.
Changing the group threshold reuses model comparisons but recomputes decisions; different
catalogue batches require different comparison calls.

For 4,000 documents, one uncached description per document is already about 4,000 calls. Matching
adds up to ceil(current groups / batch size) calls per new content hash; highly diverse collections
may cost substantially more. Transport retries multiply underlying HTTP attempts. The default
100-call cap is a small trial budget, not a promise to classify an entire collection in one run.
No live requests were used to verify this extension.

Automatic mode no longer depends on SCREENING_PATH's legal keyword lists. An explicit manual
override still wins and avoids topic-discovery calls for that file; it is optional.
Unclassified usable documents get generic passage segmentation instead of mandatory manual tiering.
The model's uncalibrated confidence is exposed for inspection, not used as a quality certificate.
Extraction reports declaring partial/failed pages and known garbled text still exclude generation.

General passages are disjoint original character ranges bounded by GENERAL_CHUNK_CHARS.
The splitter prefers paragraph, line, then word boundaries, and records character offsets and
page markers. Very short spans under MIN_ARTICLE_CHARS do not generate QA. Full source text remains
in the existing cleaned export. If a document was structurally classified as legal but no useful
legal segments are found, automatic mode falls back to general passages with a warning.

General QA uses separate question/answer/explanation prompts with concepts, evidence, and practical
applications appropriate to the actual subject. There is no forced legal framework. General
questions may name legitimate source sections; legal questions retain their existing restriction.
Review/repair prompts likewise adapt to provenance.tier. Existing evidence, numeric, privacy,
language, dialect and review gates remain in force. Arabic-only script validation still rejects
Latin technical terminology; that existing language policy is separate from theme discovery.

Artifacts under STAGE_OUTPUT_DIR/synthesis:

| Artifact | Purpose |
| --- | --- |
| topics.json | Every analyzed file, its discovered theme/summary/confidence, grouping status, warnings, group catalogue and members |
| topic-cache/*.json | Validated model descriptions and semantic comparisons; no keys |
| topics-summary.json | Discovery-only summary/plan; does not overwrite existing QA files |
| plan.json | Screening, passages, QA selection cap, discovery counts and pending counts |
| qa.jsonl | Generated candidates with topic/group/character provenance for automatic mode |

The GUI shows a group table (theme, number of documents, member paths) and downloadable topics.json.
Files remain in their original folders. A dry run never calls a model, including discovery; it
reuses available caches or reports pending themes. Discovery-only runs do not write/replace
qa.jsonl, failed.jsonl, or the prior generation summary.json.

| Environment / config field | Default | Effect |
| --- | --- | --- |
| SYNTHESIS_SCREENING_MODE / synthesis_screening_mode | legal in CLI/config; automatic selected initially in GUI | Legacy policy or model discovery |
| TOPIC_SAMPLE_CHARS / topic_sample_chars | 12000 | Combined source excerpt character budget, plus separators |
| TOPIC_MAX_CALLS / topic_max_calls | 100 | New model generation attempts for discovery and grouping per invocation |
| TOPIC_GROUP_BATCH_SIZE / topic_group_batch_size | 20 | Maximum learned groups in one semantic comparison prompt |
| TOPIC_SIMILARITY_THRESHOLD / topic_similarity_threshold | 0.75 | Minimum model-scored semantic similarity to join a group |
| GENERAL_CHUNK_CHARS / general_chunk_chars | 4000 | Maximum general passage characters; must be >= MIN_ARTICLE_CHARS |

CLI: --discover-topics selects automatic mode and stops before QA. --synthesize --screening-mode
automatic runs both discovery and QA. --topic-max-calls overrides the discovery budget. All are
compatible with --folder, --dry-run, --stage-output, and --max-samples (QA cap only).

New modules: synthesis/topics.py handles open-topic discovery, semantic matching and cached/bounded
model calls; synthesis/passages.py provides generic character/page spans. The new topic/topic_match
and general_* prompt files are tracked. No embedding API, torch, transformers, GPU or weight
download is required. They use the existing configured compatible text-model client.

Example commands:

~~~powershell
# Preview generic passages without contacting a model.
.\.venv\Scripts\python.exe cli.py --discover-topics --dry-run --folder data/out
# Perform a bounded topic-only pass. Increase budget deliberately or rerun to continue.
.\.venv\Scripts\python.exe cli.py --discover-topics --topic-max-calls 100 --folder data/out
# Reuse cached discovery and generate at most ten QA candidates across general documents.
.\.venv\Scripts\python.exe cli.py --synthesize --screening-mode automatic --max-samples 10 --folder data/out
~~~

This guide describes the runnable implementation, including its limits. The source of truth is
[Corpus_Processing_Pipline.md](../Corpus_Processing_Pipline.md). Run commands from the repository root.
Python entry points are [cli.py](../cli.py) and the Gradio [app.py](../app.py), which exposes every
pipeline stage and benchmark comparison in tabs.
Configuration is exposed through [config.py](../config.py): its PipelineConfig inherits default-off
training fields from [settings.py](../corpus_pipeline/settings.py).

## Decisions and unresolved inputs

* §2.2 says eight semantic classes but lists seven. All seven listed families are represented;
  HTML and adjacent duplicates share the existing format-cleaning implementations and counters.
* §5.1 says eight review dimensions but lists nine. Review scores all nine, including the dialect veto.
* Screening signals, wrong-word substitutions, date corrections, foreign-glyph mappings, and footer
  identifiers were not supplied. Their tracked starter files are empty or commented examples.
  Missing optional files behave as empty data; malformed present files fail explicitly.
* SYNTHESIS_MODEL has no default. Choosing a real model, provider access, and a budget is an operator
  task. No live requests were made to develop or verify these stages.
* The source's six reasoning types are not all named. Only its four named types are rotated:
  conditional application, exception analysis, procedural tracing, and deadline calculation.
  Perspectives are first person and third person with generic roles, never invented party names.
* “Related articles” means consecutive articles within the same explicit chapter/section/law heading.
  It is a structural approximation; there is no claim of semantic clustering. Unpaired final articles
  are excluded. Law titles/numbers are supplied in manual metadata, never fabricated.
* No deterministic program can prove that every natural-language fact follows from a source or that
  all personal names have been removed. Evidence/numeric checks are partial; semantic grounding,
  privacy, and scenario quality stay pending until semantic/human review.
* Hidden native model chain of thought is neither requested nor stored. The third phase produces a
  public explanation; native_chain_of_thought is null and rationale_kind is public_explanation.
* No trained/validated dialect classifier was provided. The default lexicon is explicitly degraded,
  not a replacement for a validated model. Default review therefore does not auto-accept Arabic QA.

## 1. End-to-end workflow and contracts

The default command still performs extraction plus format cleaning. All remaining stages require a
flag or an explicitly enabled configuration setting. Stage selectors choose one optional stage per
CLI invocation; they do not silently cascade several paid operations. The GUI exposes the same stages
as separate tabs: Extract & clean, Synthesize, Govern, Review, and Benchmarks. Its extraction tab keeps
the original uploads, OCR settings, progress, page reports, previews, and downloads.

~~~mermaid
flowchart TD
    Raw["Raw PDF / TXT / MD / DOCX"] --> S1["1: per-page extraction and routing"]
    S1 --> Gate{"Clean embedded text?"}
    Gate -->|yes| Digital["Keep digital text"]
    Gate -->|no| OCR["Cached provider OCR"]
    Digital --> S2["2: format cleaning + optional semantic rules"]
    OCR --> S2
    S2 --> Text["TXT + page report"]
    Text --> S3["3: screen, segment, generate, validate"]
    S3 --> Candidates["QA candidates + evidence + provenance"]
    Candidates --> S4["4: dedup, language, dialect"]
    S4 -->|pass| S5["5: review and targeted repair"]
    S4 -->|uncertain| Manual["Manual queue"]
    S4 -->|intercept| Reject["Rejected audit records"]
    S5 -->|all dimensions pass| Accepted["Accepted JSONL"]
    S5 -->|uncertain| Manual
    S5 -->|veto or exhausted| Reject
~~~

ASCII: Raw -> extraction/OCR -> cleaning -> TXT -> synthesis -> QA -> governance ->
review -> accepted; every uncertain gate -> manual, every hard failure -> rejected audit.

### Stage 1: files and pages

The file SHA-256 and output-affecting settings are computed before processing. A complete manifest
record with matching source/settings and existing TXT/report files causes a skip. Partial records
are never treated as complete. The adapter can be constructed without credentials; a digital-only
file does not need a provider request.

For each PDF page, extraction measures character count, valid-character ratio, and image coverage.
Corrupt text layers are SUSPECT. Mostly-image or short-text pages are SCANNED. Other pages are
DIGITAL. Digital pages retain the Arabic glyph-origin lam-alef repair in extraction.py. Only OCR
pages are rendered. Page futures may finish in any order; indexed slots restore original order.
Failed OCR pages leave empty page text with warnings and a failed_pages entry; other pages continue.

OCR cache identities include rendered PNG SHA-256, provider, model, endpoint, and language hints.
Retries apply to transient network errors, timeouts, HTTP 429, and server failures. Authentication
and 404 configuration failures stop that page promptly. HTTP errors never serialize request headers;
known key values and recognizable key patterns are masked in provider details.

~~~mermaid
sequenceDiagram
    participant CLI as CLI / GUI
    participant File as process_file
    participant DB as WAL SQLite
    participant PDF as PdfPageSource
    participant API as OCR adapter
    CLI->>File: path and runtime config
    File->>File: hash input and effective settings
    File->>DB: complete manifest check
    alt complete artifacts exist
        DB-->>File: skip
        File-->>CLI: skipped result
    else new, changed, forced, or partial
        File->>PDF: text signals and page route
        loop scanned / suspect pages
            File->>PDF: render PNG under PDF lock
            File->>DB: page OCR cache lookup
            alt cache miss
                File->>API: OCR under global semaphore
                API-->>File: text or safe error
                File->>DB: store successful transcription
            end
        end
        File->>File: ordered pages, format and optional rules
        File->>File: export TXT and JSON report
        File->>DB: mark complete or partial
        File-->>CLI: previews and paths
    end
~~~

ASCII: hash -> manifest? -> skip OR route -> cache? -> OCR on miss -> ordered cleaning ->
TXT/report -> complete/partial manifest.

### Concurrency remains bounded and ordered

FILE_CONCURRENCY bounds file workers; each PDF has PAGE_CONCURRENCY workers. A single semaphore
created by process_batch caps provider requests across all those workers at OCR_CONCURRENCY.
Setting any of the three to 0 removes that particular limit: every file, every candidate page, or
every provider call starts immediately. To send all pages of every active document at once, set all
three to 0; page and OCR limits are independent, so `PAGE_CONCURRENCY=0` alone still respects the
global OCR cap. Unlimited values trade memory and rate-limit risk for wall-clock time.
PyMuPDF access (including rendering and lam-alef glyph tracing) stays under the existing global
re-entrant lock. SQLite uses WAL, a busy timeout, and connections per operation. Schema setup is
serialized. No new training stage changes this arrangement.

~~~mermaid
flowchart LR
    Batch["Batch input order"] --> Files["File pool: FILE_CONCURRENCY"]
    Files --> P1["File A page pool"]
    Files --> P2["File B page pool"]
    P1 --> Lock["Global PyMuPDF render / extraction lock"]
    P2 --> Lock
    Lock --> Sem["Shared OCR semaphore: OCR_CONCURRENCY"]
    Sem --> API["Provider requests"]
    API --> Slots["Indexed page result slots"]
    Slots --> Output["Original page and file order"]
    P1 <--> DB["WAL SQLite cache / manifest"]
    P2 <--> DB
~~~

ASCII: file pool -> per-file page pools -> serialized PDF access -> shared OCR semaphore ->
indexed results -> ordered export. SQLite is shared through independent connections.

Stage 3 uses its own bounded text-generation pool. It does not share the OCR semaphore, because it is
a separate CLI command and not part of a concurrently running OCR batch. Run one optional command per
output directory at a time; concurrent processes writing the same JSON output/checkpoint paths are
not supported.

### Stage 2: format and semantic cleaning

The format path retains ftfy, line-ending normalization, anchored metadata removal, HTML removal,
conservative repeated-line collapse (runs of at least three), and adjacent-word collapse.
Counters keep their existing names and behavior. ftfy and shared HTML/duplicate functions run once,
not once per stage label.

With semantic_rules_enabled false, neither new semantic counters nor semantic report metadata are
added. The old settings fingerprint is unchanged. With it true, the file fingerprint additionally
includes the semantic-rule version and the contents of both rule-data files. A mapping change
therefore reruns cleaning while reusing successful page OCR.

~~~mermaid
flowchart TD
    Input["Page text"] --> Format["Existing format rules and counters"]
    Format --> Enabled{"Semantic rules enabled?"}
    Enabled -->|no| Output["Cleaned text and hit counts"]
    Enabled -->|yes| Digits["Article I/l digit repair"]
    Digits --> Scripts["Reviewed foreign glyph mappings"]
    Scripts --> Dates["Exact reviewed date mappings"]
    Dates --> Footers["Exact configured footer markers"]
    Footers --> Words["Curated wrong_words.tsv"]
    Words --> Stable{"Second application unchanged?"}
    Stable -->|yes| Output
    Stable -->|no| Error["Reject non-idempotent rule configuration"]
    Digits -.-> Counts["Per-family hit counters"]
    Scripts -.-> Counts
    Dates -.-> Counts
    Footers -.-> Counts
    Words -.-> Counts
~~~

ASCII: format -> optional [article digits -> glyphs -> dates -> footers -> words] ->
stability check -> text + per-rule counts.

| Rule module | Conservative scope | Counter |
| --- | --- | --- |
| rules/digits.py | I/l -> 1 after Article, Art., or المادة; if Arabic-Indic digits already occur in that number, use ١. Existing digits are never converted. | article_digit_confusion |
| rules/scripts.py | Single Hebrew/Cyrillic/Armenian glyph between Arabic letters; explicit Arabic replacement required. Foreign-language spans are preserved. | foreign_script_corrections |
| rules/html.py | Existing br -> newline and ASCII HTML-tag stripping, shared with the format path. | html_residue |
| rules/dates.py | Exact whole-phrase date mappings only; no implicit month/day or calendar conversion. | explicit_date_corrections |
| rules/duplicates.py | Existing consecutive equal-word collapse. | adjacent_word_duplicates |
| rules/footers.py | Exact configured footer markers between non-space body characters; no blanket deletion of legal numbers. | embedded_page_numbers |
| rules/tables.py | Reviewed, boundary-aware TSV phrase replacements; chains/cycles rejected. Missing TSV is a no-op. | curated_wrong_words |

Article I can be a genuine Roman numeral. Enabling the digit rule asserts that the targeted article
labels use decimal numbering. Keep semantic rules disabled for documents where that assumption is
not valid. Rule counts measure corrected matches, not individual glyphs. Aggregate counters are
available through --rules-report; skipped files contribute their saved report counters.

Example data shape (illustrative only; do not copy as a legal correction policy):

~~~json
{
  "foreign_glyphs": {},
  "dates": {"12/25/2024": "25/12/2024"},
  "footer_markers": ["[page 12]"]
}
~~~

wrong_words.tsv has a tab-separated wrong/correct header. Its commented spelling examples are
inactive. Mappings must be terminal across all families and formatting; if a second pass would
change the result, cleaning fails instead of silently accumulating changes.

### Stage 3: screening, segmentation, generation

Input is cleaned UTF-8 TXT/MD, not raw PDF. The runner records byte sizes and keyword hits. Manual
overrides use a resolved path or filename and supply tier, title/number, and known sensitive values.
Prefer absolute paths when different folders contain the same filename.

The screening JSON accepts statutory_law, case_law, administrative_noise, exclude_article_terms
lists, and a validated boolean. Conflicting or absent tier signals are manual. Keyword-only
screening remains degraded unless validated is true; unvalidated matches cannot enter generation.
An explicit manual tier overrides keyword screening. Administrative noise is excluded. A sibling
extraction report declaring failed pages/partial status also excludes that input from generation.

Statutory articles are located by explicit Article/Art./المادة headings and grouped within explicit
heading boundaries, using 2–4 articles. Preambles and mastheads before the first article are left
out; dotted index rows and too-short articles are excluded. Configured procedural/non-substantive
terms exclude articles and split groups. Unknown procedural content is not guessed. Case law is
kept as one complete unit. Known replacement/private-use corruption rejects segmentation; scrambled
but valid Arabic is still a semantic/manual concern.

The prompts are text files loaded with string.Template. Questions rotate perspective and the four
specified reasoning tasks. The answer receives source plus question. The explanatory rationale
receives only question/answer, never the raw source. Treating source text as untrusted is stated in
each source-consuming prompt.

~~~mermaid
flowchart TD
    Input["Cleaned text + extraction report"] --> Screen["Tier signals / manual override"]
    Screen -->|manual or partial| Queue["Plan: manual / excluded"]
    Screen -->|eligible| Seg["Whole ruling or 2-4 article groups"]
    Seg --> Cost["Select at most max-samples"]
    Cost --> Dry{"Dry run?"}
    Dry -->|yes| Plan["Plan JSON; no model calls"]
    Dry -->|no| Pre["Configured key/model + compatible client"]
    Pre --> Q["Generate question"]
    Q --> VQ{"Validate text and evidence"}
    VQ -->|retry budget| Q
    VQ -->|valid| A["Generate grounded answer"]
    A --> VA{"Validate source/article/language/privacy signals"}
    VA -->|retry budget| A
    VA -->|valid| R["Generate public explanation from Q + A only"]
    R --> VR{"Validate language/numbers/length"}
    VR -->|retry budget| R
    VR -->|valid| QA["Checkpoint and QA JSONL: pending review"]
    VQ -->|exhausted| Failed["Failed JSONL"]
    VA -->|exhausted| Failed
    VR -->|exhausted| Failed
~~~

ASCII: tier -> segment -> sample cap -> dry-run plan OR question/validate -> answer/validate ->
explanation/validate -> pending QA. Each invalid phase retries within a bound; failures are isolated.

Observable validators enforce:

* Nonempty target-script text. Arabic output containing Latin or other non-Arabic letters fails.
* Article references must occur verbatim in both source and unit metadata; no digit-script conversion.
* Numeric tokens must occur in the source; for the explanation, in the question/answer.
* Questions cannot name the supplied law title/number or an explicitly numbered article.
* Known sensitive strings, email/phone patterns, and some explicit identity/case/birth/address labels
  fail. Unknown names require semantic review.
* Question/answer responses must include claim fragments and exact source quotations. Both ends
  are checked. Coverage and semantic entailment cannot be established by string checks alone.
* The public explanation must be longer than the answer.

The QA schema contains id, question, answer, reasoning_chain, native_chain_of_thought (null),
rationale_kind, language, question_type, reasoning_type, scores, validations, pending_checks,
provider/model, evidence, source_text, and provenance. Provenance includes the original source/hash
when an extraction report is available, cleaned source/hash, unit id, title/number, article numbers,
page markers, structural category, and tier. Missing page markers produce an empty pages list,
not invented page numbers.

Source text and evidence are internal audit data and can contain source personal information.
They stay under ignored data/. Generated fields must satisfy privacy gates. The final accepted
training export excludes source_text, sensitive_values, evidence, governance internals, and review
history; the separate audit file retains traceability.

Each successful phase writes an atomic checkpoint. Identity covers unit/source/metadata, prompt
contents, model, provider/endpoint, language, token cap, and rotation position. Successful phases
are revalidated and reused on rerun. Failures retry without losing completed phases. Changing
prompts/model/source invalidates the corresponding identity. --force applies to extraction only;
remove a specific synthesis checkpoint to regenerate it. Never edit checkpoints to mark invalid
content as valid.

--max-samples caps units scheduled per invocation, including resumed units. It is a sample/request
guard, not a dollar budget. At most selected_units * 3 * SYNTHESIS_MAX_ATTEMPTS generation
attempts are possible, each with at most OCR_MAX_RETRIES transport retries. Live model responses
are not guaranteed byte-deterministic even at temperature zero; saved validated checkpoints stabilize
reruns. Live startup configuration is validated before queuing work; there is no startup network probe.

### Stage 4: deduplication and governance

Questions are normalized using NFKC, casefolding, diacritic/tatweel removal, alef/hamza and alef-maqsura
variants, punctuation-as-space, and whitespace collapse. Only normalized exact duplicates collapse.
The first record wins in input order; duplicates retain duplicate_of and original metadata for audit.
No embedding model or near-duplicate claim is involved.

Language governance measures letter scripts independently for question, answer, and rationale.
The mixing ratio is min(Arabic letters, Latin letters) / all letters. Above the configured threshold,
the sample is intercepted. Arabic-majority records get an Arabic assessment and misplaced-label
flag when originally labeled otherwise. Latin script alone cannot distinguish French from English:
it stays manual with detected=latin. Other scripts and empty fields are manual or invalid.

~~~mermaid
flowchart TD
    QA["QA JSONL in stable order"] --> Norm["Normalize question"]
    Norm --> Dup{"Seen exact normalized question?"}
    Dup -->|yes| Dups["duplicates.jsonl"]
    Dup -->|no| Lang["Per-field script/language assessment"]
    Lang --> Mix{"Mixing exceeds threshold?"}
    Mix -->|yes| Intercept["intercept.jsonl"]
    Mix -->|no| Classify["Per-field dialect classifier"]
    Classify --> Validated{"Calibrated and validated?"}
    Validated -->|yes| Low{"Moroccan score below 0.2?"}
    Low -->|yes| Pass["pass.jsonl"]
    Low -->|no| High{"Above 0.5?"}
    High -->|yes| Intercept
    High -->|no| Manual["manual.jsonl"]
    Validated -->|no| Fallback["Lexicon or unvalidated model"]
    Fallback -->|uncertain / lexicon| Manual
    Fallback -->|calibrated model above 0.95| Intercept
~~~

ASCII: normalize -> duplicate? -> language/mixing -> dialect per field ->
pass/manual/intercept. Default lexicon always yields manual; empty lexicon does not imply MSA.

DialectClassifier.classify returns DialectPrediction with class scores, backend, degraded/calibrated
flags, and evidence. Lexicon scores are hit indicators, not calibrated probabilities; even a hit
does not automatically delete a record. A validated injected/local model uses strict <0.2 pass,
0.2–0.5 inclusive manual, >0.5 intercept. Unvalidated calibrated models require >0.95 to intercept;
all other cases remain manual. The most restrictive field decides the sample.

The optional MARBERT adapter requires an existing local fine-tuned sequence-classification
checkpoint and an explicit model-label -> MSA/Algerian/Moroccan label_map in the lexicon JSON.
The MARBERTv2 pretrained backbone alone is insufficient. Imports of transformers/torch are lazy,
tokenizer/model loading uses local_files_only=True, inference is CPU by default, and missing
dependencies/files/labels trigger an explicit degraded fallback. Long fields are chunked; reported
per-label scores are maxima across chunks, not a normalized field-level probability distribution.
Threshold calibration must account for that aggregation.

Before enabling DIALECT_VALIDATED, complete the specification's out-of-domain Kappa >=0.7 and
in-domain macro-F1 >=0.85 gates with manual annotation. This repository does not train a classifier
or claim those metrics; the boolean is the operator's attestation.

### Stage 5: review and cause-targeted repair

Each dimension scores 0=fail, 1=uncertain/skipped, or 2=pass. Acceptance requires all nine scores to
be 2 and no pending/skipped checks.

| Dimension | Implementation | Unavailable/uncertain behavior |
| --- | --- | --- |
| Question accuracy | Injectable semantic reviewer plus deterministic named-law/article veto | 1/manual |
| Reasoning completeness | Injectable semantic reviewer checks answer and public rationale | 1/manual |
| Question self-containedness | Injectable semantic reviewer | 1/manual |
| Privacy compliance | Known-identifier rules plus semantic reviewer for names/context | 1/manual; rule failures cannot be overruled |
| Language purity | Unicode script checks per field | 0 on failure; language-identity ambiguity remains manual |
| Markdown compliance | Basic fence/raw-HTML/link-delimiter checks | 0 on failure; not a full Markdown parser |
| Fluency/no repetition | Adjacent word and repeated-line checks | 0 on failure; grammatical fluency still limited |
| Corpus accuracy | Exact articles, numeric-source checks, encoding residue, supplied evidence | 1/manual if source missing |
| Dialect compliance | Classifier score and immutable existing dialect veto | 0/reject on veto; 1/manual on uncertainty |

~~~mermaid
stateDiagram-v2
    [*] --> Score
    Score --> Reject: Moroccan dialect veto
    Score --> Accept: all scores 2 and no skipped checks
    Score --> Manual: uncertain or skipped
    Score --> Repair: repairable score 0
    Repair --> Manual: no repair adapter
    Repair --> Reject: budget exhausted or invalid patch
    Repair --> Score: patch allowed fields only
    Accept --> [*]
    Reject --> [*]
    Manual --> [*]
~~~

ASCII: score -> accept / manual / reject; repairable fail -> scoped repair -> rescore;
dialect veto -> reject immediately, never repair.

Cause-to-field routing:

* Rationale language error: reasoning_chain only; question/answer are untouched and no raw source is
  sent in that repair request.
* Answer language/article/source-number error: answer only.
* Defective question: question and answer.
* Incomplete answer reasoning: answer.
* Privacy failure: all generated fields, since leaking personal data can affect the whole sample.
* Markdown/repetition failures: the affected fields; if the question changes, also regenerate answer.

A repair adapter receives a deep copy and must return every requested text field and optional
evidence for changed fields. Unknown fields/provenance changes, empty output, exceptions, and
malformed evidence patches reject the sample. Every valid patch is rescored through the same gates.
Semantically stale dependent fields are assessed by the semantic reviewer; they are not assumed valid.
No adapter means manual review rather than simulated repair. The default offline review never invokes
a model. --llm-review opts in to both compatible semantic review and repair, with the synthesis model.

Review is sequential and bounded by REVIEW_MAX_ATTEMPTS per record; the initial score is not a repair
attempt. Semantic review/repair do not have persistent caches: rerunning paid review recomputes them.
Each operation still has bounded HTTP retries. Use a deliberately small JSONL subset for paid review.
Review artifacts are atomically replaced; audit.jsonl stores per-attempt scores, skipped dimensions,
decisions, repaired fields, and failure reasons.

## 2. Artifacts, caching, and recomputation

~~~mermaid
flowchart LR
    Raw["data/in"] --> Extract["Extraction + cleaning"]
    DB[("data/pipeline.sqlite3")]
    Extract <--> DB
    Extract --> TXT["data/out/stem-hash.txt"]
    Extract --> Report["data/out/stem-hash.report.json"]
    Extract --> Metrics["data/out/stem-hash.metrics.json"]
    Extract --> Batch["data/out/summary.json"]
    Metrics --> History["data/benchmarks/history.jsonl"]
    History --> Compare["comparison.json + comparison.csv"]
    TXT --> Synth["training/synthesis"]
    Report --> Synth
    Synth --> Plan["plan.json + dry-run-summary.json"]
    Synth --> Check["checkpoints/id.json"]
    Synth --> QA["qa.jsonl + failed.jsonl"]
    QA --> Gov["training/governance"]
    Gov --> Buckets["pass / manual / intercept / duplicates JSONL"]
    Buckets --> Review["training/review"]
    Review --> Final["accepted / rejected / manual JSONL"]
    Review --> Audit["audit.jsonl + summary.json"]
~~~

ASCII: raw -> SQLite cache + TXT/report -> synthesis plan/checkpoints/QA ->
governance buckets -> review training exports + audit. All default runtime artifacts live under data/.

| Artifact | Reused/skipped | Recomputed |
| --- | --- | --- |
| Successful OCR page | Reused on matching page/provider/model/endpoint/languages | On page rendering/provider settings change or missing cache |
| File TXT/report | Skipped only with complete matching manifest and existing artifacts | Partial, forced, changed source/settings/rules |
| Per-document metrics | Rewritten on every processed run; skipped files keep their existing file | Forced or changed source/settings/provider/model |
| Benchmark history/comparison | `history.jsonl` is append-only; `comparison.json/.csv` rebuilt from it | Every completed batch; inspect old rows to compare models over time |
| Failed OCR page | Never cached as successful | Retried on partial-file rerun |
| Synthesis plan | No model calls in dry run | Screening/segmentation rebuilt each invocation |
| Successful synthesis phase | Valid checkpoint identity reused and revalidated | Changed identity, missing checkpoint, failed validation |
| Governance buckets | No hidden cross-run dedup cache | Deterministically rebuilt from supplied records |
| Review buckets/audit | No paid-review checkpoint | Rescored each invocation, including repairs |

New JSON/JSONL writes use UTF-8, LF, sorted keys, and atomic replacement. Legacy TXT/report writing
was intentionally not changed, including platform line endings. Do not compare LF-only new artifacts
to legacy CRLF exports as if their newline convention were a regression.

## 3. Module-by-module map

Paths below are relative to repository root.

| File | Responsibility / key entry points |
| --- | --- |
| config.py | ProviderProfile catalogue, OCR/base settings, PipelineConfig, key resolution |
| cli.py | Backward-compatible extraction flags and explicit optional stage dispatch |
| app.py | Gradio tabs for extraction, synthesis, governance, review, and benchmarks; shared text-generation settings; no CLI dispatch |
| corpus_pipeline/__init__.py | Lazy public process_file wrapper, preventing config import cycles |
| corpus_pipeline/settings.py | TrainingSettings dataclass, env_bool, load_json, settings validation |
| corpus_pipeline/models.py | OCRResult, PageSignals, PageRoute, PageReport |
| corpus_pipeline/routing.py | measure_page, classify_page |
| corpus_pipeline/extraction.py | PdfPageSource, rendering lock, _page_text, _fix_ligature_pairs, TXT/DOCX extraction |
| corpus_pipeline/cleaning.py | clean_text, legacy counters, optional semantic hook, idempotency guard |
| corpus_pipeline/cache.py | PipelineCache page results and file manifest; WAL and thread-safe initialization |
| corpus_pipeline/processor.py | process_file/process_batch, shared semaphore, ordered _process_pdf, _config_hash |
| corpus_pipeline/logging_utils.py | JSONFormatter and configure_logging |
| corpus_pipeline/artifacts.py | fingerprint, atomic_text, read_jsonl, write_json/write_jsonl |
| corpus_pipeline/benchmark.py | Per-document quality/OCR/cleaning/reference metrics, history, comparison |
| corpus_pipeline/ocr/base.py | OCRAdapter protocol, probe_image, shared retries and safe error redaction |
| corpus_pipeline/ocr/openai_compatible.py | Endpoint construction, vision payload/response parsing |
| corpus_pipeline/ocr/openai.py | OpenAI adapter defaults |
| corpus_pipeline/ocr/heiyu.py | Heiyu compatible adapter defaults |
| corpus_pipeline/ocr/mistral.py | Mistral compatible adapter defaults |
| corpus_pipeline/ocr/anthropic.py | Messages API image/auth normalization |
| corpus_pipeline/ocr/google.py | Gemini image/auth normalization |
| corpus_pipeline/ocr/azure.py | Azure deployment URL and key normalization |
| corpus_pipeline/ocr/registry.py | create_adapter/register_adapter/provider ordering |
| corpus_pipeline/ocr/__init__.py | Adapter factory exports |
| corpus_pipeline/rules/__init__.py | apply_rules order and per-family counters |
| corpus_pipeline/rules/digits.py | Context-limited article digit repair |
| corpus_pipeline/rules/scripts.py | Explicit foreign glyph corrections inside Arabic |
| corpus_pipeline/rules/html.py | Shared HTML cleanup |
| corpus_pipeline/rules/duplicates.py | Shared adjacent-word cleanup |
| corpus_pipeline/rules/dates.py | Explicit reviewed date replacement |
| corpus_pipeline/rules/footers.py | Exact reviewed footer marker removal |
| corpus_pipeline/rules/tables.py | TSV loader and terminal whole-phrase replacements |
| corpus_pipeline/synthesis/__init__.py | Stage package |
| corpus_pipeline/synthesis/screening.py | ScreeningResult, screen_document |
| corpus_pipeline/synthesis/segmentation.py | SynthesisUnit, article regex, segment_document |
| corpus_pipeline/synthesis/validation.py | ValidationResult, source/article/script/privacy checks |
| corpus_pipeline/synthesis/client.py | GenerationAdapter, compatible HTTP text client/preflight |
| corpus_pipeline/synthesis/runner.py | synthesize, generate_unit, prompt loading, checkpoints |
| corpus_pipeline/synthesis/prompts/*.txt | Separate question, answer, public explanation instructions |
| corpus_pipeline/governance/__init__.py | Stage package |
| corpus_pipeline/governance/dedup.py | normalize_question |
| corpus_pipeline/governance/language.py | assess_language |
| corpus_pipeline/governance/dialect.py | DialectClassifier, lexicon/local model, dialect_verdict |
| corpus_pipeline/governance/runner.py | govern_record, process_corpus, four output buckets |
| corpus_pipeline/review/__init__.py | Stage package |
| corpus_pipeline/review/checks.py | ReviewResult, deterministic_review, dimension names |
| corpus_pipeline/review/adapters.py | Injectable review/repair protocols and model implementations |
| corpus_pipeline/review/runner.py | score_record, route_review, repair_fields, review_record/review_corpus |
| corpus_pipeline/review/prompts/*.txt | Scoring rubric and cause-targeted repair contract |
| examples/offline_workflow.py | Synthetic all-stage demonstration, explicitly fake scores |
| tests/conftest.py | Prevent accidental requests-based network calls throughout pytest |
| tests/test_semantic_rules.py | Positive/negative/idempotent rules and legacy output regression |
| tests/test_synthesis.py | Screening, grouping, validators, provenance, retries, resume/dry-run |
| tests/test_governance.py | Normalization, mixing, threshold boundaries, fallback |
| tests/test_review.py | Nine scores, manual/skipped checks, veto and repair scope/budget |
| tests/test_optional_workflows.py | CLI integration, page/chapter provenance, redaction/idempotency edge cases |
| tests/test_offline_example.py | All-stage fake-adapter workflow |
| tests/test_config.py, test_providers.py, test_registry.py | Original provider/default/key/probe tests retained |
| tests/test_parallel.py, test_mixed_pdf.py, test_routing_cleaning.py | Original ordering, WAL, ligature, mixed-PDF, cleaner tests retained |
| tests/test_benchmark.py | Normalisation, CER/WER, quality proxies, references, per-document and history artifacts |

## 4. Configuration reference

Existing extraction settings remain module constants/defaults in config.py. New fields are accessible
on the same PipelineConfig and through the environment names below. The .env file is loaded without
overriding shell variables. CLI flags override stage selection and explicitly provided runtime paths.
No complete config object is serialized because it can contain session credentials.

### Existing extraction settings

| Constant / field | Default | Effect |
| --- | --- | --- |
| INPUT_DIR / input_dir | data/in | CLI fallback input folder |
| OUTPUT_DIR / output_dir | data/out | Legacy TXT/report/summary directory |
| SUPPORTED_LANGUAGES / languages | ar, fr, en | OCR language hints; routing remains script-agnostic |
| INCLUDE_PAGE_MARKERS / include_page_markers | true | Add markers before pages |
| PAGE_MARKER_TEMPLATE / page_marker_template | two newlines + "--- Page {page} ---" + two newlines | Page origin trace in TXT |
| MIN_EXTRACTED_CHARS / min_extracted_chars | 80 | Short extracted pages route to OCR |
| MIN_VALID_CHAR_RATIO / min_valid_char_ratio | 0.85 | Corruption threshold for SUSPECT |
| FULL_PAGE_IMAGE_RATIO / full_page_image_ratio | 0.65 | Mostly-image page threshold |
| PDF_RENDER_DPI / render_dpi | 220 | OCR rendering resolution and page hash |
| OCR_PROVIDER / ocr_provider | heiyu | Active adapter/profile |
| OCR_API_KEY / ocr_api_key | empty | Legacy/session fallback; keys belong in env/UI |
| session_api_key_override | false | Nonempty GUI key wins for this request/session |
| OCR_BASE_URL / ocr_base_url | empty -> provider profile | Endpoint override |
| OCR_MODEL / ocr_model | empty -> provider profile | OCR vision model |
| OCR_TIMEOUT_S / ocr_timeout_s | 60 | HTTP timeout, reused by text model calls |
| OCR_MAX_RETRIES / ocr_max_retries | 3 | Transport retries beyond initial request |
| OCR_RETRY_BASE_S / ocr_retry_base_s | 1.0 | Exponential backoff base |
| OCR_TEST_CONNECTION_ON_STARTUP | false | Existing constant is not automatically consumed; use --test-connection |
| FILE_CONCURRENCY / file_concurrency | 4 | Concurrent file workers; 0 = all files at once; existing env supported |
| PAGE_CONCURRENCY / page_concurrency | 4 | OCR page workers per PDF; 0 = all candidate pages at once; existing env supported |
| OCR_CONCURRENCY / ocr_concurrency | 4 | Batch-wide OCR provider-call cap; 0 = no cap; existing env supported |
| CACHE_DB / cache_db | data/pipeline.sqlite3 | Page OCR and completion manifest |
| LOG_LEVEL | INFO | JSON logger level |

ProviderProfile fields are id, label, api_key_env, base_url, model, requires_base_url, and notes.
Current profile key variables: HEIYU_API_KEY, OPENAI_API_KEY, ANTHROPIC_API_KEY, GOOGLE_API_KEY,
MISTRAL_API_KEY, AZURE_OPENAI_API_KEY, and OCR_API_KEY for generic compatibility. Resolution:
nonempty GUI override > selected-provider environment > legacy OCR_API_KEY > config fallback.
Shell variables take priority over .env. Profile endpoints/models remain documented in README.md.

### Optional-stage settings: environment -> dataclass field

All positive sizes/limits and threshold ordering are validated at stage entry. Booleans accept
1/true/yes (case-insensitive); other values are false. Paths are relative to the working directory
unless absolute, so run from repository root. New fields are evaluated when PipelineConfig is built.

| Environment | Field | Default | Effect |
| --- | --- | --- | --- |
| SEMANTIC_RULES_ENABLED | semantic_rules_enabled | false | Enables Stage 2.2 inside existing cleaning |
| RULES_DATA_PATH | rules_data_path | corpus_pipeline/rules/data/rules.json | Explicit glyph/date/footer mappings |
| WRONG_WORDS_PATH | wrong_words_path | corpus_pipeline/rules/data/wrong_words.tsv | Reviewed word/phrase replacements |
| SYNTHESIS_ENABLED | synthesis_enabled | false | Selects Stage 3 in CLI absent an explicit stage flag |
| SYNTHESIS_MODEL | synthesis_model | empty | Mandatory live text model; no OCR-model fallback |
| SYNTHESIS_LANGUAGE | synthesis_language | ar | Target output script/language |
| SYNTHESIS_CONCURRENCY | synthesis_concurrency | 2 | Maximum simultaneous generation unit workers |
| SYNTHESIS_MAX_SAMPLES | synthesis_max_samples | 10 | Unit scheduling cap |
| SYNTHESIS_MAX_ATTEMPTS | synthesis_max_attempts | 2 | Validation/generation attempts per phase |
| SYNTHESIS_MAX_TOKENS | synthesis_max_tokens | 2048 | Output token cap per text call |
| SCREENING_PATH | screening_path | corpus_pipeline/synthesis/data/screening.json | Tier keywords/exclusion terms and validation attestation |
| MANUAL_OVERRIDES_PATH | manual_overrides_path | unset | Document tier/legal metadata/sensitive values |
| ARTICLE_GROUP_SIZE | article_group_size | 3 | Must be 2, 3, or 4 |
| MIN_ARTICLE_CHARS | min_article_chars | 40 | Minimum retained article span size |
| CORPUS_PROCESSING_ENABLED | corpus_processing_enabled | false | Selects Stage 4 in CLI |
| LANGUAGE_MIXING_THRESHOLD | language_mixing_threshold | 0.2 | Intercept when minority Arabic/Latin letter share is greater; valid range 0–0.5 |
| DIALECT_BACKEND | dialect_backend | lexicon | lexicon or marbert |
| DIALECT_MODEL_PATH | dialect_model_path | unset | Existing local fine-tuned classifier directory |
| DIALECT_LEXICON_PATH | dialect_lexicon_path | corpus_pipeline/governance/data/lexicon.json | MSA/Algerian/Moroccan lists and optional label_map |
| DIALECT_PASS_THRESHOLD | dialect_pass_threshold | 0.2 | Strict lower bound for validated-model pass |
| DIALECT_INTERCEPT_THRESHOLD | dialect_intercept_threshold | 0.5 | Strict upper bound for validated-model intercept |
| DIALECT_DEGRADED_INTERCEPT_THRESHOLD | dialect_degraded_intercept_threshold | 0.95 | Unvalidated calibrated model high-confidence intercept |
| DIALECT_VALIDATED | dialect_validated | false | Attests completed annotation/metric gates for local model |
| REVIEW_ENABLED | review_enabled | false | Selects Stage 5 in CLI |
| REVIEW_LLM_ENABLED | review_llm_enabled | false | Enables compatible model semantic review and repair |
| REVIEW_MAX_ATTEMPTS | review_max_attempts | 2 | Repairs after initial score; zero means no repair attempts |
| STAGE_OUTPUT_DIR | stage_output_dir | data/training | Root of synthesis/governance/review artifacts |

### Benchmark settings: environment -> dataclass field

| Environment | Field | Default | Effect |
| --- | --- | --- | --- |
| BENCHMARK_ENABLED | benchmark_enabled | true | Write per-document metrics and history; `--no-benchmark` disables |
| BENCHMARK_DIR | benchmark_dir | output_dir parent + `/benchmarks` (`data/benchmarks` by default) | Append-only `history.jsonl` and `comparison.json/.csv` |
| BENCHMARK_REFERENCE_DIR | benchmark_reference_dir | data/reference | Folder of `<source-stem>.txt` reference transcriptions |
| BENCHMARK_MAX_EVAL_CHARS | benchmark_max_eval_chars | 100000 | Cap per comparison; larger documents set `truncated=true` |
| BENCHMARK_STRIP_DIACRITICS | benchmark_strip_diacritics | true | Remove Arabic harakat and tatweel before comparison |
| BENCHMARK_UNIFY_ALEF | benchmark_unify_alef | false | Map أإآٱ to ا before comparison |
| BENCHMARK_UNIFY_YA | benchmark_unify_ya | false | Map ى to ي before comparison |

Reference-free metrics are cheap and always on. A reference transcription is never uploaded or sent
to a provider; it is read from disk only. The comparison cap applies to both sides of one comparison,
so raw and cleaned error rates remain comparable.

Explicit --synthesize, --process-corpus, or --review wins over the environment's stage selectors.
More than one selected stage is an error. Setting several stage-enabled booleans and providing no
CLI selector is likewise an error. --clean separately enables semantic rules on extraction.
Synthesis/governance/review functions do not gate themselves on their enable flag: explicitly calling
the stage API is itself opt-in. Only the CLI uses those flags to select a command. The Gradio app
runs a stage only when its tab button is pressed; environment stage selectors do not auto-run stages
there. The generation provider list is restricted to OpenAI-compatible providers because Synthesis
and LLM Review share the text client.

## 5. Failure modes and recovery

| Failure | Observable result | Recovery |
| --- | --- | --- |
| OCR timeout/rate limit/server error | Bounded backoff; page failure after retries | Lower OCR concurrency, check service, rerun partial file |
| OCR page failure | Empty page text, partial status, failed_pages; later pages continue | Rerun; successful page cache is reused |
| Invalid/missing OCR key | Safe provider/key-variable error; no key in logs | Correct env/UI key, restart to reload .env |
| Wrong endpoint/model | Safe 404/HTTP message; no request dump | Correct provider profile/base/model |
| Password-protected PDF | File error | Supply an unencrypted authorized copy; no password workflow added |
| Legacy cache corruption | SQLite error/file failure; no silent reset | Stop writers, preserve DB and WAL/SHM for diagnosis; move the corrupt DB set aside and rerun with a fresh CACHE_DB |
| Missing optional domain table | Empty/no-op family, degraded metadata/docs | Provide reviewed table/path and rerun; fingerprint changes |
| Malformed/cyclic/non-idempotent rule table | Clear validation error | Correct mappings; no partial correction accepted |
| Unclassified/conflicting document | Manual tier in plan; no generation | Add reviewed override or validated screening signals |
| Partial extraction passed to synthesis | No eligible units, plan warning | Repair extraction before generation |
| Missing live synthesis model/key | Preflight error before queued generation | Set explicit model/key, or use --dry-run/fake adapter |
| Invalid model JSON/validation | Failed unit JSONL after bounded attempts | Inspect failed reason, fix source/prompt; successful earlier phases resume |
| Corrupt synthesis checkpoint | Unit generation_failed; checkpoint not trusted | Preserve then remove the affected checkpoint and rerun |
| Missing/unvalidated classifier | Explicit degraded/manual queue | Review manually or provide locally validated checkpoint/labels |
| Mixed scripts / known corruption | Intercept or deterministic fail | Inspect source; repair affected field only when appropriate |
| Skipped/failed semantic reviewer | Score 1/manual, never auto-pass | Configure reviewer or human adjudication |
| Moroccan dialect veto | Rejected audit, no repair calls | Exclude from training; calibrate classifier separately if adjudication disputes it |
| Bad repair/out-of-scope patch | Reject with safe reason; provenance preserved | Inspect adapter/prompt, retain audit, rerun intentionally |
| Exhausted repairs | rejected.jsonl with history | Manual investigation; do not raise limits blindly |
| Missing or misaligned reference | `reference.matched=false` or `page_aligned=false` | Create `<source-stem>.txt`; keep page markers when per-page diagnostics matter |
| Truncated reference comparison | `truncated=true` plus evaluated sizes | Raise BENCHMARK_MAX_EVAL_CHARS or compare shorter documents; treat truncated CER as a sample |
| Suspect output but CER is zero/low | Check `text_quality` proxies, cleaning counters, and confidence values | Proxies detect damage patterns, not truth; add a reference or human sample |

Raw audit artifacts are not training-ready exports. Empty accepted.jsonl in degraded/offline mode is
expected, not evidence that the pipeline crashed. Manual adjudication feedback ingestion/training
and an automated “release to production” workflow are not implemented.

## 6. How to run and worked example

Use the existing Python 3.15 venv. No installs/downloads are necessary to exercise the offline
example or current tests.

~~~powershell
# Existing default behavior; digital-only documents need no OCR call.
.\.venv\Scripts\python.exe cli.py data/in/law.pdf
.\.venv\Scripts\python.exe app.py

# A: opt into semantic cleaning with aggregate counters.
.\.venv\Scripts\python.exe cli.py --clean --rules-report data/rules-hits.json data/in/law.pdf

# B: review/edit a copy of the manual override starter. Use the actual cleaned filename.
Copy-Item corpus_pipeline/synthesis/data/overrides.example.json data/overrides.json
.\.venv\Scripts\python.exe cli.py --synthesize --dry-run --max-samples 10 --manual-overrides data/overrides.json data/out/law-HASH.txt

# Live B only when the operator intentionally supplies a real key and model in .env/config:
# SYNTHESIS_MODEL=<chosen text-capable model>
.\.venv\Scripts\python.exe cli.py --synthesize --max-samples 10 --manual-overrides data/overrides.json data/out/law-HASH.txt

# C: offline script/lexicon governance. Default output will include manual cases.
.\.venv\Scripts\python.exe cli.py --process-corpus data/training/synthesis/qa.jsonl

# D: offline deterministic checks; semantic dimensions are explicitly skipped.
.\.venv\Scripts\python.exe cli.py --review data/training/governance/manual.jsonl

# Optional live review/repair, deliberately separate from offline commands:
.\.venv\Scripts\python.exe cli.py --review --llm-review data/training/governance/pass.jsonl

# Benchmarking is on by default; disable it for one run if you do not need metrics.
.\.venv\Scripts\python.exe cli.py --no-benchmark data/in/law.pdf

# Real accuracy needs a reference transcription: data/reference/<source-stem>.txt
.\.venv\Scripts\python.exe cli.py --reference-dir data/reference data/in/law.pdf

# Compare every saved run, grouped by provider and model.
.\.venv\Scripts\python.exe cli.py --compare-benchmarks
.\.venv\Scripts\python.exe cli.py --compare-benchmarks --benchmark-report data/reports/ocr-comparison.json

# All stages with synthetic input and FAKE generation/classification/review:
.\.venv\Scripts\python.exe -m examples.offline_workflow

# Fully offline test suite; project-local temp directory avoids Windows TEMP ACL problems.
.\.venv\Scripts\python.exe -m pytest -q -p no:cacheprovider --basetemp data/pytest-run
~~~

The GUI mirrors this sequence in five tabs. **Extract & clean** runs the default command and shows
per-file/page reports plus previews; its Concurrency accordion exposes the file, page, and OCR limits
with `0` meaning no limit. **Synthesize** consumes cleaned TXT/MD, defaults to a dry run, and
accepts manual overrides. **Govern** and **Review** consume JSONL produced by the previous stage; review
is deterministic unless model review/repair is enabled. **Benchmarks** is read-only: it refreshes the
provider/model comparison from `BENCHMARK_DIR` and displays CER/WER, reference coverage, OCR failure and
latency summaries. The collapsible Text generation settings section supplies the provider, key, base URL,
and `SYNTHESIS_MODEL` for the two text stages; the OCR vision model is separate. Explicit uploads take
precedence over prefilled folder/path values, so a file is not processed twice.

The offline demonstration writes data/demo/synthetic.txt, extracts/cleans it, adds a manual
statutory-law override, groups two synthetic articles, generates a fake QA/explanation, performs
fake validated dialect assessment, and scores with a fake reviewer. Expected summaries are one
generated, one governance pass, and one accepted demonstration record. Every final record is marked
demonstration_only. These samples and fake quality scores must not be used as validated legal data.

Inspect data/demo/training/{synthesis,governance,review}. Running the demonstration again skips the
completed extraction and reuses synthesis phases while recomputing governance/review. The tests
exercise the same workflow and block HTTP requests globally.

To use a local classifier, set DIALECT_BACKEND=marbert, DIALECT_MODEL_PATH to an existing
fine-tuned checkpoint, and explicit label_map values in the configured lexicon JSON. Heavy
dependencies remain optional and are not added to requirements.txt. Provision them separately only
if you choose this deployment path. Do not set DIALECT_VALIDATED merely because weights load.

## 7. Benchmarking OCR and pipeline stages

Each processed document writes `<stem>.metrics.json` beside its TXT and report. Every completed batch
also appends one row per document to `history.jsonl` and rebuilds a provider/model comparison. This
answers "did the model, endpoint, DPI, or cleaning change help?" without rerunning unrelated stages.

~~~mermaid
flowchart LR
    Doc["Processed document"] --> Raw["Raw OCR text"]
    Doc --> Clean["Cleaned text"]
    Ref["Reference &lt;stem&gt;.txt"] --> Normalize["Normalise + cap"]
    Raw --> Normalize
    Clean --> Normalize
    Normalize --> Align["difflib alignment"]
    Align --> Scores["CER / WER / precision / recall / F1"]
    Clean --> Proxies["Reference-free quality proxies"]
    Doc --> Ops["OCR latency, cache, failures"]
    Doc --> CleanStats["Cleaning counters"]
    Scores --> Metrics["stem.metrics.json"]
    Proxies --> Metrics
    Ops --> Metrics
    CleanStats --> Metrics
    Metrics --> History["history.jsonl (append-only)"]
    History --> Compare["comparison.json + .csv by provider/model"]
~~~

ASCII: raw OCR + cleaned text + optional reference -> normalise -> align -> CER/WER and proxies;
OCR/cleaning counters -> per-document metrics -> history -> provider/model comparison.

### What each record contains

| Group | Fields | Interpretation |
| --- | --- | --- |
| Identity | source, source SHA-256, provider, model, base URL, languages, DPI, status | Group runs that are actually comparable |
| OCR operations | pages routed/attempted/succeeded/failed, cache hits, failure rate, latency min/mean/median/p95/max, pages per minute, wall time, provider confidence | Cost and reliability; latency excludes cache hits and queueing |
| Reference-free proxies | character/word/line counts, valid-character ratio, replacement/private-use/control ratios, Arabic/Latin/digit/foreign-script ratios, repeated 4-gram ratio, longest repeated-line run, single-character token ratio, empty-page ratio | **Diagnostics only, not accuracy**; a low score is suspicious, a high score is not proof of correctness |
| Cleaning | characters before/after, delta, per-rule hit counts, total hits | Whether cleaning changes output, and how much |
| Reference metrics | character CER, accuracy, precision, recall, F1; word WER and the same scores; exact page alignment; per-page CER; raw-vs-cleaned delta; evaluated size and `truncated` | The only true accuracy numbers; computed only when a reference exists |
| Stage snapshot | latest `synthesis/`, `governance/`, `review/` `summary.json` under `stage_output_dir` | Batch-level context; these stages are not per-document |

`accuracy` is `max(0, 1 - error_rate)`. `precision` is matched units divided by hypothesis units and
`recall` is matched units divided by reference units. CER/WER use `(substitutions + deletions +
insertions) / reference units`; words anchor the alignment and changed word blocks are compared
character by character, so intra-word OCR errors remain visible. Character units exclude whitespace
because normalisation collapses spacing. `difflib` approximates Levenshtein when replacement spans
have different lengths. Metrics are rounded to six decimals.

### Reference workflow

Place a corrected transcription at `data/reference/<source-stem>.txt`, where `source-stem` is the
input filename without its extension (for example `data/reference/law.txt` for `data/in/law.pdf`).
Lookup order is: benchmark reference directory `<stem>.txt`, then `<stem>.ref.txt` there, then
`<stem>.ref.txt` beside the source, then `reference/<stem>.txt` beside the source. Missing references
are not an error; the record simply has `reference.matched=false`. To benchmark a document that was
processed before metrics existed, rerun it with `--force`; successful OCR pages are loaded from the
cache, so no provider calls are repeated.

Keep `--- Page N ---` markers in the reference to get per-page CER for the same page numbers. Without
markers, only the whole-document comparison is produced. Before comparison, text is NFC-normalised,
page markers are dropped, tatweel and Arabic harakat are removed (configurable), control characters
are replaced by spaces, and whitespace is collapsed. Alef/ya unification and the evaluation cap are
configurable; the cap applies to both raw and cleaned text so their delta stays fair. Use the
`raw_vs_cleaned.cer_improvement` field to see whether cleaning helped; a negative value means cleaning
made the text less similar to the reference.

### Comparing runs

`history.jsonl` is append-only. `comparison.json` and `comparison.csv` group rows by provider and
model and report document/page counts, OCR failure rate, pages per minute, mean document wall time,
mean valid-character ratio, mean CER/WER and their precision/recall/F1, mean raw CER, mean CER
improvement, and the low-CER leader. Cache-hit pages are excluded from `ocr_attempted` and from
latency, so throughput reflects provider calls rather than cache reuse. To compare models fairly,
process the same references with the same DPI, languages, concurrency, and cleaning settings; provider
and model are part of the cache identity, so switching either produces fresh OCR rather than reused
pages.

## 8. Implementation status

| Capability | State | Remaining production requirement |
| --- | --- | --- |
| Page-level route, OCR, cache/retries, ordered concurrency, lam-alef repair | Implemented; existing tests retained | Provider access for scanned live inputs |
| Previous-page OCR context stitching / multiple endpoint load balancing | Specification-only | Current parallel independent-page API does not supply previous output |
| General embedded-font reconstruction / local PaddleOCR | Specification-only | No replacement decoder or local engine added |
| Format cleaning and legacy byte output | Implemented, regression-tested | Domain sampling |
| Article-digit correction, explicit glyph/date/footer/word engines | Implemented; mappings degraded/empty by design | Reviewed domain mappings and Roman-numeral scope |
| Screening and 2–4 article segmentation | Implemented, conservative | Validated tier/exclusion lexicon or manual overrides |
| Semantic relatedness of article groups / scrambled-Arabic detection | Specification-only beyond structural checks | Domain model/manual adjudication |
| Three-step synthesis, bounded retries, cost cap, phase checkpoints | Implemented with fake-adapter coverage | Explicit compatible text model/key; no live quality claim |
| Exact evidence/numeric/article/script/known-PII checks | Implemented, partial assurance | Semantic entailment, fact-coverage, and unknown-name review |
| Hidden native chain of thought | Intentionally not implemented | Public explanation stored instead |
| Question normalized-exact dedup | Implemented | Near-duplicate detection not specified here |
| Arabic/Latin script governance | Implemented; language identification degraded for Latin | Validated French/English language classifier/manual adjudication |
| Dialect interface and lexicon | Implemented, explicitly degraded | Curated lexicon, model validation data |
| Local MARBERT classifier | Optional/lazy implementation; not live-tested | Fine-tuned local checkpoint, label mapping, optional dependencies |
| Classifier training / Kappa and macro-F1 acceptance measurement | Specification-only | Annotated gold data and training/evaluation work |
| Nine listed review dimensions and targeted repair state machine | Implemented with fake adapters | Semantic reviewer and calibrated classifier for automatic acceptance |
| Deterministic review offline | Implemented | Skipped checks intentionally prevent auto-acceptance |
| Paid review caching / cross-process writer coordination | Not implemented | Use one process per output directory; paid review reruns recompute |
| Per-document OCR/cleaning/stage benchmarking | Implemented; reference-free by default, reference metrics only when supplied | Curated references for true CER/WER; proxies are not accuracy |
| New training stages in GUI | Implemented as one tab per stage, plus a benchmark tab | Visual/manual smoke test; automated callbacks are covered offline |

## 9. Test evidence and change map

Baseline: 20 existing tests passed before changes. Phase A: 37 passed. Phase B: 49 passed.
Phase C: 57 passed. Phase D: 65 passed. Final integration/edge suite: 76 passed, including all 20
original tests. Benchmark additions: 87 passed, including the 11 new benchmark tests. GUI stage
regression tests bring the suite to 96 passed, including unlimited-concurrency coverage. The command
above reproduces the suite.
The only observed dependency warning is PyMuPDF's deprecated ByteString use on Python 3.15.

Existing files changed minimally: config.py exposes inherited settings and hides keys in repr;
cleaning.py uses shared equivalent HTML/duplicate functions plus the optional rule hook;
processor.py passes config, fingerprints enabled rule data, and adds enabled-only semantic metadata;
ocr/base.py masks exact arbitrary credential values before truncating error details;
cli.py dispatches optional stages without changing positional extraction calls; README.md and
.env.example describe the new paths. app.py now presents each stage as a tab while reusing the same
stage and extraction APIs; extraction.py, cache.py, routing.py, and the existing concurrency tests
remain functionally unchanged in this work. models.py only gains optional OCR timing/confidence/cache
fields with defaults, and processor.py adds benchmark artifacts.

All stage implementations, prompts, empty starter data, regression tests, the offline example,
and this guide are additions. No keys were inspected or staged, no network/model calls or downloads
were performed, no packages were installed, and no commits/pushes/history edits were made.
