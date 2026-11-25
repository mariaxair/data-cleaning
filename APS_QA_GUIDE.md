# APS question and answer generation guide

`aps_generate_qa.py` reads the packaged APS JSONL. Each article stays whole.
The language and original metadata come from the input, not from a model.
Edit `prompts/aps_questions.txt` and `prompts/aps_answers.txt` to change the tasks.
Both templates apply to all article topics and support Arabic, French and English.
The question template's category placeholder is filled from each article's APS
category. It includes non-exhaustive examples of unsuitable questions; these are
instructions about what to avoid, not questions to copy. An empty list means no
useful supported questions remain, not simply that the article is short.

The question model decides how many distinct, answerable questions to produce.
There is no numeric question-count setting. Repeated questions differing only in
case, punctuation or whitespace are collapsed. This is not a semantic review.
Short excerpts can yield few questions or none; richer articles can yield more.

The answer model receives the original article, its metadata and every generated
question. It answers the list in one call. Responses must cover every question ID.
Unanswerable questions are saved separately. No review model or scoring runs here.

## Prepare a small preview

From the project directory:

```powershell
python aps_generate_qa.py --question-model gpt-6.1-sol --answer-model gpt-6-astra
```

This makes **zero API calls**. The default sample has 10 articles per language,
rotating across categories with reproducible random selection within categories.
The names above are candidates listed in the project's Heiyu configuration;
their availability, quality on this task and current prices have not been verified.
Choose model names that your provider account actually supports.

Open `data/aps-qa-pilot/prompt-preview.jsonl` to see the filled prompts. The answer
preview retains `{{QUESTIONS_FROM_QUESTION_MODEL}}` until real questions exist.
Inspect `plan.json` for selection counts, role settings and request limits.
No credentials are placed in these files. Previewing requires no API key.

To prepare exactly four articles total, use:

```powershell
python aps_generate_qa.py --total 4 --output data/aps-qa-4-preview --question-model gpt-6.1-sol --answer-model gpt-6-astra
```

The four are distributed deterministically as **2 Arabic, 1 French and 1 English**,
then one category is selected for each language. The plan records the exact UUIDs.
This selection happens locally and makes no model calls.

## Launch the paid pilot yourself

The default gateway is `https://www.heiyucode.com/v1`. The script reads the
project's existing `.env`; both roles use `HEIYU_API_KEY` by default. Keys can
also be supplied via environment variables. Never put the key in a CLI argument.

After inspecting the prompts and verifying the two model names:

```powershell
python aps_generate_qa.py --total 4 --output data/aps-qa-4 --question-model gpt-6.1-sol --answer-model gpt-6-astra --run
```

Normally this makes two requests per article: questions, then answers. Articles
with no questions skip the answer call. Schema and transient HTTP failures can
cause additional attempts, bounded by `--max-attempts` (default 3 per phase).
HTTP 429 responses respect `Retry-After`; 429 and server/network failures share
the same limiter as normal requests and retries.

`--rpm 10` is the default shared start-time limit across both models and all
workers. It is a conservative starting setting, **not a verified Heiyu quota**.
`--concurrency 2` allows two articles in progress. It does not multiply the RPM
limit. Model-specific quotas and token-per-minute limits can still apply.

The technical output budgets are `--question-output-tokens 8192` and
`--answer-output-tokens 16384`. These do not set a number of questions. The
provider's own context/output limits still apply. A reported truncated response
is recorded as a failure, including its token usage; partial JSON is not accepted.
If increasing a budget or changing a model/prompt, use a new output directory.
OpenAI-compatible gateways that need `max_completion_tokens` can use
`--question-token-parameter max_completion_tokens` and the corresponding answer option.

## Output and resume

- `questions.jsonl`: original articles with their generated question lists. Each
  question is an object with `question_id` (a stable UUID) and `question` (its text).
- `qa-candidates.jsonl`: one record per answered question. Each retains the eight
  original article fields, uses a new QA `uuid`, and adds `article_uuid`,
  `instruction` and `output`. The article text is retained verbatim.
- `unanswered.jsonl`: questions the answer model reports it cannot answer, retaining
  their `question_id` and `article_uuid`.
- `failures.jsonl`: article IDs and failure codes for incomplete generation.
- `checkpoints/`: saved successful question and answer phases.
- `usage.jsonl` and `api-audit.jsonl`: all API attempts, including retries and
  responses later rejected for schema errors. Audit data stays outside QA records.
- `summary.json`: completion counts, candidate counts and reported token usage.

There are no local PC paths, image URLs, scraping fields, scores, hashes or
processing tags added to QA records. Existing article content is not rewritten.
The parent export's `summary.json` supplies public-excerpt information to prompts;
its absence never makes the script assume the source is complete.

Rerun the **same live command and output directory** to resume successful phases
without paying for them again. The aggregate result files are rebuilt from
checkpoints. Token usage includes earlier attempts in the same run directory.
Pending questions survive an answer failure. Changed source content, model names,
prompts, token budgets or selected articles require a new `--output` directory.
An interrupted run can leave `.tmp` aggregate files until the resumed run completes.

All pairs remain **candidates awaiting the later review stage**. Syntax, matching
IDs and exact-duplicate checks do not establish factual or language quality.

### How questions and answers stay linked

After generating question text, the code assigns each question a UUID derived from
the parent article UUID and normalized question text. Identical questions in
different articles get different IDs; reordering a list does not change its IDs.
The question IDs are saved in the checkpoint before the answer call and included
with the question text in that call. The answer model must copy those IDs exactly.

The parser rejects unknown, repeated or missing answer IDs. Returned items may be
in any order: the code joins them to questions by ID, not list position. The
question's `question_id` becomes the final pair's `uuid`; `article_uuid` links the
pair back to the original article. This avoids storing a duplicate question ID
field in each final QA record. An unanswerable question keeps its ID in the
separate unanswered file. IDs enforce data linkage, not semantic answer accuracy.

These prompt/schema changes require a new output directory if an earlier live
run already used the old prompts. Preview directories without a live run can be
regenerated by the operator with the updated code.

## Cost reporting

Supply the provider's actual **USD prices per million tokens** using:

```text
--question-input-price ... --question-output-price ...
--answer-input-price ... --answer-output-price ...
```

Without prices, cost is `null`; reported token counts are still recorded. Usage
includes paid responses that are truncated or fail JSON validation. Network
errors and missing usage are counted as unknown billing, not treated as free.
Token costs are estimates: cached-input discounts, provider markups, currencies
and other charges may differ from the supplied rates. No tokens are downloaded
or counted with an assumed universal characters-per-token conversion.

An illustration, **not a Heiyu quote**: assume an average of five pairs per article,
1,000 question input tokens and 250 question output tokens, plus 1,400 answer input
tokens and 900 answer output tokens per article, with no retries. This is a cost
scenario, not a generation limit or a measured token prediction.

| Hypothetical USD prices per million tokens | 30 articles | 44,396 articles |
| --- | ---: | ---: |
| Both roles: input $1, output $4 | $0.21 | $310.77 |
| Questions: $1/$4; answers: $3/$12 | $0.51 | $754.73 |

Actual question counts, answer lengths, reasoning-token charges and retries can
change these amounts substantially. Estimate the real run from the paid pilot's
per-language usage and the confirmed provider prices.

## Full run, after the pilot

```powershell
python aps_generate_qa.py --all --output data/aps-qa-final --question-model gpt-6.1-sol --answer-model gpt-6-astra
```

This still only prepares a local preview. Add `--run` yourself when ready to
generate. The source JSONL and selection remain untouched. Providers can be
configured independently through `--question-base-url`, `--answer-base-url`,
`--question-key-env` and `--answer-key-env`; both must support Chat Completions.
`APS_QUESTION_MODEL`, `APS_ANSWER_MODEL` and the corresponding `*_BASE_URL` and
`*_KEY_ENV` environment variables provide optional defaults.
