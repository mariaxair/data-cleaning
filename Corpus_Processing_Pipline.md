__Background____:__

The performance of large language models depends heavily on the quality of training data. In the Arabic legal domain, publicly available corpora are scarce, yet a large volume of authoritative literature lies "dormant" in scanned PDFs:

-   Official gazettes (JORADP, the Official Gazette of Algeria, spanning several decades);
    
-   Court decisions (constitutional court rulings, supreme court judgments);
    
-   Ministerial laws and regulations (Ministry of Labour and Social Security, the Veterans' Code, Ministry of Justice documents, etc.);
    
-   Academic paper repositories (the ASJP library, tens of thousands of PDFs).
    

These documents cannot be used for model training directly. They must pass through a complete processing pipeline that turns "scanned paper" into "machine-readable text," and then into question–answer pairs that teach a model to reason. Our pipeline is as follows:

![](https://icenterapi.zte.com.cn/group2/M04/52/19/CimNG2razfSAGn38ABcz7xOLmyE64.jpeg?h=1254&w=1254)

Every stage has its own quality checks. Any sample that fails is intercepted, repaired, or discarded — every record that ultimately enters the corpus has passed multiple rounds of verification.

  

# 1 Stage 1: OCR (PDF → Readable Text)

## 1.1 Why "Copy-Paste" Doesn't Work

PDFs come in two kinds: text-based PDFs, whose characters can be extracted directly, and scanned/image-based PDFs, which are essentially one image per page. Our documents, however, face a subtler problem — corrupted font encoding.

We ran a systematic health check on nearly 40,000 academic-paper PDFs. On the surface, about 95% of the files "have a text layer" and appear directly extractable. Digging deeper, however, we found that due to font defects introduced at production time, roughly 20–30% of the Arabic content and 18–25% of the French content is garbled (letters replaced with wrong characters, rare glyphs dropped, characters out of order, etc.). Visually the PDFs look fine — the viewer renders the glyphs — but the text you copy out is broken.

Conclusion: direct text extraction is unreliable; a visual-recognition approach is required — render each page as an image and have a multimodal large model "read the picture," transcribing page by page into Markdown.

## 1.2 Key OCR Steps

The core approach is "page-by-page imaging + large-model transcription":

1.  Rendering: each page of the PDF is rendered into a high-resolution image;
    
2.  Transcription: a multimodal large model reads the page image and outputs well-formed Markdown;
    
3.  Context stitching: when transcribing page n, the model receives the actual output of page n−1 as context, ensuring tables and paragraphs that span pages are joined correctly;
    
4.  Parallel acceleration: multiple PDFs are processed in parallel, with load balanced across multiple model-serving endpoints.
    

For the characteristics of legal literature, the transcription model follows a strict set of rules:

| Rule | Description |
| --- | --- | 
| Faithful to the source | Transcribe only — no translation, no rewriting. Foreign-language fragments embedded in a page (terms, personal names, institution names) are preserved verbatim |
| Legal structure hierarchy | Law titles, articles (المادة / Article), chapters, and sections are marked with standard heading levels, facilitating later article-level segmentation |
| No numeral conversion | Eastern Arabic numerals (٠١٢٣) and Western digits (0123) are preserved as-is, never converted |
| Tables in their true direction | Arabic tables are laid out right-to-left; each cell is transcribed in its actual column order |
| Dates preserved verbatim | Hijri–Gregorian paired dates are kept in full, with no conversion |

## 1.3 Engineering Safeguards

-   Resumable execution: jobs covering tens of thousands of documents run for days; after any interruption, processing resumes from the last checkpoint and completed files are skipped automatically;
    
-   Automatic retries: a failed page is retried automatically and only marked as bad after repeated failures;
    
-   Measured throughput: with two vision models running in parallel, combined throughput is about 9–12 pages per second. The main body of the third batch — nearly 40,000 papers (about 780,000 pages) — was transcribed in just over a day, with a final completion rate of 94.5% (37,670/39,859 papers).
    

## 1.4 Model Selection

We benchmarked recognition models of different scales:

-   1.6B small model: produces "tape-loop" repetitive output and frequently corrupts numbers — unusable;
    
-   27B model: repetitive loops and digit corruption essentially eliminated; significantly better recognition quality — became the workhorse;
    
-   35B-class models: used for larger batches, with stable quality.
    

  

# 2 Stage 2: OCR Post-Processing (Cleaning + Error Correction)

The output of large-model OCR still contains two classes of residual problems: formatting issues and semantic errors. We address them with "three-step format cleaning + rule-based semantic correction."

## 2.1 Format Cleaning (Three Steps)

| Step | What is cleaned | In plain terms |
| --- | --- | --- | 
| ① Remove metadata header | Conversion notes generated by the model at the top of each file | Like peeling the shipping label off a parcel |
| ② Remove HTML residue | <sup>, stray tags, <br> outside tables, etc. | Sweeping "web-code debris" out of the body text |
| ③ Remove repetition loops | Occasional consecutive repeated lines from the model | Cutting out the "stuck-record" passages, conservatively, to avoid deleting legitimate content |

Measured on the full third batch: header metadata removed from all 37,670 files, HTML residue cleaned from 2,806 files, and repetition removed from 895 files.

## 2.2 Semantic Error Correction (8 Classes of Deterministic Rules)

Beyond formatting, OCR also makes "wrong-character" semantic errors. From a full scan of real gazette data we distilled 8 high-frequency error classes into a set of deterministic correction rules — no model judgment involved; when a rule matches, the fix is applied, and the result can be re-verified again and again:

| Error type | Example | Handling |
| --- | --- | --- | 
| Digit–letter confusion | The "1" in an article number recognized as the letter "I" | Context-aware substitution |
| Foreign-script contamination | Hebrew, Cyrillic, or Armenian letters inside Arabic text (they resemble certain Arabic letters) | Replacement with the correct Arabic characters |
| High-frequency wrong words | Fixed phrases such as gazette-masthead institution names systematically misrecognized | Curated wrong-word table, corrected item by item |
| Residual line-break tags | <br> mixed into body text | Removed |
| Reversed date components | "Month–day" recognized as "day–month" | Restored per date rules |
| Adjacent-word duplication | The same word appearing twice in a row | Deduplicated |
| Page numbers interleaved in text | Footer page numbers embedded mid-paragraph | Detected and deleted |

These rules complement the format-cleaning scripts: format scripts handle "layout," rule scripts handle "character recognition." The two run in series without interfering with each other. Every rule carries a hit counter, so every correction is fully traceable and auditable.

# 3 Stage 3: Corpus Synthesis (Text → Question–Answer Pairs)

Cleaned text is still just raw material. Model training requires question–answer (QA) pairs — a question grounded in a realistic usage scenario, paired with a complete, accurate answer that carries its reasoning. This is the heart of the pipeline.

## 3.1 Step 1: High-Quality Document Screening

Each batch of documents first gets a "health check": we measure file sizes, coarsely classify documents by keywords (legal-provision signal words vs. administrative signal words), and confirm with manual sampling. Documents are then sorted into three tiers:

| Tier | Content type | Handling |
| --- | --- | --- | 
| ✅ Statutory law | Ministerial laws, decrees, codes (uniformly structured provision text) | Primary material, processed normally |
| ✅ Case law | Court rulings (after filtering out mixed-in press releases and event notices) | Processed, with filtering |
| ❌ Administrative noise | Meeting agendas, procurement announcements, rejected-applicant lists containing personal data | Excluded outright |

The rationale for excluding administrative noise is practical: meeting agendas can only yield rote-recall questions ("what time does the forum start?"), never legal-consultation questions of the "conditions → reasoning → conclusion" kind; applicant lists containing personal information (names, birth dates, rejection reasons) also pose privacy-compliance risks. The shape of a corpus caps what it can be used for — quality over quantity.

## 3.2 Step 2: Intelligent Segmentation

A single law can contain dozens of articles — far more than one question can carry — while a single article leaves the context too narrow. We use "article-group segmentation": every 2–4 related articles are grouped, together with their category, numbers, and the full law title, into one synthesis unit. This gives answers room to unfold while multiplying usable material roughly sixfold (from 221 to 1,379 units in our measurements).

Segmentation applies multiple layers of filtering: mastheads and indexes are stripped; procedural content such as appointments and ratifications is removed; articles with no substantive content ("shall take effect this month") are dropped; and files with scrambled character order are discarded. Court rulings, by contrast, are kept whole as a single segment — a judgment's reasoning is an indivisible whole.

## 3.3 Step 3: Two-Phase Synthesis with Dual Reasoning Chains

Generation of each QA pair is split into three steps, each with its own prompts and validations:

__Phase 1 — Question generation (deep thinking disabled, prioritizing precise scenarios)__

-   Questions are phrased as real legal consultations: "I'm in this situation — what conditions / deadlines apply, is it compliant?" — not "What does Article X of Law Y say?";
    
-   Hard rules: every fact in the question must be traceable to the source text; fabricating conditions, numbers, institutions, or dates is forbidden; the question must never name the specific law or article (just as a real client would not know which law applies);
    
-   Dialect constraint: the output is explicitly required to be Modern Standard Arabic (الفصحى) or Algerian local usage; Moroccan dialect is forbidden (see the "dialect contamination" box at the end of this chapter);
    
-   Perspective rotation (first person / third person with names) and reasoning-type rotation (conditional application, exception analysis, procedural tracing, deadline calculation, and six types in total) keep the questions diverse.
    
-   real personal data from the source texts, such as the names of parties, must never appear (parties are referred to generically)。
    
      
    

__Phase 2 — Answer generation (deep thinking enabled, strictly grounded in the source)__

-   A complete syllogism is required: legal framework → article-by-article analysis → conclusion — not a one-line verdict;
    
-   The model's raw thinking process is stored separately as an "English chain of thought," decoupled from the final answer;
    
-   __Privacy / no real personal data:__ never copy real personal data from the source text — no party/litigant/judge/lawyer names, no case numbers, no ID/national numbers, no dates of birth, no home addresses. When referring to parties, use generic role terms only (e.g. الطرف / الموظف / المستأنف / المدعي / المجني عليه). Exception: institution names that are part of the cited legal framework (e.g. the issuing authority of the law) are allowed.
    
-   Two hard validations, with retries on failure:
    
    -   Article-number consistency: every article number appearing in the answer must be found verbatim in the source — this specifically catches digit hallucinations such as "94-112" becoming "94-1112";
        
    -   Language purity: an Arabic answer containing Latin letters, Chinese, or other foreign characters is rejected and rewritten.
        
          
        

__Phase 3 — Target-language reasoning-chain completion__

-   The model is given only the question and answer — not the source text — and asked to rewrite a detailed reasoning chain in the target language (Arabic for Arabic samples). "No source text" is a deliberate design: it forces the model to reconstruct its reasoning from the already-validated answer, introducing no new facts;
    
-   This chain is required to be longer and more detailed than the answer, unfolding the full legal reasoning process.
    

Each final record carries a complete field set: question, answer, target-language reasoning chain, native chain of thought, source legal metadata (law title / number / article numbers), question type, and reasoning type — every record is traceable to its source text.

## 3.4 A Problem Peculiar to Arabic: "Dialect Contamination"

Arabic is not a single "language." Modern Standard Arabic (MSA) is the language of writing, law, and news; in addition, each country has its own dialect (Darija) — Morocco has Moroccan Darija, Algeria has Algerian Darija. The two share common roots but differ markedly in vocabulary, pronunciation, and written conventions; native speakers can tell them apart instantly.

The problem is especially tricky in our setting, and its root lies in the base model: in the Arabic data used to train the Qwen model family, Moroccan Darija accounts for a large share (Morocco leads the Arab world in open-source data availability). During synthesis, the model therefore tends to slip into Moroccan dialect — a Moroccan turn of phrase in a legal Q&A is almost unavoidable.

However, our goal is to train an Algerian model: the desired output is Standard Arabic or Algerian Darija, and Moroccan Darija is explicitly unwanted. The pipeline therefore sets up defenses at three points:

| Stage | Defense | Measure |
| --- | --- | --- | 
| ③ Corpus synthesis | Up-front prompting | Synthesis prompts explicitly require Standard Arabic output and forbid Moroccan Darija |
| ④ Corpus post-processing | Downstream screening | A language/dialect classifier adjudicates each synthesized sample; suspicious samples are intercepted (see 4.3) |
| ⑤ Corpus scoring | Veto power | A new "dialect compliance" dimension in scoring: any sample judged Moroccan Darija is deleted outright, with no repair attempt |

The three defenses back each other up: prompting reduces the probability of contamination, the classifier catches whatever slips through, and the scoring dimension delivers the final verdict.

# 4 Stage 4: Corpus Processing (Dedup + Language Governance + Targeted Repair)

## 4.1 Deduplication

Question text is normalized (unified character forms, diacritics stripped, letter variants unified) before comparison; exact duplicates are reduced to a single copy.

## 4.2 Language Governance

Multilingual corpora hide a subtle class of problems — language misplacement: files under the French directory that are actually Arabic papers. We ran two rounds of governance (third batch as the example):

1.  Re-adjudicated the language of every sample, correcting 422 misplaced samples;
    
2.  Deleted 1,122 Arabic–French mixed samples (any sample exceeding the mixing threshold is discarded);
    
3.  Governance revealed that the French directory consisted almost entirely of Arabic papers mislabeled due to their French abstract pages — the language manifest of the whole batch was corrected accordingly.
    

## 4.3 Dialect Screening

To address the dialect contamination problem, a dialect screening step is added after synthesis: a language/dialect classifier adjudicates each sample's question, answer, and reasoning-chain fields separately, classifying the output as __Standard Arabic (MSA), Algerian Darija, or Moroccan Darija__; samples judged Moroccan Darija are intercepted directly.

### 4.3.1 Approach Analysis

__Contamination is real and measurable — and it worsens with reasoning depth.__ We ran two rounds of generation experiments on 1,000 authentic Algerian Darija legal consultations, tallying the most distinctive discriminative words (ديال is the Moroccan possessive; its Algerian counterpart is تاع):

| Generation mode | Containing Moroccan possessive ديال | Containing Moroccan time word دابا | Containing Tunisian possessive متاع | Explicit foreign-law citations |
| --- | --- | --- | --- | --- | 
| Non-thinking | 7 / 1000 | 1 / 1000 | 8 / 1000 | 2 (e.g., Moroccan lease law 67.12) |
| Thinking | __146 / 1000__ | __90 / 1000__ | __123 / 1000__ | 13 (Tunisian labor code "مجلة الشغل", Moroccan contract law, etc.) |

Up-front prompting constraints in synthesis (defense ③) only lower the probability — they cannot suppress the long tail. A hard downstream gate is required.

Mainstream approaches to Arabic dialect identification fall into four families: self-review, general-purpose LLM judgment, pure rules, and dedicated classifiers. We adopt the dedicated-classifier approach, as the other three have verified drawbacks:

| Alternative approach | Drawback |
| --- | --- | 
| Self-review (the generating model checks itself) | Examiner and examinee are the same model — prone to circular self-validation |
| Zero-shot dialect judgment by general-purpose LLMs | Refuted publicly: GPT-4 achieves only 26.92 F1 on zero-shot country-level dialect identification |
| Pure rule / lexicon matching | Dialect-exclusive discriminators have low coverage; a complete lexicon is nearly impossible to compile |

### 4.3.2 Classifier Design

An off-the-shelf mainstream Arabic multi-dialect classifier can be used directly — publicly available models already ship with Standard Arabic, Algerian Darija, and Moroccan Darija among their classes. If performance falls short of expectations, a dedicated classifier can also be trained by continued fine-tuning on the base model (this requires manually annotated data):

| Design item | Choice | Rationale |
| --- | --- | --- | 
| Base model | MARBERTv2 (~130M parameters) | Continued pre-training on ~1B dialectal tweets; the strongest general-purpose backbone across the QADI/NADI shared-task series |
| Training data (optional) | QADI (540K tweets, balanced across 18 countries, 91.5% label accuracy on manual audit) + MADAR (25-city parallel corpus, incl. Algiers, Rabat, Fez, Tunis) + manually annotated data | QADI provides scale; MADAR is the only parallel corpus with fine-grained coverage of the Algeria–Morocco–Tunisia triangle; manual data further sharpens performance |
| Training strategy (optional) | Two stages: full 18-way pre-training first (aligning with the ~65% community level) → fine-tuning narrowed to "MSA + Algerian + Moroccan" | Build general recognition ability first, then narrow the class set to concentrate discriminative power on the boundary that actually matters |
| Deployment | Resident on a single offline GPU, in the same environment as the existing inference service | The encoder is small; a full batch of tens of thousands of samples is processed within minutes |

### 4.3.3 Decision Logic and Acceptance

Each sample yields a classification matrix plus a decision (pass / intercept / manual), driven by the Moroccan Darija probability:

| Moroccan Darija probability | Verdict | Follow-up |
| --- | --- | --- | 
| < 0.2 | pass | Proceeds to the scoring gate (which retains its own dialect veto, defense ⑤) |
| 0.2 – 0.5 | manual | Routed to the manual adjudication queue; spot-check outcomes feed back into threshold calibration |
| > 0.5 | intercept | Intercepted and deleted outright |

The classifier must clear two acceptance gates:

1.  __Out-of-domain validation__: it must first run on our 2,000 completed model responses (1,000 thinking + 1,000 non-thinking); automated verdicts are trusted only once agreement with human annotation reaches Cohen's Kappa ≥ 0.7;
    
2.  __In-domain gold standard__: stratified samples from corpus batches (all automated-confidence bands, plus the lexicon-hit band) are manually annotated; sentence-level three-class macro-F1 ≥ 0.85 is required before full deployment. Until the bar is met, the screen runs in degraded mode — only samples with high-confidence classifier hits are auto-intercepted; everything else goes to manual review.
    

# 5 Stage 5: Corpus Reviewing (the Quality Gate)

## 5.1 Eight-Dimension Reviewing

Before entering the corpus, every sample must pass 8 checks, each scored 0/1/2; all dimensions must pass:

| Dimension | Adjudication | What it examines |
| --- | --- | --- | 
| Question accuracy | LLM review | Is the question a realistic consultation scenario, with no fabricated facts |
| Reasoning-chain completeness | LLM review | Does the answer carry a full "framework → analysis → conclusion" syllogism |
| Question self-containedness | LLM review | Is the question understandable without external context |
| Privacy compliance | LLM review | no real person names (parties, litigants, judges, lawyers,officials), no case numbers, no ID/national numbers, no dates of birth, no addresses; parties are referred to generically. |
| Language purity | Deterministic rules | Any foreign-language characters mixed in |
| Markdown compliance | Deterministic rules | Is the formatting well-formed |
| Fluency / no repetition | Deterministic rules | Any repetition or breakage |
| Corpus accuracy | Deterministic rules | Any article-number hallucination or garbled-character residue |
| Dialect compliance | Classifier + veto | If the output is Moroccan Darija, delete outright — no repair |

## 5.2 A Key Lesson

Practice revealed an important pattern: when a large model acts as the reviewer, it systematically misjudges certain dimensions — clean samples get unfairly low scores. In one batch, LLM-based scoring inflated the pass rate by 12 percentage points; manual sampling confirmed extensive misjudgment.

This led us to two design principles for the scoring system:

-   Anything expressible as an explicit rule uses deterministic rules (article-number checks, language detection, repetition detection, format checks — machine adjudication, zero false positives, fully reproducible);
    
-   Only dimensions that genuinely require "understanding meaning" go to the LLM (does the question read like a real consultation, is the reasoning complete — things regular expressions genuinely cannot judge).
    

This "determinism-first" principle runs through synthesis as well: every quality point that can be machine-verified (article numbers, language, format) is checked in place at generation time, reserving LLM review for where it truly matters.

## 5.3 Cause-Targeted Repair

Samples that fail scoring are not blindly rerun; they are repaired according to the specific cause of failure:

| Failure cause | Repair action |
| --- | --- | 
| Foreign language in the reasoning chain | Rewrite the reasoning chain only (with tightened language constraints); question and answer untouched |
| Language / article-number problem in the answer | Regenerate the answer only |
| Defective question (fabricated facts, not self-contained) | Regenerate question and answer together |
| Incomplete reasoning in the answer | Regenerate the answer |

After repair, samples re-enter the same scoring process. In the second batch, 311 of 767 failed samples were recovered this way — a repair recovery rate of about 40%, substantially reducing waste.

Note: samples judged to be Moroccan Darija are the sole exception — dialect identity is a holistic property of a sample and cannot be repaired locally, so they are deleted outright and never enter the repair flow (see the scoring dimensions in 5.1).

# 6 Quality Principles That Run Through the Entire Pipeline

In summary, the pipeline is governed by four ever-present principles:

1.  If in doubt, don't use it — unreliable text layers mean full visual recognition; a small model with repetition defects is replaced by a larger one; noisy documents are excluded outright.
    
2.  Validation at every step — in-place checks at generation time (article numbers, language), an eight-dimension gate at scoring, targeted repair and re-verification for failures.
    
3.  Determinism first — anything a rule can adjudicate never relies on subjective model judgment, keeping results reproducible and false-positive-free.
    
4.  End-to-end traceability — every record stores its source document, law number, article numbers, and per-stage scores; any record can be traced back to the exact page of the original PDF.
    

# 7 Deliverables at a Glance

Output of the pipeline to date:

| Deliverable | Scale |
| --- | --- | 
| High-quality legal QA corpus (Batch 1) | 20,000 items (Arabic/French, with reasoning chains) |
| High-quality legal QA corpus (Batch 2) | 2,207 items (Arabic/French, with reasoning chains) |
| High-quality QA corpus (Batch 3, full run) | 23,306 items (all passed language governance and the full scoring gate) |
| OCR correction rule library | 8 classes of deterministic rules + a high-frequency wrong-word table |
| Quality-inspection rule library | 4 deterministic scoring rules + 3 LLM-reviewed dimensions |

All corpus records carry full provenance (source document, law title, article numbers) and can be used directly for supervised fine-tuning (SFT) of large models.
