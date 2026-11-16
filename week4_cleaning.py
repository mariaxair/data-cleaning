"""Week 4 source cleaning and audit for the non-PDF corpus.

The command keeps source files untouched and writes a manifest, cleaned text,
quality flags, duplicate groups, and a stratified review sample. PDF files are
deliberately excluded because their OCR delivery is handled separately.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

from config import PipelineConfig
from corpus_pipeline.cleaning import clean_text
from corpus_pipeline.extraction import extract_docx, extract_text_file

NON_PDF = {".txt", ".md", ".html", ".htm", ".json", ".csv", ".xlsx", ".docx"}
SKIP_PARTS = {"_debug", ".git", ".venv", "data"}
PII_PATTERNS = {
    "email": re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d[\s().-]?){9,15}(?!\d)"),
    "url_query": re.compile(r"https?://[^\s]+[?&](?:id|token|key|email|phone)=", re.I),
    "identity_label": re.compile(
        r"(?i)(?:case\s*(?:no|number)|date\s+of\s+birth|national\s*(?:id|number)|"
        r"passport|رقم\s*(?:الهوية|القضية)|تاريخ\s*الميلاد|العنوان)\s*[:#]?\s*\S+"
    ),
}
LANGUAGE_PARTS = {
    "ar": "ar", "fr": "fr", "en": "en", "es": "es", "ru": "ru", "cn": "zh",
    "zh": "zh", "tamazight-arb": "tamazight", "tamazight-tal": "tamazight",
    "tamazight-tif": "tamazight",
}


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize(text: str) -> str:
    value = unicodedata.normalize("NFKC", text).casefold()
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    return re.sub(r"\s+", " ", value).strip()


def simhash(text: str) -> int:
    tokens = re.findall(r"\w+", normalize(text), re.UNICODE)
    if not tokens:
        return 0
    weights = [0] * 64
    for token in set(tokens):
        value = int(hashlib.blake2b(token.encode(), digest_size=8).hexdigest(), 16)
        for bit in range(64):
            weights[bit] += 1 if value & (1 << bit) else -1
    return sum((1 << bit) for bit, weight in enumerate(weights) if weight >= 0)


def hamming(left: int, right: int) -> int:
    return (left ^ right).bit_count()


def language_bucket(path: Path, text: str) -> tuple[str, str]:
    parts = [part.casefold() for part in path.parts]
    claimed = next((LANGUAGE_PARTS[part] for part in parts if part in LANGUAGE_PARTS), "unknown")
    counts = Counter()
    for char in text:
        if unicodedata.category(char).startswith("L"):
            name = unicodedata.name(char, "")
            counts["ar" if "ARABIC" in name else "latin" if "LATIN" in name else
                   "cyrillic" if "CYRILLIC" in name else "other"] += 1
    total = sum(counts.values()) or 1
    detected = max(counts, key=counts.get) if counts else "unknown"
    if counts["ar"] / total > 0.2 and counts["latin"] / total > 0.2:
        detected = "mixed_ar_latin"
    if claimed == "tamazight" and detected == "ar":
        detected = "tamazight_arabic_script"
    return claimed, detected


def aps_body(text: str) -> tuple[str, dict]:
    lines = text.splitlines(keepends=True)
    fields = {}
    known = {"Title", "Subtitle", "URL", "Language", "Category", "Published", "Modified",
             "Tags", "Image", "Article-ID", "Scraped", "Protected"}
    for index, line in enumerate(lines[:100]):
        if re.fullmatch(r"={20,}\s*", line):
            if {"Title", "URL", "Language"}.issubset(fields):
                return "".join(lines[index + 1:]).lstrip("\r\n"), fields
            break
        match = re.match(r"^([A-Za-z][A-Za-z-]*):\s*(.*?)(?:\r?\n)?$", line)
        if match and match.group(1) in known:
            fields[match.group(1)] = match.group(2).strip()
        elif fields and line.strip() and not line.startswith(("http://", "https://")):
            # Scraped titles can wrap across physical lines before the separator.
            previous = next(reversed(fields))
            fields[previous] = (fields[previous] + " " + line.strip()).strip()
        elif not fields:
            break
    return text, {}


def html_text(text: str) -> str:
    text = re.sub(r"(?is)<(script|style)[^>]*>.*?</\1>", " ", text)
    text = re.sub(r"(?i)<\s*br\s*/?\s*>", "\n", text)
    text = re.sub(r"<[^>]+>", " ", text)
    return html.unescape(text)


def mask_pii(text: str) -> tuple[str, dict[str, int]]:
    counts = {}
    for name, pattern in PII_PATTERNS.items():
        if name == "url_query":
            replacement = "[REDACTED_URL_PARAMETER]"
        elif name == "identity_label":
            replacement = "[REDACTED_IDENTITY]"
        else:
            replacement = f"[REDACTED_{name.upper()}]"
        text, hits = pattern.subn(replacement, text)
        if hits:
            counts[name] = hits
    return text, counts


def read_source(path: Path) -> tuple[str, dict, list[str]]:
    suffix = path.suffix.casefold()
    warnings = []
    if suffix in {".txt", ".md"}:
        raw = extract_text_file(path)
        body, fields = aps_body(raw)
        if "\ufffd" in raw:
            warnings.append("replacement_character")
        return body, {"aps": fields}, warnings
    if suffix in {".html", ".htm"}:
        return html_text(extract_text_file(path)), {}, warnings
    if suffix == ".docx":
        return extract_docx(path), {}, warnings
    if suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        records = value if isinstance(value, list) else [value]
        texts = []
        for record in records:
            if isinstance(record, dict):
                body = next((record.get(key) for key in ("contenu", "content", "text", "body")
                             if isinstance(record.get(key), str)), None)
                if body is not None:
                    title = record.get("titre", record.get("title", ""))
                    texts.append((str(title) + "\n\n" if title else "") + body)
                else:
                    texts.append(json.dumps(record, ensure_ascii=False))
            else:
                texts.append(str(record))
        return "\n\n".join(texts), {"records": len(records)}, warnings
    if suffix == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.reader(handle, delimiter=";"))
        return "\n".join(" | ".join(row) for row in rows), {"rows": len(rows)}, warnings
    if suffix == ".xlsx":
        from openpyxl import load_workbook
        workbook = load_workbook(path, read_only=True, data_only=False)
        parts = []
        try:
            for sheet in workbook:
                parts.append(f"Sheet: {sheet.title}")
                for row in sheet.iter_rows(values_only=True):
                    if any(value is not None for value in row):
                        parts.append(" | ".join("" if value is None else str(value) for value in row))
        finally:
            workbook.close()
        warnings.append("spreadsheet_values_are_structured_text")
        return "\n".join(parts), {}, warnings
    raise ValueError(f"Unsupported extension: {suffix}")


def source_files(root: Path) -> list[Path]:
    return [path for path in sorted(root.rglob("*"), key=lambda p: p.as_posix().casefold())
            if path.is_file() and path.suffix.casefold() in NON_PDF
            and not any(part in SKIP_PARTS for part in path.parts)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Week 4 non-PDF cleaning and audit")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/week4"))
    parser.add_argument("--limit", type=int, help="Process only the first N files for a pilot run")
    parser.add_argument("--sample-per-bucket", type=int, default=20)
    parser.add_argument("--semantic-rules", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = args.source_root.expanduser().resolve()
    paths = source_files(root)
    if args.limit is not None:
        paths = paths[:max(0, args.limit)]
    output = args.output.expanduser().resolve()
    cleaned_root = output / "cleaned"
    cleaned_root.mkdir(parents=True, exist_ok=True)
    config = PipelineConfig(semantic_rules_enabled=args.semantic_rules)
    records = []
    exact = defaultdict(list)
    simhashes = []
    for number, path in enumerate(paths, 1):
        record = {"source": str(path), "relative_path": path.relative_to(root).as_posix(),
                  "extension": path.suffix.casefold(), "bytes": path.stat().st_size,
                  "status": "failed", "rejection_reasons": [], "warnings": []}
        try:
            raw, metadata, warnings = read_source(path)
            cleaned_result = clean_text(raw, config=config)
            cleaned = cleaned_result.text
            claimed, detected = language_bucket(path, cleaned)
            normalized = normalize(cleaned)
            exact_hash = digest(normalized) if normalized else ""
            fingerprint = simhash(cleaned)
            pii_flags = [name for name, pattern in PII_PATTERNS.items() if pattern.search(cleaned)]
            cleaned, pii_masks = mask_pii(cleaned)
            reasons = []
            if not normalized or len(normalized) < 80:
                reasons.append("too_short_or_empty")
            if "\ufffd" in cleaned or any(unicodedata.category(char) == "Co" for char in cleaned):
                reasons.append("garbled_or_private_use_characters")
            if detected == "mixed_ar_latin":
                reasons.append("mixed_arabic_latin")
            if pii_flags:
                reasons.append("pii_detected_and_masked")
            record.update({"status": "pass" if not reasons else "review",
                           "characters_before": len(raw), "characters_after": len(cleaned),
                           "claimed_language": claimed, "detected_script": detected,
                           "metadata": metadata, "cleaning_hits": cleaned_result.hit_counts,
                           "pii_flags": pii_flags, "pii_mask_counts": pii_masks,
                           "rejection_reasons": reasons, "warnings": warnings,
                           "normalized_sha256": exact_hash, "simhash": f"{fingerprint:016x}"})
            exact[exact_hash].append(record["relative_path"])
            simhashes.append((fingerprint, record["relative_path"]))
            destination = cleaned_root / Path(record["relative_path"]).with_suffix(".txt")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(cleaned, encoding="utf-8")
        except Exception as exc:
            record["rejection_reasons"] = ["read_or_parse_error"]
            record["error"] = str(exc)[:300]
        records.append(record)
        if number % 1000 == 0:
            print(f"processed {number}/{len(paths)}", flush=True)
    duplicate_groups = {key: values for key, values in exact.items() if key and len(values) > 1}
    for record in records:
        if record.get("normalized_sha256") in duplicate_groups:
            record["rejection_reasons"].append("exact_duplicate")
            record["status"] = "review"
    # Near duplicate candidates compare only within a coarse SimHash bucket;
    # this keeps the full-corpus pass bounded instead of O(files squared).
    near = []
    buckets = defaultdict(list)
    for value, path in simhashes:
        buckets[value >> 48].append((value, path))
    for candidates in buckets.values():
        for index, (left, left_path) in enumerate(candidates):
            for right, right_path in candidates[index + 1:]:
                if left_path == right_path:
                    continue
                if hamming(left, right) <= 3:
                    near.append({"left": left_path, "right": right_path, "hamming_distance": hamming(left, right)})
    by_bucket = defaultdict(list)
    for record in records:
        bucket = record.get("detected_script", "unknown")
        by_bucket[bucket].append(record)
    review = []
    for bucket in sorted(by_bucket):
        candidates = sorted(by_bucket[bucket], key=lambda item: (item["status"], item["relative_path"]))
        review.extend({"review_id": f"{bucket}-{index + 1}", "bucket": bucket,
                       "source": item["source"], "relative_path": item["relative_path"],
                       "status": item["status"], "rejection_reasons": item["rejection_reasons"],
                       "pii_flags": item.get("pii_flags", []), "review_decision": "pending"}
                      for index, item in enumerate(candidates[:args.sample_per_bucket]))
    summary = {"source_root": str(root), "files_seen": len(paths),
               "status_counts": dict(Counter(item["status"] for item in records)),
               "language_counts": dict(Counter(item.get("detected_script", "unknown") for item in records)),
               "reason_counts": dict(Counter(reason for item in records for reason in item["rejection_reasons"])),
               "pii_counts": dict(Counter(flag for item in records for flag in item.get("pii_flags", []))),
               "exact_duplicate_groups": len(duplicate_groups), "near_duplicate_candidates": len(near),
               "review_sample_size": len(review), "pdf_status": "pending OCR delivery"}
    (output / "manifest.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records) + "\n", encoding="utf-8")
    (output / "review-sample.jsonl").write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in review) + "\n", encoding="utf-8")
    (output / "near-duplicates.json").write_text(json.dumps(near, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
