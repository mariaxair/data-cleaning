"""Export whole-article APS JSONL locally, as a sample or full corpus.

Only automatically passed Arabic, French and English APS records are used.
Sampling rotates across available categories, with seeded random selection
within each category. Text is not split, rewritten, translated or completed.
Article records use an explicit field allowlist; local paths, image URLs,
scraping details and audit fields are never added to the record schema.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import uuid
from collections import Counter, defaultdict
from pathlib import Path


LANGUAGES = ("ar", "fr", "en")
ARTICLE_FIELDS = ("uuid", "text", "title", "lang", "source_url",
                  "article_id", "published", "category")


def io_path(path: Path) -> Path:
    value = str(path.absolute())
    if os.name == "nt" and not value.startswith("\\\\?\\"):
        value = "\\\\?\\UNC\\" + value[2:] if value.startswith("\\\\") else "\\\\?\\" + value
    return Path(value)


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be greater than zero")
    return number


def choose_records(records: list[dict], count: int, seed: int) -> list[dict]:
    groups = defaultdict(list)
    for record in sorted(records, key=lambda r: r["relative_path"]):
        category = record.get("metadata", {}).get("aps", {}).get("Category", "unknown")
        groups[category].append(record)
    rng = random.Random(seed)
    categories = sorted(groups)
    rng.shuffle(categories)
    for category in categories:
        rng.shuffle(groups[category])
    selected = []
    while len(selected) < count:
        for category in categories:
            if groups[category]:
                selected.append(groups[category].pop())
                if len(selected) == count:
                    return selected
        if not any(groups.values()):
            break
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/week4-md-final-ar-fr-en"))
    parser.add_argument("--output", type=Path,
                        help="Default: data/aps-jsonl-final with --all, otherwise data/aps-jsonl-test")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true",
                      help="Export every eligible APS record, without sampling")
    mode.add_argument("--per-language", type=positive_int,
                      help="Sample N articles per language (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.per_language is None:
        args.per_language = 10
    source_root = args.input.resolve()
    output_root = (args.output or Path("data/aps-jsonl-final" if args.all else "data/aps-jsonl-test")).resolve()
    if source_root == output_root or source_root.is_relative_to(output_root) or output_root.is_relative_to(source_root):
        parser.error("Use an output directory separate from the input directory")
    if io_path(output_root).exists() and any(io_path(output_root).iterdir()):
        parser.error("Output directory is not empty; choose a new --output directory")

    buckets = {language: [] for language in LANGUAGES}
    with io_path(source_root / "manifest.jsonl").open(encoding="utf-8-sig") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            language = record.get("selected_language")
            if language not in buckets or record.get("status") != "pass":
                continue
            if record.get("selection_status") != "pass":
                continue
            if "aps" not in {part.casefold() for part in Path(record.get("relative_path", "")).parts}:
                continue
            buckets[language].append(record)

    if not args.all:
        for language, candidates in buckets.items():
            if len(candidates) < args.per_language:
                parser.error(f"Only {len(candidates)} eligible {language} files; requested {args.per_language}")
    if not any(buckets.values()):
        parser.error("No eligible APS records found")

    articles = []
    bodies = {}
    flag_counts = Counter()
    public_excerpt_uuids = []
    selected_counts = {}
    for language in LANGUAGES:
        selected_counts[language] = len(buckets[language]) if args.all else args.per_language
    print(f"Preparing {sum(selected_counts.values()):,} APS records; no model or API calls.", flush=True)
    for language in LANGUAGES:
        selected = (sorted(buckets[language], key=lambda r: r["relative_path"]) if args.all
                    else choose_records(buckets[language], args.per_language, args.seed))
        for record in selected:
            source_path = Path(record["final_markdown_path"]).resolve()
            if not io_path(source_path).is_relative_to(io_path(source_root)):
                parser.error(f"Markdown path outside input directory: {source_path}")
            body = io_path(source_path).read_bytes()
            text = body.decode("utf-8")
            if not text.strip():
                parser.error(f"Selected article is empty: {source_path}")
            aps = record.get("metadata", {}).get("aps", {})
            identifier = record.get("document_id") or str(uuid.uuid5(
                uuid.NAMESPACE_URL, "corpus-source:" + record["relative_path"]
            ))
            identifier = str(uuid.UUID(identifier))
            if identifier in bodies:
                parser.error(f"Duplicate document ID: {identifier}")
            protected = str(aps.get("Protected", "")).casefold().startswith(("yes", "true"))
            flags = ["source_marked_public_excerpt"] if protected else []
            if aps.get("Language") and aps["Language"] != language:
                flags.append("metadata_language_disagrees")
            if not aps.get("Title"):
                flags.append("missing_title_metadata")
            if not aps.get("URL"):
                flags.append("missing_source_url")
            flag_counts.update(flags)
            if protected:
                public_excerpt_uuids.append(identifier)
            articles.append({
                "uuid": identifier,
                "text": text,
                "title": aps.get("Title"),
                "lang": language,
                "source_url": aps.get("URL"),
                "article_id": aps.get("Article-ID"),
                "published": aps.get("Published"),
                "category": aps.get("Category"),
            })
            bodies[identifier] = body
            if len(articles) % 1000 == 0:
                print(f"Read and validated {len(articles):,} articles.", flush=True)

    # All selected sources are read and validated before creating the export.
    io_path(output_root).mkdir(parents=True, exist_ok=True)
    with io_path(output_root / "articles.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for index, article in enumerate(articles, 1):
            exported = {field: article[field] for field in ARTICLE_FIELDS}
            handle.write(json.dumps(exported, ensure_ascii=False) + "\n")
            if not args.all:
                sample_path = io_path(output_root / "samples" / article["lang"] / (article["uuid"] + ".md"))
                sample_path.parent.mkdir(parents=True, exist_ok=True)
                sample_path.write_bytes(bodies[article["uuid"]])
            if index % 1000 == 0:
                print(f"Exported {index:,} articles.", flush=True)

    summary = {
        "records_exported": len(articles),
        "language_counts": dict(Counter(a["lang"] for a in articles)),
        "available_aps_counts": {language: len(items) for language, items in buckets.items()},
        "mode": "all" if args.all else "sample",
        "sampling": None if args.all else {"per_language": args.per_language, "seed": args.seed,
                     "method": "Round-robin categories with seeded random selection; pilot, not a corpus quality estimate"},
        "by_language": {language: {
            "categories": len({a["category"] for a in articles if a["lang"] == language}),
            "minimum_characters": min((len(a["text"]) for a in articles if a["lang"] == language), default=None),
            "maximum_characters": max((len(a["text"]) for a in articles if a["lang"] == language), default=None),
        } for language in LANGUAGES},
        "content_flag_counts": dict(flag_counts),
        "public_excerpt_uuids": public_excerpt_uuids,
        "policy": "One source file per record; exact available text retained, including public excerpts; no splitting or API calls",
        "human_review_status": "not_reviewed",
    }
    io_path(output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    io_path(output_root / "README.md").write_text(
        "# APS whole-article export\n\n"
        "articles.jsonl contains one record for each selected source document.\n"
        "Sample mode also creates samples/ar, samples/fr and samples/en for reading.\n"
        "Full mode exports JSONL without duplicating all Markdown files.\n\n"
        "Text, including original line endings, is retained exactly. Metadata stays separate.\n"
        "Article records contain only uuid, text, title, lang, source_url, article_id, published and category.\n"
        "Local paths, image URLs, scraping details and technical audit fields are excluded from article records.\n"
        "Language comes from existing labels; it has not been independently classified.\n"
        "Source metadata marked Protected is listed by UUID in summary.json as a public excerpt.\n"
        "Absence of this flag does not prove an article is complete.\n"
        "Packaging is not human quality approval or a statistical quality estimate.\n\n"
        "Check actual language, title/body alignment, completeness, encoding, repetition and masking.\n"
        "No text splitting, model calls or question/answer generation was performed.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
