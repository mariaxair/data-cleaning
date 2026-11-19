"""Source-preserving chunk proposals; Chonkie stays optional and lazy."""

import hashlib
import re
from importlib.metadata import version

from config import PipelineConfig
from .legal import pages_for


def _semantic(config: PipelineConfig):
    """Explicit enablement may initialize the operator-selected embedding model."""
    from chonkie import SemanticChunker
    if version("chonkie") != "1.5.2":
        raise ValueError("Unsupported optional chunker version.")
    if not config.metadata_embedding_model.strip():
        raise ValueError("Set an embedding model explicitly.")
    return SemanticChunker(
        embedding_model=config.metadata_embedding_model, threshold=config.metadata_semantic_threshold,
        chunk_size=config.metadata_chunk_tokens, similarity_window=config.metadata_similarity_window,
        skip_window=0, delim=config.metadata_delimiters, include_delim=config.metadata_include_delim)


def recursive_spans(text: str, max_chars: int, max_tokens: int) -> list[tuple[int, int]]:
    """Prefer paragraphs/sentences; UTF-8 byte counts bound fallback token estimates."""
    spans, start = [], 0
    while start < len(text):
        end = start
        size = 0
        while end < len(text) and end - start < max_chars:
            cost = len(text[end].encode("utf-8"))
            if size + cost > max_tokens and end > start:
                break
            size += cost
            end += 1
        if end < len(text):
            for separator in ("\n\n", "\n", "؟", "؛", ". ", " "):
                found = text.rfind(separator, start + (end - start) // 2, end)
                if found >= start:
                    end = found + len(separator)
                    break
        spans.append((start, end))
        start = end
    return spans


def chunk_document(text: str, config: PipelineConfig, *, legal_spans: list[dict] | None = None,
                   dry_run: bool = False, chunker=None) -> tuple[list[dict], dict]:
    """Validate reconstruction; legal spans always take priority, including dry runs."""
    config.validate_metadata()
    reason, selected = None, "legal_structure" if legal_spans is not None else "paragraph_recursive"
    spans = legal_spans
    if spans is None:
        if config.metadata_semantic_enabled and not dry_run:
            try:
                proposal = (chunker or _semantic(config)).chunk(text) if text else []
                candidate, cursor = [], 0
                for item in proposal:
                    start, end = item.start_index, item.end_index
                    if (type(start) is not int or type(end) is not int or start != cursor or
                            not start < end <= len(text) or text[start:end] != item.text or
                            type(item.token_count) is not int or item.token_count > config.metadata_chunk_tokens or
                            end - start > config.metadata_chunk_chars):
                        raise ValueError("Unverified semantic chunk offsets/size.")
                    candidate.append({"start": start, "end": end, "token_count": item.token_count})
                    cursor = end
                if cursor != len(text):
                    raise ValueError("Incomplete semantic reconstruction.")
                spans, selected = candidate, "chonkie_semantic_1.5.2"
            except Exception:
                # Exceptions can contain paths/credentials; keep a fixed, actionable reason.
                reason = "Semantic chunking unavailable or failed source/size validation; using deterministic fallback."
        elif config.metadata_semantic_enabled:
            reason = "Dry run: semantic embeddings were not initialized; fallback boundaries are provisional."
        else:
            reason = "Semantic chunking disabled; using deterministic fallback (not semantic boundaries)."
        if spans is None:
            spans = [{"start": start, "end": end} for start, end in recursive_spans(
                text, config.metadata_chunk_chars, config.metadata_chunk_tokens)]
    chunks = []
    for index, span in enumerate(spans):
        start, end = span["start"], span["end"]
        body = text[start:end]
        headings = [m.group() for m in re.finditer(r"(?m)^\s*#{1,6}[^\n]*", text[:end])]
        chunks.append({**span, "index": index, "text": body, "char_count": len(body),
                       "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
                       "token_count": span.get("token_count"), "token_estimate_upper": len(body.encode("utf-8")),
                       "pages": pages_for(text, start, end), "level": span.get("level", "passage"),
                       "structural_path": span.get("structural_path", headings[-1:]),
                       "article_numbers": span.get("article_numbers", [])})
    if "".join(chunk["text"] for chunk in chunks) != text:
        raise ValueError("Chunk reconstruction must exactly match the source.")
    return chunks, {"chunker": selected, "degraded": reason is not None, "fallback_reason": reason,
                    "reconstruction_exact": True, "intentional_gaps": []}
