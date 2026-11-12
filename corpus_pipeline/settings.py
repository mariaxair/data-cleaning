"""Default-off settings shared by the optional training stages."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def env_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).lower() in {"1", "true", "yes"}


def load_json(path: Path | None) -> dict[str, Any]:
    """Missing optional domain data is an explicit empty configuration."""
    if path is None or not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path.name}.")
    return value


@dataclass
class TrainingSettings:
    """Environment settings are evaluated when a pipeline config is created."""

    semantic_rules_enabled: bool = field(default_factory=lambda: env_bool("SEMANTIC_RULES_ENABLED"))
    rules_data_path: Path = field(default_factory=lambda: Path(os.getenv(
        "RULES_DATA_PATH", "corpus_pipeline/rules/data/rules.json")))
    wrong_words_path: Path = field(default_factory=lambda: Path(os.getenv(
        "WRONG_WORDS_PATH", "corpus_pipeline/rules/data/wrong_words.tsv")))
    synthesis_enabled: bool = field(default_factory=lambda: env_bool("SYNTHESIS_ENABLED"))
    synthesis_model: str = field(default_factory=lambda: os.getenv("SYNTHESIS_MODEL", ""))
    synthesis_language: str = field(default_factory=lambda: os.getenv("SYNTHESIS_LANGUAGE", "ar"))
    synthesis_concurrency: int = field(default_factory=lambda: int(os.getenv("SYNTHESIS_CONCURRENCY", "2")))
    synthesis_max_samples: int = field(default_factory=lambda: int(os.getenv("SYNTHESIS_MAX_SAMPLES", "10")))
    synthesis_max_attempts: int = field(default_factory=lambda: int(os.getenv("SYNTHESIS_MAX_ATTEMPTS", "2")))
    synthesis_max_tokens: int = field(default_factory=lambda: int(os.getenv("SYNTHESIS_MAX_TOKENS", "2048")))
    synthesis_screening_mode: str = field(default_factory=lambda: os.getenv("SYNTHESIS_SCREENING_MODE", "legal"))
    topic_sample_chars: int = field(default_factory=lambda: int(os.getenv("TOPIC_SAMPLE_CHARS", "12000")))
    topic_max_calls: int = field(default_factory=lambda: int(os.getenv("TOPIC_MAX_CALLS", "100")))
    topic_group_batch_size: int = field(default_factory=lambda: int(os.getenv("TOPIC_GROUP_BATCH_SIZE", "20")))
    topic_similarity_threshold: float = field(default_factory=lambda: float(os.getenv("TOPIC_SIMILARITY_THRESHOLD", "0.75")))
    general_chunk_chars: int = field(default_factory=lambda: int(os.getenv("GENERAL_CHUNK_CHARS", "4000")))
    screening_path: Path = field(default_factory=lambda: Path(os.getenv(
        "SCREENING_PATH", "corpus_pipeline/synthesis/data/screening.json")))
    manual_overrides_path: Path | None = field(default_factory=lambda: (
        Path(os.environ["MANUAL_OVERRIDES_PATH"]) if os.getenv("MANUAL_OVERRIDES_PATH") else None))
    article_group_size: int = field(default_factory=lambda: int(os.getenv("ARTICLE_GROUP_SIZE", "3")))
    min_article_chars: int = field(default_factory=lambda: int(os.getenv("MIN_ARTICLE_CHARS", "40")))
    corpus_processing_enabled: bool = field(default_factory=lambda: env_bool("CORPUS_PROCESSING_ENABLED"))
    language_mixing_threshold: float = field(default_factory=lambda: float(os.getenv("LANGUAGE_MIXING_THRESHOLD", "0.2")))
    dialect_backend: str = field(default_factory=lambda: os.getenv("DIALECT_BACKEND", "lexicon"))
    dialect_model_path: Path | None = field(default_factory=lambda: (
        Path(os.environ["DIALECT_MODEL_PATH"]) if os.getenv("DIALECT_MODEL_PATH") else None))
    dialect_lexicon_path: Path = field(default_factory=lambda: Path(os.getenv(
        "DIALECT_LEXICON_PATH", "corpus_pipeline/governance/data/lexicon.json")))
    dialect_pass_threshold: float = field(default_factory=lambda: float(os.getenv("DIALECT_PASS_THRESHOLD", "0.2")))
    dialect_intercept_threshold: float = field(default_factory=lambda: float(os.getenv("DIALECT_INTERCEPT_THRESHOLD", "0.5")))
    dialect_degraded_intercept_threshold: float = field(default_factory=lambda: float(os.getenv(
        "DIALECT_DEGRADED_INTERCEPT_THRESHOLD", "0.95")))
    dialect_validated: bool = field(default_factory=lambda: env_bool("DIALECT_VALIDATED"))
    review_enabled: bool = field(default_factory=lambda: env_bool("REVIEW_ENABLED"))
    review_llm_enabled: bool = field(default_factory=lambda: env_bool("REVIEW_LLM_ENABLED"))
    review_max_attempts: int = field(default_factory=lambda: int(os.getenv("REVIEW_MAX_ATTEMPTS", "2")))
    stage_output_dir: Path = field(default_factory=lambda: Path(os.getenv("STAGE_OUTPUT_DIR", "data/training")))

    def validate_training(self) -> None:
        """Reject unsafe cost controls and invalid threshold ordering."""
        if self.synthesis_screening_mode not in {"legal", "automatic"}:
            raise ValueError("SYNTHESIS_SCREENING_MODE must be legal or automatic.")
        if min(self.topic_sample_chars, self.topic_max_calls, self.topic_group_batch_size,
               self.general_chunk_chars) < 1:
            raise ValueError("Topic limits and general chunk size must be positive.")
        if self.general_chunk_chars < self.min_article_chars:
            raise ValueError("GENERAL_CHUNK_CHARS must be >= MIN_ARTICLE_CHARS.")
        if not 0 <= self.topic_similarity_threshold <= 1:
            raise ValueError("TOPIC_SIMILARITY_THRESHOLD must be between 0 and 1.")
        if self.article_group_size not in {2, 3, 4}:
            raise ValueError("ARTICLE_GROUP_SIZE must be 2, 3, or 4.")
        if min(self.synthesis_concurrency, self.synthesis_max_samples,
               self.synthesis_max_attempts, self.synthesis_max_tokens, self.min_article_chars) < 1:
            raise ValueError("Synthesis limits, concurrency, and article size must be positive.")
        if self.review_max_attempts < 0:
            raise ValueError("REVIEW_MAX_ATTEMPTS cannot be negative.")
        if not 0 <= self.language_mixing_threshold <= 0.5:
            raise ValueError("LANGUAGE_MIXING_THRESHOLD must be between 0 and 0.5.")
        if not 0 <= self.dialect_pass_threshold <= self.dialect_intercept_threshold <= 1:
            raise ValueError("Dialect thresholds must satisfy 0 <= pass <= intercept <= 1.")
        if not self.dialect_intercept_threshold <= self.dialect_degraded_intercept_threshold <= 1:
            raise ValueError("Degraded interception threshold must be >= the interception threshold.")
