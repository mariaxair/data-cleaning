"""Injectable dialect classifier and offline-only optional local MARBERT adapter."""

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from config import PipelineConfig
from ..settings import load_json

LABELS = ("MSA", "Algerian", "Moroccan")


@dataclass
class DialectPrediction:
    probabilities: dict[str, float]
    backend: str
    degraded: bool = True
    calibrated: bool = False
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


class DialectClassifier(Protocol):
    def classify(self, text: str) -> DialectPrediction: ...


class LexiconDialectClassifier:
    """Signals only; scores are not calibrated probabilities and never auto-pass."""

    def __init__(self, path: Path, reason: str = "No validated classifier configured."):
        self.lexicon = load_json(path)
        self.reason = reason

    def classify(self, text: str) -> DialectPrediction:
        tokens = set(re.findall(r"\w+", text.casefold()))
        hits = {label: sorted(tokens.intersection(word.casefold()
                            for word in self.lexicon.get(label, []) if word)) for label in LABELS}
        return DialectPrediction(
            {label: float(bool(words)) for label, words in hits.items()},
            "lexicon", True, False,
            tuple([self.reason] + [f"{label}:{word}" for label, words in hits.items() for word in words]))


class MarbertDialectClassifier:
    """Loads a LOCAL fine-tuned classification checkpoint; never downloads weights."""

    def __init__(self, path: Path, label_map: dict[str, str], validated: bool = False):
        if not path.is_dir():
            raise ValueError("DIALECT_MODEL_PATH must be a local checkpoint directory.")
        if set(label_map.values()) != set(LABELS):
            raise ValueError("Provide model-label -> MSA/Algerian/Moroccan in lexicon JSON label_map.")
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(path), local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(str(path), local_files_only=True)
        self.model.eval()
        self.label_map = label_map
        self.validated = validated
        self.labels = {str(self.model.config.id2label[i]) for i in range(self.model.config.num_labels)}
        if not self.labels <= set(label_map):
            raise ValueError("Every classifier label must have an explicit dialect mapping.")

    def classify(self, text: str) -> DialectPrediction:
        import torch
        values = {label: 0.0 for label in LABELS}
        # Short chunks prevent silent truncation of long reasoning fields.
        chunks = [text[start:start + 800] for start in range(0, len(text), 800)] or [""]
        for chunk in chunks:
            encoded = self.tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                scores = torch.softmax(self.model(**encoded).logits, dim=-1)[0].tolist()
            for index, probability in enumerate(scores):
                label = self.label_map[str(self.model.config.id2label[index])]
                values[label] = max(values[label], float(probability))
        return DialectPrediction(values, "local_marbert", not self.validated, True)


def create_classifier(config: PipelineConfig) -> DialectClassifier:
    if config.dialect_backend == "marbert":
        try:
            if config.dialect_model_path is None:
                raise ValueError("Local classifier path not configured.")
            return MarbertDialectClassifier(config.dialect_model_path,
                load_json(config.dialect_lexicon_path).get("label_map", {}), config.dialect_validated)
        except (ImportError, OSError, ValueError, RuntimeError):
            return LexiconDialectClassifier(config.dialect_lexicon_path,
                "Local MARBERT unavailable/unconfigured; using manual lexicon fallback.")
    if config.dialect_backend != "lexicon":
        raise ValueError("DIALECT_BACKEND must be lexicon or marbert.")
    return LexiconDialectClassifier(config.dialect_lexicon_path)


def dialect_verdict(prediction: DialectPrediction, config: PipelineConfig) -> str:
    value = prediction.probabilities.get("Moroccan", float("nan"))
    if not math.isfinite(value) or not 0 <= value <= 1:
        return "manual"
    if not prediction.calibrated:
        return "manual"
    if prediction.degraded:
        return "intercept" if value > config.dialect_degraded_intercept_threshold else "manual"
    if value < config.dialect_pass_threshold:
        return "pass"
    if value > config.dialect_intercept_threshold:
        return "intercept"
    return "manual"
