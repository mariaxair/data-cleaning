"""Conservative configurable document tiering and manual overrides."""

from dataclasses import dataclass, field
from pathlib import Path

from ..settings import load_json

TIERS = {"statutory_law", "case_law", "administrative_noise", "manual", "general"}


@dataclass
class ScreeningResult:
    tier: str
    degraded: bool
    signals: dict[str, int] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


def screen_document(text: str, source: Path, policy: dict, overrides: dict) -> ScreeningResult:
    """Manual overrides use a resolved file path, or an unambiguous file name."""
    override = overrides.get(str(source.resolve()), overrides.get(source.name))
    if override:
        tier = override.get("tier", "manual")
        if tier not in TIERS:
            raise ValueError("Manual tier must be statutory_law, case_law, administrative_noise, manual, or general.")
        return ScreeningResult(tier, False, metadata=override)
    lower = text.casefold()
    counts = {tier: sum(lower.count(word.casefold()) for word in policy.get(tier, []) if word)
              for tier in ("statutory_law", "case_law", "administrative_noise")}
    matched = [tier for tier, count in counts.items() if count]
    # Conflicting tiers are not resolved by guessing.
    tier = matched[0] if len(matched) == 1 else "manual"
    return ScreeningResult(tier, not bool(policy.get("validated", False)), counts)
