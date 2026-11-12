"""Script-based language governance; Latin script does not prove French."""

import unicodedata
from collections import Counter


def assess_language(text: str, claimed: str, mixing_threshold: float) -> dict:
    counts = Counter({"arabic": 0, "latin": 0, "other": 0})
    for char in text:
        if not unicodedata.category(char).startswith("L"):
            continue
        name = unicodedata.name(char, "")
        script = "arabic" if "ARABIC" in name else "latin" if "LATIN" in name else "other"
        counts[script] += 1
    total = sum(counts.values())
    mixing = min(counts["arabic"], counts["latin"]) / max(total, 1)
    if not total:
        detected, decision = "unknown", "manual"
    elif counts["other"]:
        detected, decision = "other", "manual"
    elif mixing > mixing_threshold:
        detected, decision = "ar_latin_mixed", "intercept"
    elif counts["arabic"] > counts["latin"]:
        detected, decision = "ar", "pass"
    else:
        detected, decision = "latin", "manual"
    return {"claimed": claimed, "detected": detected, "mixing_ratio": mixing,
            "counts": dict(counts), "decision": decision,
            "misplaced": bool(claimed and detected == "ar" and claimed != "ar"),
            "degraded": detected != "ar"}
