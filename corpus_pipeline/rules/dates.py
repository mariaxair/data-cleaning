"""Exact reviewed date substitutions; never infer an ambiguous numeric date."""


def apply(text: str, mappings: dict[str, str]) -> tuple[str, int]:
    from .tables import replace_phrases
    return replace_phrases(text, mappings)
