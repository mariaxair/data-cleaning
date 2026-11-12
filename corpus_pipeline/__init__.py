"""PDF and text corpus extraction and cleaning."""

from typing import Any

__all__ = ["process_file"]


def process_file(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """Load the orchestrator lazily so configuration has no import cycle."""
    from .processor import process_file as process
    return process(*args, **kwargs)
