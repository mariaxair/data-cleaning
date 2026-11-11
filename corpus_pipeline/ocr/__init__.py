"""Provider-agnostic OCR adapters."""

from .registry import create_adapter, register_adapter

__all__ = ["create_adapter", "register_adapter"]
