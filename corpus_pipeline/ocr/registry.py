"""Small OCR adapter registry; providers can be swapped through configuration."""

from collections.abc import Callable

from config import PROVIDER_PROFILES, PipelineConfig
from .anthropic import AnthropicAdapter
from .azure import AzureAdapter
from .base import OCRAdapter
from .google import GoogleAdapter
from .heiyu import HeiyuAdapter
from .mistral import MistralAdapter
from .openai import OpenAIAdapter
from .openai_compatible import OpenAICompatibleAdapter

AdapterFactory = Callable[[PipelineConfig], OCRAdapter]

_REGISTRY: dict[str, AdapterFactory] = {
    "heiyu": HeiyuAdapter,
    "openai": OpenAIAdapter,
    "openai_compatible": OpenAICompatibleAdapter,
    "anthropic": AnthropicAdapter,
    "google": GoogleAdapter,
    "mistral": MistralAdapter,
    "azure": AzureAdapter,
}


def register_adapter(name: str, factory: AdapterFactory) -> None:
    """Register or replace an adapter factory, useful for extensions and tests."""
    _REGISTRY[name.lower()] = factory


def create_adapter(config: PipelineConfig) -> OCRAdapter:
    """Create the configured provider adapter."""
    name = config.ocr_provider.lower()
    try:
        factory = _REGISTRY[name]
    except KeyError:
        choices = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Unknown OCR_PROVIDER '{name}'. Available providers: {choices}.") from None
    return factory(config)


def registered_providers() -> tuple[str, ...]:
    """Return registered provider names, ordered like the provider profiles."""
    ordered = [name for name in PROVIDER_PROFILES if name in _REGISTRY]
    extras = sorted(name for name in _REGISTRY if name not in PROVIDER_PROFILES)
    return tuple(ordered + extras)
