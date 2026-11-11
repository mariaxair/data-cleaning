"""Central configuration for the corpus processing pipeline."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


# Load this project's .env without overriding variables already set by the shell.
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env", override=False)


@dataclass(frozen=True)
class ProviderProfile:
    """Defaults and credential location for one OCR provider."""

    id: str
    label: str
    api_key_env: str
    base_url: str = ""
    model: str = ""
    requires_base_url: bool = False
    notes: str = ""


# Provider catalogue. Adding a provider means adding one entry here plus one
# adapter module registered in corpus_pipeline/ocr/registry.py.
PROVIDER_PROFILES: dict[str, ProviderProfile] = {
    "heiyu": ProviderProfile(
        id="heiyu",
        label="HeiyuCode (OpenAI-compatible)",
        api_key_env="HEIYU_API_KEY",
        base_url="https://www.heiyucode.com/v1",
        model="gpt-6-sol",
        notes="Gateway models: gpt-6.1-terra, gpt-6.1-sol, gpt-6-astra, gpt-6-sol, gpt-5.6-sol, gpt-5.6-terra.",
    ),
    "openai": ProviderProfile(
        id="openai",
        label="OpenAI",
        api_key_env="OPENAI_API_KEY",
        base_url="https://api.openai.com/v1",
        model="gpt-4o-mini",
    ),
    "anthropic": ProviderProfile(
        id="anthropic",
        label="Anthropic",
        api_key_env="ANTHROPIC_API_KEY",
        model="claude-sonnet-4-5",
        notes="Base URL defaults to https://api.anthropic.com.",
    ),
    "google": ProviderProfile(
        id="google",
        label="Google Gemini",
        api_key_env="GOOGLE_API_KEY",
        model="gemini-2.5-flash",
        notes="Base URL defaults to https://generativelanguage.googleapis.com/v1beta.",
    ),
    "mistral": ProviderProfile(
        id="mistral",
        label="Mistral",
        api_key_env="MISTRAL_API_KEY",
        base_url="https://api.mistral.ai/v1",
        model="pixtral-12b-2409",
    ),
    "azure": ProviderProfile(
        id="azure",
        label="Azure OpenAI",
        api_key_env="AZURE_OPENAI_API_KEY",
        requires_base_url=True,
        notes="OCR_MODEL is the Azure deployment name; the resource URL is required.",
    ),
    "openai_compatible": ProviderProfile(
        id="openai_compatible",
        label="Other OpenAI-compatible",
        api_key_env="OCR_API_KEY",
        notes="Set a base URL and model; the key is read from OCR_API_KEY.",
    ),
}

DEFAULT_PROVIDER = "heiyu"


def provider_profile(name: str) -> ProviderProfile:
    """Return the profile for a provider id, with a helpful error if unknown."""
    key = (name or "").strip().lower()
    try:
        return PROVIDER_PROFILES[key]
    except KeyError:
        choices = ", ".join(sorted(PROVIDER_PROFILES))
        raise ValueError(f"Unknown OCR provider '{name}'. Available providers: {choices}.") from None


def provider_choices() -> list[tuple[str, str]]:
    """Return (label, id) pairs for GUI dropdowns."""
    return [(profile.label, profile.id) for profile in PROVIDER_PROFILES.values()]


# Paths and language hints
INPUT_DIR = Path("data/in")
OUTPUT_DIR = Path("data/out")
SUPPORTED_LANGUAGES = ["ar", "fr", "en"]
PAGE_MARKER_TEMPLATE = "\n\n--- Page {page} ---\n\n"
INCLUDE_PAGE_MARKERS = True

# Per-page PDF routing thresholds
MIN_EXTRACTED_CHARS = 80
MIN_VALID_CHAR_RATIO = 0.85
FULL_PAGE_IMAGE_RATIO = 0.65
PDF_RENDER_DPI = 220

# OCR provider settings. Keys live in .env, one variable per provider; the
# active provider's variable is resolved by PipelineConfig.effective_api_key().
# An empty base URL or model falls back to the provider profile above.
OCR_PROVIDER = DEFAULT_PROVIDER
OCR_API_KEY = ""
OCR_BASE_URL = ""
OCR_MODEL = ""
OCR_TIMEOUT_S = 60
OCR_MAX_RETRIES = 3
OCR_RETRY_BASE_S = 1.0
OCR_TEST_CONNECTION_ON_STARTUP = False

# Persistent cache and file completion manifest
CACHE_DB = Path("data/pipeline.sqlite3")
LOG_LEVEL = "INFO"


@dataclass
class PipelineConfig:
    """Runtime settings, including session-only GUI overrides."""

    input_dir: Path = INPUT_DIR
    output_dir: Path = OUTPUT_DIR
    languages: list[str] = field(default_factory=lambda: list(SUPPORTED_LANGUAGES))
    page_marker_template: str = PAGE_MARKER_TEMPLATE
    include_page_markers: bool = INCLUDE_PAGE_MARKERS
    min_extracted_chars: int = MIN_EXTRACTED_CHARS
    min_valid_char_ratio: float = MIN_VALID_CHAR_RATIO
    full_page_image_ratio: float = FULL_PAGE_IMAGE_RATIO
    render_dpi: int = PDF_RENDER_DPI
    ocr_provider: str = OCR_PROVIDER
    ocr_api_key: str = OCR_API_KEY
    session_api_key_override: bool = False
    ocr_base_url: str = OCR_BASE_URL
    ocr_model: str = OCR_MODEL
    ocr_timeout_s: int = OCR_TIMEOUT_S
    ocr_max_retries: int = OCR_MAX_RETRIES
    ocr_retry_base_s: float = OCR_RETRY_BASE_S
    cache_db: Path = CACHE_DB

    def __post_init__(self) -> None:
        # Fill base URL and model from the provider profile unless overridden.
        # Unknown provider ids are left untouched so custom adapters keep working.
        profile = PROVIDER_PROFILES.get((self.ocr_provider or "").strip().lower())
        if profile:
            self.ocr_provider = profile.id
            if not self.ocr_base_url.strip():
                self.ocr_base_url = profile.base_url
            if not self.ocr_model.strip():
                self.ocr_model = profile.model

    @property
    def api_key_env(self) -> str:
        """Environment variable that holds this provider's API key."""
        profile = PROVIDER_PROFILES.get((self.ocr_provider or "").strip().lower())
        return profile.api_key_env if profile else "OCR_API_KEY"

    def effective_api_key(self) -> str:
        """Resolve the key: session override, provider env, legacy env, config."""
        if self.session_api_key_override and self.ocr_api_key.strip():
            return self.ocr_api_key.strip()
        provider_key = os.environ.get(self.api_key_env, "").strip()
        if provider_key:
            return provider_key
        # Legacy single-key variable kept for existing scripts and tests.
        return os.environ.get("OCR_API_KEY", self.ocr_api_key).strip()
