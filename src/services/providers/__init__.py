from .base import LLMProvider
from .ollama_provider import OllamaProvider
from .openai_compat import OpenAICompatibleProvider
from .gemini_provider import GeminiProvider

# ---------------------------------------------------------------------------
# Registry  (provider_name -> (class, default_config))
# ---------------------------------------------------------------------------

REGISTRY: dict[str, type[LLMProvider]] = {}

# Default field definitions for the UI
PROVIDER_DEFS: dict[str, dict] = {
    "ollama": {
        "label": "Ollama (Local)",
        "fields": [
            {"key": "base_url", "label": "Base URL", "placeholder": "http://localhost:11434"},
        ],
    },
    "openai": {
        "label": "OpenAI",
        "fields": [
            {"key": "api_key", "label": "API Key", "secret": True},
            {"key": "base_url", "label": "Base URL", "placeholder": "https://api.openai.com/v1"},
        ],
    },
    "alibaba": {
        "label": "Alibaba Cloud (DashScope)",
        "fields": [
            {"key": "api_key", "label": "API Key", "secret": True},
            {"key": "base_url", "label": "Base URL", "placeholder": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        ],
    },
    "gemini": {
        "label": "Google Gemini",
        "fields": [
            {"key": "api_key", "label": "API Key", "secret": True},
        ],
    },
}


def register_provider(name: str, cls: type[LLMProvider]) -> None:
    REGISTRY[name] = cls


def get_provider(name: str, config: dict | None = None) -> LLMProvider:
    cls = REGISTRY.get(name)
    if cls is None:
        msg = f"Unknown provider {name!r}. Available: {list(REGISTRY)}"
        raise ValueError(msg)
    return cls(config)


def get_current_provider() -> LLMProvider:
    """Shortcut: read the active provider from settings and return an instance."""
    from ..llm_service import get_settings
    settings = get_settings(include_secrets=True)
    name = settings.get("provider", "ollama")
    provider_config = settings.get("providers", {}).get(name, {})
    return get_provider(name, provider_config)


def get_available_providers() -> dict[str, dict]:
    """Return provider metadata (without secrets) for the API."""
    from ..llm_service import get_settings
    settings = get_settings()
    current = settings.get("provider", "ollama")
    result = {}
    for name, defn in PROVIDER_DEFS.items():
        configured = settings.get("providers", {}).get(name, {})
        fields = []
        for f in defn["fields"]:
            val = configured.get(f["key"], "")
            fields.append({
                "key": f["key"],
                "label": f["label"],
                "secret": f.get("secret", False),
                "has_value": bool(val),
                "placeholder": f.get("placeholder", ""),
            })
        result[name] = {
            "label": defn["label"],
            "current": name == current,
            "fields": fields,
        }
    return result


# Register built-in providers
register_provider("ollama", OllamaProvider)
register_provider("openai", OpenAICompatibleProvider)
register_provider("alibaba", OpenAICompatibleProvider)
register_provider("gemini", GeminiProvider)


__all__ = [
    "LLMProvider",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "GeminiProvider",
    "get_provider",
    "get_current_provider",
    "get_available_providers",
    "PROVIDER_DEFS",
]
