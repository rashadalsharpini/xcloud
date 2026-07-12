import json
import os
import re
from dataclasses import dataclass, field

from .providers import get_current_provider, get_provider

# ---- Settings persistence ------------------------------------------------- #

SETTINGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "settings.json")
)

DEFAULT_SETTINGS = {
    "provider": "ollama",
    "default_model": "auto",
    "providers": {
        "ollama": {"base_url": "http://localhost:11434"},
        "openai": {"api_key": "", "base_url": "https://api.openai.com/v1"},
        "alibaba": {"api_key": "", "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1"},
        "gemini": {"api_key": ""},
    },
    "embedding": {
        "provider": "ollama",
        "model": "nomic-embed-text:latest",
    },
}

SECRET_KEYS = {"api_key"}


def _load_settings() -> dict:
    if not os.path.exists(SETTINGS_PATH):
        _save_settings(DEFAULT_SETTINGS)
        return dict(DEFAULT_SETTINGS)
    try:
        with open(SETTINGS_PATH, "r") as f:
            stored = json.load(f)
        merged = dict(DEFAULT_SETTINGS)
        merged.update(stored)
        for key in ("providers", "embedding"):
            if key in stored:
                merged[key] = stored[key]
        return merged
    except (json.JSONDecodeError, OSError):
        _save_settings(DEFAULT_SETTINGS)
        return dict(DEFAULT_SETTINGS)


def _save_settings(settings: dict) -> None:
    with open(SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)


def get_settings(include_secrets: bool = False) -> dict:
    settings = _load_settings()
    if not include_secrets:
        providers = settings.get("providers", {})
        sanitized = {}
        for pname, pconf in providers.items():
            sanitized[pname] = {
                k: ("*****" if k in SECRET_KEYS and v else v)
                for k, v in pconf.items()
            }
        settings = dict(settings)
        settings["providers"] = sanitized
    return settings


def get_default_model() -> str | None:
    settings = _load_settings()
    model_pref = settings.get("default_model", "auto")

    if model_pref and model_pref != "auto":
        return model_pref

    provider = get_current_provider()
    models = provider.get_chat_models()
    if models:
        return models[0]

    # Fallback auto-pull only for Ollama
    if settings.get("provider") == "ollama":
        return _auto_pull_ollama_model()

    return None


def _auto_pull_ollama_model() -> str | None:
    import platform
    import subprocess

    vram_gb = 0.0
    try:
        if platform.system() in ["Linux", "Windows"]:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                stderr=subprocess.DEVNULL, text=True,
            )
            vram_gb = int(output.strip().split("\n")[0]) / 1024.0
    except Exception:
        pass

    target_model = "qwen3:8b" if vram_gb >= 8 else "qwen3:1.7b"

    print(f"No LLM found. VRAM detected: {vram_gb:.1f}GB. Pulling {target_model} via Ollama...")
    try:
        from ollama import pull
        pull(target_model)
        print(f"Successfully pulled {target_model}")

        index_model = "nomic-embed-text:latest"
        print(f"Checking for indexing model {index_model}...")
        pull(index_model)
        print(f"Successfully ensured {index_model} is available.")

        return target_model
    except Exception as e:
        print(f"Failed to pull models: {e}")

    return None


def resolve_model(preferred: str | None) -> str:
    """
    Return `preferred` if the active provider actually serves it, otherwise
    fall back to the provider's default model.

    Chats persist the model they were created with, so after switching
    providers a stored model (e.g. an Ollama model) may not exist on the new
    provider (e.g. Gemini). Passing it through would 404 mid-stream.
    """
    if preferred:
        try:
            if get_current_provider().is_model_supported(preferred):
                return preferred
        except Exception:
            pass
    return get_default_model() or ""


def friendly_error(exc: Exception) -> str:
    """
    Compress provider exceptions into a short, human-readable message for the
    chat UI (raw payloads can be hundreds of characters of JSON).
    """
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    if code == 429:
        return (
            "The model is rate-limited or out of quota. "
            "Try again in a minute, pick another model, or check your plan/billing."
        )
    if code == 503:
        return (
            "The model is overloaded right now (provider-side). "
            "Please try again in a few seconds."
        )
    if code in (401, 403):
        return "The provider rejected the API key. Check the provider configuration."
    if code == 404:
        return "The selected model is not available on this provider. Pick another model."

    # Fall back to the exception's own message, truncated.
    msg = str(exc)
    # google-genai errors embed a JSON blob; prefer its 'message' field.
    m = re.search(r"'message': '([^']+)'", msg)
    if m:
        msg = m.group(1)
    return msg[:300]


def save_default_model(model_name: str) -> dict:
    settings = _load_settings()
    settings["default_model"] = model_name
    _save_settings(settings)
    return settings


def save_provider_config(provider_name: str, config: dict) -> dict:
    settings = _load_settings()
    providers = settings.setdefault("providers", {})
    existing = providers.get(provider_name, {})
    existing.update(config)
    providers[provider_name] = existing
    settings["providers"] = providers
    _save_settings(settings)
    return get_settings()


def set_active_provider(provider_name: str) -> dict:
    settings = _load_settings()
    settings["provider"] = provider_name
    _save_settings(settings)
    return get_settings()


SYSTEM_PROMPT = """You are Xcloud, an intelligent AI assistant created by Rashad.
You are knowledgeable, precise, and helpful. You communicate clearly and concisely.

Core traits:
- You think step-by-step when solving complex problems.
- You are honest about what you know and don't know.
- When given context from documents or web search, you use it accurately and cite sources.
- You are conversational but professional.
- You can help with coding, analysis, writing, math, and general knowledge.

When provided with context from various sources (documents, web search results, etc.),
use the provided context to answer the user's question accurately.
If the context contains web search results, cite the sources.
If you don't know the answer even with the provided context, say so honestly."""

SUGGESTED_PROMPTS = [
    {"title": "Explain a concept", "prompt": "Explain how {topic} works in simple terms", "category": "learning"},
    {"title": "Write code", "prompt": "Write a {language} function that {description}", "category": "coding"},
    {"title": "Debug help", "prompt": "Help me debug this error: {error_message}", "category": "coding"},
    {"title": "Summarize text", "prompt": "Summarize the following text in bullet points: {text}", "category": "writing"},
    {"title": "Compare options", "prompt": "Compare the pros and cons of {option_a} vs {option_b}", "category": "analysis"},
    {"title": "Brainstorm ideas", "prompt": "Give me 5 creative ideas for {topic}", "category": "creative"},
    {"title": "Translate text", "prompt": "Translate the following to {language}: {text}", "category": "language"},
    {"title": "Review code", "prompt": "Review this code for bugs and improvements:\n```\n{code}\n```", "category": "coding"},
]


def read_context_from_folder(folder_path: str) -> str:
    combined_text = ""
    for filename in os.listdir(folder_path):
        if filename.endswith(".md"):
            with open(os.path.join(folder_path, filename), "r") as f:
                combined_text += f.read() + "\n"
    return combined_text


def get_available_models():
    try:
        provider = get_current_provider()
        return provider.list_models()
    except Exception as e:
        return {"error": str(e)}


def get_available_llm_models():
    try:
        provider = get_current_provider()
        return provider.get_chat_models()
    except Exception as e:
        return {"error": str(e)}


def get_suggested_prompts(category: str = None) -> list:
    if category:
        return [p for p in SUGGESTED_PROMPTS if p["category"] == category]
    return SUGGESTED_PROMPTS


# ---------------------------------------------------------------------------
# LLM Session
# ---------------------------------------------------------------------------


@dataclass
class LLMSession:
    model: str = ""
    extra_context: str = ""
    conversation_history: list = field(default_factory=list)

    def __post_init__(self):
        if not self.model:
            self.model = get_default_model() or ""

    def clear_history(self):
        self.conversation_history = []

    async def stream(self, prompt: str, think: bool = False):
        provider = get_current_provider()

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
        ]
        if self.extra_context:
            messages.append({"role": "system", "content": f"Context:\n{self.extra_context}"})
        messages.extend(self.conversation_history)
        messages.append({"role": "user", "content": prompt})

        assistant_reply = ""
        thinking_content = ""

        async for chunk in provider.chat_stream(
            model=self.model,
            messages=messages,
            think=think,
        ):
            if chunk.get("thinking"):
                thinking_content += chunk["thinking"]
                yield json.dumps({"type": "thinking", "content": chunk["thinking"]}) + "\n"

            if chunk.get("content"):
                assistant_reply += chunk["content"]
                yield json.dumps({"type": "content", "content": chunk["content"]}) + "\n"

        self.conversation_history.append({"role": "user", "content": prompt})
        self.conversation_history.append({"role": "assistant", "content": assistant_reply})

        yield json.dumps({"type": "done", "thinking": thinking_content if thinking_content else None}) + "\n"


SUMMARIZE_SYSTEM_PROMPT = """You are a meeting summarizer. Summarize the following meeting transcript concisely.
Extract key points, decisions, action items, and important discussions.
Format the summary with clear sections."""


async def summarize_text(text: str) -> str:
    model = get_default_model() or ""
    if not model:
        return "No LLM model available for summarization."

    provider = get_current_provider()
    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": f"Summarize this meeting transcript:\n\n{text}"},
    ]
    return await provider.chat(model=model, messages=messages)


# Global session for backwards compat (used by non-authed endpoints)
session = LLMSession(model=get_default_model() or "")
