import asyncio
import re
from typing import AsyncIterator

from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types

from .base import LLMProvider

# Transient upstream failures worth retrying: overloaded (503) and
# rate-limited (429). Non-transient errors (400/401/404) fail immediately.
_RETRYABLE_CODES = {429, 503}
_MAX_ATTEMPTS = 3
_BACKOFF_SECONDS = (2, 5)


def _is_retryable(exc: Exception) -> bool:
    return (
        isinstance(exc, genai_errors.APIError)
        and getattr(exc, "code", None) in _RETRYABLE_CODES
    )

# Only expose stable, chat-usable Gemini models to the UI. Google's list API
# returns dozens of variants (previews, experiments, dated snapshots, aqa,
# imagen, deep-research, ...) which are confusing and often not usable for
# plain generateContent chat.
_CHAT_MODEL_RE = re.compile(r"^gemini-\d+(\.\d+)?-(pro|flash)(-lite)?$")


class GeminiProvider(LLMProvider):
    def __init__(self, config: dict | None = None):
        self._config = config or {}
        api_key = self._config.get("api_key", "")
        self._client = genai.Client(api_key=api_key) if api_key else genai.Client()

    @property
    def name(self) -> str:
        return "gemini"

    def _to_gemini_messages(self, messages: list) -> list:
        """Convert OpenAI-format messages to Gemini content list."""
        gemini_msgs = []
        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                gemini_msgs.append(
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=f"[System instruction]: {content}")],
                    )
                )
                gemini_msgs.append(
                    genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text="Understood, I will follow these instructions.")],
                    )
                )
            elif role == "user":
                gemini_msgs.append(
                    genai_types.Content(
                        role="user",
                        parts=[genai_types.Part(text=content)],
                    )
                )
            elif role == "assistant":
                gemini_msgs.append(
                    genai_types.Content(
                        role="model",
                        parts=[genai_types.Part(text=content)],
                    )
                )
            elif role == "tool":
                gemini_msgs.append(
                    genai_types.Content(
                        role="user",
                        parts=[
                            genai_types.Part(
                                function_response=genai_types.FunctionResponse(
                                    name=msg.get("name", ""),
                                    response={"result": content},
                                )
                            )
                        ],
                    )
                )
        return gemini_msgs

    def _to_gemini_tools(self, tools: list) -> list[genai_types.Tool] | None:
        if not tools:
            return None

        declarations = []
        for t in tools:
            func = t.get("function", {})
            declarations.append(
                genai_types.FunctionDeclaration(
                    name=func.get("name", ""),
                    description=func.get("description", ""),
                    parameters=func.get("parameters"),
                )
            )

        return [genai_types.Tool(function_declarations=declarations)]

    async def chat_stream(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        gemini_messages = self._to_gemini_messages(messages)
        gemini_tools = self._to_gemini_tools(tools)

        config = genai_types.GenerateContentConfig(tools=gemini_tools) if gemini_tools else None

        # Retry transient upstream failures (503 overloaded / 429 rate limit),
        # but only before any content has been yielded — never mid-stream.
        response = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.models.generate_content_stream(
                    model=model,
                    contents=gemini_messages,
                    config=config,
                )
                break
            except Exception as e:
                if attempt < _MAX_ATTEMPTS - 1 and _is_retryable(e):
                    await asyncio.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                raise

        chunk_iter = iter(response)
        first_chunk = None
        for attempt in range(_MAX_ATTEMPTS):
            try:
                first_chunk = next(chunk_iter, None)
                break
            except Exception as e:
                if attempt < _MAX_ATTEMPTS - 1 and _is_retryable(e):
                    await asyncio.sleep(_BACKOFF_SECONDS[attempt])
                    response = self._client.models.generate_content_stream(
                        model=model,
                        contents=gemini_messages,
                        config=config,
                    )
                    chunk_iter = iter(response)
                    continue
                raise

        def chunks():
            if first_chunk is not None:
                yield first_chunk
            yield from chunk_iter

        for chunk in chunks():
            chunk_out = {}
            if chunk.text:
                chunk_out["content"] = chunk.text

            if chunk.candidates:
                candidate = chunk.candidates[0]
                if candidate.content and candidate.content.parts:
                    for part in candidate.content.parts:
                        if part.function_call:
                            fn_call = part.function_call
                            chunk_out["tool_calls"] = [
                                {
                                    "id": fn_call.name,
                                    "type": "function",
                                    "function": {
                                        "name": fn_call.name,
                                        "arguments": str(fn_call.args) if fn_call.args else "{}",
                                    },
                                }
                            ]

            if chunk_out:
                yield chunk_out

    async def chat(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        **kwargs,
    ) -> str:
        gemini_messages = self._to_gemini_messages(messages)
        gemini_tools = self._to_gemini_tools(tools)

        config = genai_types.GenerateContentConfig(tools=gemini_tools) if gemini_tools else None

        for attempt in range(_MAX_ATTEMPTS):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=gemini_messages,
                    config=config,
                )
                return response.text or ""
            except Exception as e:
                if attempt < _MAX_ATTEMPTS - 1 and _is_retryable(e):
                    await asyncio.sleep(_BACKOFF_SECONDS[attempt])
                    continue
                raise
        return ""

    def list_models(self) -> list[str]:
        """Return the curated chat model list (what the UI should show)."""
        return self.get_chat_models()

    def get_chat_models(self) -> list[str]:
        """Only stable `gemini-X-pro/flash(-lite)` models, newest first."""
        try:
            models = self._client.models.list()
        except Exception:
            return []

        names = {
            m.name.replace("models/", "")
            for m in models
            if "generateContent" in (m.supported_actions or [])
        }
        curated = [n for n in names if _CHAT_MODEL_RE.match(n)]

        def sort_key(name: str) -> tuple:
            version = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
            v = float(version.group(1)) if version else 0.0
            tier = 0 if "-pro" in name else 1 if "-lite" not in name else 2
            return (-v, tier)

        return sorted(curated, key=sort_key)

    def is_model_supported(self, model: str) -> bool:
        """Accept any generateContent-capable model, not just curated ones,
        so power users can pin e.g. a preview model as their default."""
        try:
            models = self._client.models.list()
            return any(
                m.name.replace("models/", "") == model
                and "generateContent" in (m.supported_actions or [])
                for m in models
            )
        except Exception:
            return False
