from typing import AsyncIterator

from ollama import AsyncClient
from ollama import list as list_models

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    def __init__(self, config: dict | None = None):
        self._config = config or {}
        base_url = self._config.get("base_url", "")
        self._client = AsyncClient(host=base_url) if base_url else AsyncClient()

    @property
    def name(self) -> str:
        return "ollama"

    async def chat_stream(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        params = dict(model=model, messages=messages, stream=True)
        if tools:
            params["tools"] = tools

        think = kwargs.get("think", False)
        if think:
            params["think"] = True

        async for part in await self._client.chat(**params):
            msg = part.get("message", {})
            chunk = {}
            if msg.get("thinking"):
                chunk["thinking"] = msg["thinking"]
            if msg.get("content"):
                chunk["content"] = msg["content"]
            if msg.get("tool_calls"):
                chunk["tool_calls"] = msg["tool_calls"]
            if chunk:
                yield chunk

    async def chat(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        **kwargs,
    ) -> str:
        params = dict(model=model, messages=messages, stream=False)
        if tools:
            params["tools"] = tools
        response = await self._client.chat(**params)
        return response["message"]["content"]

    def list_models(self) -> list[str]:
        try:
            response = list_models()
            return [m.model for m in response.models]
        except Exception:
            return []

    def get_chat_models(self) -> list[str]:
        """Filter out embedding-only models by name heuristics."""
        embedding_keywords = ["embed", "nomic-embed-text", "all-minilm", "mxbai-embed"]
        all_models = self.list_models()
        return [
            m for m in all_models
            if not any(kw in m.lower() for kw in embedding_keywords)
        ]
