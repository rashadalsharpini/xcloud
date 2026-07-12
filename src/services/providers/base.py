from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMProvider(ABC):
    """Abstract interface for LLM providers."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique provider key (ollama, openai, gemini, etc.)."""
        ...

    @abstractmethod
    async def chat_stream(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """
        Stream a chat response.

        Yields dicts with optional keys:
          - content: str       – text fragment
          - thinking: str      – reasoning / chain-of-thought
          - tool_calls: list   – function-call objects (on the final relevant chunk)
        """
        ...

    @abstractmethod
    async def chat(
        self,
        model: str,
        messages: list,
        tools: list | None = None,
        **kwargs,
    ) -> str:
        """Non-streaming chat.  Returns the full response text."""
        ...

    @abstractmethod
    def list_models(self) -> list[str]:
        """Return all model names visible to this provider."""
        ...

    def get_chat_models(self) -> list[str]:
        """Return only chat-capable models (override if provider distinguishes)."""
        return self.list_models()

    def is_model_supported(self, model: str) -> bool:
        """Whether this provider can serve `model` (override for providers
        whose `list_models()` is curated rather than exhaustive)."""
        return model in self.list_models()
