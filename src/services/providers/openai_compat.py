from typing import AsyncIterator

from openai import AsyncOpenAI

from .base import LLMProvider


class OpenAICompatibleProvider(LLMProvider):
    def __init__(self, config: dict | None = None):
        self._config = config or {}
        api_key = self._config.get("api_key", "")
        base_url = self._config.get("base_url", "https://api.openai.com/v1").rstrip("/")
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @property
    def name(self) -> str:
        return "openai_compat"

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

        stream = await self._client.chat.completions.create(**params)

        tool_calls_acc: dict[int, dict] = {}

        async for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta is None:
                continue

            chunk_out = {}

            if delta.content:
                chunk_out["content"] = delta.content

            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": "",
                            "type": "function",
                            "function": {"name": "", "arguments": ""},
                        }
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["function"]["name"] += tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["function"]["arguments"] += tc.function.arguments

            if chunk_out:
                yield chunk_out

        if tool_calls_acc:
            yield {
                "tool_calls": [
                    tool_calls_acc[i] for i in sorted(tool_calls_acc)
                ]
            }

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
        response = await self._client.chat.completions.create(**params)
        return response.choices[0].message.content or ""

    def list_models(self) -> list[str]:
        try:
            response = self._client.models.list()
            return sorted(m.id for m in response)
        except Exception:
            return []
