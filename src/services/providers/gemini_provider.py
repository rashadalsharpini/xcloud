from typing import AsyncIterator

from google import genai
from google.genai import types as genai_types

from .base import LLMProvider


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

        response = self._client.models.generate_content_stream(
            model=model,
            contents=gemini_messages,
            config=config,
        )

        for chunk in response:
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

        response = self._client.models.generate_content(
            model=model,
            contents=gemini_messages,
            config=config,
        )
        return response.text or ""

    def list_models(self) -> list[str]:
        try:
            models = self._client.models.list()
            return sorted(m.name.replace("models/", "") for m in models)
        except Exception:
            return []

    def get_chat_models(self) -> list[str]:
        """Gemini lists all models; filter to chat-capable ones."""
        all_models = self.list_models()
        return [m for m in all_models if "embedding" not in m.lower()]
