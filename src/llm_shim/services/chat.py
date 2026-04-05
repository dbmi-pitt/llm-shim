"""Service for OpenAI-compatible chat completions."""

import logging
import time
from functools import lru_cache
from typing import Any, cast
from uuid import uuid4

from pydantic_ai import Agent
from pydantic_ai.settings import ModelSettings
from pydantic_ai.usage import RunUsage

from llm_shim.api.schemas.openai import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseMessage,
    ChatCompletionUsage,
)
from llm_shim.core.config import Settings, get_settings
from llm_shim.core.exceptions import ProviderCallError
from llm_shim.core.utils import environment_override_lock, patched_environ

logger = logging.getLogger(__name__)


class ChatService:
    """Service for creating OpenAI-compatible chat completion responses."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service dependencies."""
        self._settings = settings or get_settings()

    @staticmethod
    def _build_model_settings(
        provider_settings: dict[str, Any],
        request_kwargs: dict[str, Any],
    ) -> ModelSettings | None:
        """Merge provider defaults with request-level chat settings."""
        allowed = {
            "max_tokens",
            "temperature",
            "top_p",
            "timeout",
            "parallel_tool_calls",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logit_bias",
            "stop_sequences",
            "extra_headers",
            "extra_body",
            "thinking",
        }
        merged: dict[str, Any] = {
            key: value for key, value in provider_settings.items() if key in allowed
        }
        for key in ("max_tokens", "temperature", "top_p"):
            value = request_kwargs.get(key)
            if value is not None:
                merged[key] = value
        return cast(ModelSettings, merged) if merged else None

    @staticmethod
    def _build_response(
        configured_model: str,
        content: str,
        usage: RunUsage,
    ) -> ChatCompletionResponse:
        """Build an OpenAI-compatible chat completion response."""
        return ChatCompletionResponse(
            id=f"chatcmpl-{uuid4().hex}",
            created=int(time.time()),
            model=configured_model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatCompletionResponseMessage(content=content),
                    finish_reason="stop",
                ),
            ],
            usage=ChatCompletionUsage(
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                total_tokens=usage.input_tokens + usage.output_tokens,
            ),
        )

    @staticmethod
    def _messages_to_prompt(request: ChatCompletionRequest) -> str:
        """Flatten OpenAI-style messages into a single prompt."""
        lines = [f"{message.role}: {message.content}" for message in request.messages]
        return "\n".join(lines)

    async def _run_text_model(
        self,
        model_name: str,
        prompt: str,
        model_settings: ModelSettings | None,
    ) -> tuple[str, RunUsage]:
        """Execute plain text generation with pydantic-ai."""
        result = await Agent(
            model_name,
            output_type=str,
            model_settings=model_settings,
        ).run(prompt)
        return str(result.output), result.usage()

    async def create(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        """Create a provider chat completion normalized to OpenAI response format.

        Args:
            request (ChatCompletionRequest): OpenAI-compatible request.

        Returns:
            ChatCompletionResponse: OpenAI-compatible chat completion response.
        """
        provider_id, resolved_model, provider = self._settings.resolve_chat_provider(
            request.model
        )
        configured_model = f"{provider_id}:{resolved_model}"
        prompt = self._messages_to_prompt(request)
        create_kwargs = request.chat_kwargs()
        model_settings = self._build_model_settings(
            provider_settings=provider.chat_model_settings,
            request_kwargs=create_kwargs,
        )

        try:
            async with environment_override_lock:
                with patched_environ(provider.env):
                    response_text, usage = await self._run_text_model(
                        model_name=configured_model,
                        prompt=prompt,
                        model_settings=model_settings,
                    )
        except Exception as error:
            logger.exception("Chat completion failed for %s", configured_model)
            raise ProviderCallError(
                f"Provider chat completion failed: {error}"
            ) from error

        logger.info(
            "Chat completion for %s: %d input, %d output tokens",
            configured_model,
            usage.input_tokens,
            usage.output_tokens,
        )
        return self._build_response(
            configured_model=configured_model,
            content=response_text,
            usage=usage,
        )


@lru_cache(maxsize=1)
def get_chat_service() -> ChatService:
    """Singleton factory for ChatService.

    Returns:
        ChatService: Cached service instance.
    """
    return ChatService()
