"""Tests for chat API endpoint error handling."""

from types import SimpleNamespace
from typing import Any, cast

import pytest

from llm_shim.api.chat import create_chat_completion
from llm_shim.api.schemas.openai import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
)
from llm_shim.core.config import SecuritySettings
from llm_shim.core.exceptions import BadRequestError, ProviderCallError
from llm_shim.services.chat import ChatService


class FakeFailingChatServiceBadRequest(ChatService):
    """Service that raises BadRequestError."""

    async def create(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise BadRequestError("Invalid prompt")


class FakeFailingChatServiceProvider(ChatService):
    """Service that raises ProviderCallError."""

    async def create(self, request: ChatCompletionRequest) -> ChatCompletionResponse:
        raise ProviderCallError("Provider chat completion failed: connection refused")


@pytest.fixture
def chat_request() -> ChatCompletionRequest:
    return ChatCompletionRequest(
        messages=[ChatMessage(role="user", content="hello")],
    )


@pytest.mark.asyncio
async def test_chat_endpoint_returns_400_on_bad_request(
    chat_request: ChatCompletionRequest,
) -> None:
    """Chat endpoint should return 400 for BadRequestError."""
    with pytest.raises(BadRequestError, match="Invalid prompt"):
        await create_chat_completion(
            chat_request,
            http_request=cast(
                Any,
                SimpleNamespace(
                    app=SimpleNamespace(
                        state=SimpleNamespace(
                            settings=SimpleNamespace(security=SecuritySettings())
                        )
                    )
                ),
            ),
            service=cast(ChatService, FakeFailingChatServiceBadRequest()),
        )


@pytest.mark.asyncio
async def test_chat_endpoint_returns_502_on_provider_error(
    chat_request: ChatCompletionRequest,
) -> None:
    """Chat endpoint should return 502 for ProviderCallError."""
    with pytest.raises(ProviderCallError, match="Provider chat completion failed"):
        await create_chat_completion(
            chat_request,
            http_request=cast(
                Any,
                SimpleNamespace(
                    app=SimpleNamespace(
                        state=SimpleNamespace(
                            settings=SimpleNamespace(security=SecuritySettings())
                        )
                    )
                ),
            ),
            service=cast(ChatService, FakeFailingChatServiceProvider()),
        )
