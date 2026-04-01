"""Tests for chat API endpoint error handling."""

from typing import Any, cast

import pytest
from fastapi import HTTPException

from llm_shim.api import chat
from llm_shim.api.schemas.openai import ChatCompletionRequest, ChatMessage
from llm_shim.services.chat import ChatService


class FakeFailingChatService:
    """Service that raises ValueError."""

    async def create(self, request: ChatCompletionRequest) -> None:
        raise ValueError("Invalid prompt")


class FakeFailingChatServiceRuntime:
    """Service that raises RuntimeError with initialization prefix."""

    async def create(self, request: ChatCompletionRequest) -> None:
        raise RuntimeError("Failed to initialize provider client: connection refused")


class FakeFailingChatServiceRuntimeOther:
    """Service that raises RuntimeError without initialization prefix."""

    async def create(self, request: ChatCompletionRequest) -> None:
        raise RuntimeError("Model not found")


@pytest.mark.asyncio
async def test_chat_endpoint_returns_400_on_value_error(monkeypatch: Any) -> None:
    """Chat endpoint should return 400 for ValueError."""
    monkeypatch.setattr(
        chat, "ChatService", lambda: cast(ChatService, FakeFailingChatService())
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat.create_chat_completion(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hello")],
            )
        )

    assert exc_info.value.status_code == 400
    assert "Invalid prompt" in exc_info.value.detail


@pytest.mark.asyncio
async def test_chat_endpoint_returns_500_on_initialization_runtime_error(
    monkeypatch: Any,
) -> None:
    """Chat endpoint should return 500 for RuntimeError with init message."""
    monkeypatch.setattr(
        chat,
        "ChatService",
        lambda: cast(ChatService, FakeFailingChatServiceRuntime()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat.create_chat_completion(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hello")],
            )
        )

    assert exc_info.value.status_code == 500
    assert "Failed to initialize provider client" in exc_info.value.detail


@pytest.mark.asyncio
async def test_chat_endpoint_returns_502_on_other_runtime_error(
    monkeypatch: Any,
) -> None:
    """Chat endpoint should return 502 for RuntimeError without init message."""
    monkeypatch.setattr(
        chat,
        "ChatService",
        lambda: cast(ChatService, FakeFailingChatServiceRuntimeOther()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await chat.create_chat_completion(
            ChatCompletionRequest(
                messages=[ChatMessage(role="user", content="hello")],
            )
        )

    assert exc_info.value.status_code == 502
    assert "Model not found" in exc_info.value.detail
