"""Tests for embeddings API endpoint error handling."""

from typing import Any, cast

import pytest
from fastapi import HTTPException

from llm_shim.api import embeddings
from llm_shim.api.schemas.openai import EmbeddingsRequest
from llm_shim.services.embeddings import EmbeddingsService


class FakeFailingEmbeddingsService:
    """Service that raises ValueError."""

    async def create(self, request: EmbeddingsRequest) -> None:
        raise ValueError("Invalid input")


class FakeFailingEmbeddingsServiceRuntime:
    """Service that raises RuntimeError with initialization prefix."""

    async def create(self, request: EmbeddingsRequest) -> None:
        raise RuntimeError("Failed to initialize provider client: auth failed")


class FakeFailingEmbeddingsServiceRuntimeOther:
    """Service that raises RuntimeError without initialization prefix."""

    async def create(self, request: EmbeddingsRequest) -> None:
        raise RuntimeError("Service unavailable")


@pytest.mark.asyncio
async def test_embeddings_endpoint_returns_400_on_value_error(monkeypatch: Any) -> None:
    """Embeddings endpoint should return 400 for ValueError."""
    monkeypatch.setattr(
        embeddings,
        "EmbeddingsService",
        lambda: cast(EmbeddingsService, FakeFailingEmbeddingsService()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await embeddings.create_embeddings(EmbeddingsRequest(input="hello"))

    assert exc_info.value.status_code == 400
    assert "Invalid input" in exc_info.value.detail


@pytest.mark.asyncio
async def test_embeddings_endpoint_returns_500_on_initialization_runtime_error(
    monkeypatch: Any,
) -> None:
    """Embeddings endpoint should return 500 for RuntimeError with init message."""
    monkeypatch.setattr(
        embeddings,
        "EmbeddingsService",
        lambda: cast(EmbeddingsService, FakeFailingEmbeddingsServiceRuntime()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await embeddings.create_embeddings(EmbeddingsRequest(input="hello"))

    assert exc_info.value.status_code == 500
    assert "Failed to initialize provider client" in exc_info.value.detail


@pytest.mark.asyncio
async def test_embeddings_endpoint_returns_502_on_other_runtime_error(
    monkeypatch: Any,
) -> None:
    """Embeddings endpoint should return 502 for RuntimeError without init message."""
    monkeypatch.setattr(
        embeddings,
        "EmbeddingsService",
        lambda: cast(EmbeddingsService, FakeFailingEmbeddingsServiceRuntimeOther()),
    )

    with pytest.raises(HTTPException) as exc_info:
        await embeddings.create_embeddings(EmbeddingsRequest(input="hello"))

    assert exc_info.value.status_code == 502
    assert "Service unavailable" in exc_info.value.detail
