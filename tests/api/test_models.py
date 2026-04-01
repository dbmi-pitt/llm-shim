"""Tests for models list API endpoint."""

from typing import Any

import pytest

from llm_shim.api import models
from llm_shim.api.schemas.openai import ModelListItem, ModelListResponse
from llm_shim.core.config import ProviderSettings, ServerSettings, Settings
from llm_shim.services.models import ModelsService


@pytest.mark.asyncio
async def test_models_endpoint_lists_chat_and_embedding_models(
    monkeypatch: Any,
) -> None:
    settings = Settings.model_construct(
        providers={
            "openai": ProviderSettings(
                chat_models=["gpt-4o-mini", "gpt-4.1*"],
                embedding_models=["text-embedding-3-small"],
            ),
            "bedrock": ProviderSettings(
                chat_models=["haiku*"],
                embedding_models=["amazon.titan-embed-text-*"],
            ),
        },
        server=ServerSettings(),
    )

    class FakeModelsService(ModelsService):
        def list(self) -> ModelListResponse:
            return ModelListResponse(
                data=[
                    ModelListItem(
                        id=model,
                        created=123,
                        owned_by=provider_id,
                    )
                    for provider_id, model in [
                        *settings.list_chat_models(),
                        *settings.list_embedding_models(),
                    ]
                ]
            )

    monkeypatch.setattr(
        models, "get_models_service", lambda: FakeModelsService(settings=settings)
    )

    response = await models.list_models()

    assert response.object == "list"
    assert len(response.data) == 5
    ids = [entry.id for entry in response.data]
    owners = [entry.owned_by for entry in response.data]

    assert "gpt-4o-mini" in ids
    assert "gpt-4.1*" in ids
    assert "text-embedding-3-small" in ids
    assert "haiku*" in ids
    assert "amazon.titan-embed-text-*" in ids

    assert "openai" in owners
    assert "bedrock" in owners
    assert all(entry.object == "model" for entry in response.data)
