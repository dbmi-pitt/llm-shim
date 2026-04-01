"""Service for OpenAI-compatible models listing responses."""

import time
from functools import lru_cache

from llm_shim.api.schemas.openai import ModelListItem, ModelListResponse
from llm_shim.core.config import Settings, get_settings


class ModelsService:
    """Service for listing configured model routes."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize service dependencies."""
        self._settings = settings or get_settings()

    def list(self) -> ModelListResponse:
        """List configured chat and embedding model routes in OpenAI format."""
        created = int(time.time())

        data: list[ModelListItem] = []
        for provider_id, model in self._settings.list_chat_models():
            data.append(ModelListItem(id=model, created=created, owned_by=provider_id))
        for provider_id, model in self._settings.list_embedding_models():
            data.append(ModelListItem(id=model, created=created, owned_by=provider_id))

        return ModelListResponse(data=data)


@lru_cache(maxsize=1)
def get_models_service() -> ModelsService:
    """Singleton factory for ModelsService."""
    return ModelsService()
