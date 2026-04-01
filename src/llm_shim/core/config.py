"""Configuration models and helpers."""

import fnmatch
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import (
    BaseSettings,
    EnvSettingsSource,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

__all__ = ["ProviderSettings", "Settings", "get_data_dir", "get_settings"]


def get_data_dir() -> Path:
    """Return the path to the data from the environment or default location.

    Returns:
        Path: Path to the data directory where config.yaml should exist.
    """
    data_dir = os.getenv("LLM_SHIM_DATA_DIR", "data/")
    return Path(data_dir)


class ProviderSettings(BaseModel):
    """Configuration for a single provider routing entry."""

    chat_models: str | list[str] = Field(default_factory=list)
    embedding_models: str | list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    chat_model_settings: dict[str, Any] = Field(default_factory=dict)
    embedding_model_settings: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_models(self) -> ProviderSettings:
        """Normalize model patterns and require at least one route."""

        def normalize(value: str | list[str]) -> list[str]:
            items = [value] if isinstance(value, str) else value
            return [item for item in items if item]

        self.chat_models = normalize(self.chat_models)
        self.embedding_models = normalize(self.embedding_models)
        if not self.chat_models and not self.embedding_models:
            raise ValueError(
                "Provider must define at least one of chat_models or embedding_models"
            )
        return self


class ServerSettings(BaseModel):
    """Runtime server configuration for uvicorn."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    workers: int = 1
    log_level: str = "info"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(extra="ignore")

    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    server: ServerSettings = Field(default_factory=ServerSettings)

    def resolve_provider(
        self,
        requested_model: str | None,
        mode: Literal["chat", "embedding"],
    ) -> tuple[str, str, ProviderSettings]:
        """Resolve provider id and concrete model from mode-specific patterns."""
        field_name = "chat_models" if mode == "chat" else "embedding_models"

        if requested_model is not None:
            for provider_id, provider in self.providers.items():
                patterns = getattr(provider, field_name)
                if any(
                    fnmatch.fnmatchcase(requested_model, pattern)
                    for pattern in patterns
                ):
                    return provider_id, requested_model, provider

            raise ValueError(
                f"Requested model is not configured for {field_name}. "
                f"Add a matching provider {field_name} pattern."
            )

        for provider_id, provider in self.providers.items():
            patterns = getattr(provider, field_name)
            for pattern in patterns:
                if not any(token in pattern for token in ("*", "?", "[")):
                    return provider_id, pattern, provider

        raise ValueError(
            f"Request model is required when provider {field_name} use wildcard "
            "patterns"
        )

    def resolve_chat_provider(
        self, requested_model: str | None
    ) -> tuple[str, str, ProviderSettings]:
        """Resolve provider for chat model routing."""
        return self.resolve_provider(requested_model=requested_model, mode="chat")

    def resolve_embedding_provider(
        self, requested_model: str | None
    ) -> tuple[str, str, ProviderSettings]:
        """Resolve provider for embedding model routing."""
        return self.resolve_provider(requested_model=requested_model, mode="embedding")

    def list_chat_models(self) -> list[tuple[str, str]]:
        """Return configured chat model patterns as (provider_id, model)."""
        models: list[tuple[str, str]] = []
        for provider_id, provider in self.providers.items():
            models.extend((provider_id, model) for model in provider.chat_models)
        return models

    def list_embedding_models(self) -> list[tuple[str, str]]:
        """Return configured embedding model patterns as (provider_id, model)."""
        models: list[tuple[str, str]] = []
        for provider_id, provider in self.providers.items():
            models.extend((provider_id, model) for model in provider.embedding_models)
        return models

    @model_validator(mode="before")
    @classmethod
    def parse_provider_entries(cls, data: Any) -> Any:
        """Enforce provider nesting and reject removed schemas."""
        if not isinstance(data, dict):
            return data

        if "global_config" in data or "profiles" in data:
            raise ValueError("global_config/profiles schema is no longer supported")

        if "providers" not in data:
            raise ValueError("Configuration must define providers as a top-level key")

        return data

    @model_validator(mode="after")
    def validate_profiles(self) -> Settings:
        """Validate configured providers."""
        if not self.providers:
            raise ValueError("At least one provider entry must be configured")

        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Customize the order of configuration sources."""
        return (
            init_settings,
            YamlConfigSettingsSource(
                settings_cls, yaml_file=get_data_dir() / "config.yaml"
            ),
            EnvSettingsSource(
                settings_cls,
                env_prefix="LLM_SHIM_",
                env_nested_delimiter="__",
                env_parse_none_str="null",
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for cached settings instance.

    Returns:
        Settings: Cached settings instance.
    """
    return Settings()
