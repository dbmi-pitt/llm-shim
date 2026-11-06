"""Configuration models and helpers."""

import fnmatch
import ipaddress
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

from llm_shim.core.exceptions import BadRequestError, ProviderConfigError

__all__ = [
    "ProviderSettings",
    "SecuritySettings",
    "ServerSettings",
    "Settings",
    "TeiProviderSettings",
    "get_data_dir",
    "get_settings",
]


def get_data_dir() -> Path:
    """Return the path to the data from the environment or default location.

    Returns:
        Path: Path to the data directory where config.yaml should exist.
    """
    data_dir = os.getenv("LLM_SHIM_DATA_DIR", "data/")
    return Path(data_dir)


def _is_loopback(host: str) -> bool:
    """Return whether a bind host is unambiguously local."""
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        # DNS names can resolve differently after startup, so fail closed.
        return False


class TeiProviderSettings(BaseModel):
    """Configuration for a HuggingFace TEI embedding backend."""

    base_url: str
    endpoint: str = "/embed"
    auth_token_env: str | None = None
    request_timeout_seconds: float = 30.0
    input_prefix_template: str | None = None


class ProviderSettings(BaseModel):
    """Configuration for a single provider routing entry."""

    chat_models: str | list[str] = Field(default_factory=list)
    embedding_models: str | list[str] = Field(default_factory=list)
    backend: Literal["pydantic_ai", "tei"] = "pydantic_ai"
    environment: dict[str, str] = Field(default_factory=dict)
    env: dict[str, str] = Field(default_factory=dict)
    chat_model_settings: dict[str, Any] = Field(default_factory=dict)
    embedding_model_settings: dict[str, Any] = Field(default_factory=dict)
    tei: TeiProviderSettings | None = None

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
        if self.backend == "tei":
            if self.chat_models:
                raise ValueError("TEI providers only support embedding_models")
            if self.tei is None:
                raise ValueError("TEI providers must define tei settings")
        if self.environment.keys() & self.env.keys():
            raise ValueError("environment and env may not set the same variable")
        return self

    def environment_overrides(self) -> dict[str, str]:
        """Resolve provider credential references without putting values in YAML.

        Each ``env`` value is the name of the environment variable holding the
        credential; its key is the variable Pydantic AI expects.
        """
        overrides = dict(self.environment)
        for target, source in self.env.items():
            value = os.getenv(source)
            if not value:
                raise ProviderConfigError(
                    "A required provider credential is unavailable"
                )
            overrides[target] = value
        return overrides


class ServerSettings(BaseModel):
    """Runtime server configuration for uvicorn."""

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    reload: bool = False
    workers: int = 1
    log_level: str = "info"


class SecuritySettings(BaseModel):
    """Inbound access and resource controls; key material stays outside YAML."""

    expose_network: bool = False
    allow_anonymous_loopback: bool = False
    api_keys_env: str = "LLM_SHIM_API_KEYS"
    api_keys_file: Path | None = None
    docs_enabled: bool = False
    request_body_max_bytes: int = Field(default=1_048_576, ge=1024, le=16_777_216)
    max_messages: int = Field(default=64, ge=1, le=1024)
    max_message_chars: int = Field(default=32_768, ge=1, le=1_000_000)
    max_embedding_inputs: int = Field(default=128, ge=1, le=10_000)
    max_embedding_chars: int = Field(default=32_768, ge=1, le=1_000_000)
    max_total_input_chars: int = Field(default=131_072, ge=1, le=4_000_000)
    default_max_tokens: int = Field(default=1024, ge=1, le=100_000)
    max_tokens: int = Field(default=4096, ge=1, le=100_000)
    max_concurrent_requests: int = Field(default=8, ge=1, le=1000)
    queue_timeout_seconds: float = Field(default=0.1, gt=0, le=60)
    request_timeout_seconds: float = Field(default=120, gt=0, le=600)
    rate_limit_requests: int = Field(default=60, ge=1, le=100_000)
    rate_limit_window_seconds: float = Field(default=60, gt=0, le=3600)

    @model_validator(mode="after")
    def validate_token_limits(self) -> SecuritySettings:
        """Ensure the default output budget is within the hard limit."""
        if self.default_max_tokens > self.max_tokens:
            raise ValueError("default_max_tokens cannot exceed max_tokens")
        return self

    def client_keys(self) -> dict[str, str]:
        """Load ``name:key`` (or bare) keys from environment and secret file."""
        entries: list[str] = []
        if value := os.getenv(self.api_keys_env):
            entries.extend(value.replace(",", "\n").splitlines())
        if self.api_keys_file is not None:
            try:
                entries.extend(self.api_keys_file.read_text().splitlines())
            except OSError as error:
                raise ValueError("Unable to read configured API key file") from error
        keys: dict[str, str] = {}
        for number, raw in enumerate(entries, start=1):
            entry = raw.strip()
            if not entry or entry.startswith("#"):
                continue
            name, separator, secret = entry.partition(":")
            if not separator:
                name, secret = f"key-{number}", name
            if not name.strip() or not secret.strip():
                raise ValueError("API key entries must be non-empty")
            keys[name.strip()] = secret.strip()
        return keys


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(extra="ignore")

    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    server: ServerSettings = Field(default_factory=ServerSettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)

    def resolve_provider(
        self,
        requested_model: str | None,
        mode: Literal["chat", "embedding"],
    ) -> tuple[str, str, ProviderSettings]:
        """Resolve provider id and concrete model from mode-specific patterns."""
        field_name = "chat_models" if mode == "chat" else "embedding_models"

        if requested_model is None:
            raise BadRequestError(
                "Request model is required and must use provider:model format"
            )

        provider_id, sep, model_name = requested_model.partition(":")
        if not sep or not provider_id or not model_name:
            raise BadRequestError("Request model must use provider:model format")

        provider = self.providers.get(provider_id)
        if provider is None:
            raise BadRequestError("Requested provider is not configured")

        patterns = getattr(provider, field_name)
        if any(fnmatch.fnmatchcase(model_name, pattern) for pattern in patterns):
            return provider_id, model_name, provider

        raise BadRequestError("Requested model is not configured for this provider")

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
        """Validate providers and fail closed before a socket is opened."""
        if not self.providers:
            raise ValueError("At least one provider entry must be configured")
        is_local = _is_loopback(self.server.host)
        keys = self.security.client_keys()
        if not is_local:
            if not self.security.expose_network:
                raise ValueError(
                    "Non-loopback binding requires security.expose_network=true"
                )
            if not keys:
                raise ValueError(
                    "Non-loopback binding requires configured inbound API keys"
                )
            if self.server.reload:
                raise ValueError("reload must be disabled for non-loopback binding")
            if self.security.docs_enabled:
                raise ValueError("docs must be disabled for non-loopback binding")
        elif not keys and not self.security.allow_anonymous_loopback:
            raise ValueError(
                "Configure inbound API keys or explicitly enable "
                "allow_anonymous_loopback"
            )
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
            EnvSettingsSource(
                settings_cls,
                env_prefix="LLM_SHIM_",
                env_nested_delimiter="__",
                env_parse_none_str="null",
            ),
            YamlConfigSettingsSource(
                settings_cls, yaml_file=get_data_dir() / "config.yaml"
            ),
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Singleton accessor for cached settings instance.

    Returns:
        Settings: Cached settings instance.
    """
    return Settings()
