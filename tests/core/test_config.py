"""Tests for core configuration module."""

import os
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from llm_shim.core.config import (
    ProviderSettings,
    ServerSettings,
    Settings,
    get_data_dir,
    get_settings,
)


def test_get_data_dir_default() -> None:
    with mock.patch.dict(os.environ, {}, clear=False):
        if "LLM_SHIM_DATA_DIR" in os.environ:
            del os.environ["LLM_SHIM_DATA_DIR"]
        assert get_data_dir() == Path("data/")


def test_get_data_dir_from_env() -> None:
    with mock.patch.dict(os.environ, {"LLM_SHIM_DATA_DIR": "/custom/path"}):
        assert get_data_dir() == Path("/custom/path")


def test_provider_settings_accepts_string_models_and_normalizes() -> None:
    provider = ProviderSettings(
        chat_models="gpt-4o-mini",
        embedding_models="text-embedding-3-small",
        env={"OPENAI_API_KEY": "sk-test", "google_api_key": "abc"},
    )
    assert provider.chat_models == ["gpt-4o-mini"]
    assert provider.embedding_models == ["text-embedding-3-small"]
    assert provider.env["OPENAI_API_KEY"] == "sk-test"
    assert provider.env["google_api_key"] == "abc"


def test_provider_settings_requires_non_empty_models() -> None:
    with pytest.raises(ValidationError):
        ProviderSettings(chat_models=[], embedding_models=[])


def test_server_settings_defaults() -> None:
    settings = ServerSettings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.reload is False
    assert settings.workers == 1
    assert settings.log_level == "info"


def test_settings_resolve_provider_matches_exact_and_wildcard() -> None:
    settings = Settings.model_construct(
        providers={
            "bedrock": ProviderSettings(chat_models=["haiku*", "sonnet*"]),
            "openai": ProviderSettings(chat_models=["gpt-4o-mini"]),
        },
        server=ServerSettings(),
    )

    provider_id, model_name, _ = settings.resolve_chat_provider("haiku-3-5")
    assert provider_id == "bedrock"
    assert model_name == "haiku-3-5"

    provider_id, model_name, _ = settings.resolve_chat_provider("gpt-4o-mini")
    assert provider_id == "openai"
    assert model_name == "gpt-4o-mini"


def test_settings_resolve_embedding_provider_matches_patterns() -> None:
    settings = Settings.model_construct(
        providers={
            "bedrock": ProviderSettings(embedding_models=["amazon.titan-embed-*"]),
            "openai": ProviderSettings(embedding_models=["text-embedding-3-small"]),
        },
        server=ServerSettings(),
    )

    provider_id, model_name, _ = settings.resolve_embedding_provider(
        "amazon.titan-embed-text-v2:0"
    )
    assert provider_id == "bedrock"
    assert model_name == "amazon.titan-embed-text-v2:0"


def test_settings_resolve_provider_not_found() -> None:
    settings = Settings.model_construct(
        providers={"openai": ProviderSettings(chat_models=["gpt-4o-mini"])},
        server=ServerSettings(),
    )

    with pytest.raises(ValueError):
        settings.resolve_chat_provider("anthropic")


def test_settings_requires_provider_entries(tmp_path: Path) -> None:
    with mock.patch.dict(os.environ, {"LLM_SHIM_DATA_DIR": str(tmp_path)}):
        with pytest.raises(ValidationError):
            Settings(providers={}, server=ServerSettings())


def test_settings_resolve_provider_defaults_to_first_exact_pattern_when_model_missing() -> (
    None
):
    settings = Settings.model_construct(
        providers={
            "openai": ProviderSettings(
                chat_models=["gpt-4o-mini", "gpt-4.1-mini"],
                embedding_models=["text-embedding-3-small"],
            ),
            "bedrock": ProviderSettings(chat_models=["haiku*"]),
        },
        server=ServerSettings(),
    )

    provider_id, model_name, _ = settings.resolve_chat_provider(None)
    assert provider_id == "openai"
    assert model_name == "gpt-4o-mini"


def test_settings_requires_model_if_only_wildcard_patterns_exist() -> None:
    settings = Settings.model_construct(
        providers={"openai": ProviderSettings(chat_models=["*"])},
        server=ServerSettings(),
    )

    with pytest.raises(ValueError, match="Request model is required"):
        settings.resolve_chat_provider(None)


def test_settings_requires_providers_key(tmp_path: Path) -> None:
    with mock.patch.dict(os.environ, {"LLM_SHIM_DATA_DIR": str(tmp_path)}):
        with pytest.raises(ValidationError, match="must define providers"):
            Settings(
                **{
                    "openai": {
                        "chat_models": ["gpt-*"],
                        "env": {"OPENAI_API_KEY": "sk-test"},
                    },
                    "server": {"port": 9000},
                }
            )


def test_settings_accepts_nested_providers_key() -> None:
    validated = Settings(
        providers={
            "openai": {
                "chat_models": ["gpt-*"],
                "embedding_models": ["text-embedding-3-*"],
                "env": {"OPENAI_API_KEY": "sk-test"},
            }
        },
        server={"port": 9000},
    )

    assert "openai" in validated.providers
    assert validated.providers["openai"].chat_models == ["gpt-*"]
    assert validated.server.port == 9000


def test_settings_rejects_legacy_schema() -> None:
    with pytest.raises(ValidationError, match="no longer supported"):
        Settings(
            **{
                "global_config": {"provider": "openai"},
                "profiles": {"default": {"provider": "openai"}},
            }
        )


def test_get_settings_returns_cached_instance() -> None:
    get_settings.cache_clear()
    assert get_settings() is get_settings()
