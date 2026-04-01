"""Tests for core configuration module."""

import os
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError

from llm_shim.core.config import (
    InstructorSettings,
    ServerSettings,
    Settings,
    get_data_dir,
    get_settings,
)


def test_get_data_dir_default() -> None:
    """get_data_dir should use default when env var not set."""
    with mock.patch.dict(os.environ, {}, clear=False):
        if "LLM_SHIM_DATA_DIR" in os.environ:
            del os.environ["LLM_SHIM_DATA_DIR"]
        result = get_data_dir()
        assert result == Path("data/")


def test_get_data_dir_from_env() -> None:
    """get_data_dir should use environment variable when set."""
    with mock.patch.dict(os.environ, {"LLM_SHIM_DATA_DIR": "/custom/path"}):
        result = get_data_dir()
        assert result == Path("/custom/path")


def test_instructor_settings_provider_name_with_slash() -> None:
    """provider_name should extract prefix before slash."""
    settings = InstructorSettings(model="openai/gpt-4o")
    assert settings.provider_name == "openai"


def test_instructor_settings_requires_provider_format() -> None:
    """InstructorSettings requires provider/model format."""
    with pytest.raises(ValidationError):
        InstructorSettings(model="gpt-4o")


def test_instructor_settings_instructor_kwargs_with_mode() -> None:
    """instructor_kwargs should include mode when set."""
    from instructor import Mode

    settings = InstructorSettings(model="openai/gpt-4o", mode=Mode.JSON)
    kwargs = settings.instructor_kwargs()
    assert kwargs["mode"] == Mode.JSON


def test_instructor_settings_instructor_kwargs_without_mode() -> None:
    """instructor_kwargs should not include mode when None."""
    settings = InstructorSettings(model="openai/gpt-4o", mode=None)
    kwargs = settings.instructor_kwargs()
    assert "mode" not in kwargs


def test_instructor_settings_instructor_kwargs_with_extra() -> None:
    """instructor_kwargs should include extra fields."""
    settings = InstructorSettings(
        model="openai/gpt-4o",
        extra={"api_key": "sk-test", "base_url": "https://api.openai.com"},
    )
    kwargs = settings.instructor_kwargs()
    assert kwargs["api_key"] == "sk-test"
    assert kwargs["base_url"] == "https://api.openai.com"


def test_instructor_settings_validate_model_format_requires_slash() -> None:
    """Model validation should require provider/model format."""
    with pytest.raises(ValidationError) as exc_info:
        InstructorSettings(model="gpt-4o")

    assert "provider/model format" in str(exc_info.value).lower()


def test_instructor_settings_validate_model_format_accepts_slash() -> None:
    """Model validation should accept provider/model format."""
    settings = InstructorSettings(model="openai/gpt-4o")
    assert str(settings.model) == "openai/gpt-4o"


def test_server_settings_defaults() -> None:
    """ServerSettings should have sensible defaults."""
    settings = ServerSettings()
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.reload is False
    assert settings.workers == 1
    assert settings.log_level == "info"


def test_server_settings_custom_values() -> None:
    """ServerSettings should accept custom values."""
    settings = ServerSettings(host="127.0.0.1", port=9000, workers=4, log_level="debug")
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000
    assert settings.workers == 4
    assert settings.log_level == "debug"


def test_settings_resolve_provider_by_default() -> None:
    """resolve_provider should resolve default provider when model is None."""
    settings = Settings(
        default_provider="main",
        providers={
            "main": InstructorSettings(model="openai/gpt-4o-mini"),
            "alt": InstructorSettings(model="anthropic/claude-3"),
        },
    )

    name, provider = settings.resolve_provider(None)
    assert name == "main"
    assert str(provider.model) == "openai/gpt-4o-mini"


def test_settings_resolve_provider_by_alias() -> None:
    """resolve_provider should resolve by provider alias."""
    settings = Settings(
        default_provider="main",
        providers={
            "main": InstructorSettings(model="openai/gpt-4o-mini"),
            "fast": InstructorSettings(model="mistral/mistral-7b"),
        },
    )

    name, provider = settings.resolve_provider("fast")
    assert name == "fast"
    assert str(provider.model) == "mistral/mistral-7b"


def test_settings_resolve_provider_by_model_id() -> None:
    """resolve_provider should resolve by exact model ID."""
    settings = Settings(
        default_provider="main",
        providers={
            "main": InstructorSettings(model="openai/gpt-4o-mini"),
            "other": InstructorSettings(model="openai/gpt-4-turbo"),
        },
    )

    name, provider = settings.resolve_provider("openai/gpt-4-turbo")
    assert name == "other"
    assert str(provider.model) == "openai/gpt-4-turbo"


def test_settings_resolve_provider_not_found() -> None:
    """resolve_provider should raise ValueError when model not found."""
    settings = Settings(
        default_provider="main",
        providers={
            "main": InstructorSettings(model="openai/gpt-4o-mini"),
        },
    )

    with pytest.raises(ValueError) as exc_info:
        settings.resolve_provider("anthropic/claude-3")

    assert "not configured" in str(exc_info.value).lower()


def test_settings_works_with_valid_providers() -> None:
    """Settings should accept valid provider configuration."""
    settings = Settings(
        default_provider="main",
        providers={"main": InstructorSettings(model="openai/gpt-4o")},
    )
    assert settings.default_provider == "main"
    assert "main" in settings.providers


def test_settings_validate_providers_default_must_exist() -> None:
    """Validation should require default provider to exist in providers."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            default_provider="main",
            providers={
                "other": InstructorSettings(model="openai/gpt-4o"),
            },
        )

    assert "default_provider" in str(exc_info.value).lower()


def test_get_settings_returns_cached_instance() -> None:
    """get_settings should return the same instance on subsequent calls."""
    # Clear the cache first
    get_settings.cache_clear()

    instance1 = get_settings()
    instance2 = get_settings()

    assert instance1 is instance2


def test_get_settings_cache_clear() -> None:
    """get_settings cache can be cleared."""
    instance1 = get_settings()
    get_settings.cache_clear()
    instance2 = get_settings()

    # They should be different instances
    assert instance1 is not instance2


def test_settings_validate_providers_requires_default_in_dict() -> None:
    """Settings validation should fail if default not in providers dict."""
    with pytest.raises(ValidationError) as exc_info:
        Settings(
            default_provider="missing",
            providers={
                "other": InstructorSettings(model="openai/gpt-4o"),
            },
        )
    assert "default_provider" in str(exc_info.value).lower()


def test_instructor_settings_extra_fields() -> None:
    """InstructorSettings should handle extra configuration fields."""
    settings = InstructorSettings(
        model="openai/gpt-4o",
        extra={
            "api_key": "sk-123",
            "base_url": "https://api.openai.com/v1",
            "timeout": 30,
        },
    )

    kwargs = settings.instructor_kwargs()
    assert kwargs["api_key"] == "sk-123"
    assert kwargs["base_url"] == "https://api.openai.com/v1"
    assert kwargs["timeout"] == 30


def test_server_settings_custom_reload() -> None:
    """ServerSettings should accept custom reload setting."""
    settings = ServerSettings(reload=True)
    assert settings.reload is True


def test_server_settings_custom_port() -> None:
    """ServerSettings should accept custom port."""
    settings = ServerSettings(port=3000)
    assert settings.port == 3000


def test_settings_resolve_provider_by_multiple_models() -> None:
    """resolve_provider should handle multiple providers with different models."""
    settings = Settings(
        default_provider="gpt4",
        providers={
            "gpt4": InstructorSettings(model="openai/gpt-4o"),
            "gpt35": InstructorSettings(model="openai/gpt-3.5-turbo"),
            "claude": InstructorSettings(model="anthropic/claude-3"),
        },
    )

    # By alias
    name, provider = settings.resolve_provider("gpt35")
    assert name == "gpt35"
    assert str(provider.model) == "openai/gpt-3.5-turbo"

    # By model ID
    name, provider = settings.resolve_provider("anthropic/claude-3")
    assert name == "claude"

    # Default
    name, provider = settings.resolve_provider(None)
    assert name == "gpt4"
