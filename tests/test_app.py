"""Integration tests for main app and initialization."""

from typing import Any

import pytest
from fastapi.testclient import TestClient

from llm_shim import app, create_app
from llm_shim.core.config import (
    ProviderSettings,
    SecuritySettings,
    ServerSettings,
    Settings,
)
from llm_shim.core.exceptions import (
    BadRequestError,
    ProviderCallError,
    ProviderConfigError,
    ShimError,
)


def _secured_app(monkeypatch: pytest.MonkeyPatch):
    """Create an app with two rotating test keys and no provider calls."""
    monkeypatch.setenv("TEST_SHIM_KEYS", "old:old-key,new:new-key")
    settings = Settings.model_construct(
        providers={"test": ProviderSettings(chat_models=["*"], embedding_models=["*"])},
        server=ServerSettings(),
        security=SecuritySettings(api_keys_env="TEST_SHIM_KEYS"),
    )
    return create_app(settings)


def _route_paths() -> list[str]:
    paths: list[str] = []
    for route in app.routes:
        path = getattr(route, "path", None)
        if isinstance(path, str):
            paths.append(path)
    return paths


def test_app_created() -> None:
    """Test that app is properly initialized."""
    assert app is not None
    assert app.title == "llm-shim"


def test_create_app_returns_fastapi_instance() -> None:
    """create_app should return a FastAPI instance."""
    from fastapi import FastAPI

    test_app = create_app()
    assert isinstance(test_app, FastAPI)
    assert test_app.title == "llm-shim"


def test_app_has_routers_configured(client: Any | None = None) -> None:
    """App should have chat and embeddings routers configured."""
    # Check that routes are registered
    routes = _route_paths()
    assert "/v1/chat/completions" in routes
    assert "/v1/embeddings" in routes
    assert "/v1/models" in routes


def test_health_endpoints() -> None:
    """App should have health check endpoints."""
    routes = _route_paths()
    assert "/livez" in routes
    assert "/healthz" in routes


def test_shim_error_status_codes() -> None:
    """ShimError subclasses should have correct status_code attributes."""
    assert ShimError.status_code == 500
    assert BadRequestError.status_code == 400
    assert ProviderConfigError.status_code == 500
    assert ProviderCallError.status_code == 502


def test_shim_error_inheritance() -> None:
    """All custom exceptions should inherit from ShimError."""
    assert issubclass(BadRequestError, ShimError)
    assert issubclass(ProviderConfigError, ShimError)
    assert issubclass(ProviderCallError, ShimError)


def test_health_endpoint_returns_ok() -> None:
    """Health endpoints should return ok status."""
    client = TestClient(app)
    for path in ("/healthz", "/livez"):
        response = client.get(path)
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


def test_request_id_header_assigned() -> None:
    """Responses should include x-request-id header."""
    client = TestClient(app)
    response = client.get("/healthz")
    assert "x-request-id" in response.headers


def test_request_id_header_echoed() -> None:
    """When client sends x-request-id, response should echo it back."""
    client = TestClient(app)
    response = client.get("/healthz", headers={"x-request-id": "test-123"})
    assert response.headers["x-request-id"] == "test-123"


def test_unsafe_request_id_is_replaced() -> None:
    """Client-controlled request IDs cannot inject log/header content."""
    client = TestClient(app)
    response = client.get("/healthz", headers={"x-request-id": "bad value"})
    assert response.headers["x-request-id"] != "bad value"


@pytest.mark.parametrize(
    ("method", "path", "payload"),
    [
        ("get", "/v1/models", None),
        ("post", "/v1/chat/completions", {"model": "test:chat", "messages": []}),
        ("post", "/v1/embeddings", {"model": "test:embed", "input": "x"}),
    ],
)
def test_all_v1_endpoints_require_authentication(
    monkeypatch: pytest.MonkeyPatch,
    method: str,
    path: str,
    payload: dict[str, Any] | None,
) -> None:
    """Authentication rejects every model-facing endpoint before route work."""
    client = TestClient(_secured_app(monkeypatch))
    response = client.get(path) if method == "get" else client.post(path, json=payload)
    assert response.status_code == 401
    assert response.json()["error"]["message"] == "Invalid authentication credentials."


def test_models_accepts_rotated_bearer_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both active named keys are accepted during a rotation window."""
    client = TestClient(_secured_app(monkeypatch))
    for key in ("old-key", "new-key"):
        assert (
            client.get(
                "/v1/models", headers={"Authorization": f"Bearer {key}"}
            ).status_code
            == 200
        )


def test_docs_are_disabled_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """Interactive documentation is absent unless explicitly enabled locally."""
    assert TestClient(_secured_app(monkeypatch)).get("/docs").status_code == 404
