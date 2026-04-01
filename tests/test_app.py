"""Integration tests for main app and initialization."""

from typing import Any

from llm_shim import app, create_app


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


def test_health_endpoints() -> None:
    """App should have health check endpoints."""
    routes = _route_paths()
    assert "/livez" in routes
    assert "/healthz" in routes
