"""Utility functions/helpers."""

import asyncio
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager

environment_override_lock = asyncio.Lock()


@contextmanager
def patched_environ(overrides: Mapping[str, str]) -> Iterator[None]:
    """Temporarily apply environment variable overrides and restore afterwards."""
    if not overrides:
        yield
        return

    previous: dict[str, str | None] = {}
    injected: set[str] = set()

    for key, value in overrides.items():
        if not key:
            continue

        normalized_key = key.strip()
        string_value = str(value)

        for candidate in {normalized_key, normalized_key.upper()}:
            if not candidate:
                continue
            if candidate not in injected:
                previous[candidate] = os.environ.get(candidate)
                injected.add(candidate)
            os.environ[candidate] = string_value

    try:
        yield
    finally:
        for key in injected:
            old_value = previous[key]
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value
