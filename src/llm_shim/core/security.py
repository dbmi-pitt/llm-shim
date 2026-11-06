"""Small, process-local access controls for the HTTP boundary."""

import asyncio
import hmac
import re
import time
from collections import defaultdict, deque

from llm_shim.core.config import SecuritySettings
from llm_shim.core.exceptions import OverloadedError, RateLimitError

REQUEST_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def safe_request_id(value: str | None, generated: str) -> str:
    """Accept only short, log-safe request IDs; otherwise use a generated ID."""
    return value if value and REQUEST_ID_RE.fullmatch(value) else generated


class AccessController:
    """Authenticate and bound inference for one application process."""

    def __init__(self, settings: SecuritySettings) -> None:
        """Initialize local key, rate, and concurrency state."""
        self._settings = settings
        self._keys = settings.client_keys()
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_requests)

    def authenticate(self, authorization: str | None) -> str | None:
        """Return a key label after constant-time bearer-key matching."""
        if not self._keys:
            return (
                "anonymous-loopback"
                if self._settings.allow_anonymous_loopback
                else None
            )
        if not authorization or not authorization.startswith("Bearer "):
            return None
        supplied = authorization[7:]
        for name, expected in self._keys.items():
            if hmac.compare_digest(supplied, expected):
                return name
        return None

    def check_rate_limit(self, key_name: str) -> None:
        """Apply a bounded per-key sliding-window counter."""
        now = time.monotonic()
        requests = self._requests[key_name]
        cutoff = now - self._settings.rate_limit_window_seconds
        while requests and requests[0] <= cutoff:
            requests.popleft()
        if len(requests) >= self._settings.rate_limit_requests:
            raise RateLimitError("Rate limit exceeded")
        requests.append(now)

    async def acquire(self) -> None:
        """Acquire capacity without an unbounded inference wait queue."""
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(), timeout=self._settings.queue_timeout_seconds
            )
        except TimeoutError as error:
            raise OverloadedError("Service is busy") from error

    def release(self) -> None:
        """Release a previously acquired inference slot."""
        self._semaphore.release()
