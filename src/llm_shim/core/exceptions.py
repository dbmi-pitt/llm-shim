"""Custom exception hierarchy for llm-shim."""


class ShimError(Exception):
    """Base exception for all llm-shim errors."""

    status_code: int = 500


class BadRequestError(ShimError):
    """Client sent an invalid or malformed request (400)."""

    status_code = 400


class ProviderConfigError(ShimError):
    """Provider initialization or configuration failed (500)."""

    status_code = 500


class ProviderCallError(ShimError):
    """Upstream provider API call failed (502)."""

    status_code = 502


class AuthenticationError(ShimError):
    """Client did not present a valid shim credential (401)."""

    status_code = 401


class RateLimitError(ShimError):
    """Client exceeded a local fixed-window rate limit (429)."""

    status_code = 429


class OverloadedError(ShimError):
    """The bounded inference queue is full (503)."""

    status_code = 503


class RequestTimeoutError(ShimError):
    """Inference exceeded the configured per-request timeout (504)."""

    status_code = 504


class RequestTooLargeError(ShimError):
    """HTTP request body exceeded its configured size limit (413)."""

    status_code = 413
