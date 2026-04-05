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
