#!/usr/bin/env python3
"""
Utility functions for the Klea API layer.

File: klea_utils/api/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import httpx
from pydantic import AnyUrl
from pydantic import ValidationError as PydanticValidationError
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    stop_after_delay,
    wait_random_exponential,
)


def validate_url(value: str) -> str:
    """Return *value* if it is a valid HTTP(S) URL, else raise ``ValueError``."""
    try:
        AnyUrl(value)
    except PydanticValidationError:
        raise ValueError(f"'{value}' is not a valid HTTP(S) URL")
    return value


def _make_retryer(attempts: int | None = None, timeout: float = 180.0) -> AsyncRetrying:
    """Create an ``AsyncRetrying`` that retries transient API call errors.

    :param attempts: If set, bound the number of probe attempts.  Takes
        precedence over *timeout* (useful for fast single-shot probes).
    :param timeout: Total wall-clock seconds to keep probing when
        *attempts* is unset.  Generous by default so a server that is slow
        to initialize (MCP servers, embedding model downloads) is not given
        up on prematurely.
    :returns: A configured :class:`tenacity.AsyncRetrying` instance
    """
    if attempts is not None:
        stop = stop_after_attempt(attempts)
    else:
        stop = stop_after_delay(timeout)
    return AsyncRetrying(
        wait=wait_random_exponential(multiplier=1, max=10),
        stop=stop,
        retry=retry_if_exception_type(
            (
                httpx.ConnectError,
                httpx.HTTPStatusError,
                httpx.ReadError,
                httpx.ReadTimeout,
            )
        ),
        reraise=True,
    )


async def _get_ready(url: str) -> dict:
    """GET the health endpoint and return its JSON, raising on non-2xx."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.json()


async def check_api_is_ready(
    url: str, attempts: int | None = None, timeout: float = 180.0
):
    """Exponentially back off checking that the API is ready.

    :param url: Health check endpoint URL
    :param attempts: If set, maximum number of probe attempts (overrides
        *timeout*)
    :param timeout: Total wall-clock seconds to keep probing when
        *attempts* is unset
    """
    retryer = _make_retryer(attempts, timeout)
    return await retryer(_get_ready, url)
