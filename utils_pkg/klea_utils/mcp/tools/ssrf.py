#!/usr/bin/env python3
"""
SSRF (Server-Side Request Forgery) protection for outbound HTTP tools.

File: klea_utils/mcp/tools/ssrf.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import ipaddress
import logging
import socket
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


def is_private_or_reserved(ip: ipaddress.IPv4Address | ipaddress.IPv6Address) -> bool:
    """Return True for addresses an SSRF guard should refuse to fetch.

    Blocks loopback, private (RFC1918/ULA), link-local (incl. the cloud
    metadata address 169.254.169.254), reserved, and multicast ranges.

    :param ip: Address to classify.
    """
    return (
        ip.is_loopback
        or ip.is_private
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
    )


def check_ssrf(url: str) -> str | None:
    """Return an error message if *url* resolves to a private/internal host.

    Resolves the hostname and rejects the request when any resolved address
    is private, loopback, link-local, reserved, or multicast.  Returns
    ``None`` when the request is allowed.

    NOTE: only the initial *url* is checked; an ``httpx`` client that follows
    redirects could still be redirected to an internal host.  This is a
    known best-effort limitation, acceptable for typical use.

    :param url: Absolute URL to check.
    :returns: An error message describing the denial, or ``None`` when the
        URL is allowed.
    """
    host = urlparse(url).hostname
    if not host:
        return "URL has no host."
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        logger.warning(f"Could not resolve host {host}: {exc}")
        return f"Could not resolve host {host}: {exc}"
    logger.debug(f"Resolved {host} -> {[info[4][0] for info in infos]}")
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if is_private_or_reserved(ip):
            logger.warning(f"SSRF guard: {host} resolves to {ip} (blocked)")
            return f"Blocked request to private/internal address: {ip}"
    return None
