#!/usr/bin/env python3
"""
Regex-based bibliographic metadata extraction

File: klea_utils/biblio/regex.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import re
from typing import Any

#: Loose DOI pattern: a DOI begins with "10." followed by a 4-9 digit
#: registrant prefix and a slash.
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;]+")
#: Loose URL pattern.
URL_RE = re.compile(r"https?://[^\s,;]+")

#: Labeled keyword list header (Keywords:, Key words:, ...).
_KEYWORDS_RE = re.compile(
    r"(?im)^\s*(?:key\s*words?\s*(?:and\s*phrases?)?|key\s*terms?)\s*:\s*(.+)$"
)
#: Labeled author list header, without the bare "By:" variant.
_AUTHORS_LABELED_RE = re.compile(r"(?im)^\s*authors?\s*\(?s?\)?\s*:\s*(.{3,500})$")
#: Bare "By:" author line, only consulted when no labeled header matched.
_AUTHORS_BY_RE = re.compile(r"(?im)^\s*by\s*:\s*(.{3,500})$")
#: Labeled DOI line.
_DOI_LABELED_RE = re.compile(r"(?im)^\s*doi\s*:\s*(10\.\d{4,9}/[^\s,;]+)")
#: Labeled URL line (URL:, Website:, Homepage:, Webpage:).
_URL_LABELED_RE = re.compile(r"(?im)^\s*(?:url|website|homepage|webpage)\s*:\s*(\S+)")

#: Default number of leading characters scanned.  Bibliographic headers
#: (authors, keywords, DOI, URL) live on the first page, so scanning the
#: whole document is unnecessary and would raise false-positive noise.
DEFAULT_SCAN_LIMIT = 3000


def extract_regex_metadata(
    text: str, limit: int = DEFAULT_SCAN_LIMIT
) -> dict[str, Any]:
    """Extract bibliographic fields from *text* with regex heuristics.

    Scans the first *limit* characters of *text* for labeled keyword and
    author lists, and for a DOI (labeled or loose pattern) and a labeled
    URL.  This is a pre-population aid, not robust extraction: regex
    matches are acknowledged to be noisy, and the caller (the metadata
    extraction cascade) falls back to it only when more authoritative
    sources have failed.  Only non-empty keys are returned.

    :param text: Text to scan (e.g. the joined body text of a document)
    :param limit: Number of leading characters to scan
    :returns: Mapping with any of ``keywords`` (list), ``authors``
        (list), ``doi`` (str), ``url`` (str)
    """
    scan_text = text[:limit]

    result: dict[str, Any] = {}
    keywords = _scan_keywords(scan_text)
    if keywords:
        result["keywords"] = keywords
    authors = _scan_authors(scan_text)
    if authors:
        result["authors"] = authors
    doi = _scan_doi(scan_text)
    if doi:
        result["doi"] = doi
    url = _scan_url(scan_text)
    if url:
        result["url"] = url
    return result


def _scan_keywords(text: str) -> list[str]:
    """Return the keyword list from a labeled keyword header, if any."""
    match = _KEYWORDS_RE.search(text)
    if not match:
        return []
    return _split_terms(match.group(1))


def _scan_authors(text: str) -> list[str]:
    """Return the author list from a labeled header, preferring
    ``Author(s):``/``Authors:`` over a bare ``By:`` line."""
    match = _AUTHORS_LABELED_RE.search(text)
    if not match:
        match = _AUTHORS_BY_RE.search(text)
    if not match:
        return []
    return _split_terms(match.group(1))


def _scan_doi(text: str) -> str | None:
    """Return a DOI from a labeled line, or from the loose pattern."""
    match = _DOI_LABELED_RE.search(text)
    if match:
        return _rstrip_punct(match.group(1))
    match = DOI_RE.search(text)
    if match:
        return _rstrip_punct(match.group(0))
    return None


def _scan_url(text: str) -> str | None:
    """Return a URL from a labeled line (URL:, Website:, ...)."""
    match = _URL_LABELED_RE.search(text)
    if not match:
        return None
    candidate = _rstrip_punct(match.group(1))
    url_match = URL_RE.search(candidate)
    if url_match:
        return _rstrip_punct(url_match.group(0))
    if candidate.startswith("www."):
        return candidate
    return None


def _split_terms(value: str) -> list[str]:
    """Split a comma/semicolon/newline-separated term list."""
    terms = re.split(r"[,;\n]", value)
    return [term.strip().rstrip(".") for term in terms if term.strip()]


def _rstrip_punct(value: str) -> str:
    """Strip trailing punctuation that may follow a DOI/URL in a value."""
    return value.rstrip(".,;:)]}")
