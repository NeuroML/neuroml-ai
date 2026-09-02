#!/usr/bin/env python3
"""
Regex-based bibliographic metadata extraction

File: klea_utils/biblio/regex.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: Loose DOI pattern: a DOI begins with "10." followed by a 4-9 digit
#: registrant prefix and a slash.
DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;]+")
#: Loose URL pattern.
URL_RE = re.compile(r"https?://[^\s,;]+")

#: Labeled keyword list header.  Accepts an optional colon and the list
#: on the same line or the following line, since publishers format these
#: in several ways (e.g. ``Keywords: a, b``, ``KEYWORDS`` then the list,
#: ``Keywords :`` then the list on the next line).
_KEYWORDS_RE = re.compile(
    r"(?im)^\s*(?:key\s*words?\s*(?:and\s*phrases?)?|key\s*terms?)\s*:?\s*(.+)$"
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

#: Keyword headers can sit well below the author/DOI region (e.g. after an
#: abstract), so keywords are scanned over a wider window than the other
#: fields, which stay at :data:`DEFAULT_SCAN_LIMIT` to avoid false
#: positives on prose deeper in the document.
KEYWORD_SCAN_LIMIT = 8000


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
    # Keywords are scanned over their own wider window (they can sit below
    # the abstract); the other fields stay within *limit*.
    keywords = _scan_keywords(text[:KEYWORD_SCAN_LIMIT])
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
    logger.debug(f"regex extraction over {len(scan_text)} characters: {result}")
    return result


def _scan_keywords(text: str) -> list[str]:
    """Return the keyword list from a labeled keyword header, if any.

    Accepts ``Keywords: a, b``, ``KEYWORDS`` with the list on the next
    line, and ``Keywords :`` with the list on the next line.
    """
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


def _sanitize_doi(doi: str) -> str | None:
    """Return a valid DOI from a possibly-garbled match, or ``None``.

    A DOI is ``10.<registrant>/<suffix>`` where the suffix contains no
    further ``/``.  URL-based matches often drag in a trailing path
    (e.g. ``10.1073/pnas.2201699120/-/DCSupplemental``) and text matches
    can carry markdown/URL continuation junk (e.g. ``](https...``) or be
    truncated by whitespace Docling inserted mid-DOI; this keeps only
    the ``prefix/suffix`` part.  Returns ``None`` when *doi* is not a
    well-formed DOI after sanitizing.
    """
    cleaned = _rstrip_punct(doi)
    match = DOI_RE.match(cleaned)
    if not match:
        return None
    candidate = match.group(0)
    # Bound the suffix at markdown/URL continuations and trailing
    # punctuation, then strip a trailing URL path (a valid DOI suffix
    # has no further slash).
    candidate = re.split(r"[\])}\s]", candidate, maxsplit=1)[0]
    candidate = candidate.rstrip(".,;:)]}")
    parts = candidate.split("/", 1)
    if len(parts) == 2 and "/" in parts[1]:
        candidate = parts[0] + "/" + parts[1].split("/", 1)[0]
    return candidate or None


def _scan_dois(text: str) -> list[str]:
    """Return all distinct DOI candidates found in *text*.

    Unlike :func:`_scan_doi`, which returns a single match, this returns
    every DOI-like match (sanitized and deduplicated) so the extraction
    cascade can try candidates in order instead of trusting the first
    match, which may be a broken or journal-level DOI.

    :param text: Text to scan
    :returns: Deduplicated list of sanitized DOI strings
    """
    seen: list[str] = []
    seen_set: set[str] = set()
    for match in DOI_RE.finditer(text):
        doi = _sanitize_doi(match.group(0))
        if doi and doi not in seen_set:
            seen_set.add(doi)
            seen.append(doi)
    return seen


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
