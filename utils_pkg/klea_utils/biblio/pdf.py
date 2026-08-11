#!/usr/bin/env python3
"""
PDF bibliographic metadata extraction

File: klea_utils/biblio/pdf.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

#: PDF Info-dict fields with bibliographic value, in preference order.
#: Date and tooling fields (Creator, Producer) are omitted.
_BIBLIO_FIELDS = ("Title", "Author", "Keywords", "Subject")

#: Loose DOI pattern, shared with the regex extraction tier.
_DOI_RE = re.compile(r"\b10\.\d{4,9}/[^\s,;]+")
#: Loose URL pattern.
_URL_RE = re.compile(r"https?://[^\s,;]+")

#: Standard Info-dict keys that pypdfium2 returns, scanned in this order
#: for an embedded DOI/URL.  pdfium only exposes these standard keys --
#: arbitrary custom keys (e.g. ``DOI``/``URL``) are dropped, so a DOI or
#: URL is picked up from the *value* of these fields (journals commonly
#: place the DOI in ``Subject``).
_SCAN_ORDER = ("Subject", "Title", "Author", "Keywords", "Creator", "Producer")


def extract_pdf_info(path: str) -> dict[str, str]:
    """Extract bibliographic metadata from a PDF's Info dict.

    Reads the document's metadata fields (Title, Author, Keywords,
    Subject) with pypdfium2, which is already installed as a Docling
    dependency.  A DOI or URL is also picked up from the values of the
    standard fields (journals commonly embed the DOI in ``Subject``).
    Only non-empty fields are returned, keyed lower-case (``title``,
    ``author``, ``keywords``, ``subject``, ``doi``, ``url``).

    Returns an empty dict for non-PDF files, files without metadata,
    and when the backend is unavailable, so callers can simply fall
    through to the next extraction tier.

    :param path: Path to the PDF file
    :returns: Lower-case mapping of bibliographic PDF metadata fields to
        their string values (only non-empty fields)
    """
    # Lazy: importing pypdfium2 loads the native pdfium bindings.  It is
    # only needed when a PDF is being ingested, so defer the import.
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.error("pypdfium2 not available, skipping PDF metadata")
        return {}

    pdf_path = Path(path)
    if pdf_path.suffix.lower() != ".pdf" or not pdf_path.is_file():
        return {}

    try:
        doc = pdfium.PdfDocument(str(pdf_path))
    except (pdfium.PdfiumError, OSError) as e:
        logger.error(f"Could not open PDF {pdf_path.name}: {e}")
        return {}

    try:
        metadata = doc.get_metadata_dict() or {}
    except (pdfium.PdfiumError, OSError) as e:
        logger.error(f"Could not read metadata from {pdf_path.name}: {e}")
        return {}
    finally:
        doc.close()

    result = {}
    for field in _BIBLIO_FIELDS:
        value = metadata.get(field)
        if value is not None and str(value).strip():
            result[field.lower()] = str(value).strip()

    doi, url = _find_doi_url(metadata)
    if doi:
        result["doi"] = doi
    if url:
        result["url"] = url
    return result


def _find_doi_url(metadata: dict) -> tuple[str | None, str | None]:
    """Pick a DOI and URL out of a PDF Info dict.

    Scans the standard metadata field values (Subject first -- journals
    commonly embed the DOI there) for a DOI or URL pattern.  pypdfium2
    does not return arbitrary custom keys, so this works on the values
    of the fields it does return.

    :param metadata: Metadata dict from pypdfium2
    :returns: ``(doi, url)``, either or both possibly ``None``
    """
    doi: str | None = None
    url: str | None = None

    for key in _SCAN_ORDER:
        value = metadata.get(key)
        if not value or not str(value).strip():
            continue
        value_str = str(value)
        if doi is None:
            match = _DOI_RE.search(value_str)
            if match:
                doi = _rstrip_punct(match.group(0))
        if url is None:
            match = _URL_RE.search(value_str)
            if match:
                url = _rstrip_punct(match.group(0))
        if doi is not None and url is not None:
            break

    return doi, url


def _rstrip_punct(value: str) -> str:
    """Strip trailing punctuation that may follow a DOI/URL in a value."""
    return value.rstrip(".,;:)]}")
