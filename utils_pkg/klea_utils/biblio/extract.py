#!/usr/bin/env python3
"""
Bibliographic metadata extraction cascade

File: klea_utils/biblio/extract.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import re
from typing import Protocol

from .docling import extract_docling_structured, extract_layout_region
from .doi import BiblioRecord
from .pdf import extract_pdf_info
from .regex import extract_regex_metadata

logger = logging.getLogger(__name__)


class Resolver(Protocol):
    """Protocol for objects that can resolve a DOI to a record.

    :class:`~klea_utils.biblio.doi.DoiResolver` implements this; tests
    and other callers may substitute any object with a compatible
    ``resolve`` method.
    """

    def resolve(self, doi: str) -> BiblioRecord | None:
        """Resolve *doi* to a record, or ``None`` on failure."""


def extract_metadata(
    dl_doc,
    file_path: str,
    pdf_path: str | None = None,
    resolver: Resolver | None = None,
) -> dict:
    """Extract bibliographic metadata for a document via the tiered cascade.

    This is the whole metadata extraction pipeline.  Each tier tries to
    be more reliable than the next, and the tiers run most-authoritative
    first; a tier only fills fields the tiers above it have not already
    set (gap-fill), so a resolved record always wins:

    **Tiers**

    1. ``pdf-info`` -- the PDF Info dict (title, authors, keywords, doi,
       url), read with pypdfium2.  Local and fast, but often empty: many
       publishers ship no bibliographic fields in it.

    2. ``docling`` -- the free structured signals from Docling's layout
       model: the title item, the origin mimetype/URI, and the
       hyperlinks on text items.

    3. ``layout-regex`` -- regex over the focused first-page header
       region (the top fraction of page one, selected via the layout
       bounding boxes), where authors, keywords and the DOI live.

    4. ``regex`` -- regex over the first ``DEFAULT_SCAN_LIMIT``
       characters of the whole document text.  A broader net than the
       layout region, so it still catches a DOI/URL sitting in a
       first-page footer; it is deliberately bounded, since the
       bibliographic fields live in the front matter.

    5. ``doi-service`` -- if a DOI was discovered by any tier above, it
       is resolved via Crossref/OpenAlex/Semantic Scholar (round-robin
       across calls, fallback on rate limits, disk-cached).  A resolved
       record is authoritative: its title, authors, year, venue and DOI
       override everything below.  This tier is skipped when *resolver*
       is ``None``.

    **Output** -- a flat dict of non-empty fields (``title``,
    ``authors``, ``keywords``, ``year``, ``venue``, ``doi``, ``url``,
    ``source_type``, ``source_url``, ``urls``) plus two internal keys:

    - ``_metadata_complete`` -- ``True`` only when a full DOI record
      (title + authors + year) or a full PDF Info dict (title + author +
      keywords) was obtained; otherwise ``False``, signalling the
      researcher to review the pre-populated map.
    - ``_sources`` -- the tiers that contributed at least one field, in
      priority order (e.g. ``["doi-service", "regex"]``).

    The abstract of a resolved record is used only as completeness
    evidence and is deliberately NOT included in the output: the abstract
    is already part of the chunked document, so persisting it per-chunk
    would duplicate it.

    :param dl_doc: A Docling ``DoclingDocument``
    :param file_path: Path of the source file
    :param pdf_path: Path to a PDF file, when the source is a PDF
    :param resolver: Optional :class:`Resolver`; when ``None`` the
        DOI-service tier is skipped (no network)
    :returns: Flat metadata dict including the ``_metadata_complete``
        and ``_sources`` internal keys
    """
    metadata: dict = {}
    sources: list[str] = []

    logger.debug(f"extracting metadata for {file_path = }\n{pdf_path = }")

    # Tier 0 -- join the document's text items for the regex tiers.
    full_text = _document_text(dl_doc)

    # Tier 1 -- PDF Info dict.
    pdf_info = extract_pdf_info(pdf_path) if pdf_path else {}
    pdf_fields = _normalize_pdf_info(pdf_info)
    logger.debug(f"pdf-info tier: {pdf_fields}")

    # Tier 2 -- Docling structured signals.
    docling_info = extract_docling_structured(dl_doc, file_path)
    logger.debug(f"docling tier: {docling_info}")

    # Tier 3 -- focused first-page header text.
    layout_text = extract_layout_region(dl_doc)
    layout_regex = extract_regex_metadata(layout_text) if layout_text else {}
    logger.debug(f"layout-regex tier: {layout_regex}")

    # Tier 4 -- broader front-matter text.
    front_regex = extract_regex_metadata(full_text) if full_text else {}
    logger.debug(f"front-regex tier: {front_regex}")

    # Tier 5 -- DOI discovery: the first tier to find one wins.
    doi = _discover_doi(pdf_fields, layout_regex, front_regex)
    logger.debug(f"discovered {doi = }")

    # Tier 6 -- authoritative record from the DOI services.
    record: BiblioRecord | None = None
    if doi and resolver is not None:
        record = resolver.resolve(doi)
    if record is not None:
        logger.info(
            f"resolved DOI {doi} via DOI services\n"
            f"{record.title = }\n"
            f"{record.authors = }\n"
            f"{record.year = }"
        )

    # Merge, most-authoritative first; each tier only fills gaps.
    if record is not None:
        _gap_fill(metadata, _record_fields(record), "doi-service", sources)
    _gap_fill(metadata, pdf_fields, "pdf-info", sources)
    _gap_fill(metadata, docling_info, "docling", sources)
    _gap_fill(metadata, layout_regex, "layout-regex", sources)
    _gap_fill(metadata, front_regex, "regex", sources)

    if record is not None:
        complete = bool(record.title and record.authors and record.year)
    else:
        complete = bool(
            pdf_fields.get("title")
            and pdf_fields.get("authors")
            and pdf_fields.get("keywords")
        )
    metadata["_metadata_complete"] = complete
    metadata["_sources"] = sources

    logger.debug(
        f"metadata extraction done for {file_path = }\n"
        f"{metadata = }\n"
        f"{sources = }\n"
        f"{complete = }"
    )

    return metadata


def _document_text(dl_doc) -> str:
    """Join the document's text items into a single string."""
    return "\n".join(item.text for item in dl_doc.texts if item.text.strip())


def _normalize_pdf_info(pdf_info: dict) -> dict:
    """Normalise PDF Info fields to the canonical metadata key set."""
    result: dict = {}
    if pdf_info.get("title"):
        result["title"] = pdf_info["title"]
    if pdf_info.get("author"):
        result["authors"] = _split_terms(pdf_info["author"])
    if pdf_info.get("keywords"):
        result["keywords"] = _split_terms(pdf_info["keywords"])
    if pdf_info.get("doi"):
        result["doi"] = pdf_info["doi"]
    if pdf_info.get("url"):
        result["url"] = pdf_info["url"]
    return result


def _discover_doi(*sources: dict) -> str | None:
    """Return the first DOI found across *sources*."""
    for source in sources:
        doi = source.get("doi")
        if doi:
            return doi
    return None


def _record_fields(record: BiblioRecord) -> dict:
    """Convert a resolved record to canonical metadata fields.

    The abstract is deliberately excluded (it is already part of the
    chunked document).
    """
    fields = {
        "title": record.title,
        "authors": record.authors,
        "year": record.year,
        "venue": record.venue,
        "doi": record.doi,
    }
    return {key: value for key, value in fields.items() if value not in (None, [], "")}


def _gap_fill(metadata: dict, source: dict, label: str, sources: list[str]) -> None:
    """Fill unset *metadata* fields from *source*, recording *label*.

    Fields already present in *metadata* are never overwritten, which is
    what gives the cascade its most-authoritative-first precedence.
    """
    added = False
    for key, value in source.items():
        if key in metadata or not value:
            continue
        metadata[key] = value
        added = True
    if added and label not in sources:
        sources.append(label)


def _split_terms(value: str) -> list[str]:
    """Split a comma/semicolon-separated term list."""
    return [term.strip() for term in re.split(r"[,;]", value) if term.strip()]
