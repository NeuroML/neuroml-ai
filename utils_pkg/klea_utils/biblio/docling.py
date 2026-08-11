#!/usr/bin/env python3
"""
Docling-based bibliographic metadata extraction

File: klea_utils/biblio/docling.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_docling_structured(dl_doc, file_path: str) -> dict:
    """Extract the free, reliably-structured bibliographic signals from a
    Docling document.

    These come straight from Docling's layout model and document origin,
    so they do not depend on regex heuristics:

    - ``title`` -- the first ``TitleItem``'s text, falling back to the
      source file's stem
    - ``source_type`` -- the document's ``origin.mimetype``
    - ``source_url`` -- the ``origin.uri`` when it is an ``http(s)`` URL
      (web-sourced inputs only)
    - ``urls`` -- deduplicated ``http(s)`` hyperlink URLs found on any
      text item (from markdown/HTML links)

    :param dl_doc: A Docling ``DoclingDocument``
    :param file_path: Path of the source file (for the title fallback)
    :returns: Dict with any of ``title``, ``source_type``,
        ``source_url``, ``urls``
    """
    result: dict = {}
    result["title"] = _first_title_text(dl_doc) or Path(file_path).stem

    origin = getattr(dl_doc, "origin", None)
    if origin is not None:
        mimetype = getattr(origin, "mimetype", None)
        if mimetype:
            result["source_type"] = mimetype
        uri = getattr(origin, "uri", None)
        if uri is not None:
            uri_str = str(uri)
            if uri_str.startswith(("http://", "https://")):
                result["source_url"] = uri_str

    urls = _hyperlink_urls(dl_doc)
    if urls:
        result["urls"] = urls

    logger.debug(f"docling structured signals: {result}")
    return result


def extract_layout_region(dl_doc, page: int = 1, frac: float = 0.35) -> str | None:
    """Return the text of the top *frac* of *page* as a single string.

    Filters the document's text items to those whose bounding box starts
    within the top *frac* of *page* (TOPLEFT origin) and joins their
    text.  The first-page header is where authors, keywords and the DOI
    live, so this gives the regex tier a focused region to scan rather
    than the whole document.

    :param dl_doc: A Docling ``DoclingDocument``
    :param page: Page number to inspect
    :param frac: Fraction of the page height that counts as the header
    :returns: Joined region text, or ``None`` when there is nothing in
        the region
    """
    page_item = dl_doc.pages.get(page)
    if page_item is None:
        return None
    height = page_item.size.height

    region_texts = []
    for item in dl_doc.texts:
        if not item.prov:
            continue
        prov = item.prov[0]
        if prov.page_no != page:
            continue
        if prov.bbox.t <= frac * height:
            region_texts.append(item.text)

    if not region_texts:
        return None
    region_text = "\n".join(region_texts)
    logger.debug(
        f"layout region (page {page}, top {frac:.0%}): "
        f"{len(region_text)} characters from {len(region_texts)} items"
    )
    return region_text


def _first_title_text(dl_doc) -> str | None:
    """Return the first title item's text, or ``None`` if there is none."""
    # Lazy: importing docling_core pulls in the pydantic document models.
    # Only needed when a Docling document is being inspected.
    from docling_core.types.doc.document import DocItemLabel

    for item in dl_doc.texts:
        if item.label == DocItemLabel.TITLE and item.text.strip():
            return item.text.strip()
    return None


def _hyperlink_urls(dl_doc) -> list[str]:
    """Return deduplicated ``http(s)`` hyperlink URLs from text items."""
    urls: list[str] = []
    seen: set[str] = set()
    for item in dl_doc.texts:
        hyperlink = getattr(item, "hyperlink", None)
        if hyperlink is None:
            continue
        value = str(hyperlink)
        if not value.startswith(("http://", "https://")):
            continue
        if value not in seen:
            seen.add(value)
            urls.append(value)
    return urls
