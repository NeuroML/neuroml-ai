#!/usr/bin/env python3
"""
Pre-check source directories to decide whether OCR is needed per PDF.

File: klea_utils/stores/precheck.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from dataclasses import dataclass
from pathlib import Path

from .ingestion import CACHE_DIR_NAME

logger = logging.getLogger(__name__)

#: Output directory for scanned/image PDFs (OCR required).
OCR_DIR_NAME = "ocr"

#: Output directory for text-based PDFs (and all non-PDF files).
NO_OCR_DIR_NAME = "no-ocr"

#: A page whose extracted text has fewer than this many non-whitespace
#: characters is treated as an image page (a scan or a figure-only page).
#: Text-based PDFs easily produce thousands of characters per page, while
#: a scanned page yields near-zero (just embedded text stamps/watermarks).
IMAGE_PAGE_CHAR_THRESHOLD = 50

#: A PDF is classified as needing OCR when at least this fraction of its
#: pages are image pages.
IMAGE_PAGE_FRACTION = 0.5

#: A PDF with no readable pages (e.g. zero pages) is conservatively
#: classified as needing OCR -- it is almost always a bad scan, and the
#: safe default is to OCR it rather than silently store empty chunks.
NO_PAGES_NEEDS_OCR = True

#: Directories inside a source dir that are generated output, never
#: source documents.  The chunk cache is already skipped by ingestion;
#: the organise output dirs are also skipped so a re-run does not
#: re-classify copies.
_SKIP_DIR_NAMES = frozenset({CACHE_DIR_NAME, OCR_DIR_NAME, NO_OCR_DIR_NAME})


@dataclass(frozen=True)
class PageStats:
    """Per-PDF page text statistics for the pre-check report."""

    #: Number of pages read from the PDF.
    pages: int = 0
    #: Number of pages that fell below :data:`IMAGE_PAGE_CHAR_THRESHOLD`.
    image_pages: int = 0
    #: Total non-whitespace characters extracted across all pages.
    total_chars: int = 0


@dataclass(frozen=True)
class PdfClassification:
    """The OCR decision and statistics for a single PDF."""

    path: Path
    needs_ocr: bool
    stats: PageStats


def classify_pdf(path: Path) -> PdfClassification:
    """Decide whether a PDF needs OCR, based on its embedded text layer.

    Reads the PDF with pypdfium2 (already a Docling dependency) and counts
    non-whitespace characters per page.  A page under
    :data:`IMAGE_PAGE_CHAR_THRESHOLD` characters is an image page; a PDF
    with at least :data:`IMAGE_PAGE_FRACTION` image pages needs OCR.
    Text-based (born-digital) PDFs carry a full text layer and classify as
    no-OCR; scanned/image PDFs have almost none and classify as needs-OCR.

    A PDF that cannot be read (missing, corrupt, or pypdfium2 unavailable)
    is conservatively classified as needing OCR.

    :param path: Path to the PDF file
    :returns: A :class:`PdfClassification` with the decision and stats
    """
    # Lazy: importing pypdfium2 loads the native pdfium bindings.  Only
    # needed when a PDF is actually being inspected.
    try:
        import pypdfium2 as pdfium
    except ImportError:
        logger.error("pypdfium2 not available, cannot pre-check PDFs")
        return PdfClassification(path=path, needs_ocr=True, stats=PageStats())

    if path.suffix.lower() != ".pdf" or not path.is_file():
        return PdfClassification(path=path, needs_ocr=True, stats=PageStats())

    try:
        doc = pdfium.PdfDocument(str(path))
    except (pdfium.PdfiumError, OSError) as e:
        logger.error(f"Could not open PDF {path.name}: {e}")
        return PdfClassification(path=path, needs_ocr=True, stats=PageStats())

    try:
        page_count = len(doc)
        image_pages = 0
        total_chars = 0
        for page in doc:
            text = page.get_textpage().get_text_bounded()
            chars = sum(1 for ch in text if not ch.isspace())
            total_chars += chars
            if chars < IMAGE_PAGE_CHAR_THRESHOLD:
                image_pages += 1
            page.close()
    except (pdfium.PdfiumError, OSError) as e:
        logger.error(f"Could not read text from {path.name}: {e}")
        return PdfClassification(path=path, needs_ocr=True, stats=PageStats())
    finally:
        doc.close()

    stats = PageStats(
        pages=page_count, image_pages=image_pages, total_chars=total_chars
    )
    if page_count == 0:
        needs_ocr = NO_PAGES_NEEDS_OCR
    else:
        needs_ocr = image_pages / page_count >= IMAGE_PAGE_FRACTION

    logger.debug(
        f"{path.name}: pages={page_count} image={image_pages} chars={total_chars} "
        f"needs_ocr={needs_ocr}"
    )
    return PdfClassification(path=path, needs_ocr=needs_ocr, stats=stats)


def classify_directory(source_dir: Path) -> dict[Path, PdfClassification]:
    """Classify every PDF under *source_dir*.

    Recursively walks ``**/*.pdf``, skipping the chunk cache and the
    ``ocr``/``no-ocr`` organise output directories (generated artifacts,
    not source documents -- see :data:`_SKIP_DIR_NAMES`).

    Logs progress at ``INFO`` roughly every 10% of PDFs (every file for
    small corpora) so a long pre-check does not sit silently before the
    final report prints.

    :param source_dir: Directory to walk recursively
    :returns: Mapping of PDF path to its :class:`PdfClassification`
    """
    source_resolved = source_dir.resolve()
    pdfs = [
        f
        for f in sorted(source_dir.rglob("*"))
        if f.is_file()
        and f.suffix.lower() == ".pdf"
        and not any(
            part in _SKIP_DIR_NAMES for part in f.relative_to(source_resolved).parts
        )
    ]

    # Report roughly every 10% of PDFs; for small corpora (<= 10 files)
    # this collapses to reporting every file so the user still sees the
    # first result promptly.
    step = max(1, len(pdfs) // 10)

    results: dict[Path, PdfClassification] = {}
    for index, path in enumerate(pdfs):
        if index % step == 0:
            pct = 100 * index // len(pdfs) if pdfs else 100
            logger.info(f"Pre-checking PDFs: {index}/{len(pdfs)} ({pct}%)")
        results[path] = classify_pdf(path)
    return results


def format_precheck_report(classifications: dict[Path, PdfClassification]) -> str:
    """Render a :func:`classify_directory` result as compact text.

    :param classifications: Mapping from :func:`classify_directory`
    :returns: Multi-line summary of the classification outcome
    """
    items = sorted(classifications.values(), key=lambda c: c.path.name)
    ocr = [c for c in items if c.needs_ocr]
    no_ocr = [c for c in items if not c.needs_ocr]

    lines = [
        (
            f"Pre-check: {len(items)} PDFs, {len(ocr)} need OCR "
            f"(image-based), {len(no_ocr)} are text-based (no OCR needed)"
        )
    ]
    if items:
        lines.append("")
        for c in items:
            label = "OCR" if c.needs_ocr else "no-OCR"
            stats = c.stats
            lines.append(
                f"  {label:6} {c.path.name:40} "
                f"pages={stats.pages} image_pages={stats.image_pages} "
                f"chars={stats.total_chars}"
            )
    return "\n".join(lines)


def organise_directory(
    source_dir: Path,
    classifications: dict[Path, PdfClassification] | None = None,
) -> tuple[Path, Path]:
    """Copy classified files into ``ocr/`` and ``no-ocr/`` subdirectories.

    Copies (never moves) files so the researcher's original bibliography
    directory is left untouched.  Scanned/image PDFs go to ``ocr/``;
    text-based PDFs go to ``no-ocr/``.  Non-PDF files (which never need
    OCR) are also copied to ``no-ocr/`` so a single ``chunk no-ocr/`` pass
    ingests them alongside the text PDFs.

    Re-running is idempotent: existing output directories are reused and
    files are overwritten, and :func:`classify_directory` skips the output
    dirs so copies are never re-classified.

    :param source_dir: Directory containing the source documents
    :param classifications: Optional precomputed classifications (from
        :func:`classify_directory`); computed if not given
    :returns: ``(ocr_dir, no_ocr_dir)`` the two output directory paths
    """
    if classifications is None:
        classifications = classify_directory(source_dir)

    ocr_dir = source_dir / OCR_DIR_NAME
    no_ocr_dir = source_dir / NO_OCR_DIR_NAME
    ocr_dir.mkdir(exist_ok=True)
    no_ocr_dir.mkdir(exist_ok=True)

    no_ocr_names = {c.path.name for c in classifications.values() if not c.needs_ocr}

    for f in sorted(source_dir.rglob("*")):
        if not f.is_file():
            continue
        relative = f.relative_to(source_dir.resolve())
        if any(part in _SKIP_DIR_NAMES for part in relative.parts):
            continue
        if f.suffix.lower() == ".pdf":
            # A PDF goes to no-OCR only when explicitly classified as
            # text-based; anything unclassified (e.g. it failed to read)
            # is conservatively sent to OCR.
            target = no_ocr_dir if f.name in no_ocr_names else ocr_dir
        else:
            target = no_ocr_dir
        dest = target / f.name
        if not dest.exists() or dest.stat().st_size != f.stat().st_size:
            import shutil

            shutil.copy2(f, dest)
            logger.info(f"Copied {f.name} -> {dest.relative_to(source_dir)}")

    return ocr_dir, no_ocr_dir
