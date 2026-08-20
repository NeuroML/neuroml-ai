#!/usr/bin/env python3
"""
LLM-free linting of a stored BM25/vector corpus (post-store check).

File: klea_utils/stores/postcheck.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from collections import Counter
from typing import Any

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

#: A chunk whose ``page_content`` has fewer than this many characters is
#: flagged as suspicious -- typically a failed conversion or an OCR miss
#: on a scanned PDF.
MIN_CHUNK_CHARS = 20

#: Metadata keys that, when absent from a chunk, suggest the file's
#: DEFAULT entry did not apply or extraction failed.  ``file_name`` is
#: machine-set so its presence is expected; the others come from the map.
REQUIRED_METADATA_KEYS = ("file_name", "title", "year", "doi")

#: Maximum characters of ``page_content`` shown per sample chunk.
SAMPLE_TEXT_WIDTH = 200

#: How many short chunks / missing-metadata files to list before truncating
#: the report (the rest are counted in the summary line).
MAX_LISTED = 20

#: Contiguous chunks shown at each sampling location.  The user reviews a
#: small run of consecutive chunks per location rather than one isolated
#: chunk, to check that chunking stays coherent across boundaries.
PER_ANCHOR_SAMPLES = 3


def lint_store(docs: list[Document]) -> dict[str, Any]:
    """Return a structured report of issues in a stored chunk corpus.

    Runs only deterministic checks (no LLM): a corpus summary, suspicious
    chunks (near-empty text, missing bibliographic metadata), and
    structural problems (chunks without a ``file_name``, invalid types).
    *docs* is the unpickled BM25 corpus (a list of
    :class:`langchain_core.documents.Document`).

    :param docs: List of stored chunk documents
    :returns: A dict with ``total``, ``files``, ``chunks_per_file``,
        ``total_chars``, ``empty`` (list of ``(file_name, chars)``),
        ``missing_metadata`` (``{file_name: [missing keys]}``),
        ``no_file_name`` (count of chunks lacking ``file_name``),
        ``invalid_content`` (count of non-string ``page_content``), and
        ``invalid_year`` (count of non-int ``year``)
    """
    total = len(docs)
    total_chars = sum(
        len(doc.page_content) if isinstance(doc.page_content, str) else 0
        for doc in docs
    )

    files: Counter[str] = Counter()
    for doc in docs:
        file_name = doc.metadata.get("file_name")
        if isinstance(file_name, str) and file_name:
            files[file_name] += 1

    chunks_per_file: dict[str, int | float] = {}
    if files:
        counts = list(files.values())
        chunks_per_file = {
            "min": min(counts),
            "max": max(counts),
            "avg": round(sum(counts) / len(counts), 2),
        }

    # Suspicious: near-empty content (conversion/OCR miss).
    empty: list[tuple[str, int]] = []
    for doc in docs:
        text = doc.page_content
        if not isinstance(text, str) or len(text.strip()) < MIN_CHUNK_CHARS:
            file_name = doc.metadata.get("file_name", "?")
            empty.append((file_name, len(text) if isinstance(text, str) else 0))

    # Suspicious: missing bibliographic metadata.
    missing_metadata: dict[str, list[str]] = {}
    for doc in docs:
        file_name = doc.metadata.get("file_name", "?")
        missing = [k for k in REQUIRED_METADATA_KEYS if k not in doc.metadata]
        if missing:
            missing_metadata.setdefault(file_name, []).extend(missing)

    # Structural: chunks that cannot be attributed to a source file.
    no_file_name = sum(
        1 for doc in docs if not isinstance(doc.metadata.get("file_name"), str)
    )

    # Structural: invalid page_content / year types.
    invalid_content = sum(1 for doc in docs if not isinstance(doc.page_content, str))
    invalid_year = sum(
        1
        for doc in docs
        if "year" in doc.metadata and not isinstance(doc.metadata["year"], int)
    )

    return {
        "total": total,
        "files": len(files),
        "chunks_per_file": chunks_per_file,
        "total_chars": total_chars,
        "empty": empty,
        "missing_metadata": missing_metadata,
        "no_file_name": no_file_name,
        "invalid_content": invalid_content,
        "invalid_year": invalid_year,
    }


def select_sample_windows(
    docs: list[Document],
    anchors: int,
    per_anchor: int = PER_ANCHOR_SAMPLES,
) -> list[list[Document]]:
    """Pick ``anchors`` evenly-spaced windows of contiguous chunks to review.

    Each window is ``per_anchor`` consecutive chunks starting at an anchor
    position, so the user sees a small coherent run at several different
    places in the corpus (early/middle/late) rather than the first few
    chunks.  Anchor starts are evenly spaced across the whole corpus and
    avoid the very first/last chunk, giving roughly 25% / 50% / 75% for
    three anchors.

    For a corpus smaller than one window, a single window of the whole
    corpus is returned.  The selection is deterministic -- no RNG -- so
    re-runs show the same samples for easy comparison.

    :param docs: List of stored chunk documents
    :param anchors: Number of sampling locations (``--samples``)
    :param per_anchor: Contiguous chunks to show at each location
    :returns: List of windows (each a list of :class:`Document`); empty
        when ``anchors <= 0`` or the corpus is empty
    """
    total = len(docs)
    if anchors <= 0 or total == 0:
        return []
    if total <= per_anchor:
        return [docs]
    starts = [round(total * i / (anchors + 1)) for i in range(1, anchors + 1)]
    return [docs[start : start + per_anchor] for start in starts]


def format_chunk_sample(doc: Document, width: int = SAMPLE_TEXT_WIDTH) -> str:
    """Render one chunk for human review: truncated text + metadata line.

    :param doc: A stored chunk document
    :param width: Maximum characters of ``page_content`` to show
    :returns: Multi-line sample text
    """
    text = doc.page_content if isinstance(doc.page_content, str) else ""
    truncated = text if len(text) <= width else text[:width] + "..."
    meta = doc.metadata
    head = " > ".join(meta.get("headings") or [])
    bits = [
        f"file={meta.get('file_name', '?')}",
        f"title={meta.get('title', '?')}",
        f"year={meta.get('year', '?')}",
        f"doi={meta.get('doi', '?')}",
    ]
    if head:
        bits.append(f"headings={head}")
    return f"{truncated}\n  [{', '.join(bits)}]"


def format_store_lint_report(
    report: dict[str, Any], samples: list[list[Document]]
) -> str:
    """Render a :func:`lint_store` report as compact text.

    :param report: Report dict from :func:`lint_store`
    :param samples: List of sample windows (each a list of
        :class:`Document`) to print, from :func:`select_sample_windows`
    :returns: Multi-line report, including the given sample windows
    """
    lines: list[str] = []
    total = report["total"]
    files = report["files"]
    cpf = report["chunks_per_file"]
    summary = (
        f"Store: {total} chunks across {files} files, {report['total_chars']} chars"
    )
    if cpf:
        summary += f" (chunks/file min={cpf['min']} max={cpf['max']} avg={cpf['avg']})"
    lines.append(summary)

    empty = report["empty"]
    if empty:
        lines.append("")
        lines.append(
            f"Suspiciously short chunks ({len(empty)} total; possible "
            "conversion/OCR miss):"
        )
        for file_name, chars in empty[:MAX_LISTED]:
            lines.append(f"  {file_name}: {chars} chars")

    missing = report["missing_metadata"]
    if missing:
        lines.append("")
        lines.append("Chunks missing bibliographic metadata:")
        for file_name in sorted(missing)[:MAX_LISTED]:
            keys = sorted(set(missing[file_name]))
            lines.append(f"  {file_name}: missing {', '.join(keys)}")

    if report["no_file_name"]:
        lines.append("")
        lines.append(f"Chunks with no file_name: {report['no_file_name']}")

    if report["invalid_content"]:
        lines.append("")
        lines.append(f"Chunks with non-string content: {report['invalid_content']}")

    if report["invalid_year"]:
        lines.append("")
        lines.append(f"Chunks with non-integer year: {report['invalid_year']}")

    if samples:
        lines.append("")
        lines.append("=====================================")
        lines.append(
            f"Sample windows -- {len(samples)} locations x up to "
            f"{len(samples[0])} contiguous chunks"
        )
        lines.append("=====================================")
        for i, window in enumerate(samples, 1):
            lines.append("")
            lines.append(f"--- Location {i} of {len(samples)} ---")
            for j, doc in enumerate(window, 1):
                lines.append("")
                lines.append(f"[{i}.{j}]")
                lines.append(format_chunk_sample(doc))
        lines.append("")
        lines.append("=====================================")

    has_issues = any(
        [
            empty,
            missing,
            report["no_file_name"],
            report["invalid_content"],
            report["invalid_year"],
        ]
    )
    if not has_issues:
        lines.append("")
        lines.append("No issues found.")

    return "\n".join(lines)
