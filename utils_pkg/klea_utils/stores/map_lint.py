#!/usr/bin/env python3
"""
LLM-free linting of a metadata map.

File: klea_utils/stores/map_lint.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from klea_utils.stores.metadata import (
    ALWAYS_STORED_METADATA_KEYS,
    MACHINE_SET_METADATA_KEYS,
)

logger = logging.getLogger(__name__)

#: Core bibliographic fields a complete DEFAULT entry should carry -- the
#: always-stored schema minus the keys the pipeline sets itself.  Derived
#: (not duplicated) so a change to the stored schema is picked up here.
CORE_FIELDS = tuple(sorted(ALWAYS_STORED_METADATA_KEYS - MACHINE_SET_METADATA_KEYS))

#: DEFAULT entries with more than this many ``url*`` keys almost certainly
#: picked up reference URLs during extraction (a paper typically carries a
#: DOI page, a journal page, and a couple of extras).
URL_WARN_THRESHOLD = 5

#: Earliest plausible publication year.  Anything before this is flagged.
YEAR_MIN = 1800

#: Latest plausible publication year: the current year plus a small
#: margin for in-press/early-access items (a hardcoded far-future cap
#: would go stale as the current year advances).
YEAR_MAX = datetime.now(UTC).year + 2

#: Tokenize a filename stem for the year-vs-filename heuristic, e.g.
#: ``SinhaEtAl2025.pdf`` -> the ``2025`` group.
_STEM_YEAR_RE = re.compile(r"(\d{4})")

#: A DOI is ``prefix/suffix`` -- two segments, neither a URL path tail.
#: Anything containing ``.pdf`` or a second slash is suspicious.
_BOGUS_DOI_RE = re.compile(r"\.pdf\b|/[^/]+/")

#: Titles that indicate the stem-fallback (or an unhelpful extraction)
#: was used instead of a real title.
_SUSPICIOUS_TITLES = {"untitled", "untitled document", "no title", "unknown"}


def _filename_year(file_name: str) -> int | None:
    """Return a 4-digit year found in a source filename stem, if any.

    Looks for a 4-digit group that falls in the plausible publication
    range; ``None`` when the name carries no usable year.
    """
    stem = Path(file_name).stem
    for match in _STEM_YEAR_RE.findall(stem):
        year = int(match)
        if YEAR_MIN <= year <= YEAR_MAX:
            return year
    return None


def lint_file_metadata(file_name: str, entry: dict[str, Any]) -> list[str]:
    """Return human-readable issues for one metadata-map file entry.

    *entry* is the per-file dict from a metadata map: a ``"DEFAULT"``
    metadata dict plus one dict per heading chain (empty ``{}``
    placeholders in a generated template).

    :param file_name: Source filename (used for the year-vs-stem check)
    :param entry: Per-file metadata-map entry
    :returns: Sorted list of issue strings; empty when the entry is clean
    """
    issues: list[str] = []
    default = entry.get("DEFAULT", {})

    missing = [f for f in CORE_FIELDS if f not in default]
    if missing:
        issues.append(f"missing: {', '.join(missing)}")

    title = default.get("title")
    if isinstance(title, str):
        if title.strip().lower() in _SUSPICIOUS_TITLES:
            issues.append(f"suspicious title: {title!r} (looks like a fallback)")
        elif title.strip().lower() == Path(file_name).stem.lower():
            issues.append(
                f"title matches the filename stem: {title!r} (extraction failed?)"
            )

    doi = default.get("doi")
    if isinstance(doi, str) and _BOGUS_DOI_RE.search(doi):
        issues.append(f"suspicious DOI: {doi!r}")

    year = default.get("year")
    if isinstance(year, int):
        if not (YEAR_MIN <= year <= YEAR_MAX):
            issues.append(f"implausible year: {year}")
        else:
            stem_year = _filename_year(file_name)
            if stem_year is not None and stem_year != year:
                issues.append(
                    f"year {year} differs from the year in the filename ({stem_year})"
                )
    elif year is not None and not isinstance(year, int):
        issues.append(f"year is not an integer: {year!r}")

    if "venue" in default and "journal" not in default:
        issues.append("stale 'venue' key: rename to 'journal'")

    url_keys = [k for k in default if k.startswith("url")]
    if len(url_keys) > URL_WARN_THRESHOLD:
        issues.append(
            f"{len(url_keys)} url* keys (exceeds {URL_WARN_THRESHOLD}; "
            "looks like reference URLs leaked into the metadata)"
        )

    return issues


def lint_metadata_map(
    data: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Lint a whole metadata map and return a structured report.

    :param data: Parsed metadata-map JSON (``{file_name: entry}``)
    :returns: dict with ``files`` (total), ``complete`` (count of
        ``_metadata_complete`` DEFAULTs), ``issues`` (``{file_name:
        [issue, ...]}`` for files with at least one issue), and
        ``placeholders`` (``{file_name: int}`` count of empty heading
        placeholders per file)
    """
    issues: dict[str, list[str]] = {}
    placeholders: dict[str, int] = {}
    complete = 0

    for file_name, entry in data.items():
        file_issues = lint_file_metadata(file_name, entry)
        if file_issues:
            issues[file_name] = file_issues

        default = entry.get("DEFAULT", {})
        if default.get("_metadata_complete"):
            complete += 1

        heading_placeholders = sum(
            1 for key, value in entry.items() if key != "DEFAULT" and value == {}
        )
        if heading_placeholders:
            placeholders[file_name] = heading_placeholders

    return {
        "files": len(data),
        "complete": complete,
        "issues": issues,
        "placeholders": placeholders,
    }


def format_metadata_lint_report(report: dict[str, Any]) -> str:
    """Render a :func:`lint_metadata_map` report as compact text."""
    files = report["files"]
    complete = report["complete"]
    issues = report["issues"]
    placeholders = report["placeholders"]

    lines = [
        (
            f"Metadata map: {files} files, {complete} complete, "
            f"{files - complete} need review"
        )
    ]

    if issues:
        lines.append("")
        lines.append("Needs review:")
        for file_name in sorted(issues):
            lines.append(f"  {file_name}")
            for issue in issues[file_name]:
                lines.append(f"    - {issue}")

    if placeholders:
        total = sum(placeholders.values())
        largest = max(placeholders.items(), key=lambda item: item[1])
        lines.append("")
        lines.append(
            f"Empty heading placeholders: {total} across {len(placeholders)} files "
            f"(most: {largest[0]} with {largest[1]}) -- optional, DEFAULT is enough"
        )

    return "\n".join(lines)
