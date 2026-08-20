#!/usr/bin/env python3
"""
Tests for the LLM-free metadata-map linter.

File: tests/test_stores_map_lint.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from klea_utils.stores.map_lint import (
    format_metadata_lint_report,
    lint_file_metadata,
    lint_metadata_map,
)


def _entry(default: dict, headings: list[str] | None = None) -> dict:
    """Build a per-file map entry from a DEFAULT dict + heading names."""
    entry = {"DEFAULT": default}
    for heading in headings or []:
        entry[heading] = {}
    return entry


def test_clean_entry_has_no_issues():
    issues = lint_file_metadata(
        "SinhaEtAl2025.pdf",
        _entry(
            {
                "title": "A real title",
                "authors": ["A. Sinha"],
                "year": 2025,
                "journal": "eLife",
                "doi": "10.7554/elife.95135",
                "keywords": ["neuroscience"],
                "url_1": "https://doi.org/10.7554/elife.95135",
            },
            ["Introduction"],
        ),
    )
    assert issues == []


def test_missing_core_fields_listed():
    issues = lint_file_metadata("ChenEtAl2006.pdf", _entry({"title": "Something"}))
    assert any("missing:" in issue for issue in issues)
    assert "authors" in issues[0]
    assert "year" in issues[0]


def test_suspicious_title_stem_fallback():
    issues = lint_file_metadata(
        "WhiteEtAl1986.pdf",
        _entry({"title": "WhiteEtAl1986"}),
    )
    assert any("filename stem" in issue for issue in issues)

    issues = lint_file_metadata(
        "GleesonEtAl2018.pdf",
        _entry({"title": "untitled"}),
    )
    assert any("fallback" in issue for issue in issues)


def test_suspicious_doi():
    issues = lint_file_metadata(
        "GleesonEtAl2018.pdf",
        _entry({"doi": "10.1098/rstb.2017.0379/254766/rstb.2017.0379.pdf"}),
    )
    assert any("suspicious DOI" in issue for issue in issues)


def test_year_mismatch_with_filename_stem():
    issues = lint_file_metadata(
        "SinhaEtAl2025.pdf",
        _entry({"title": "Real title", "year": 2024}),
    )
    assert any("differs from the year" in issue for issue in issues)


def test_implausible_year():
    issues = lint_file_metadata(
        "Foo2025.pdf",
        _entry({"title": "Real title", "year": 1700}),
    )
    assert any("implausible year" in issue for issue in issues)


def test_future_year_beyond_cap_flagged():
    from klea_utils.stores.map_lint import YEAR_MAX

    issues = lint_file_metadata(
        "Foo2025.pdf",
        _entry({"title": "Real title", "year": YEAR_MAX + 10}),
    )
    assert any("implausible year" in issue for issue in issues)


def test_year_not_integer():
    issues = lint_file_metadata(
        "Foo2025.pdf", _entry({"title": "Real title", "year": "2025"})
    )
    assert any("not an integer" in issue for issue in issues)


def test_stale_venue_key():
    issues = lint_file_metadata(
        "Foo2025.pdf",
        _entry({"title": "Real title", "venue": "eLife"}),
    )
    assert any("venue" in issue and "journal" in issue for issue in issues)


def test_excess_url_keys():
    issues = lint_file_metadata(
        "Molina2025.pdf",
        _entry(
            {
                "title": "Real title",
                **{f"url_{i}": f"https://ref/{i}" for i in range(1, 9)},
            }
        ),
    )
    assert any("url* keys" in issue for issue in issues)


def test_lint_metadata_map_report():
    data = {
        "Good2025.pdf": {
            "DEFAULT": {
                "title": "Good",
                "authors": ["A"],
                "year": 2025,
                "journal": "J",
                "doi": "10.1/x",
                "keywords": ["k"],
                "_metadata_complete": True,
            }
        },
        "Bad2025.pdf": {
            "DEFAULT": {"title": "untitled", "year": 2024},
            "Intro": {},
            "Methods": {},
        },
    }
    report = lint_metadata_map(data)
    assert report["files"] == 2
    assert report["complete"] == 1
    assert "Bad2025.pdf" in report["issues"]
    assert report["placeholders"] == {"Bad2025.pdf": 2}
    assert report["missing_keys"] == []
    assert report["unknown_keys"] == []


def test_lint_metadata_map_with_matching_source_files_passes():
    data = {
        "Good2025.pdf": {"DEFAULT": {"title": "Good", "authors": ["A"], "year": 2025}}
    }
    report = lint_metadata_map(data, source_files=["Good2025.pdf"])
    assert report["missing_keys"] == []
    assert report["unknown_keys"] == []


def test_lint_metadata_map_reports_missing_and_unknown_keys():
    data = {
        "Chapter > Section": {"title": "Heading-keyed"},
        "Open Source Brain": {"title": "Another heading"},
        "Good2025.pdf": {"DEFAULT": {"title": "Good"}},
    }
    report = lint_metadata_map(
        data,
        source_files=["Good2025.pdf", "Missing2025.pdf"],
    )
    assert report["missing_keys"] == ["Missing2025.pdf"]
    assert report["unknown_keys"] == ["Chapter > Section", "Open Source Brain"]


def test_lint_metadata_map_skips_key_checks_without_source_files():
    data = {"Chapter > Section": {"title": "Heading-keyed"}}
    report = lint_metadata_map(data)
    assert report["missing_keys"] == []
    assert report["unknown_keys"] == []


def test_old_flat_format_entry_reports_structural_issue():
    data = {
        "Open Source Brain": {
            "title": "Open Source Brain",
            "url": "https://docs.opensourcebrain.org",
            "authors": ["A. Sinha"],
            "year": 2026,
            "keywords": ["neuroscience"],
        }
    }
    report = lint_metadata_map(data)
    issues = report["issues"]["Open Source Brain"]
    assert any("old flat format" in issue for issue in issues)
    assert not any("missing:" in issue for issue in issues)

    issues = lint_file_metadata(
        "Open Source Brain",
        {"title": "Open Source Brain", "url": "https://docs.opensourcebrain.org"},
    )
    assert any("old flat format" in issue for issue in issues)


def test_format_metadata_lint_report_shows_key_sections():
    report = lint_metadata_map(
        {
            "Chapter > Section": {"title": "Heading-keyed"},
            "Good2025.pdf": {"DEFAULT": {"title": "Good"}},
        },
        source_files=["Good2025.pdf", "Missing2025.pdf"],
    )
    text = format_metadata_lint_report(report)
    assert "Source files with no entry (store will FAIL): 1" in text
    assert "Missing2025.pdf" in text
    assert "Map keys that are not source files (heading-keyed or stale): 1" in text
    assert "Chapter > Section" in text


def test_format_metadata_lint_report_mentions_issues_and_placeholders(capsys):
    report = lint_metadata_map(
        {
            "Bad2025.pdf": {
                "DEFAULT": {"title": "untitled"},
                "Intro": {},
            }
        }
    )
    text = format_metadata_lint_report(report)
    assert "1 files, 0 complete, 1 need review" in text
    assert "Needs review:" in text
    assert "Bad2025.pdf" in text
    assert "Empty heading placeholders:" in text
