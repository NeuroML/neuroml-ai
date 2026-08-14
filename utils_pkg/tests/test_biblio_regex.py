#!/usr/bin/env python3
"""
Test regex-based bibliographic metadata extraction.

File: tests/test_biblio_regex.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

from klea_utils.biblio.regex import extract_regex_metadata

logger = logging.getLogger(__name__)


def test_keywords_labeled():
    """A labeled keyword line is split into terms."""
    text = "Keywords: worm, NeuroML; c302\nSome body text here."
    result = extract_regex_metadata(text)
    logger.info(f"keywords result: {result}")
    assert result["keywords"] == ["worm", "NeuroML", "c302"]


def test_keywords_label_variants():
    """Key words:, Key words and phrases: and Key terms: are all accepted."""
    for label in ("Key words:", "Key words and phrases:", "Key terms:"):
        result = extract_regex_metadata(f"{label} alpha, beta")
        assert result["keywords"] == ["alpha", "beta"], label


def test_keywords_no_colon_same_line():
    """A 'Keywords' label without a colon, list on the same line."""
    text = "Keywords Agentic AI, Systems biology, Foundational models"
    result = extract_regex_metadata(text)
    assert result["keywords"] == [
        "Agentic AI",
        "Systems biology",
        "Foundational models",
    ]


def test_keywords_uppercase_no_colon_next_line():
    """Uppercase KEYWORDS with the list on the following line."""
    text = "some preamble text.\nKEYWORDS\nC. elegans, connectome\nbody text"
    result = extract_regex_metadata(text)
    assert result["keywords"] == ["C. elegans", "connectome"]


def test_keywords_colon_next_line():
    """Keywords : with the list on the following line."""
    text = (
        "intro.\nKeywords :\nC. elegans, Connectome, Bilateral symmetry\n2 Introduction"
    )
    result = extract_regex_metadata(text)
    assert result["keywords"] == ["C. elegans", "Connectome", "Bilateral symmetry"]


def test_keywords_beyond_default_scan_limit():
    """Keywords past the 3000-char default limit are still found."""
    text = "x" * 3500 + "\nKeywords: worm, NeuroML"
    result = extract_regex_metadata(text)
    assert result["keywords"] == ["worm", "NeuroML"]


def test_keywords_in_prose_not_matched():
    """A prose sentence mentioning keywords is not treated as a list."""
    text = "We studied keywords in this paper and found the results informative."
    result = extract_regex_metadata(text)
    assert "keywords" not in result


def test_authors_labeled():
    """A labeled Author(s): line is split into names."""
    text = "Author(s): Ankur Sinha, Padraig Gleeson"
    result = extract_regex_metadata(text)
    logger.info(f"authors result: {result}")
    assert result["authors"] == ["Ankur Sinha", "Padraig Gleeson"]


def test_authors_by_fallback():
    """A bare By: line is used when no Author(s): header exists."""
    text = "By: John Smith, Jane Doe"
    result = extract_regex_metadata(text)
    assert result["authors"] == ["John Smith", "Jane Doe"]


def test_authors_by_false_positive_guard():
    """'by' without a colon is never treated as an author line."""
    text = "The model was improved by contrast and by design.\nNo authors here."
    result = extract_regex_metadata(text)
    assert "authors" not in result


def test_doi_labeled():
    """A labeled DOI line wins over the loose pattern."""
    text = "DOI: 10.1234/abc.5678."
    result = extract_regex_metadata(text)
    logger.info(f"doi result: {result}")
    assert result["doi"] == "10.1234/abc.5678"


def test_doi_loose_pattern():
    """An unlabeled DOI is found with the loose pattern."""
    text = "Available at https://doi.org/10.1000/xyz.123 or elsewhere."
    result = extract_regex_metadata(text)
    assert result["doi"] == "10.1000/xyz.123"


def test_url_labeled():
    """A labeled URL line is captured."""
    for label in ("URL:", "Website:", "Homepage:"):
        result = extract_regex_metadata(f"{label} https://example.com/paper")
        assert result["url"] == "https://example.com/paper", label


def test_missing_fields_return_empty_dict():
    """Plain text without labeled fields yields an empty dict."""
    text = "Just some plain body text, nothing labeled in here at all."
    assert extract_regex_metadata(text) == {}


def test_scan_limit_truncates_authors_but_not_keywords():
    """Authors are capped at `limit`; keywords use their own wider window."""
    text = ("x" * 3000) + "\nKeywords: worm\nAuthor(s): Hidden"
    result = extract_regex_metadata(text)
    # Keywords have a dedicated, wider scan window...
    assert result["keywords"] == ["worm"]
    # ...while the author header sits beyond the default limit.
    assert "authors" not in result


if __name__ == "__main__":
    import pytest

    pytest.main()
