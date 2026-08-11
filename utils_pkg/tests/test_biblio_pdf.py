#!/usr/bin/env python3
"""
Test PDF bibliographic metadata extraction.

File: tests/test_biblio_pdf.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path

import pytest
from klea_utils.biblio.pdf import extract_pdf_info

logger = logging.getLogger(__name__)

TESTS_DIR = Path(__file__).resolve().parent
ELIFE_PDF = TESTS_DIR / "elife-sinha2025.pdf"
PLOS_CB_PDF = TESTS_DIR / "plos-cb-sinha2021a.pdf"

pytest.importorskip("pypdfium2")


def _escape_literal(value: str) -> str:
    """Escape a string for a PDF literal-string object."""
    return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _build_pdf(path: Path, metadata: dict[str, str] | None = None) -> None:
    """Write a minimal single-page PDF, optionally with an Info dict."""
    objs: list[str] = []
    objs.append("<< /Type /Catalog /Pages 2 0 R >>")
    objs.append("<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    objs.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>")
    if metadata:
        info = (
            "<< "
            + " ".join(
                f"/{key} ({_escape_literal(value)})" for key, value in metadata.items()
            )
            + " >>"
        )
        objs.append(info)

    out = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (len(objs) + 1)
    for index, body in enumerate(objs, start=1):
        offsets[index] = len(out)
        out += f"{index} 0 obj\n{body}\nendobj\n".encode()

    xref_pos = len(out)
    out += f"xref\n0 {len(objs) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()

    trailer = f"<< /Size {len(objs) + 1} /Root 1 0 R"
    if metadata:
        trailer += f" /Info {len(objs)} 0 R"
    trailer += " >>\n"
    out += f"trailer\n{trailer}startxref\n{xref_pos}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def test_elife_pdf_has_no_info_dict():
    """The eLife paper ships no bibliographic PDF metadata."""
    result = extract_pdf_info(str(ELIFE_PDF))
    logger.info(f"eLife metadata: {result}")
    assert result == {}


def test_plos_cb_pdf_has_title_and_author():
    """The PLOS CB paper carries title and author in its Info dict."""
    result = extract_pdf_info(str(PLOS_CB_PDF))
    logger.info(f"PLOS CB metadata: {result}")
    assert result["title"] == (
        "Growth rules for the repair of Asynchronous Irregular neuronal "
        "networks after peripheral lesions"
    )
    assert "Ankur Sinha" in result["author"]
    assert "Volker Steuber" in result["author"]
    # Keywords and Subject are empty in this file, so they are omitted
    assert "keywords" not in result
    assert "subject" not in result


def test_synthetic_pdf_with_full_metadata(tmp_path):
    """A generated PDF with the standard fields returns them all."""
    metadata = {
        "Title": "Synthetic Paper Title",
        "Author": "Jane Doe",
        "Keywords": "alpha; beta",
        "Subject": "Fixture subject",
    }
    pdf_path = tmp_path / "synth.pdf"
    _build_pdf(pdf_path, metadata)

    result = extract_pdf_info(str(pdf_path))
    logger.info(f"synthetic metadata: {result}")
    assert result == {
        "title": "Synthetic Paper Title",
        "author": "Jane Doe",
        "keywords": "alpha; beta",
        "subject": "Fixture subject",
    }


def test_synthetic_pdf_doi_and_url_in_subject(tmp_path):
    """A DOI/URL embedded in the Subject field is picked up as a fallback.

    pypdfium2 only returns the standard Info-dict keys, so custom
    ``DOI``/``URL`` keys are invisible; journals commonly embed the DOI
    in the ``Subject`` value instead.
    """
    metadata = {
        "Subject": "Open access paper, DOI 10.1000/xyz.123. More at https://example.com/x."
    }
    pdf_path = tmp_path / "subject-doi.pdf"
    _build_pdf(pdf_path, metadata)

    result = extract_pdf_info(str(pdf_path))
    logger.info(f"subject-doi metadata: {result}")
    assert result["doi"] == "10.1000/xyz.123"
    assert result["url"] == "https://example.com/x"


def test_synthetic_pdf_without_metadata(tmp_path):
    """A generated PDF with no Info dict returns an empty dict."""
    pdf_path = tmp_path / "bare.pdf"
    _build_pdf(pdf_path)

    result = extract_pdf_info(str(pdf_path))
    logger.info(f"bare metadata: {result}")
    assert result == {}


def test_non_pdf_returns_empty(tmp_path):
    """Non-PDF files are skipped."""
    text_path = tmp_path / "not-a-pdf.txt"
    text_path.write_text("plain text")

    result = extract_pdf_info(str(text_path))
    logger.info(f"non-PDF result: {result}")
    assert result == {}


def test_missing_file_returns_empty(tmp_path):
    """Missing files are skipped."""
    result = extract_pdf_info(str(tmp_path / "nope.pdf"))
    logger.info(f"missing-file result: {result}")
    assert result == {}


if __name__ == "__main__":
    pytest.main()
