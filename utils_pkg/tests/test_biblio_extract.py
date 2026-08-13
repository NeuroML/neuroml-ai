#!/usr/bin/env python3
"""
Test the bibliographic metadata extraction cascade.

File: tests/test_biblio_extract.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path

from docling_core.types.doc.document import (
    BoundingBox,
    DocItemLabel,
    DoclingDocument,
    DocumentOrigin,
    ProvenanceItem,
    Size,
)
from klea_utils.biblio.doi import BiblioRecord
from klea_utils.biblio.extract import extract_metadata, extract_metadata_from_text

logger = logging.getLogger(__name__)

RESOLVED = BiblioRecord(
    title="Resolved Title",
    authors=["Jane Doe", "John Smith"],
    year=2024,
    journal="Journal of Samples",
    abstract="Full abstract text, not to be persisted.",
    doi="10.1234/abc.5678",
)


class _StubResolver:
    """Minimal resolver stub that records the DOIs it is asked about."""

    def __init__(self, record: BiblioRecord | None):
        self.record = record
        self.calls: list[str] = []

    def resolve(self, doi: str) -> BiblioRecord | None:
        self.calls.append(doi)
        return self.record


def _make_doc(
    title: str | None = None,
    header_texts: tuple[str, ...] = (),
    body_texts: tuple[str, ...] = (),
) -> DoclingDocument:
    """Build a synthetic DoclingDocument for testing."""
    doc = DoclingDocument(name="paper")
    doc.add_page(page_no=1, size=Size(width=612, height=792))
    doc.origin = DocumentOrigin(
        mimetype="application/pdf", binary_hash=0, filename="paper.pdf"
    )
    if title:
        doc.add_title(text=title)
    for text in header_texts:
        doc.add_text(
            text=text,
            label=DocItemLabel.TEXT,
            prov=ProvenanceItem(
                page_no=1,
                bbox=BoundingBox(l=50, t=100, r=300, b=120),
                charspan=(0, len(text)),
            ),
        )
    for text in body_texts:
        doc.add_text(text=text, label=DocItemLabel.TEXT)
    return doc


def _escape_literal(value: str) -> str:
    """Escape a string for a PDF literal-string object."""
    return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _build_pdf(path: Path, metadata: dict[str, str]) -> None:
    """Write a minimal single-page PDF with an Info dict."""
    objs: list[str] = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
        "<< "
        + " ".join(
            f"/{key} ({_escape_literal(value)})" for key, value in metadata.items()
        )
        + " >>",
    ]
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
    trailer = f"<< /Size {len(objs) + 1} /Root 1 0 R /Info {len(objs)} 0 R >>\n"
    out += f"trailer\n{trailer}startxref\n{xref_pos}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def test_resolved_record_is_authoritative_and_complete():
    """A resolved record overrides weaker tiers and sets complete=True."""
    doc = _make_doc(
        title="Layout Title",
        body_texts=("DOI: 10.1234/abc.5678", "Keywords: alpha, beta"),
    )
    resolver = _StubResolver(RESOLVED)

    result = extract_metadata(doc, "paper.pdf", resolver=resolver)
    logger.info(f"resolved result: {result}")

    assert resolver.calls == ["10.1234/abc.5678"]
    assert result["title"] == "Resolved Title"
    assert result["authors"] == ["Jane Doe", "John Smith"]
    assert result["year"] == 2024
    assert result["journal"] == "Journal of Samples"
    assert result["doi"] == "10.1234/abc.5678"
    assert "abstract" not in result
    assert result["_metadata_complete"] is True
    assert "doi-service" in result["_sources"]


def test_no_doi_falls_back_to_regex():
    """Without a DOI the regex tiers fill keywords and authors."""
    doc = _make_doc(
        title="Some Paper",
        body_texts=("Keywords: alpha, beta", "By: Jane Doe"),
    )

    result = extract_metadata(doc, "paper.pdf")
    logger.info(f"regex-fallback result: {result}")

    assert result["title"] == "Some Paper"
    assert result["keywords"] == ["alpha", "beta"]
    assert result["authors"] == ["Jane Doe"]
    assert result["_metadata_complete"] is False
    assert "regex" in result["_sources"]


def test_pdf_info_overrides_stem_title(tmp_path):
    """PDF Info fields win over the docling stem fallback."""
    pdf_path = tmp_path / "paper.pdf"
    _build_pdf(pdf_path, {"Title": "Pdf Title", "Author": "Jane Doe"})
    doc = _make_doc()  # no title item -> stem "paper"

    result = extract_metadata(doc, "paper.pdf", pdf_path=str(pdf_path))
    logger.info(f"pdf-info result: {result}")

    assert result["title"] == "Pdf Title"
    assert result["authors"] == ["Jane Doe"]
    assert "pdf-info" in result["_sources"]


def test_resolved_overrides_pdf_and_docling(tmp_path):
    """The DOI record beats both the PDF Info dict and the layout title."""
    pdf_path = tmp_path / "paper.pdf"
    _build_pdf(pdf_path, {"Title": "Pdf Title"})
    doc = _make_doc(title="Layout Title", body_texts=("DOI: 10.1234/abc.5678",))

    result = extract_metadata(
        doc, "paper.pdf", pdf_path=str(pdf_path), resolver=_StubResolver(RESOLVED)
    )
    logger.info(f"precedence result: {result}")

    assert result["title"] == "Resolved Title"


def test_complete_when_full_pdf_info(tmp_path):
    """A PDF Info dict with title+author+keywords sets complete=True."""
    pdf_path = tmp_path / "paper.pdf"
    _build_pdf(
        pdf_path,
        {"Title": "Pdf Title", "Author": "Jane Doe", "Keywords": "alpha, beta"},
    )
    doc = _make_doc()

    result = extract_metadata(doc, "paper.pdf", pdf_path=str(pdf_path))
    logger.info(f"pdf-complete result: {result}")

    assert result["_metadata_complete"] is True
    assert result["keywords"] == ["alpha", "beta"]


def test_layout_region_doi_is_resolved():
    """A DOI in the first-page header region is discovered and resolved."""
    doc = _make_doc(title="T", header_texts=("DOI: 10.2345/def.6789",))
    resolver = _StubResolver(RESOLVED)

    result = extract_metadata(doc, "paper.pdf", resolver=resolver)
    logger.info(f"layout-doi result: {result}")

    assert resolver.calls == ["10.2345/def.6789"]
    assert result["doi"] == "10.1234/abc.5678"


def test_missing_everything():
    """With no signals at all, only the stem title and flags remain."""
    doc = _make_doc()

    result = extract_metadata(doc, "paper.pdf")
    logger.info(f"empty result: {result}")

    assert result["title"] == "paper"
    assert result["_metadata_complete"] is False
    assert result["_sources"] == ["docling"]


def test_from_text_resolves_doi_record():
    """A DOI in the text is resolved and the record is authoritative."""
    resolver = _StubResolver(RESOLVED)
    result = extract_metadata_from_text(
        "Title here. DOI: 10.1234/abc.5678.\nKeywords: alpha, beta",
        "paper.pdf",
        resolver=resolver,
    )
    logger.info(f"text-resolved result: {result}")

    assert resolver.calls == ["10.1234/abc.5678"]
    assert result["title"] == "Resolved Title"
    assert result["authors"] == ["Jane Doe", "John Smith"]
    assert result["year"] == 2024
    assert result["journal"] == "Journal of Samples"
    assert result["doi"] == "10.1234/abc.5678"
    assert result["keywords"] == ["alpha", "beta"]
    assert "abstract" not in result
    assert result["_metadata_complete"] is True
    assert "doi-service" in result["_sources"]


def test_from_text_regex_only_without_doi():
    """Without a DOI, only the regex fields are filled."""
    result = extract_metadata_from_text(
        "Keywords: alpha, beta\nBy: Jane Doe",
        "paper.pdf",
    )
    logger.info(f"text-regex result: {result}")

    assert result["keywords"] == ["alpha", "beta"]
    assert result["authors"] == ["Jane Doe"]
    assert result["_metadata_complete"] is False
    assert result["_sources"] == ["regex"]


def test_from_text_includes_pdf_info(tmp_path):
    """A PDF path adds the pdf-info tier on top of the regex tier."""
    pdf_path = tmp_path / "paper.pdf"
    _build_pdf(
        pdf_path,
        {"Title": "Pdf Title", "Author": "Jane Doe", "Keywords": "alpha, beta"},
    )

    result = extract_metadata_from_text(
        "Some front-matter text without headers.",
        "paper.pdf",
        pdf_path=str(pdf_path),
    )
    logger.info(f"text-pdf result: {result}")

    assert result["title"] == "Pdf Title"
    assert result["authors"] == ["Jane Doe"]
    assert result["keywords"] == ["alpha", "beta"]
    assert result["_metadata_complete"] is True
    assert "pdf-info" in result["_sources"]


def test_from_text_resolved_beats_pdf_and_regex(tmp_path):
    """A resolved record beats both the PDF Info dict and the regex tier."""
    pdf_path = tmp_path / "paper.pdf"
    _build_pdf(pdf_path, {"Title": "Pdf Title"})

    result = extract_metadata_from_text(
        "DOI: 10.1234/abc.5678",
        "paper.pdf",
        pdf_path=str(pdf_path),
        resolver=_StubResolver(RESOLVED),
    )
    logger.info(f"text-precedence result: {result}")

    assert result["title"] == "Resolved Title"


def test_from_text_empty_falls_back_to_stem():
    """Empty text yields only the stem title and flags."""
    result = extract_metadata_from_text("", "paper.pdf")
    logger.info(f"text-empty result: {result}")

    assert result["title"] == "paper"
    assert result["_metadata_complete"] is False
    assert result["_sources"] == []


if __name__ == "__main__":
    import pytest

    pytest.main()
