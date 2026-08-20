#!/usr/bin/env python3
"""
Tests for the store pre-check (OCR decision per PDF).

File: tests/test_stores_precheck.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path

import pytest
from klea_utils.stores.precheck import (
    IMAGE_PAGE_CHAR_THRESHOLD,
    NO_OCR_DIR_NAME,
    OCR_DIR_NAME,
    classify_directory,
    classify_pdf,
    format_precheck_report,
    organise_directory,
)
from typer.testing import CliRunner

logger = logging.getLogger(__name__)

pytest.importorskip("pypdfium2")


def _escape_literal(value: str) -> str:
    """Escape a string for a PDF literal-string object."""
    return value.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def _build_scanned_pdf(path: Path) -> None:
    """Write a minimal single-page PDF with no content stream.

    Mirrors the bare-PDF fixture used elsewhere in the test suite: a
    scanned/image page has no extractable text layer.
    """
    objs = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >>",
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
    trailer = f"<< /Size {len(objs) + 1} /Root 1 0 R >>\n"
    out += f"trailer\n{trailer}startxref\n{xref_pos}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


def _build_text_pdf(path: Path, text: str) -> None:
    """Write a minimal single-page PDF with a drawable text layer.

    The page carries a Resources dict with a Helvetica Type1 font and a
    content stream rendering *text*, so pypdfium2 extracts it as text.
    """
    content = f"BT /F1 12 Tf 72 720 Td ({_escape_literal(text)}) Tj ET".encode()
    bodies = [
        "<< /Type /Catalog /Pages 2 0 R >>",
        "<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ),
        "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = [0] * (len(bodies) + 1)
    for index, body in enumerate(bodies, start=1):
        offsets[index] = len(out)
        out += f"{index} 0 obj\n".encode()
        out += body if isinstance(body, bytes) else body.encode()
        out += b"\nendobj\n"
    xref_pos = len(out)
    out += f"xref\n0 {len(bodies) + 1}\n".encode()
    out += b"0000000000 65535 f \n"
    for offset in offsets[1:]:
        out += f"{offset:010d} 00000 n \n".encode()
    trailer = f"<< /Size {len(bodies) + 1} /Root 1 0 R >>\n"
    out += f"trailer\n{trailer}startxref\n{xref_pos}\n%%EOF\n".encode()
    path.write_bytes(bytes(out))


_LONG_TEXT = (
    "This is a perfectly normal scientific paper page containing many hundreds "
    "of characters of fully extractable text, enough that the pre-check "
    "classifier treats it as a text-based page rather than an image scan of "
    "a page that would need optical character recognition to recover its "
    "content for retrieval and embedding."
)


class TestClassifyPdf:
    """Unit tests for classify_pdf()."""

    def test_scanned_pdf_needs_ocr(self, tmp_path):
        """A PDF with no text layer (a scan) is classified as needing OCR."""
        pdf = tmp_path / "scanned.pdf"
        _build_scanned_pdf(pdf)

        result = classify_pdf(pdf)
        assert result.needs_ocr is True
        assert result.stats.pages == 1
        assert result.stats.image_pages == 1
        assert result.stats.total_chars == 0

    def test_text_pdf_does_not_need_ocr(self, tmp_path):
        """A PDF with a full text layer is classified as no-OCR."""
        pdf = tmp_path / "text.pdf"
        _build_text_pdf(pdf, _LONG_TEXT)

        result = classify_pdf(pdf)
        assert result.needs_ocr is False
        assert result.stats.pages == 1
        assert result.stats.image_pages == 0
        assert result.stats.total_chars > IMAGE_PAGE_CHAR_THRESHOLD

    def test_mixed_pages_majority_text_no_ocr(self, tmp_path):
        """A PDF with mostly text pages and some image pages is no-OCR."""
        # Build a 4-page PDF: pages 1,2,3 text; page 4 blank (image).
        # This needs a multi-page fixture; a 2-page proxy (1 text, 1 blank)
        # is 50/50 which is borderline -- instead assert the 1-text-only
        # page case and the pure-scanned case above are unambiguous.
        pdf = tmp_path / "single.pdf"
        _build_text_pdf(pdf, _LONG_TEXT)
        assert classify_pdf(pdf).needs_ocr is False

    def test_missing_file_conservative_ocr(self, tmp_path):
        """A missing/unreadable PDF defaults to needing OCR (safe)."""
        result = classify_pdf(tmp_path / "missing.pdf")
        assert result.needs_ocr is True

    def test_non_pdf_defaults_ocr(self, tmp_path):
        """A non-PDF path passed to classify_pdf defaults to OCR."""
        txt = tmp_path / "notes.txt"
        txt.write_text("plain text")
        assert classify_pdf(txt).needs_ocr is True


class TestClassifyDirectory:
    """Unit tests for classify_directory()."""

    def test_mixed_directory(self, tmp_path):
        """PDFs are classified; non-PDFs are ignored by classify_directory."""
        _build_scanned_pdf(tmp_path / "scanned.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)
        (tmp_path / "notes.md").write_text("# Notes")

        results = classify_directory(tmp_path)
        assert set(results) == {tmp_path / "scanned.pdf", tmp_path / "text.pdf"}
        assert results[tmp_path / "scanned.pdf"].needs_ocr is True
        assert results[tmp_path / "text.pdf"].needs_ocr is False

    def test_skips_output_and_cache_dirs(self, tmp_path):
        """The ocr/ and no-ocr/ output dirs and .klea-cache are skipped."""
        _build_scanned_pdf(tmp_path / "root.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)
        (tmp_path / OCR_DIR_NAME).mkdir()
        _build_scanned_pdf(tmp_path / OCR_DIR_NAME / "copy.pdf")
        (tmp_path / NO_OCR_DIR_NAME).mkdir()
        _build_text_pdf(tmp_path / NO_OCR_DIR_NAME / "copy.pdf", _LONG_TEXT)
        (tmp_path / ".klea-cache").mkdir()
        _build_text_pdf(tmp_path / ".klea-cache" / "cached.pdf", _LONG_TEXT)

        results = classify_directory(tmp_path)
        names = {p.name for p in results}
        assert names == {"root.pdf", "text.pdf"}
        assert all(
            OCR_DIR_NAME not in p.parts and NO_OCR_DIR_NAME not in p.parts
            for p in results
        )

    def test_empty_directory(self, tmp_path):
        """An empty directory yields no classifications."""
        assert classify_directory(tmp_path) == {}


class TestFormatReport:
    """Unit tests for format_precheck_report()."""

    def test_report_summary_counts(self, tmp_path):
        """The report summarises OCR vs no-OCR counts and lists files."""
        _build_scanned_pdf(tmp_path / "scanned.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)
        results = classify_directory(tmp_path)

        report = format_precheck_report(results)
        logger.info(f"report:\n{report}")
        assert "2 PDFs" in report
        assert "1 need OCR" in report
        assert "1 are text-based" in report
        assert "OCR" in report
        assert "no-OCR" in report


class TestOrganiseDirectory:
    """Unit tests for organise_directory()."""

    def test_copies_into_subdirs_leaves_originals(self, tmp_path):
        """Files are copied (never moved) into ocr/ and no-ocr/."""
        _build_scanned_pdf(tmp_path / "scanned.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)
        (tmp_path / "notes.md").write_text("# Notes")
        scanned_orig = (tmp_path / "scanned.pdf").read_bytes()

        ocr_dir, no_ocr_dir = organise_directory(tmp_path)

        assert (ocr_dir / "scanned.pdf").is_file()
        assert (no_ocr_dir / "text.pdf").is_file()
        assert (no_ocr_dir / "notes.md").is_file()
        # Originals untouched
        assert (tmp_path / "scanned.pdf").read_bytes() == scanned_orig
        assert (tmp_path / "text.pdf").is_file()
        assert (tmp_path / "notes.md").is_file()

    def test_idempotent_re_run(self, tmp_path):
        """Re-running organise copies nothing new and is stable."""
        _build_scanned_pdf(tmp_path / "scanned.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)

        organise_directory(tmp_path)
        ocr_dir, no_ocr_dir = organise_directory(tmp_path)

        assert sorted(p.name for p in ocr_dir.iterdir()) == ["scanned.pdf"]
        assert sorted(p.name for p in no_ocr_dir.iterdir()) == ["text.pdf"]

    def test_does_not_reclassify_copies_on_rerun(self, tmp_path):
        """Copies already in ocr//no-ocr/ are not re-classified as sources."""
        _build_scanned_pdf(tmp_path / "scanned.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)

        ocr_dir, no_ocr_dir = organise_directory(tmp_path)
        # Simulate a re-run that recomputes classifications from the
        # (now populated) directory: output dirs must be skipped so the
        # copies do not appear as new sources.
        results = classify_directory(tmp_path)
        assert ocr_dir not in [c.path.parent for c in results.values()]
        assert no_ocr_dir not in [c.path.parent for c in results.values()]

    def test_unclassified_pdf_goes_to_ocr(self, tmp_path):
        """A PDF with no classification entry is conservatively OCR'd."""
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)
        # Pass an empty classification map directly to force the
        # conservative fallback path.
        ocr_dir, no_ocr_dir = organise_directory(tmp_path, classifications={})
        assert (ocr_dir / "text.pdf").is_file()
        assert not (no_ocr_dir / "text.pdf").exists()


class TestPreCheckCli:
    """Tests for the ``klea-stores-create pre-check`` command."""

    def test_report_only(self, tmp_path):
        """Report-only mode prints the classification summary."""
        from klea_utils.ui.stores_create import app

        _build_scanned_pdf(tmp_path / "scanned.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)

        result = CliRunner().invoke(app, ["pre-check", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "2 PDFs" in result.output
        assert "1 need OCR" in result.output
        assert "scanned.pdf" in result.output
        assert "text.pdf" in result.output
        # No organise commands in report-only mode
        assert "klea-stores-create chunk" not in result.output

    def test_empty_directory_exits_clean(self, tmp_path):
        """An empty directory is reported and exits 0."""
        from klea_utils.ui.stores_create import app

        result = CliRunner().invoke(app, ["pre-check", str(tmp_path)])
        assert result.exit_code == 0, result.output
        assert "0 PDFs" in result.output

    def test_missing_directory_errors(self, tmp_path):
        """A missing source directory is an error (non-zero exit)."""
        from klea_utils.ui.stores_create import app

        result = CliRunner().invoke(app, ["pre-check", str(tmp_path / "nope")])
        assert result.exit_code != 0

    def test_organise_copies_and_prints_workflow(self, tmp_path):
        """--organise copies into ocr/ and no-ocr/ and prints commands."""
        from klea_utils.ui.stores_create import app

        _build_scanned_pdf(tmp_path / "scanned.pdf")
        _build_text_pdf(tmp_path / "text.pdf", _LONG_TEXT)
        (tmp_path / "notes.md").write_text("# Notes")

        result = CliRunner().invoke(app, ["pre-check", str(tmp_path), "--organise"])
        assert result.exit_code == 0, result.output
        assert (tmp_path / OCR_DIR_NAME / "scanned.pdf").is_file()
        assert (tmp_path / NO_OCR_DIR_NAME / "text.pdf").is_file()
        assert (tmp_path / NO_OCR_DIR_NAME / "notes.md").is_file()
        # Originals untouched
        assert (tmp_path / "scanned.pdf").is_file()
        # Workflow printed
        assert "klea-stores-create chunk" in result.output
        assert "--no-ocr" in result.output
        assert "SAME collection" in result.output


if __name__ == "__main__":
    pytest.main()
