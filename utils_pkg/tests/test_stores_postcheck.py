#!/usr/bin/env python3
"""
Tests for the store post-check (BM25/vector corpus lint).

File: tests/test_stores_postcheck.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import pickle
from pathlib import Path

import pytest
from klea_utils.stores.postcheck import (
    PER_ANCHOR_SAMPLES,
    format_chunk_sample,
    format_store_lint_report,
    lint_store,
    select_sample_windows,
)
from langchain_core.documents import Document
from typer.testing import CliRunner

logger = logging.getLogger(__name__)


def _doc(text, file_name="paper.pdf", **meta):
    """Build a chunk Document with the usual metadata defaults."""
    base = {
        "file_name": file_name,
        "title": "A title",
        "year": 2025,
        "doi": "10.1000/xyz",
        "headings": ["Intro"],
    }
    base.update(meta)
    return Document(page_content=text, metadata=base)


class TestLintStoreSummary:
    """Unit tests for lint_store() corpus summary."""

    def test_summary_counts(self):
        docs = [
            _doc("x" * 100, "a.pdf"),
            _doc("y" * 100, "a.pdf"),
            _doc("z" * 100, "b.pdf"),
        ]
        report = lint_store(docs)
        assert report["total"] == 3
        assert report["files"] == 2
        assert report["total_chars"] == 300
        assert report["chunks_per_file"] == {"min": 1, "max": 2, "avg": 1.5}

    def test_empty_corpus(self):
        report = lint_store([])
        assert report["total"] == 0
        assert report["files"] == 0
        assert report["chunks_per_file"] == {}
        assert report["empty"] == []

    def test_non_string_content_counts_zero_chars(self):
        docs = [_doc("hello")]
        docs[0].page_content = 12345  # type: ignore[assignment]
        report = lint_store(docs)
        assert report["total_chars"] == 0
        assert report["invalid_content"] == 1


class TestLintStoreSuspicious:
    """Unit tests for suspicious-chunk detection."""

    def test_short_chunk_flagged(self):
        docs = [_doc("tiny", "a.pdf"), _doc("x" * 100, "a.pdf")]
        report = lint_store(docs)
        assert ("a.pdf", 4) in report["empty"]

    def test_empty_chunk_flagged(self):
        docs = [_doc("", "a.pdf")]
        report = lint_store(docs)
        assert ("a.pdf", 0) in report["empty"]

    def test_missing_metadata_flagged(self):
        docs = [_doc("x" * 100, "a.pdf", title=None)]
        del docs[0].metadata["title"]
        report = lint_store(docs)
        assert "title" in report["missing_metadata"]["a.pdf"]

    def test_no_missing_metadata_when_complete(self):
        docs = [_doc("x" * 100, "a.pdf")]
        report = lint_store(docs)
        assert "a.pdf" not in report["missing_metadata"]


class TestLintStoreStructural:
    """Unit tests for structural checks."""

    def test_chunk_without_file_name(self):
        docs = [_doc("x" * 100, "a.pdf")]
        del docs[0].metadata["file_name"]
        report = lint_store(docs)
        assert report["no_file_name"] == 1

    def test_invalid_year(self):
        docs = [_doc("x" * 100, "a.pdf", year="2025")]
        report = lint_store(docs)
        assert report["invalid_year"] == 1

    def test_valid_year_not_flagged(self):
        docs = [_doc("x" * 100, "a.pdf", year=2025)]
        report = lint_store(docs)
        assert report["invalid_year"] == 0


class TestSampleRendering:
    """Unit tests for chunk sample rendering."""

    def test_format_chunk_sample_truncates(self):
        doc = _doc("x" * 500, "a.pdf", year=2020)
        sample = format_chunk_sample(doc, width=50)
        assert sample.startswith("x" * 50 + "...")
        assert "file=a.pdf" in sample
        assert "year=2020" in sample

    def test_format_chunk_sample_includes_headings(self):
        doc = _doc("hello", "a.pdf", headings=["Ch 1", "Sec 2"])
        sample = format_chunk_sample(doc)
        assert "headings=Ch 1 > Sec 2" in sample

    def test_select_sample_windows_spreads_across_corpus(self):
        docs = [_doc(f"chunk {i}") for i in range(100)]
        windows = select_sample_windows(docs, anchors=3)
        assert len(windows) == 3
        # Each window is PER_ANCHOR_SAMPLES contiguous chunks.
        assert all(len(w) == PER_ANCHOR_SAMPLES for w in windows)
        # Anchors are spread across the corpus (early / middle / late),
        # roughly at 25% / 50% / 75%.
        starts = [windows[i][0].page_content for i in range(3)]
        assert starts[0] == "chunk 25"
        assert starts[1] == "chunk 50"
        assert starts[2] == "chunk 75"

    def test_select_sample_windows_contiguous(self):
        docs = [_doc(f"chunk {i}") for i in range(100)]
        windows = select_sample_windows(docs, anchors=1)
        # A single window starting at 50% with 3 contiguous chunks.
        assert [w.page_content for w in windows[0]] == [
            "chunk 50",
            "chunk 51",
            "chunk 52",
        ]

    def test_select_sample_windows_small_corpus(self):
        docs = [_doc(f"chunk {i}") for i in range(2)]
        windows = select_sample_windows(docs, anchors=3)
        assert windows == [docs]

    def test_select_sample_windows_zero_or_empty(self):
        assert select_sample_windows([], anchors=3) == []
        docs = [_doc("x")]
        assert select_sample_windows(docs, anchors=0) == []

    def test_report_includes_sample_windows(self):
        docs = [_doc("x" * 100, "a.pdf") for _ in range(10)]
        report = lint_store(docs)
        text = format_store_lint_report(report, samples=select_sample_windows(docs, 1))
        assert "Sample windows" in text
        assert "Location 1 of 1" in text
        assert "No issues found." in text

    def test_report_no_samples_when_empty_list(self):
        docs = [_doc("x" * 100, "a.pdf")]
        report = lint_store(docs)
        text = format_store_lint_report(report, samples=[])
        assert "Sample windows" not in text

    def test_report_lists_issues(self):
        docs = [_doc("tiny", "a.pdf")]
        report = lint_store(docs)
        text = format_store_lint_report(report, samples=[])
        assert "Suspiciously short chunks" in text
        assert "No issues found." not in text


class TestStoreLintCli:
    """Tests for the ``klea-stores-create store-lint`` command."""

    def _write_corpus(self, tmp_path, docs):
        path = tmp_path / "corpus.pkl"
        with open(path, "wb") as f:
            pickle.dump(docs, f)
        return path

    def test_report_output(self, tmp_path):
        from klea_utils.ui.stores_create import app

        corpus = self._write_corpus(
            tmp_path,
            [
                _doc("x" * 100, "a.pdf"),
                _doc("tiny", "b.pdf"),
            ],
        )
        result = CliRunner().invoke(app, ["store-lint", str(corpus)])
        assert result.exit_code == 0, result.output
        assert "2 chunks across 2 files" in result.output
        assert "Sample windows" in result.output
        assert "Suspiciously short chunks" in result.output

    def test_samples_zero_suppresses(self, tmp_path):
        from klea_utils.ui.stores_create import app

        corpus = self._write_corpus(tmp_path, [_doc("x" * 100, "a.pdf")])
        result = CliRunner().invoke(app, ["store-lint", str(corpus), "--samples", "0"])
        assert result.exit_code == 0, result.output
        assert "Sample windows" not in result.output

    def test_missing_corpus_errors(self, tmp_path):
        from klea_utils.ui.stores_create import app

        result = CliRunner().invoke(app, ["store-lint", str(tmp_path / "nope.pkl")])
        assert result.exit_code != 0

    def test_non_list_corpus_errors(self, tmp_path):
        from klea_utils.ui.stores_create import app

        path = tmp_path / "not-list.pkl"
        with open(path, "wb") as f:
            pickle.dump({"not": "a list"}, f)
        result = CliRunner().invoke(app, ["store-lint", str(path)])
        assert result.exit_code != 0

    def test_store_command_auto_prints_report(self, tmp_path, monkeypatch):
        """store auto-prints a store-lint report when it writes a BM25 corpus."""
        from klea_utils.stores.ingestion import StoresBuilder
        from klea_utils.ui.stores_create import app

        source = tmp_path / "src"
        source.mkdir()
        (source / "paper.pdf").write_bytes(b"%PDF fake")

        def fake_resolve(self, source_path, metadata_map_path):
            return {"paper.pdf": {"DEFAULT": {"title": "T", "year": 2025}}}

        def fake_load(self, source_path, metadata_map):
            doc = Document(
                page_content="x" * 100,
                metadata={
                    "file_name": "paper.pdf",
                    "title": "T",
                    "year": 2025,
                    "doi": "10.1000/x",
                    "headings": ["Intro"],
                },
            )
            return [("xxh64:abc", [doc], source_path / "paper.pdf")]

        def fake_store_all(
            self,
            results,
            store_uri,
            collection_name,
            source_path,
            force=False,
            bm25_path: str | None = None,
        ):
            docs = [d for _, docs, _ in results for d in docs]
            if bm25_path is None:
                return
            Path(bm25_path).parent.mkdir(parents=True, exist_ok=True)
            with open(bm25_path, "wb") as f:
                pickle.dump(docs, f)

        monkeypatch.setattr(StoresBuilder, "_resolve_metadata_map", fake_resolve)
        monkeypatch.setattr(StoresBuilder, "_load_and_fold_results", fake_load)
        monkeypatch.setattr(StoresBuilder, "store_all", fake_store_all)

        bm25 = tmp_path / "corpus.pkl"
        result = CliRunner().invoke(
            app,
            [
                "store",
                str(source),
                "--collection",
                "col",
                "--store",
                f"chroma:{tmp_path / 'store'}",
                "--bm25-store",
                str(bm25),
            ],
        )
        assert result.exit_code == 0, result.output
        assert bm25.is_file()
        assert "1 chunks across 1 files" in result.output
        assert "Sample windows" in result.output

    def test_store_command_errors_on_empty_source(self, tmp_path, monkeypatch):
        """store reports "No files" when the source directory is empty.

        _load_and_fold_results is a lazy generator (store_all consumes it
        per file), so the CLI checks emptiness on the source directory
        instead of len() on the results.
        """
        from klea_utils.stores.ingestion import StoresBuilder
        from klea_utils.ui.stores_create import app

        source = tmp_path / "src"
        source.mkdir()

        def fake_resolve(self, source_path, metadata_map_path):
            return {"paper.pdf": {"DEFAULT": {"title": "T"}}}

        monkeypatch.setattr(StoresBuilder, "_resolve_metadata_map", fake_resolve)

        result = CliRunner().invoke(
            app,
            [
                "store",
                str(source),
                "--collection",
                "col",
                "--store",
                f"chroma:{tmp_path / 'store'}",
            ],
        )
        # The guard aborts with a non-zero exit before store_all runs; the
        # message goes through logging (not result.output), so only the
        # exit code is asserted here.
        assert result.exit_code == 1


if __name__ == "__main__":
    pytest.main()
