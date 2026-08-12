#!/usr/bin/env python3
"""
Test vector store ingestion.

File: tests/test_stores_ingestion.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import pickle
import tempfile
from pathlib import Path

import pytest
from klea_utils.stores.ingestion import StoresBuilder
from klea_utils.stores.utils import instantiate_vector_store
from langchain_core.documents import Document
from ollama import ResponseError

TEST_MD_CONTENT = """# Test Document
## Section 1
This is some content in section 1. It has enough text to produce at least
one chunk for the vector store.

## Section 2
This is content in section 2. It also has enough text to produce at least
one chunk for the vector store. We need enough text here to make sure
the chunker actually produces some chunks.
"""

TEST_MD_TWO = """# Second Document
## Overview
This is another test file. It will be used to test incremental
ingestion where new files are added to an existing store.
"""


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


class TestIngestion:
    """Test vector store ingestion."""

    def setup_method(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.tmpdir_path = Path(self.tmpdir.name)
        self.logger = logging.getLogger("test_ingestion")

    def teardown_method(self):
        self.tmpdir.cleanup()

    @pytest.mark.localonly
    def test_build_chroma(self):
        """Test building a chroma store from a source directory."""
        md_file = self.tmpdir_path / "test.md"
        md_file.write_text(TEST_MD_CONTENT)
        self.logger.info(f"Wrote test file: {md_file}")

        store_dir = self.tmpdir_path / "chroma_store"
        store_uri = f"chroma:{store_dir}"
        collection_name = "test_build"
        self.logger.info(f"Store URI: {store_uri}, collection: {collection_name}")

        try:
            builder = StoresBuilder(
                embedding_model="ollama:bge-m3:latest",
                logger=self.logger,
            )
            self.logger.info("Builder set up with Ollama embeddings")
            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Build completed")

            store = instantiate_vector_store(
                store_uri, collection_name, builder.embeddings, self.logger
            )
            result = store.get()
            self.logger.info(f"Store contains {len(result['ids'])} chunks")
            assert result["ids"], "No documents found in store"
            assert len(result["ids"]) > 0

            hnsw_space = store._chroma_collection.configuration["hnsw"]["space"]
            self.logger.info(f"Chroma HNSW space: {hnsw_space}")
            assert hnsw_space == "cosine"

            first_meta = result["metadatas"][0]
            assert "file_hash" in first_meta
            assert "file_name" in first_meta
            assert "source_path" in first_meta
            assert "headings" in first_meta
            assert first_meta["file_name"] == "test.md"
            self.logger.info(
                f"First chunk metadata verified: headings={first_meta['headings']}"
            )

        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))

    @pytest.mark.localonly
    def test_build_idempotent(self):
        """Test that building twice produces the same result."""
        md_file = self.tmpdir_path / "test.md"
        md_file.write_text(TEST_MD_CONTENT)
        self.logger.info(f"Wrote test file: {md_file}")

        store_dir = self.tmpdir_path / "chroma_store"
        store_uri = f"chroma:{store_dir}"
        collection_name = "test_idempotent"

        try:
            builder = StoresBuilder(
                embedding_model="ollama:bge-m3:latest",
                logger=self.logger,
            )
            self.logger.info("Builder set up with Ollama embeddings")

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("First build completed")

            store = instantiate_vector_store(
                store_uri, collection_name, builder.embeddings, self.logger
            )
            first_result = store.get()
            first_count = len(first_result["metadatas"])
            self.logger.info(f"First build: {first_count} chunks")

            hnsw_space = store._chroma_collection.configuration["hnsw"]["space"]
            self.logger.info(f"Chroma HNSW space: {hnsw_space}")
            assert hnsw_space == "cosine"

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Second (idempotent) build completed")

            second_result = store.get()
            second_count = len(second_result["metadatas"])
            self.logger.info(f"Second build: {second_count} chunks")

            assert first_count == second_count, (
                f"Expected same count after idempotent build "
                f"({first_count} != {second_count})"
            )
            self.logger.info("Idempotency verified -- counts match")

        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))

    @pytest.mark.localonly
    def test_build_incremental(self):
        """Test adding a new file after initial build."""
        md_file = self.tmpdir_path / "test.md"
        md_file.write_text(TEST_MD_CONTENT)
        self.logger.info(f"Wrote initial test file: {md_file}")

        store_dir = self.tmpdir_path / "chroma_store"
        store_uri = f"chroma:{store_dir}"
        collection_name = "test_incremental"

        try:
            builder = StoresBuilder(
                embedding_model="ollama:bge-m3:latest",
                logger=self.logger,
            )
            self.logger.info("Builder set up with Ollama embeddings")

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Initial build completed")

            store = instantiate_vector_store(
                store_uri, collection_name, builder.embeddings, self.logger
            )
            first_count = len(store.get()["metadatas"])
            self.logger.info(f"Initial build: {first_count} chunks")

            md_two = self.tmpdir_path / "another.md"
            md_two.write_text(TEST_MD_TWO)
            self.logger.info(f"Added new test file: {md_two}")

            builder.build(
                source_dir=str(self.tmpdir_path),
                store_uri=store_uri,
                collection_name=collection_name,
            )
            self.logger.info("Incremental build completed")

            second_count = len(store.get()["metadatas"])
            self.logger.info(f"Incremental build: {second_count} chunks")

            assert second_count > first_count, (
                f"Expected more chunks after incremental build "
                f"({second_count} <= {first_count})"
            )
            self.logger.info(
                f"Incremental verified -- added {second_count - first_count} more chunks"
            )

        except ResponseError as e:
            pytest.skip(str(e))
        except ConnectionError as e:
            pytest.skip(str(e))

    def test_do_ocr_defaults_to_true(self):
        """OCR is enabled by default, preserving the previous behaviour."""
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        assert builder.do_ocr is True

    def test_do_ocr_false_constructs_converter(self):
        """do_ocr=False is stored and builds a converter without error."""
        builder = StoresBuilder(embedding_model="", logger=self.logger, do_ocr=False)
        assert builder.do_ocr is False
        converter = builder._get_converter()
        assert converter is not None
        assert converter is builder._get_converter()

    def test_cache_round_trips_extracted_metadata(self):
        """Cached entries restore both chunks and extracted metadata."""
        doc = Document(
            page_content="Some content.", metadata={"headings": ["Section 1"]}
        )
        extracted = {
            "title": "T",
            "_metadata_complete": True,
            "_sources": ["docling"],
        }
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        builder._save_to_cache([doc], extracted, self.tmpdir_path, "xxh64:abc")

        loaded = builder._load_from_cache(self.tmpdir_path, "xxh64:abc")
        assert loaded is not None
        docs, restored = loaded
        assert docs[0].page_content == "Some content."
        assert restored == extracted

    def test_cache_load_handles_legacy_format(self):
        """Legacy plain-list cache entries load with empty extraction."""
        doc = Document(page_content="Legacy.", metadata={})
        cache_dir = self.tmpdir_path / ".klea-cache"
        cache_dir.mkdir()
        with open(cache_dir / "xxh64_legacy.pkl", "wb") as f:
            pickle.dump([doc], f)

        builder = StoresBuilder(embedding_model="", logger=self.logger)
        loaded = builder._load_from_cache(self.tmpdir_path, "xxh64:legacy")
        assert loaded == ([doc], {})

    @pytest.mark.localonly
    def test_chunk_all_prefills_metadata_template(self):
        """chunk_all pre-fills the per-file DEFAULT template entry.

        Runs the biblio extraction cascade on a real PDF conversion: the
        PDF Info dict (title/author/keywords) lands in the DEFAULT entry
        along with the internal ``_metadata_complete`` / ``_sources``
        flags.  A re-run hits the cache and must restore the same
        extraction rather than degrading to regex-only.
        """
        _build_pdf(
            self.tmpdir_path / "paper.pdf",
            {
                "Title": "Synthetic Title",
                "Author": "Jane Doe",
                "Keywords": "alpha, beta",
            },
        )

        builder = StoresBuilder(embedding_model="", logger=self.logger, do_ocr=False)
        _, file_headings = builder.chunk_all(self.tmpdir_path)

        default = file_headings["paper.pdf"]["DEFAULT"]
        self.logger.info(f"pre-filled DEFAULT: {default}")
        assert default["title"] == "Synthetic Title"
        assert default["authors"] == ["Jane Doe"]
        assert default["keywords"] == ["alpha", "beta"]
        assert default["_metadata_complete"] is True
        assert "pdf-info" in default["_sources"]

        # Cache hit: the persisted full extraction is restored unchanged.
        _, file_headings2 = builder.chunk_all(self.tmpdir_path)
        assert file_headings2["paper.pdf"]["DEFAULT"] == default

    def test_write_bm25_store(self):
        """write_bm25_store pickles the combined chunked documents."""
        doc = Document(
            page_content="Some content for the BM25 corpus.",
            metadata={"file_name": "test.md", "headings": ["Section 1"]},
        )
        results = [("xxh64:abc", [doc], Path("test.md"))]
        out_path = self.tmpdir_path / "bm25_corpus.pkl"

        builder = StoresBuilder(embedding_model="", logger=self.logger)
        builder.write_bm25_store(results, str(out_path))

        assert out_path.is_file()
        with open(out_path, "rb") as f:
            loaded = pickle.load(f)
        assert len(loaded) == 1
        assert loaded[0].page_content == doc.page_content
        assert loaded[0].metadata == doc.metadata

    def test_write_bm25_store_empty(self):
        """write_bm25_store skips without creating a file for empty results."""
        out_path = self.tmpdir_path / "empty_bm25_corpus.pkl"

        builder = StoresBuilder(embedding_model="", logger=self.logger)
        builder.write_bm25_store([], str(out_path))

        assert not out_path.exists()


if __name__ == "__main__":
    pytest.main()
