#!/usr/bin/env python3
"""
Test vector store ingestion.

File: tests/test_stores_ingestion.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import pickle
import tempfile
from pathlib import Path

import pytest
from klea_utils.stores.ingestion import (
    TEMPLATE_FILE_NAME,
    StoresBuilder,
    _ensure_doi_url,
    _normalize_extracted_metadata,
    _split_url_list,
)
from klea_utils.stores.utils import instantiate_vector_store, normalize_text
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


class TestNormalizeText:
    """Unit tests for normalize_text()."""

    def test_soft_hyphen_rejoins_split_words(self):
        """Soft hyphens (PDF line-break artifacts) are dropped in place."""
        assert normalize_text("multi-\u00adscale") == "multi-scale"
        assert (
            normalize_text("Multi-\u00adscale Model\u00ading") == "Multi-scale Modeling"
        )

    def test_no_break_spaces_become_regular_spaces(self):
        """No-break space variants map to a regular space."""
        assert normalize_text("in\u00a0neuroscience") == "in neuroscience"
        assert normalize_text("a\u2007b\u202fc") == "a b c"

    def test_zero_width_and_bom_stripped(self):
        """BOM and zero-width characters are removed entirely."""
        assert normalize_text("\ufeffleading\u200bedge") == "leadingedge"

    def test_bidi_and_variation_selectors_stripped(self):
        """Bidi marks and variation selectors are removed entirely."""
        assert normalize_text("a\u200eb\u200fc\ufe0e") == "abc"

    def test_ligatures_folded(self):
        """NFKC folds ligatures into their plain letters."""
        assert normalize_text("e\ufb00icient") == "efficient"
        assert normalize_text("\ufb01le") == "file"
        assert normalize_text("\ufb00") == "ff"

    def test_fullwidth_folded(self):
        """NFKC folds full-width forms to ASCII."""
        assert normalize_text("\uff21\uff10\uff08") == "A0("

    def test_superscripts_folded(self):
        """NFKC folds superscript digits to plain digits."""
        assert normalize_text("10\u2070\u00b9\u00b2") == "10012"

    def test_typographic_spaces_folded(self):
        """NFKC folds en/em/thin/ideographic spaces to a regular space."""
        assert normalize_text("a\u2002b\u2003c\u3000d") == "a b c d"

    def test_non_breaking_hyphen_folded(self):
        """NFKC folds the non-breaking hyphen to a regular hyphen."""
        assert normalize_text("a\u2011b") == "a\u2010b"

    def test_typographic_dashes_preserved(self):
        """Em/en dashes are meaningful punctuation and are kept unchanged."""
        assert normalize_text("a\u2013b\u2014c") == "a\u2013b\u2014c"

    def test_nfc_composition(self):
        """Canonical unicode composition (e.g. combining accents)."""
        assert normalize_text("e\u0301") == "\u00e9"
        assert normalize_text("cafe\u0301") == "caf\u00e9"

    def test_collapses_whitespace_and_strips(self):
        """Repeated spaces/tabs collapse and edges are trimmed."""
        assert normalize_text("  a\t\tb   c  ") == "a b c"

    def test_plain_text_unchanged(self):
        """Already-clean text passes through untouched."""
        assert (
            normalize_text("Plain text stays the same") == "Plain text stays the same"
        )
        assert normalize_text("") == ""

    def test_normalize_extracted_metadata(self):
        """String fields are normalized; non-string fields pass through."""
        extracted = {
            "title": "Multi-\u00adscale Model\u00ading",
            "authors": ["A\xa0Sinha", "Jane Doe"],
            "year": 2025,
            "urls": ["https://example.org/x"],
            "_metadata_complete": True,
        }
        out = _normalize_extracted_metadata(extracted)

        assert out["title"] == "Multi-scale Modeling"
        assert out["authors"] == ["A Sinha", "Jane Doe"]
        assert out["year"] == 2025
        assert out["urls"] == ["https://example.org/x"]
        assert out["_metadata_complete"] is True
        # The original dict is not mutated.
        assert extracted["title"] == "Multi-\u00adscale Model\u00ading"

    def test_split_url_list(self):
        """A urls list is expanded into url_1/url_2 keys."""
        out = _split_url_list(
            {"title": "T", "urls": ["https://a", "https://b", "https://c"]}
        )
        assert out == {
            "title": "T",
            "url_1": "https://a",
            "url_2": "https://b",
            "url_3": "https://c",
        }

    def test_split_url_list_keeps_existing_url_keys(self):
        """A singular url/source_url key is left untouched and not reused."""
        out = _split_url_list(
            {
                "url": "https://pdf",
                "source_url": "https://source",
                "urls": ["https://a", "https://b"],
            }
        )
        assert out["url"] == "https://pdf"
        assert out["source_url"] == "https://source"
        assert out["url_1"] == "https://a"
        assert out["url_2"] == "https://b"

    def test_split_url_list_skips_taken_indices(self):
        """Numbering skips indices already present in the metadata."""
        out = _split_url_list({"url_1": "https://taken", "urls": ["https://a"]})
        assert out == {"url_1": "https://taken", "url_2": "https://a"}

    def test_split_url_list_drops_empty_or_absent(self):
        """An empty/absent urls list leaves no urls keys behind."""
        assert _split_url_list({"title": "T", "urls": []}) == {"title": "T"}
        assert _split_url_list({"title": "T"}) == {"title": "T"}

    def test_ensure_doi_url_derived(self):
        """url_doi is derived from the doi field as a resolvable URL."""
        out = _ensure_doi_url({"title": "T", "doi": "10.7554/elife.95135"})
        assert out["url_doi"] == "https://doi.org/10.7554/elife.95135"
        assert out["doi"] == "10.7554/elife.95135"

    def test_ensure_doi_url_keeps_existing_url_doi(self):
        """A researcher-provided url_doi wins over the derived value."""
        out = _ensure_doi_url({"doi": "10.x/y", "url_doi": "https://custom"})
        assert out["url_doi"] == "https://custom"

    def test_ensure_doi_url_no_doi_unchanged(self):
        """No doi field means no url_doi key is added."""
        out = _ensure_doi_url({"title": "T"})
        assert out == {"title": "T"}


class _FakeMeta:
    """Minimal stand-in for a chunk's pydantic ``meta`` object."""

    def __init__(self, headings: list[str]):
        self._headings = headings

    def model_dump(self) -> dict:
        return {"headings": self._headings}


class _FakeChunk:
    """Minimal stand-in for a Docling chunk."""

    def __init__(self, text: str, headings: list[str]):
        self._text = text
        self._headings = headings

    @property
    def meta(self) -> _FakeMeta:
        return _FakeMeta(self._headings)


class TestConvertAndChunkNormalization:
    """_convert_and_chunk normalises chunk text and headings."""

    def setup_method(self):
        self.logger = logging.getLogger("test_ingestion_normalization")

    def test_convert_and_chunk_normalizes_text_and_headings(self, monkeypatch):
        from types import SimpleNamespace

        # Skip the real biblio cascade; it operates on a Docling document
        # we are not building here.
        monkeypatch.setattr(
            "klea_utils.stores.ingestion.extract_metadata",
            lambda *args, **kwargs: {},
        )

        builder = StoresBuilder(embedding_model="", logger=self.logger)

        class _FakeConverter:
            def convert(self, path):
                return SimpleNamespace(document=SimpleNamespace())

        class _FakeChunker:
            def __init__(self, chunks):
                self._chunks = chunks

            def chunk(self, dl_doc):
                return self._chunks

            def contextualize(self, chunk):
                return chunk._text

        converter = _FakeConverter()
        chunker = _FakeChunker(
            [
                _FakeChunk(
                    "multi-\u00adscale model\u00ading in\u00a0neuroscience",
                    ["Multi-\u00adscale Model\u00ading"],
                ),
                _FakeChunk(
                    "plain content",
                    ["Introduction", "eLife Assessment"],
                ),
            ]
        )
        monkeypatch.setattr(builder, "_get_converter", lambda: converter)
        monkeypatch.setattr(builder, "_get_chunker", lambda: chunker)

        docs, extracted = builder._convert_and_chunk(Path("paper.pdf"), resolver=None)

        assert extracted == {}
        assert docs[0].page_content == "multi-scale modeling in neuroscience"
        assert docs[0].metadata["headings"] == ["Multi-scale Modeling"]
        assert docs[1].page_content == "plain content"
        assert docs[1].metadata["headings"] == ["Introduction", "eLife Assessment"]

    def test_convert_and_chunk_handles_none_headings(self, monkeypatch):
        """Chunks whose DocMeta.headings is None produce empty heading lists.

        docling's ``DocMeta.headings`` is ``Optional[list[str]]`` defaulting
        to ``None`` (chunks outside a heading hierarchy), so this exercises
        the real pydantic model rather than a mock that assumes a list.
        """
        from types import SimpleNamespace

        from docling_core.transforms.chunker.doc_chunk import DocChunk, DocMeta
        from docling_core.types.doc import TextItem
        from docling_core.types.doc.labels import DocItemLabel

        monkeypatch.setattr(
            "klea_utils.stores.ingestion.extract_metadata",
            lambda *args, **kwargs: {},
        )

        def _doc_chunk(text: str, headings: list[str] | None) -> DocChunk:
            item = TextItem(
                self_ref="#/texts/0",
                label=DocItemLabel.TEXT,
                orig=text,
                text=text,
            )
            return DocChunk(
                text=text,
                meta=DocMeta(doc_items=[item], headings=headings),
            )

        builder = StoresBuilder(embedding_model="", logger=self.logger)

        class _FakeConverter:
            def convert(self, path):
                return SimpleNamespace(document=SimpleNamespace())

        class _FakeChunker:
            def __init__(self, chunks):
                self._chunks = chunks

            def chunk(self, dl_doc):
                return self._chunks

            def contextualize(self, chunk):
                return chunk.text

        converter = _FakeConverter()
        chunker = _FakeChunker(
            [
                _doc_chunk(
                    "multi-\u00adscale model\u00ading",
                    ["Multi-\u00adscale Model\u00ading"],
                ),
                _doc_chunk("no heading chunk", None),
            ]
        )
        monkeypatch.setattr(builder, "_get_converter", lambda: converter)
        monkeypatch.setattr(builder, "_get_chunker", lambda: chunker)

        docs, extracted = builder._convert_and_chunk(Path("paper.pdf"), resolver=None)

        assert extracted == {}
        assert docs[0].metadata["headings"] == ["Multi-scale Modeling"]
        assert docs[1].page_content == "no heading chunk"
        assert docs[1].metadata["headings"] == []


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

    def test_load_metadata_map_normalizes_heading_keys(self):
        """Heading keys with typographic artifacts are normalized on load.

        Users may paste keys (e.g. from a PDF) containing soft hyphens or
        no-break spaces; they must still match the normalized chunk headings.
        """
        map_path = self.tmpdir_path / "metadata-map.json"
        map_path.write_text(
            json.dumps(
                {
                    "DEFAULT": {"topic": "fallback"},
                    "Multi-\u00adscale Model\u00ading in\u00a0neuroscience": {
                        "topic": "nml"
                    },
                }
            )
        )
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        loaded = builder._load_metadata_map(str(map_path))

        assert loaded["DEFAULT"] == {"topic": "fallback"}
        assert loaded["Multi-scale Modeling in neuroscience"] == {"topic": "nml"}
        assert (
            loaded.get("Multi-\u00adscale Model\u00ading in\u00a0neuroscience") is None
        )

    def test_resolve_metadata_matches_normalized_headings(self):
        """_resolve_metadata matches normalized chunk headings to map keys."""
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        meta = builder._resolve_metadata(
            "paper.pdf",
            ["Multi-scale Modeling"],
            {
                "paper.pdf": {
                    "DEFAULT": {"topic": "fallback"},
                    "Multi-scale Modeling": {"topic": "nml"},
                }
            },
        )
        assert meta == {"topic": "nml"}

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

    def test_write_heading_template_preserves_existing_when_empty(self):
        """An empty chunk run must not clobber an existing template."""
        existing = {"paper.pdf": {"DEFAULT": {"title": "T"}}}
        template = self.tmpdir_path / TEMPLATE_FILE_NAME
        template.write_text(json.dumps(existing))

        builder = StoresBuilder(embedding_model="", logger=self.logger)
        builder.write_heading_template({}, self.tmpdir_path)

        assert json.loads(template.read_text()) == existing

    def test_write_heading_template_writes_literal_utf8(self):
        """Accented characters are written literally, not \\u-escaped.

        The template is a human-edited file; an author such as "B\u00f3ris
        Marin" must appear as-is so the editor can see the exact text.
        """
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        builder.write_heading_template(
            {
                "paper.pdf": {
                    "DEFAULT": {
                        "title": "Caf\u00e9 science",
                        "authors": ["B\u00f3ris Marin"],
                    }
                }
            },
            self.tmpdir_path,
        )

        text = (self.tmpdir_path / TEMPLATE_FILE_NAME).read_text()
        assert "B\u00f3ris Marin" in text
        assert "\\u00f3" not in text
        assert json.loads(text)["paper.pdf"]["DEFAULT"]["authors"] == [
            "B\u00f3ris Marin"
        ]

    def test_write_heading_template_empty_no_existing_no_write(self):
        """An empty chunk run with no existing template writes nothing."""
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        builder.write_heading_template({}, self.tmpdir_path)

        assert not (self.tmpdir_path / TEMPLATE_FILE_NAME).exists()

    def test_build_raises_when_nothing_chunked(self, monkeypatch):
        """build() fails loudly instead of storing nothing and reporting done."""
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        monkeypatch.setattr(builder, "chunk_all", lambda *a, **k: ([], {}))

        with pytest.raises(RuntimeError, match="No files were successfully chunked"):
            builder.build(str(self.tmpdir_path), "chroma:/tmp/x", "c")

    def test_find_files_excludes_template(self):
        """The generated template is not treated as an ingestible file."""
        (self.tmpdir_path / TEMPLATE_FILE_NAME).write_text("{}")
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        assert builder._find_files(self.tmpdir_path) == []

    def test_find_files_excludes_loaded_metadata_map(self):
        """The --metadata-map file is excluded when it lives in source_dir."""
        src = self.tmpdir_path / "doc.md"
        src.write_text("# Doc\n")
        map_path = self.tmpdir_path / "metadata.json"
        map_path.write_text(json.dumps({"paper.pdf": {"DEFAULT": {}}}))

        builder = StoresBuilder(embedding_model="", logger=self.logger)
        builder._load_metadata_map(str(map_path))

        assert builder._find_files(self.tmpdir_path) == [src]

    def test_find_files_includes_non_metadata_json(self):
        """Other .json files (docling json_docling) stay ingestible."""
        src = self.tmpdir_path / "docling-doc.json"
        src.write_text("{}")
        builder = StoresBuilder(embedding_model="", logger=self.logger)
        assert builder._find_files(self.tmpdir_path) == [src]


if __name__ == "__main__":
    pytest.main()
