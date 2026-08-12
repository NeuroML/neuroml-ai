#!/usr/bin/env python3
"""
Store ingestion -- convert documents, chunk, embed, and write to stores

File: klea_utils/stores/ingestion.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Any

import xxhash
from langchain_core.documents import Document

from ..biblio.extract import Resolver, extract_metadata, extract_metadata_from_text
from ..llm import setup_embedding
from .utils import instantiate_vector_store, normalize_text

CACHE_DIR_NAME = ".klea-cache"
TEMPLATE_FILE_NAME = "metadata-map.template.json"


class StoresBuilder:
    """Build stores from a directory of source documents.

    Uses Docling for document conversion and token-aware chunking, then
    embeds chunks and writes them to a vector store backend.  Optionally
    also writes the combined chunked corpus for BM25 retrieval.
    """

    DEFAULT_MAX_TOKENS = 450
    DEFAULT_MERGE_PEERS = True
    DEFAULT_TOKENIZER_MODEL = "BAAI/bge-m3"

    def __init__(
        self,
        embedding_model: str,
        logger: logging.Logger,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        merge_peers: bool = DEFAULT_MERGE_PEERS,
        tokenizer_model: str = DEFAULT_TOKENIZER_MODEL,
        do_ocr: bool = True,
    ):
        """Initialise the builder.

        :param embedding_model: Embedding model identifier
            (e.g. ``"ollama:bge-m3:latest"``).  Only needed when
            :meth:`store_all` will be called.
        :param logger: Logger instance
        :param max_tokens: Maximum tokens per chunk
        :param merge_peers: Whether the chunker should merge peer
            elements (e.g. consecutive paragraphs)
        :param tokenizer_model: HuggingFace tokenizer model used for
            token-aware chunking
        :param do_ocr: Whether Docling should OCR pages during PDF
            conversion.  Keep enabled for scanned/image-based PDFs;
            disabling it speeds up conversion of text-based PDFs
            significantly.
        """
        self.embedding_model = embedding_model
        self.logger = logging.getLogger(f"{logger.name}.{self.__class__.__name__}")
        self.max_tokens = max_tokens
        self.merge_peers = merge_peers
        self.tokenizer_model = tokenizer_model
        self.do_ocr = do_ocr

        self.embeddings = None
        self._converter = None
        self._chunker = None

        self.logger.info(
            f"StoresBuilder initialised (max_tokens={max_tokens}, "
            f"merge_peers={merge_peers}, tokenizer={tokenizer_model}, "
            f"do_ocr={do_ocr})"
        )

    def build(
        self,
        source_dir: str,
        store_uri: str,
        collection_name: str,
        force: bool = False,
        metadata_map_path: str | None = None,
        bm25_path: str | None = None,
    ) -> None:
        """Full pipeline: chunk documents and write them to a vector store.

        Convenience wrapper around :meth:`chunk_all` + :meth:`store_all`.

        :param source_dir: Path to a directory containing source documents
        :param store_uri: Vector store URI (e.g. ``chroma:/path``)
        :param collection_name: Collection name for the store
        :param force: Re-process all files even if unchanged
        :param metadata_map_path: Optional path to a metadata map JSON file
        :param bm25_path: Optional path to write the combined BM25 corpus to
        """
        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        self.logger.info(
            f"Starting full pipeline: {source_dir} -> "
            f"collection '{collection_name}' at {store_uri}"
        )

        metadata_map = None
        if metadata_map_path:
            metadata_map = self._load_metadata_map(metadata_map_path)

        results, _ = self.chunk_all(source_path, metadata_map, force)
        if not results:
            raise RuntimeError(f"No files were successfully chunked from {source_path}")
        self.store_all(results, store_uri, collection_name, force, bm25_path)
        self.logger.info(f"Ingestion complete for collection '{collection_name}'")

    def chunk_all(
        self,
        source_path: Path,
        metadata_map: dict[str, dict[str, Any]] | None = None,
        force: bool = False,
    ) -> tuple[list[tuple[str, list[Document], Path]], dict[str, dict[str, Any]]]:
        """Convert, chunk, cache, and enrich metadata for all files.

        Skips converting files whose cache entry exists (unless
        ``force`` is ``True``).  Always caches newly-converted chunks.
        Heading chains are collected per file for template generation.
        The per-file ``DEFAULT`` template entry is pre-filled with the
        automatically-extracted bibliographic metadata (see
        :func:`~klea_utils.biblio.extract.extract_metadata`).

        :param source_path: Resolved source directory path
        :param metadata_map: Metadata map for heading-based enrichment,
            or ``None``
        :param force: Re-process all files even if cached
        :returns: ``(results, file_headings)`` where *results* is a
            list of ``(file_hash, docs, file_path)`` tuples and
            *file_headings* is a ``{file_name: {"DEFAULT": {extracted
            metadata}, "heading > heading": {}, ...}}`` dict
        """
        self._ensure_tokenizer()

        files = self._find_files(source_path)
        self.logger.info(f"Found {len(files)} ingestible files in {source_path}")

        resolver = self._make_resolver(source_path)

        results: list[tuple[str, list[Document], Path]] = []
        file_headings: dict[str, dict[str, Any]] = {}
        total = len(files)

        for ctr, file_path in enumerate(files, 1):
            file_hash = _hash_file(file_path)

            docs = None
            extracted: dict[str, Any] = {}
            if not force:
                cached = self._load_from_cache(source_path, file_hash)
                if cached is not None:
                    docs, extracted = cached

            if docs is None:
                self.logger.info(f"Processing: {file_path.name} ({ctr}/{total})")
                try:
                    docs, extracted = self._convert_and_chunk(file_path, resolver)
                    self._save_to_cache(docs, extracted, source_path, file_hash)
                except Exception as e:
                    self.logger.error(f"Failed to process {file_path.name}: {e}")
                    continue
            else:
                self.logger.debug(
                    f"Using cached chunks for: {file_path.name} ({ctr}/{total})"
                )
                if not extracted:
                    # No persisted extraction (e.g. a legacy cache entry
                    # written before the biblio cascade existed), so run
                    # the text-only extraction over the cached chunks:
                    # regex + pdf-info (for PDFs) + DOI resolution.
                    cached_text = "\n".join(
                        doc.page_content for doc in docs if doc.page_content
                    )
                    extracted = (
                        extract_metadata_from_text(
                            cached_text,
                            str(file_path),
                            pdf_path=(
                                str(file_path)
                                if file_path.suffix.lower() == ".pdf"
                                else None
                            ),
                            resolver=resolver,
                        )
                        if cached_text
                        else {}
                    )
                    self.logger.warning(
                        f"No cached metadata extraction for {file_path.name}, "
                        f"falling back to text-only extraction "
                        f"({len(extracted)} fields); re-run with --force to "
                        f"regenerate the full extraction"
                    )

            for doc in docs:
                doc.metadata.update(
                    {
                        "file_hash": file_hash,
                        "file_name": file_path.name,
                        "source_path": str(file_path),
                    }
                )

            if metadata_map:
                for doc in docs:
                    meta = self._resolve_metadata(
                        file_path.name,
                        doc.metadata.get("headings"),
                        metadata_map,
                    )
                    if meta:
                        doc.metadata.update(meta)

            normalized_default = _normalize_extracted_metadata(extracted)
            if normalized_default != extracted:
                self.logger.debug(f"Normalised extracted metadata for {file_path.name}")
            file_entry: dict[str, Any] = {"DEFAULT": normalized_default}
            for doc in docs:
                headings = doc.metadata.get("headings", [])
                if headings:
                    key = " > ".join(headings)
                    if key not in file_entry:
                        file_entry[key] = {}
            file_headings[file_path.name] = file_entry

            results.append((file_hash, docs, file_path))

        # The resolver's HTTP client is left for the process to clean up;
        # ingestion is a one-shot CLI run.
        return results, file_headings

    def store_all(
        self,
        results: list[tuple[str, list[Document], Path]],
        store_uri: str,
        collection_name: str,
        force: bool = False,
        bm25_path: str | None = None,
    ) -> None:
        """Write chunked documents to a vector store.

        Initialises the embedding model on first call if not already
        done.  Skips files whose hash is already present in the store
        (unless ``force`` is ``True``).  Optionally also writes the
        combined document corpus for BM25 retrieval.

        :param results: List of ``(file_hash, docs, file_path)`` tuples
            from :meth:`chunk_all`
        :param store_uri: Vector store URI
        :param collection_name: Collection name for the store
        :param force: Re-store all files even if already indexed
        :param bm25_path: Optional path to write the combined BM25 corpus to
        """
        if self.embeddings is None:
            self.logger.info(f"Initialising embedding model ({self.embedding_model})")
            self.embeddings = setup_embedding(self.embedding_model, self.logger)
        assert store_uri and collection_name

        self.logger.info(f"Opening vector store '{collection_name}' at {store_uri}")
        store = instantiate_vector_store(
            store_uri,
            collection_name,
            self.embeddings,
            self.logger,
            create=True,
        )

        total = len(results)
        for ctr, (file_hash, docs, file_path) in enumerate(results, 1):
            if not force:
                existing = store.get(where={"file_hash": file_hash})
                if existing and existing["ids"]:
                    self.logger.debug(
                        f"Skipping already indexed file: "
                        f"{file_path.name} ({ctr}/{total})"
                    )
                    continue

            store.add_documents(docs)
            self.logger.info(
                f"Added {len(docs)} chunks from {file_path.name} ({ctr}/{total})"
            )

        if bm25_path:
            self.write_bm25_store(results, bm25_path)

    def write_bm25_store(
        self,
        results: list[tuple[str, list[Document], Path]],
        bm25_path: str,
    ) -> None:
        """Write the combined chunked documents to a BM25 corpus.

        Flattens the per-file chunked documents from :meth:`chunk_all`
        into a single list and pickles it to *bm25_path*.  This file is
        the BM25 store: a :class:`BM25RetrieverManager` loads it at
        runtime to build its keyword index.  It is independent of the
        per-file ``.klea-cache``, so the cache can be removed once the
        corpus has been written.

        The corpus holds the same chunk units (and metadata) that are
        stored in the vector store, so BM25 and vector retrieval return
        consistent results.

        :param results: List of ``(file_hash, docs, file_path)`` tuples
            from :meth:`chunk_all`
        :param bm25_path: Path to write the combined corpus pickle to
        """
        all_docs = [doc for _, docs, _ in results for doc in docs]
        if not all_docs:
            self.logger.warning("No documents to write to BM25 store, skipping")
            return

        path = Path(bm25_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(all_docs, f)
        self.logger.info(f"Wrote BM25 store with {len(all_docs)} chunks to {path}")

    def write_heading_template(
        self, file_headings: dict[str, dict[str, Any]], source_dir: Path
    ) -> None:
        """Write a metadata-map template JSON file organised per source file.

        Each file gets a ``"DEFAULT"`` placeholder and one entry per
        unique heading chain found in that file.  The user fills in the
        ``{}`` with their metadata key-value pairs.

        Refuses to write when *file_headings* is empty (no files were
        chunked): an existing template is preserved rather than clobbered
        with an empty one.

        :param file_headings: ``{file_name: {"DEFAULT": {},
            "heading > heading": {}, ...}, ...}`` from :meth:`chunk_all`
        :param source_dir: Resolved source directory path (template is
            written alongside it)
        """
        out_path = source_dir / TEMPLATE_FILE_NAME
        if not file_headings:
            if out_path.is_file():
                self.logger.warning(
                    f"No files chunked; keeping existing template {out_path}"
                )
                return
            self.logger.warning(
                f"No files chunked and no existing template at {out_path}"
            )
            return
        # ensure_ascii=False keeps accented characters (e.g. "B\u00f3ris")
        # as literal UTF-8 in the file, so the human editing the template
        # can see exactly what text a heading/author contains.
        with open(out_path, "w") as f:
            json.dump(file_headings, f, indent=4, ensure_ascii=False)
            f.write("\n")
        total_chains = sum(len(v) - 1 for v in file_headings.values())
        self.logger.info(
            f"Metadata map template written to {out_path} "
            f"({len(file_headings)} files, {total_chains} heading chains)"
        )

    def _cache_dir(self, source_dir: Path) -> Path:
        """Return the cache directory path inside *source_dir*.

        :param source_dir: Resolved source directory path
        :returns: Path to ``<source_dir>/.klea-cache/``
        """
        return source_dir / CACHE_DIR_NAME

    def _cache_path(self, source_dir: Path, file_hash: str) -> Path:
        """Return the cache file path for a given file hash.

        The ``:`` in the hash is replaced with ``_`` for filesystem
        safety (``:`` is allowed in most Linux filesystems but is
        problematic on Windows and some networked FSes).

        :param source_dir: Resolved source directory path
        :param file_hash: xxhash digest of the source file
        :returns: Path to ``<cache_dir>/<file_hash>.pkl``
        """
        safe_hash = file_hash.replace(":", "_")
        return self._cache_dir(source_dir) / f"{safe_hash}.pkl"

    def _save_to_cache(
        self,
        docs: list[Document],
        extracted: dict[str, Any],
        source_dir: Path,
        file_hash: str,
    ) -> None:
        """Pickle *docs* and their extracted metadata to the cache directory.

        Creates the cache directory if it does not exist.  The cache
        entry is a ``(docs, extracted)`` tuple so that cache hits can
        restore the full-cascade bibliographic extraction instead of
        degrading to a weaker regex-only pass.  (Legacy cache entries
        hold a plain list of docs; :meth:`_load_from_cache` handles both.)

        :param docs: List of chunked documents to cache
        :param extracted: Bibliographic metadata extracted for the file
        :param source_dir: Resolved source directory path
        :param file_hash: xxhash digest of the source file
        """
        cache_dir = self._cache_dir(source_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = self._cache_path(source_dir, file_hash)
        with open(path, "wb") as f:
            pickle.dump((docs, extracted), f)
        self.logger.info(f"Cached {len(docs)} chunks to {path}")

    def _load_from_cache(
        self, source_dir: Path, file_hash: str
    ) -> tuple[list[Document], dict[str, Any]] | None:
        """Load pickled chunks and their extracted metadata from the cache.

        To inspect cached chunks from a Python shell::

            import pickle
            from pathlib import Path
            for p in Path("<source_dir>/.klea-cache/").glob("*.pkl"):
                data = pickle.load(open(p, "rb"))
                docs, extracted = data if isinstance(data, tuple) else (data, {})
                print(p.stem, docs[0].metadata.get("headings"), extracted)

        Handles legacy cache entries (a plain list of documents) by
        returning an empty extracted dict for them.

        :param source_dir: Resolved source directory path
        :param file_hash: xxhash digest of the source file
        :returns: ``(docs, extracted)``, or ``None`` if the cache file
            does not exist
        """
        path = self._cache_path(source_dir, file_hash)
        if not path.is_file():
            return None
        self.logger.debug(f"Cache hit: {path.name}")
        with open(path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, tuple):
            docs, extracted = data
            return docs, extracted or {}
        return data, {}

    # ------------------------------------------------------------------
    # Metadata map helpers
    # ------------------------------------------------------------------

    def _load_metadata_map(self, metadata_map_path: str) -> dict[str, dict[str, Any]]:
        """Load and validate a metadata map JSON file.

        The file must contain a JSON object with string keys (heading
        text) and dict values of metadata key-value pairs.  An optional
        ``DEFAULT`` key provides a fallback when no heading matches.

        :param metadata_map_path: Path to the JSON file
        :returns: Mapping of heading text to metadata dicts
        :raises FileNotFoundError: If the path does not exist
        :raises ValueError: If the JSON is not well-formed
        """
        path = Path(metadata_map_path)
        if not path.is_file():
            raise FileNotFoundError(f"Metadata map file not found: {path}")
        self.logger.info(f"Loading metadata map from {path}")
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(
                f"Metadata map must be a JSON object (dict), got {type(data).__name__}"
            )
        for k, v in data.items():
            if not isinstance(k, str):
                raise ValueError(
                    f"Metadata map keys must be strings, got {type(k).__name__}"
                )
            if not isinstance(v, dict):
                raise ValueError(
                    f"Values in metadata map must be dicts, "
                    f"got {type(v).__name__} for key {k!r}"
                )
        # Normalise heading keys so user-filled keys (possibly pasted with
        # typographic artifacts) match the normalised chunk headings.  The
        # ``DEFAULT`` key passes through unchanged.
        changed_keys = [k for k in data if normalize_text(k) != k]
        if changed_keys:
            self.logger.debug(
                f"Normalised {len(changed_keys)} heading keys in metadata map"
            )
        data = {normalize_text(k): v for k, v in data.items()}
        self.logger.info(f"Loaded metadata map with {len(data)} entries from {path}")
        return data

    def _resolve_metadata(
        self,
        file_name: str,
        headings: list[str] | None,
        metadata_map: dict[str, dict[str, Any]],
    ) -> dict[str, Any] | None:
        """Resolve a metadata dict for a chunk using the per-file metadata map.

        Looks up the file in the map, then matches the heading chain
        from most specific to least specific.  Falls back to
        ``DEFAULT`` for that file.

        :param file_name: Source filename to look up in the map
        :param headings: Heading hierarchy for the chunk (most specific
            last), or ``None``
        :param metadata_map: Per-file metadata map
            ``{file_name: {"DEFAULT": {}, "heading": {...}}}``
        :returns: Matched metadata dict, or ``None``
        """
        file_map = metadata_map.get(file_name)
        if file_map is None:
            return None
        if headings:
            for heading in reversed(headings):
                if heading in file_map:
                    self.logger.debug(f"Resolved metadata for {file_name}: '{heading}'")
                    return file_map[heading]
        fallback = file_map.get("DEFAULT")
        if fallback:
            self.logger.debug(f"Resolved DEFAULT metadata for {file_name}")
        return fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_files(self, source_dir: Path) -> list[Path]:
        """Walk ``source_dir`` and return files whose extensions are in
        docling's :attr:`~docling.datamodel.base_models.FormatToExtensions`.

        Files with unsupported extensions are logged as a warning and skipped.

        :param source_dir: Directory to walk recursively
        :returns: Sorted list of files with supported extensions
        """
        from docling.datamodel.base_models import FormatToExtensions

        all_exts: set[str] = set()
        for exts in FormatToExtensions.values():
            all_exts.update(exts)

        supported: list[Path] = []
        for f in sorted(source_dir.rglob("*")):
            if not f.is_file():
                continue
            if CACHE_DIR_NAME in f.parts:
                continue
            suffix = f.suffix.lstrip(".").lower()
            if suffix in all_exts:
                supported.append(f)
            else:
                self.logger.warning(f"Skipping unsupported file: {f.name}")

        return supported

    def _ensure_tokenizer(self) -> None:
        """Download the HuggingFace tokenizer used for token-aware chunking
        if it is not already cached locally.

        ..  TODO:: Allow overriding ``tokenizer_model`` via an environment
            variable (e.g. ``KLEA_INGEST_TOKENIZER_MODEL``) or a local
            filesystem path so that air-gapped deployments can point at
            pre-downloaded tokenizer files.
        """
        from transformers import AutoTokenizer

        self.logger.debug(f"Ensuring tokenizer '{self.tokenizer_model}' is available")
        AutoTokenizer.from_pretrained(self.tokenizer_model)

    def _get_converter(self):
        """Lazily initialise and return the Docling
        :class:`~docling.document_converter.DocumentConverter` singleton.

        When :attr:`do_ocr` is ``False``, the PDF pipeline is configured
        with OCR disabled, which speeds up conversion of text-based PDFs
        significantly (scanned/image-based PDFs then lose their
        embedded text).

        :returns: Shared :class:`~docling.document_converter.DocumentConverter`
            instance
        """
        if self._converter is None:
            self.logger.debug(
                f"Initialising Docling DocumentConverter (do_ocr={self.do_ocr})"
            )
            from docling.document_converter import (
                DocumentConverter,
                PdfFormatOption,
            )

            if self.do_ocr:
                self._converter = DocumentConverter()
            else:
                # Lazy: the pipeline-options imports pull in docling's
                # pipeline machinery; only needed when OCR is disabled.
                from docling.datamodel.base_models import InputFormat
                from docling.datamodel.pipeline_options import PdfPipelineOptions

                options = PdfPipelineOptions()
                options.do_ocr = False
                self._converter = DocumentConverter(
                    format_options={
                        InputFormat.PDF: PdfFormatOption(pipeline_options=options)
                    }
                )
        return self._converter

    def _get_chunker(self):
        """Lazily initialise and return the
        :class:`~docling.chunking.HybridChunker`
        configured with the instance tokenizer and chunking parameters.

        :returns: Configured :class:`~docling.chunking.HybridChunker` instance
        """
        if self._chunker is None:
            self.logger.debug(
                f"Initialising HybridChunker "
                f"(max_tokens={self.max_tokens}, merge_peers={self.merge_peers})"
            )
            from docling.chunking import HybridChunker
            from docling_core.transforms.chunker.tokenizer.huggingface import (
                HuggingFaceTokenizer,
            )
            from transformers import AutoTokenizer

            hf_tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_model)
            tokenizer = HuggingFaceTokenizer(
                tokenizer=hf_tokenizer, max_tokens=self.max_tokens
            )
            self._chunker = HybridChunker(
                tokenizer=tokenizer, merge_peers=self.merge_peers
            )
        return self._chunker

    def _convert_and_chunk(
        self, file_path: Path, resolver: Resolver | None
    ) -> tuple[list[Document], dict[str, Any]]:
        """Convert ``file_path`` with Docling, chunk with the
        :class:`~docling.chunking.HybridChunker`,
        and return :class:`~langchain_core.documents.Document` objects
        alongside the automatically-extracted bibliographic metadata.

        Each document's metadata includes a ``headings`` list (the heading
        hierarchy for the chunk).  The extracted metadata (see
        :func:`~klea_utils.biblio.extract.extract_metadata`) is used to
        pre-fill the per-file template ``DEFAULT`` entry; it is not
        attached to the chunks themselves.

        :param file_path: Path to the source document file
        :param resolver: DOI resolver for the extraction cascade, or
            ``None`` to skip DOI resolution
        :returns: ``(docs, extracted)`` where *docs* is the list of
            chunked :class:`~langchain_core.documents.Document` objects
            ready for embedding and *extracted* is the flat
            bibliographic metadata dict
        """
        converter = self._get_converter()
        chunker = self._get_chunker()

        self.logger.info(f"Converting {file_path.name} with Docling")
        result = converter.convert(str(file_path))
        dl_doc = result.document

        docs: list[Document] = []
        for chunk in chunker.chunk(dl_doc=dl_doc):
            raw_text = chunker.contextualize(chunk=chunk)
            chunk_text = normalize_text(raw_text)
            meta = chunk.meta.model_dump()
            # DocMeta.headings is Optional[list[str]] (default None) for
            # chunks not under a heading hierarchy; normalise to [].
            raw_headings = meta.get("headings") or []
            headings = [normalize_text(heading) for heading in raw_headings]

            if chunk_text != raw_text:
                self.logger.debug(
                    f"Normalised chunk text: {len(raw_text)} -> {len(chunk_text)} chars"
                )
            if headings != raw_headings:
                self.logger.debug(f"Normalised headings: {raw_headings} -> {headings}")

            doc = Document(
                page_content=chunk_text,
                metadata={"headings": headings},
            )
            docs.append(doc)

        extracted = extract_metadata(
            dl_doc,
            str(file_path),
            pdf_path=str(file_path) if file_path.suffix.lower() == ".pdf" else None,
            resolver=resolver,
        )

        return docs, extracted

    def _make_resolver(self, source_path: Path) -> Resolver:
        """Build a DOI resolver for a source directory.

        The resolver caches resolved DOIs under the source directory's
        ``.klea-cache/`` and picks up the ``KLEA_INGEST_MAILTO``
        polite-pool address from the environment.

        :param source_path: Resolved source directory path
        :returns: A configured
            :class:`~klea_utils.biblio.doi.DoiResolver`
        """
        # Lazy: importing the DOI resolver pulls in httpx; it is only
        # needed when documents are being converted.
        from ..biblio.doi import DoiResolver

        return DoiResolver(cache_dir=self._cache_dir(source_path))


def _hash_file(file_path: Path) -> str:
    """Return an xxhash hex digest of a file's contents.

    :param file_path: Path to the file to hash
    :returns: Hex digest string prefixed with ``"xxh64:"``
    """
    h = xxhash.xxh64()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return f"xxh64:{h.hexdigest()}"


def _normalize_extracted_metadata(extracted: dict[str, Any]) -> dict[str, Any]:
    """Return *extracted* with typographic artifacts stripped from string fields.

    The bibliographic cascade (:func:`~klea_utils.biblio.extract.extract_metadata`)
    runs over raw converted text, so fields such as ``title``, ``authors``,
    and ``urls`` can carry soft hyphens / no-break spaces.  Normalising them
    keeps ``metadata-map.template.json``'s ``DEFAULT`` entry plain text.
    Non-string values (e.g. ``year``, ``_metadata_complete``) are untouched.

    :param extracted: Metadata dict from the extraction cascade
    :returns: Copy of *extracted* with string values normalised
    """
    normalized: dict[str, Any] = {}
    for key, value in extracted.items():
        if isinstance(value, str):
            normalized[key] = normalize_text(value)
        elif isinstance(value, list):
            normalized[key] = [
                normalize_text(item) if isinstance(item, str) else item
                for item in value
            ]
        else:
            normalized[key] = value
    return normalized
