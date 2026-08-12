#!/usr/bin/env python3
"""
Vector store utilities

File: klea_utils/stores/utils.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path

from langchain_core.documents import Document

#: Metadata key holding each document's original per-source scores (e.g.
#: ``{"vector store": 0.87, "BM25": 3.21}``), set by :func:`rrf_merge`.
SOURCE_SCORES_KEY = "_source_scores"

#: HNSW distance space used for Chroma collections created by Klea.  Cosine
#: makes the vector-store relevance scores true cosine similarities
#: (``1 - cosine_distance``), so a ``score_threshold`` reads as a minimum
#: cosine similarity.
CHROMA_HNSW_SPACE = "cosine"

#: Rank offset for Reciprocal Rank Fusion.  A document at rank *r* within a
#: source's result list contributes ``1 / (RRF_K + r)`` to its fused score.
RRF_K = 60

_INTERNAL_META_KEYS = {
    "file_name",
    "source_path",
    "file_hash",
    "headings",
    SOURCE_SCORES_KEY,
    # Bibliographic extraction provenance (see klea_utils/biblio).  These
    # guide the researcher reviewing metadata-map.template.json but carry
    # no meaning for the answer LLM, so they are never serialized to it.
    "_metadata_complete",
    "_sources",
}


def rrf_merge(
    result_sets: list[tuple[str, list[tuple[Document, float]]]],
    num_refs_max: int,
) -> list[tuple[Document, float]]:
    """Fuse per-source retrieval results with Reciprocal Rank Fusion.

    Scores from different retrievers (e.g. cosine similarity vs BM25) are not
    comparable, so each document is scored purely by its rank within each
    source's result list.  The original per-source scores are preserved in
    each document's :data:`SOURCE_SCORES_KEY` metadata for display.

    :param result_sets: List of ``(source_label, results)`` pairs, where each
        *results* is a list of ``(document, score)`` tuples already ranked by
        its source
    :param num_refs_max: Maximum number of documents to return
    :returns: Documents ordered by RRF score, deduplicated by content, capped
        at *num_refs_max*
    """
    rrf_scores: dict[str, float] = {}
    doc_by_key: dict[str, Document] = {}

    for source_label, results in result_sets:
        for rank, (doc, score) in enumerate(results, start=1):
            key = doc.page_content
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            if key not in doc_by_key:
                doc_by_key[key] = doc
                doc.metadata[SOURCE_SCORES_KEY] = {}
            doc_by_key[key].metadata[SOURCE_SCORES_KEY][source_label] = score

    merged = sorted(
        ((doc_by_key[key], rrf) for key, rrf in rrf_scores.items()),
        key=lambda tup: tup[1],
        reverse=True,
    )
    return merged[:num_refs_max]


def format_source_scores(doc: Document, precision: int = 2) -> str | None:
    """Format a document's original per-source scores for display.

    :param doc: Document with :data:`SOURCE_SCORES_KEY` metadata
    :param precision: Number of decimal places per score
    :returns: Joined string like ``"vector store 0.87, BM25 3.21"``, or
        ``None`` if the document has no per-source scores
    """
    source_scores = doc.metadata.get(SOURCE_SCORES_KEY)
    if not source_scores:
        return None
    return ", ".join(f"{k} {v:.{precision}f}" for k, v in source_scores.items())


def _format_score_str(doc: Document, score: float) -> str:
    """Format a document's relevance scores for prompt context.

    Shows the original per-source scores (e.g. ``vector store 0.8723,
    BM25 3.2100``) when present in ``_source_scores`` metadata, so the LLM
    can interpret scores from different retrievers.  Falls back to the
    single relevance score for documents without per-source info.

    :param doc: Document to format
    :param score: Relevance score for *doc*
    :returns: Score string to append to a document heading
    """
    source_scores = format_source_scores(doc, precision=4)
    if source_scores:
        return f" (relevance: {source_scores})"
    return f" (relevance score: {score:.4f})"


def serialize_vs_retrieval(
    reference_material: dict[str, list[tuple[Document, float]]],
) -> str:
    """Serialize vector store retrieval results into text for use in prompt context.

    Documents are sorted by relevance score within each group.
    Uses Docling ``HybridChunker`` metadata format:

    - ``headings``: list of heading hierarchy (most specific last)
    - ``file_name``: source filename
    - ``source_path``: full path to source file
    - ``_source_scores``: optional per-retriever scores (from the RRF merge)
    - Optional custom keys from the ``--metadata-map`` (e.g., ``url``)

    :param reference_material: Dict mapping query/domain to list of (doc, score) tuples
    :returns: Formatted string representation of references
    """
    serialized = ""
    for q, sorted_refs in reference_material.items():
        ctr = 1
        serialized += f"## {q}\n"
        for r, score in sorted_refs:
            headings = r.metadata.get("headings", [])
            file_name = r.metadata.get("file_name", "")
            heading_str = " > ".join(headings) if headings else "(no heading)"
            if file_name:
                heading_str = f"[{file_name}] {heading_str}"

            score_str = _format_score_str(r, score)
            serialized += (
                f"\n### Document {ctr}/{len(sorted_refs)}: {heading_str}{score_str}\n"
            )
            custom_meta = {
                k: v for k, v in r.metadata.items() if k not in _INTERNAL_META_KEYS
            }
            if custom_meta:
                meta_str = " | ".join(f"{k}={v}" for k, v in custom_meta.items())
                serialized += f"Metadata: {meta_str}\n"
            serialized += r.page_content
            ctr += 1

    return serialized


def instantiate_vector_store(
    path: str,
    name: str,
    embeddings,
    logger: logging.Logger,
    create: bool = False,
):
    """Instantiate a vector store based on the URI scheme in path.

    Expected format: ``"scheme:location"``.

    If ``create`` is ``True``, the store is created if it does not exist
    (relevant for ChromaDB which requires a local directory).  For
    Qdrant and PGVector the flag is a no-op --- collections are created
    on first write.

    For ChromaDB, ``location`` must point at the store folder.  Chroma
    always stores its database as ``<folder>/chroma.sqlite3`` and the
    filename is not configurable, so a path pointing at an existing file
    (even the ``chroma.sqlite3`` itself) is rejected.  The collection
    ``name`` selects which collection within the store file is
    addressed: a single ChromaDB store file can hold multiple
    collections, so reusing an existing folder with a new collection
    name creates a new collection in it.

    New Chroma collections are created with the :data:`CHROMA_HNSW_SPACE`
    HNSW distance space (cosine).  The configuration is only applied at
    collection creation; loading an existing collection keeps its own
    distance space.

    :param path: URI-style string with scheme prefix
        (e.g. ``"chroma:/path/to/dir"``,
        ``"qdrant:http://localhost:6333"``,
        ``"pgvector:postgresql://localhost/db"``)
    :param name: Collection name for the vector store
    :param embeddings: Embedding function to use
    :param logger: Logger instance
    :param create: If ``True``, allow creating a new store
    :returns: Instantiated LangChain VectorStore
    :raises ValueError: If the scheme is missing or unknown
    :raises FileNotFoundError: If ``create`` is ``False`` and a local
        ChromaDB store does not exist
    """
    scheme, sep, location = path.partition(":")
    if not sep:
        raise ValueError(
            f"Invalid vector store path '{path}': "
            f"expected format 'scheme:location' (e.g. 'chroma:/path/to/store')"
        )

    match scheme.lower():
        case "chroma":
            try:
                import chromadb
                from langchain_chroma import Chroma
            except ImportError:
                raise ImportError(
                    "ChromaDB backend not installed. "
                    "Install: pip install klea_utils[chroma]"
                ) from None

            store_dir = Path(location)
            if not store_dir.is_absolute():
                store_dir = Path.cwd() / store_dir
                logger.debug(f"Store path made absolute relative to cwd: {store_dir}")

            # Chroma always stores its database in <folder>/chroma.sqlite3
            # (the filename is not configurable), so a store is addressed
            # by its folder, never by the database file.
            if store_dir.is_file():
                raise FileNotFoundError(
                    f"'{store_dir}' is a file, not a folder. Chroma stores "
                    f"its database in a folder as 'chroma.sqlite3'; pass the "
                    f"folder path instead (e.g. 'chroma:{store_dir.parent}')"
                )

            if create:
                store_dir.mkdir(parents=True, exist_ok=True)
            else:
                if not store_dir.is_dir():
                    logger.error(f"Could not find folder: {store_dir}")
                    raise FileNotFoundError(f"Could not find folder: {store_dir}")

                store_db = store_dir / "chroma.sqlite3"
                if not store_db.is_file():
                    raise FileNotFoundError(f"ChromaDB not found at {store_db}")

            logger.debug(
                f"Loading Chroma vector store '{name}' from {store_dir.absolute()}"
            )

            settings = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=str(store_dir.absolute()),
                anonymized_telemetry=False,
            )
            return Chroma(
                collection_name=name,
                embedding_function=embeddings,
                client_settings=settings,
                collection_configuration={"hnsw": {"space": CHROMA_HNSW_SPACE}},
            )

        case "qdrant":
            try:
                from langchain_qdrant import QdrantVectorStore
                from qdrant_client import QdrantClient
            except ImportError:
                raise ImportError(
                    "Qdrant backend not installed. "
                    "Install: pip install klea_utils[qdrant]"
                ) from None

            client = QdrantClient(url=location)
            logger.debug(f"Loading Qdrant vector store '{name}' at {location}")
            return QdrantVectorStore(
                client=client,
                collection_name=name,
                embedding=embeddings,
            )

        case "pgvector":
            try:
                from langchain_postgres import PGVector
            except ImportError:
                raise ImportError(
                    "PGVector backend not installed. "
                    "Install: pip install klea_utils[pgvector]"
                ) from None

            logger.debug(
                f"Loading PGVector vector store '{name}' with connection {location}"
            )
            return PGVector(
                collection_name=name,
                embeddings=embeddings,
                connection=location,
            )

        case _:
            raise ValueError(
                f"Unknown vector store scheme '{scheme}'. "
                f"Supported: chroma, qdrant, pgvector"
            )
