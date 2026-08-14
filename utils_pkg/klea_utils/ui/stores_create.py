#!/usr/bin/env python3
"""
CLI for creating stores from documents

File: klea_utils/ui/stores_create.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging

import typer

from ..plogging import setup_root_logger

app = typer.Typer(help="Create stores from documents")


@app.command()
def build(
    source_dir: str = typer.Argument(help="Directory containing source documents"),
    collection_name: str = typer.Option(
        ...,
        "--collection",
        "-n",
        help="Collection name for the vector store. Must match the "
        "'name' of the corresponding vector_stores/bm25_stores entry in "
        "the RAG config file (e.g. klea.json); a different name on an "
        "existing store file creates a new collection",
    ),
    store_path: str = typer.Option(
        ...,
        "--store",
        "-s",
        help="Vector store URI (e.g. chroma:/path/to/store). For local "
        "Chroma stores, point at the store folder: the database file "
        "inside it is always named chroma.sqlite3",
    ),
    embedding_model: str = typer.Option(
        "ollama:bge-m3:latest",
        "--model",
        "-m",
        help="Embedding model identifier",
    ),
    max_tokens: int = typer.Option(
        450, "--max-tokens", help="Maximum tokens per chunk"
    ),
    ocr: bool = typer.Option(
        True,
        "--ocr/--no-ocr",
        help="Whether to perform optical character recognition (OCR) "
        "during PDF conversion (default: on). Keep for scanned/image "
        "PDFs; disable for text-based PDFs to speed up conversion "
        "significantly",
    ),
    metadata_map_path: str = typer.Option(
        None,
        "--metadata-map",
        "-M",
        help="JSON file keyed by source filename; each file entry maps "
        "heading chains to metadata dicts (with per-file DEFAULT fallback). "
        "If not given, build generates metadata-map.template.json from the "
        "extraction and uses it without a review step",
    ),
    bm25_store: str = typer.Option(
        None,
        "--bm25-store",
        help="Write the combined document corpus to this path for BM25 "
        "retrieval (a pickle of all chunked documents). Defaults to "
        "<collection>.pkl in the current directory; the file can be "
        "moved after creation",
        show_default="<collection>.pkl",
    ),
    embed_batch_size: int = typer.Option(
        256,
        "--embed-batch-size",
        help="Number of chunks embedded per store write call. Smaller "
        "values report progress more frequently; larger values reduce "
        "per-request overhead on very large corpora",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process all files even if unchanged"
    ),
):
    """Full pipeline: chunk, embed, and write to a vector store.

    Processes all files in SOURCE_DIR: converts them with Docling,
    chunks them, embeds them, and writes to the vector store.
    Processed chunks are cached in ``<source_dir>/.klea-cache/`` so
    subsequent runs (e.g. with ``--metadata-map``) skip conversion.

    The ``--bm25-store`` option (default ``<collection>.pkl`` in the
    current directory) writes the combined chunked documents to a single
    pickle file that can be used as a BM25 store.

    The optional ``--metadata-map`` / ``-M`` flag accepts a JSON file
    organised by source file.  Within each file entry, the most specific
    heading chain match wins; a ``DEFAULT`` entry provides fallback for
    any heading not listed.

    Example metadata-map.json::

        {
            "PrimerOnCElegans.md": {
                "DEFAULT": {},
                "C. elegans tissue morphology": {
                    "url": "https://example.com/worm"
                }
            },
            "c302-paper.pdf": {
                "DEFAULT": {
                    "url": "https://example.com/c302"
                }
            }
        }
    """
    setup_root_logger("klea-stores-create")
    logger = logging.getLogger("klea-stores-create")

    logger.info(
        f"Building vector store '{collection_name}' at {store_path}"
        f"\n  Source: {source_dir}"
        f"\n  Model: {embedding_model}"
        f"\n  Max tokens: {max_tokens}"
        f"\n  Metadata map: {metadata_map_path or '(none)'}"
    )

    # Typer cannot compute a default that depends on another argument, so
    # the dynamic --bm25-store default is resolved here.
    if bm25_store is None:
        bm25_store = f"{collection_name}.pkl"
    logger.info(f"BM25 store: {bm25_store}")

    try:
        # Lazy: importing StoresBuilder pulls in ingestion.py -> llm.py ->
        # langchain_huggingface/langchain_ollama, stores/utils.py ->
        # chromadb/qdrant etc.  Deferring to function body keeps
        # --help fast (Python only needs the function signature).
        from klea_utils.stores.ingestion import StoresBuilder

        builder = StoresBuilder(
            embedding_model=embedding_model,
            logger=logger,
            max_tokens=max_tokens,
            do_ocr=ocr,
            embed_batch_size=embed_batch_size,
        )
        builder.build(
            source_dir=source_dir,
            store_uri=store_path,
            collection_name=collection_name,
            force=force,
            metadata_map_path=metadata_map_path,
            bm25_path=bm25_store,
        )
        logger.info(f"Done -- collection '{collection_name}' is ready")
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None


@app.command()
def chunk(
    source_dir: str = typer.Argument(help="Directory containing source documents"),
    max_tokens: int = typer.Option(
        450, "--max-tokens", help="Maximum tokens per chunk"
    ),
    ocr: bool = typer.Option(
        True,
        "--ocr/--no-ocr",
        help="Whether to perform optical character recognition (OCR) "
        "during PDF conversion (default: on). Keep for scanned/image "
        "PDFs; disable for text-based PDFs to speed up conversion "
        "significantly",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process all files even if unchanged"
    ),
):
    """Chunk and cache documents without writing to a vector store.

    Converts all files in SOURCE_DIR with Docling, chunks them, and
    caches the result in ``<source_dir>/.klea-cache/``.  Also writes a
    ``metadata-map.template.json`` file organised by source file, with
    empty ``{}`` placeholders for each heading chain.  Fill in the
    metadata values and pass the file to
    ``klea-stores-create store --metadata-map``.
    """
    setup_root_logger("klea-stores-create")
    logger = logging.getLogger("klea-stores-create")

    logger.info(f"Chunking documents in {source_dir}\n  Max tokens: {max_tokens}")

    try:
        # Lazy: importing StoresBuilder pulls in ingestion.py -> llm.py ->
        # langchain_huggingface/langchain_ollama, stores/utils.py ->
        # chromadb/qdrant etc.  Deferring to function body keeps
        # --help fast (Python only needs the function signature).
        from pathlib import Path

        from klea_utils.stores.ingestion import TEMPLATE_FILE_NAME, StoresBuilder

        builder = StoresBuilder(
            embedding_model="",  # not needed for chunking only
            logger=logger,
            max_tokens=max_tokens,
            do_ocr=ocr,
        )
        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        _, file_headings = builder.chunk_all(source_path, force=force)
        builder.write_heading_template(file_headings, source_path)
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None

    if not file_headings:
        logger.error(
            f"No files were successfully chunked from {source_path} -- see errors above"
        )
        raise typer.Exit(1)
    logger.info("Chunking complete -- cache is ready")
    logger.info(
        f"Review/update the metadata map before storing: "
        f"{source_path / TEMPLATE_FILE_NAME} -- fill in per-heading entries, "
        "then run 'klea-stores-create store' with --metadata-map"
    )
    logger.info("Metadata map summary:")
    try:
        import json

        from klea_utils.stores.map_lint import (
            format_metadata_lint_report,
            lint_metadata_map,
        )

        map_path = source_path / TEMPLATE_FILE_NAME
        with open(map_path) as f:
            data = json.load(f)
        print(format_metadata_lint_report(lint_metadata_map(data)))
    except Exception as e:
        logger.warning(f"Could not lint metadata map: {e}")


@app.command()
def store(
    source_dir: str = typer.Argument(help="Directory containing source documents"),
    collection_name: str = typer.Option(
        ...,
        "--collection",
        "-n",
        help="Collection name for the vector store. Must match the "
        "'name' of the corresponding vector_stores/bm25_stores entry in "
        "the RAG config file (e.g. klea.json); a different name on an "
        "existing store file creates a new collection",
    ),
    store_path: str = typer.Option(
        ...,
        "--store",
        "-s",
        help="Vector store URI (e.g. chroma:/path/to/store). For local "
        "Chroma stores, point at the store folder: the database file "
        "inside it is always named chroma.sqlite3",
    ),
    embedding_model: str = typer.Option(
        "ollama:bge-m3:latest",
        "--model",
        "-m",
        help="Embedding model identifier",
    ),
    max_tokens: int = typer.Option(
        450, "--max-tokens", help="Maximum tokens per chunk (for files not yet cached)"
    ),
    ocr: bool = typer.Option(
        True,
        "--ocr/--no-ocr",
        help="Whether to perform optical character recognition (OCR) "
        "during PDF conversion (default: on). Keep for scanned/image "
        "PDFs; disable for text-based PDFs to speed up conversion "
        "significantly",
    ),
    metadata_map_path: str = typer.Option(
        None,
        "--metadata-map",
        "-M",
        help="JSON file keyed by source filename; each file entry maps "
        "heading chains to metadata dicts (with per-file DEFAULT fallback). "
        "Defaults to metadata-map.template.json in the source "
        "directory; required when no template exists",
    ),
    bm25_store: str = typer.Option(
        None,
        "--bm25-store",
        help="Write the combined document corpus to this path for BM25 "
        "retrieval (a pickle of all chunked documents). Defaults to "
        "<collection>.pkl in the current directory; the file can be "
        "moved after creation",
        show_default="<collection>.pkl",
    ),
    embed_batch_size: int = typer.Option(
        256,
        "--embed-batch-size",
        help="Number of chunks embedded per store write call. Smaller "
        "values report progress more frequently; larger values reduce "
        "per-request overhead on very large corpora",
    ),
    force: bool = typer.Option(
        False, "--force", "-f", help="Re-process all files even if unchanged"
    ),
):
    """Write cached document chunks to a vector store.

    Reads previously cached chunks from ``<source_dir>/.klea-cache/``,
    optionally applies a metadata map (per-file format), and writes
    them to the vector store.  Unseen files are converted and chunked
    on the fly.

    The ``--bm25-store`` option (default ``<collection>.pkl`` in the
    current directory) writes the combined chunked documents to a single
    pickle file that can be used as a BM25 store.

    Run ``klea-stores-create chunk`` first to populate the cache and
    generate a ``metadata-map.template.json``.
    """
    setup_root_logger("klea-stores-create")
    logger = logging.getLogger("klea-stores-create")

    logger.info(
        f"Storing cached chunks to '{collection_name}' at {store_path}"
        f"\n  Source: {source_dir}"
        f"\n  Model: {embedding_model}"
        f"\n  Metadata map: {metadata_map_path or '(none)'}"
    )

    # Typer cannot compute a default that depends on another argument, so
    # the dynamic --bm25-store default is resolved here.
    if bm25_store is None:
        bm25_store = f"{collection_name}.pkl"
    logger.info(f"BM25 store: {bm25_store}")

    try:
        from pathlib import Path

        from klea_utils.stores.ingestion import TEMPLATE_FILE_NAME, StoresBuilder

        builder = StoresBuilder(
            embedding_model=embedding_model,
            logger=logger,
            max_tokens=max_tokens,
            do_ocr=ocr,
            embed_batch_size=embed_batch_size,
        )
        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        metadata_map = builder._resolve_metadata_map(source_path, metadata_map_path)
        if metadata_map is None:
            # store is the review-driven path: it consumes the template a
            # prior chunk run wrote (or an explicit --metadata-map), so a
            # missing map here is a misconfiguration, not a silent fallback.
            raise ValueError(
                f"No metadata map found for {source_path}. Run "
                f"'klea-stores-create chunk' to generate "
                f"{TEMPLATE_FILE_NAME} in the source directory, or pass "
                f"--metadata-map."
            )

        results, _ = builder.chunk_all(
            source_path, metadata_map=metadata_map, force=force
        )
        if not results:
            logger.error(
                f"No files were successfully chunked from "
                f"{source_path} -- see errors above"
            )
            raise typer.Exit(1)
        builder.store_all(
            results, store_path, collection_name, force=force, bm25_path=bm25_store
        )
        logger.info(f"Done -- collection '{collection_name}' is ready")
    except typer.Exit:
        raise
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None


@app.command()
def map_lint(
    source_dir: str = typer.Argument(
        help="Directory containing the metadata map (uses "
        "metadata-map.template.json unless --metadata-map is given)"
    ),
    metadata_map_path: str = typer.Option(
        None,
        "--metadata-map",
        "-M",
        help="Explicit metadata-map JSON file to lint, instead of the "
        "template in SOURCE_DIR",
    ),
):
    """Report issues in a metadata map so it can be reviewed efficiently.

    Runs only deterministic checks (missing fields, suspicious titles or
    DOIs, year/filename mismatches, stale 'venue' keys, excess url* keys,
    placeholder counts) -- no LLM is needed.  Useful after editing the
    map by hand, and printed automatically at the end of 'chunk'.
    """
    setup_root_logger("klea-stores-create")
    logger = logging.getLogger("klea-stores-create")

    try:
        import json
        from pathlib import Path

        from klea_utils.stores.ingestion import TEMPLATE_FILE_NAME
        from klea_utils.stores.map_lint import (
            format_metadata_lint_report,
            lint_metadata_map,
        )

        source_path = Path(source_dir).resolve()
        if not source_path.is_dir():
            raise FileNotFoundError(f"Source directory not found: {source_path}")

        if metadata_map_path:
            map_path = Path(metadata_map_path)
        else:
            map_path = source_path / TEMPLATE_FILE_NAME
        if not map_path.is_file():
            raise FileNotFoundError(f"Metadata map not found: {map_path}")

        with open(map_path) as f:
            data = json.load(f)
        report = lint_metadata_map(data)
        print(format_metadata_lint_report(report))
    except Exception as e:
        logger.error(f"Failed: {e}")
        raise typer.Exit(1) from None


if __name__ == "__main__":
    app()
