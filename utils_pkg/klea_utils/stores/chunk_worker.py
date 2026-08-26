#!/usr/bin/env python3
"""
Subprocess chunking workers -- isolate Docling's per-conversion memory leak

File: klea_utils/stores/chunk_worker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from __future__ import annotations

import logging
import multiprocessing
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Any

from .ingestion import StoresBuilder

logger = logging.getLogger(__name__)

#: How often (seconds) the parent polls a worker's result queue while it
#: runs, and the overall cap before a worker is considered hung and
#: killed (its batch is then marked failed and the run continues).
_WORKER_POLL_INTERVAL = 1.0
_WORKER_TIMEOUT_SECONDS = 4 * 3600

#: Console format for the worker's own stderr, matching the application's
#: ``setup_root_logger`` style so interleaved output stays consistent.
_WORKER_LOG_FORMAT = (
    "%(asctime)s %(name)s (%(levelname)s) in '%(funcName)s' >>> %(message)s"
)


@dataclass
class ChunkWorkerConfig:
    """Picklable configuration handed to one conversion worker batch.

    The parent's :class:`StoresBuilder` is not picklable (it holds the
    Docling converter, chunker, resolver, etc.), so the worker rebuilds
    its own from these plain fields.
    """

    source_path: str
    max_tokens: int
    tokenizer_model: str
    do_ocr: bool
    mem_limit_bytes: int | None
    log_level: int


@dataclass
class ChunkItemResult:
    """Outcome of converting one file inside a worker.

    ``status`` is one of:

    * ``ok`` -- converted and cached; *file_headings_entry* holds the
      file's metadata-map template entry
    * ``zero_chunks`` -- converted but produced no chunks (scanned PDF)
    * ``failed`` -- conversion raised; *error* has the message
    * ``deferred`` -- not processed because the worker hit its RSS cap;
      *file_path* lets the parent re-dispatch it to a fresh worker
    """

    file_name: str
    file_hash: str
    status: str
    file_headings_entry: dict[str, Any] | None = None
    error: str | None = None
    file_path: str | None = None


def _current_rss_bytes() -> int | None:
    """Return this process's current RSS in bytes, or ``None`` if unknown.

    Read from ``/proc/self/status`` (Linux).  On other platforms, or when
    the file is unreadable, returns ``None`` so callers fall back to
    batch-size-bounded batching instead of a memory cap.
    """
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    # Line is "VmRSS:    <kB>".
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _over_mem_limit(mem_limit_bytes: int | None) -> bool:
    """Return ``True`` when this process's RSS exceeds *mem_limit_bytes*."""
    if mem_limit_bytes is None:
        return False
    rss = _current_rss_bytes()
    if rss is None:
        return False
    return rss > mem_limit_bytes


def convert_batch_worker(
    config: ChunkWorkerConfig, items: list[tuple[str, str]]
) -> list[ChunkItemResult]:
    """Convert and cache *items* in this process, one per result.

    Runs in a spawned child for real chunking, or directly in the parent
    under tests.  Rebuilds its own :class:`StoresBuilder`, so Docling's
    per-conversion memory leak (unreclaimable in-process; see
    docling-project/docling#2788) is confined to this process and
    reclaimed by the OS when it exits.

    Before each file, if the process RSS exceeds *config.mem_limit_bytes*
    the remaining items are returned as ``deferred`` (the parent re-runs
    them in a fresh worker).  On platforms where RSS is unavailable the
    cap is skipped and the parent's fixed batch size bounds the worker
    instead.

    :param config: Worker configuration
    :param items: ``(absolute_file_path, file_hash)`` tuples to process
    :returns: One :class:`ChunkItemResult` per item, in input order
    """
    logging.basicConfig(
        level=config.log_level,
        format=_WORKER_LOG_FORMAT,
        force=False,
    )
    worker_logger = logging.getLogger("klea-stores-create.ChunkWorker")
    builder = StoresBuilder(
        embedding_model="",
        logger=worker_logger,
        max_tokens=config.max_tokens,
        tokenizer_model=config.tokenizer_model,
        do_ocr=config.do_ocr,
    )
    builder._ensure_tokenizer()
    source_path = Path(config.source_path)
    resolver = builder._make_resolver(source_path)

    results: list[ChunkItemResult] = []
    try:
        for ctr, (file_path_str, file_hash) in enumerate(items, 1):
            if _over_mem_limit(config.mem_limit_bytes):
                worker_logger.warning(
                    f"Worker RSS exceeded limit; deferring "
                    f"{len(items) - ctr + 1} remaining files to a fresh worker"
                )
                for remaining_path, remaining_hash in items[ctr - 1 :]:
                    results.append(
                        ChunkItemResult(
                            file_name=Path(remaining_path).name,
                            file_hash=remaining_hash,
                            status="deferred",
                            file_path=remaining_path,
                        )
                    )
                break

            file_path = Path(file_path_str)
            try:
                docs, extracted = builder._convert_and_chunk(file_path, resolver)
                builder._save_to_cache(docs, extracted, source_path, file_hash)
            except Exception as e:
                worker_logger.error(f"Failed to process {file_path.name}: {e}")
                results.append(
                    ChunkItemResult(
                        file_name=file_path.name,
                        file_hash=file_hash,
                        status="failed",
                        error=str(e),
                    )
                )
                continue

            if not docs:
                worker_logger.warning(
                    f"No chunks produced for {file_path.name}. This usually "
                    f"means the PDF is scanned/image-based and its text "
                    f"could not be extracted with OCR disabled. Re-run "
                    f"with OCR enabled (drop --no-ocr) or run "
                    f"'klea-stores-create pre-check' to classify it."
                )
                results.append(
                    ChunkItemResult(
                        file_name=file_path.name,
                        file_hash=file_hash,
                        status="zero_chunks",
                    )
                )
                continue

            entry = builder._build_file_headings_entry(file_path.name, extracted, docs)
            results.append(
                ChunkItemResult(
                    file_name=file_path.name,
                    file_hash=file_hash,
                    status="ok",
                    file_headings_entry=entry,
                )
            )
    finally:
        # Flush any buffered DOI resolutions written by this worker so
        # later workers see them.  The resolver is protocol-typed (only
        # ``resolve`` is guaranteed), so close defensively.
        close = getattr(resolver, "close", None)
        if close is not None:
            close()

    return results


def _run_worker_and_put(
    config: ChunkWorkerConfig,
    items: list[tuple[str, str]],
    queue: multiprocessing.Queue,
) -> None:
    """Spawn entry point: run the worker and put its results on *queue*.

    Exceptions are swallowed -- the process is about to exit and the
    parent's polling/kill logic handles a missing result.
    """
    try:
        queue.put(convert_batch_worker(config, items))
    except Exception as e:
        # The process is about to exit; the parent's polling/kill logic
        # handles a missing result.  Log for the record.
        logger.error(f"Chunk worker crashed before returning results: {e}")


def _run_one_worker(
    ctx: Any,
    config: ChunkWorkerConfig,
    batch: list[tuple[str, str]],
) -> list[ChunkItemResult] | None:
    """Spawn one worker process for *batch* and collect its results.

    Returns ``None`` when the worker died without returning results
    (e.g. it was OOM-killed) or exceeded the run-time cap; the caller
    then marks the batch as failed and the run continues.  Extracted as
    its own function so the parent's batching/deferral logic can be
    tested by monkeypatching it instead of exercising real spawns.
    """
    queue = ctx.Queue()
    process = ctx.Process(target=_run_worker_and_put, args=(config, batch, queue))
    process.start()

    batch_results: list[ChunkItemResult] | None = None
    started = time.monotonic()
    while True:
        try:
            batch_results = queue.get(timeout=_WORKER_POLL_INTERVAL)
            break
        except Empty:
            if not process.is_alive():
                # Worker died without returning results (e.g. it was
                # OOM-killed); treat the batch as failed.
                break
            if time.monotonic() - started > _WORKER_TIMEOUT_SECONDS:
                logger.error(
                    f"Worker exceeded the {_WORKER_TIMEOUT_SECONDS}s cap; killing it"
                )
                process.kill()
                process.join()
                break

    process.join(timeout=5)
    if process.is_alive():
        logger.error("Worker did not exit after producing results; killing it")
        process.kill()
        process.join()
    return batch_results


def dispatch_conversion_batches(
    parent_logger: logging.Logger,
    config: ChunkWorkerConfig,
    items: list[tuple[str, str]],
    batch_size: int,
) -> list[ChunkItemResult]:
    """Convert *items* by spawning a fresh worker process per batch.

    A fresh process per batch is deliberate: Docling leaks memory per
    conversion that cannot be freed in-process, so each worker's leak is
    reclaimed when it exits.  A worker stops at whichever comes first --
    its RSS cap (``config.mem_limit_bytes``) or *batch_size* -- marking
    the unprocessed tail ``deferred``; the parent re-dispatches those to
    a fresh worker.  On platforms where RSS is unavailable only
    *batch_size* bounds a worker.  A worker that dies without returning
    results (e.g. OOM-killed) has its whole batch marked ``failed`` and
    the run continues -- those files are simply re-converted on a later
    run.

    :param parent_logger: Logger for dispatch progress/errors
    :param config: Worker configuration
    :param items: ``(absolute_file_path, file_hash)`` items to convert
    :param batch_size: Max items handed to any single worker
    :returns: Flattened :class:`ChunkItemResult` for every item
    """
    ctx = multiprocessing.get_context("spawn")
    results: list[ChunkItemResult] = []
    pending: list[tuple[str, str]] = list(items)
    batch_no = 0

    while pending:
        batch_no += 1
        batch = pending[:batch_size]
        pending = pending[batch_size:]

        parent_logger.info(
            f"Spawning conversion worker #{batch_no} for {len(batch)} files"
        )
        batch_results = _run_one_worker(ctx, config, batch)

        if batch_results is None:
            parent_logger.error(
                f"Conversion worker #{batch_no} produced no results; "
                f"marking its {len(batch)} files as failed"
            )
            for file_path_str, file_hash in batch:
                results.append(
                    ChunkItemResult(
                        file_name=Path(file_path_str).name,
                        file_hash=file_hash,
                        status="failed",
                        error="worker produced no results",
                    )
                )
            continue

        # Re-dispatch the deferred tail to a fresh worker (clean memory);
        # the deferred results themselves are not reported -- those files
        # are covered by the re-run.
        deferred = [r for r in batch_results if r.status == "deferred" and r.file_path]
        if deferred:
            parent_logger.warning(
                f"Worker #{batch_no} deferred {len(deferred)} files to a "
                f"fresh worker (memory cap)"
            )
            pending[:0] = [
                (r.file_path, r.file_hash) for r in deferred if r.file_path is not None
            ]
        results.extend(r for r in batch_results if r.status != "deferred")

    return results
