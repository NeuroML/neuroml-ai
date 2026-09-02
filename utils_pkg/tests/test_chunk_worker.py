#!/usr/bin/env python3
"""
Tests for subprocess chunking workers.

File: tests/test_chunk_worker.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
from pathlib import Path

import pytest
from klea_utils.stores.chunk_worker import (
    ChunkItemResult,
    ChunkWorkerConfig,
    convert_batch_worker,
    dispatch_conversion_batches,
)
from klea_utils.stores.ingestion import CACHE_DIR_NAME, StoresBuilder, _hash_file

logger = logging.getLogger(__name__)

TEST_MD_CONTENT = """# Test Document
## Section 1
This is some content in section 1. It has enough text to produce at least
one chunk for the vector store.

## Section 2
This is content in section 2. It also has enough text to produce at least
one chunk for the vector store. We need enough text here to make sure
the chunker actually produces some chunks.
"""


def _config(tmp_path: Path, **overrides) -> ChunkWorkerConfig:
    """Build a worker config with sensible test defaults."""
    defaults = {
        "source_path": str(tmp_path),
        "max_tokens": 450,
        "tokenizer_model": "BAAI/bge-m3",
        "do_ocr": False,
        "mem_limit_bytes": None,
        "log_level": logging.DEBUG,
    }
    defaults.update(overrides)
    return ChunkWorkerConfig(**defaults)


def test_worker_converts_and_caches(tmp_path):
    """A worker converts an uncached file, caches it, and returns the entry."""
    md = tmp_path / "test.md"
    md.write_text(TEST_MD_CONTENT)
    file_hash = _hash_file(md)

    results = convert_batch_worker(_config(tmp_path), [(str(md), file_hash)])

    assert len(results) == 1
    result = results[0]
    assert result.status == "ok"
    assert result.file_name == "test.md"
    assert result.file_headings_entry is not None
    assert "DEFAULT" in result.file_headings_entry
    cache_file = tmp_path / CACHE_DIR_NAME / f"{file_hash.replace(':', '_')}.pkl"
    assert cache_file.is_file()


def test_worker_marks_zero_chunks(tmp_path, caplog, monkeypatch):
    """A conversion yielding no chunks is reported, not failed."""
    md = tmp_path / "test.md"
    md.write_text(TEST_MD_CONTENT)
    file_hash = _hash_file(md)

    monkeypatch.setattr(
        StoresBuilder,
        "_convert_and_chunk",
        lambda self, file_path, resolver: ([], {}),
    )
    with caplog.at_level(logging.WARNING):
        results = convert_batch_worker(_config(tmp_path), [(str(md), file_hash)])

    assert results[0].status == "zero_chunks"
    assert "No chunks produced for test.md" in caplog.text


def test_worker_marks_failure(tmp_path, caplog, monkeypatch):
    """A conversion exception is reported as failed and the run continues."""
    md = tmp_path / "test.md"
    md.write_text(TEST_MD_CONTENT)
    file_hash = _hash_file(md)

    def boom(self, file_path, resolver):
        raise RuntimeError("conversion exploded")

    monkeypatch.setattr(StoresBuilder, "_convert_and_chunk", boom)
    with caplog.at_level(logging.ERROR):
        results = convert_batch_worker(_config(tmp_path), [(str(md), file_hash)])

    assert results[0].status == "failed"
    assert results[0].error is not None
    assert "conversion exploded" in results[0].error
    assert "Failed to process test.md" in caplog.text


def test_worker_defers_remaining_when_over_mem_limit(tmp_path, caplog, monkeypatch):
    """The tail is deferred when the worker's RSS exceeds the cap."""
    md = tmp_path / "test.md"
    md.write_text(TEST_MD_CONTENT)
    file_hash = _hash_file(md)

    # Fake the RSS probe so the very first file trips the cap: nothing is
    # converted, everything comes back deferred for a fresh worker.
    monkeypatch.setattr(
        "klea_utils.stores.chunk_worker._current_rss_bytes", lambda: 10**12
    )
    with caplog.at_level(logging.WARNING):
        results = convert_batch_worker(
            _config(tmp_path, mem_limit_bytes=10**9),
            [(str(md), file_hash)],
        )

    assert [r.status for r in results] == ["deferred"]
    assert results[0].file_path == str(md)
    assert "deferring 1 remaining files" in caplog.text


def test_dispatch_batches_by_batch_size(tmp_path, monkeypatch):
    """Items are handed to workers in batch_size chunks."""
    seen: list[int] = []

    def fake_run_one_worker(ctx, config, batch):
        seen.append(len(batch))
        return [
            ChunkItemResult(file_name=Path(p).name, file_hash=h, status="ok")
            for p, h in batch
        ]

    monkeypatch.setattr(
        "klea_utils.stores.chunk_worker._run_one_worker", fake_run_one_worker
    )
    items = [(str(tmp_path / f"{i}.md"), f"xxh64:{i}") for i in range(5)]
    results = dispatch_conversion_batches(
        logger, _config(tmp_path), items, batch_size=2
    )

    assert seen == [2, 2, 1]
    assert [r.status for r in results] == ["ok"] * 5


def test_dispatch_redispatch_deferred_tail(tmp_path, monkeypatch):
    """Deferred results are handed back to a fresh worker."""
    calls: list[list[tuple[str, str]]] = []

    def fake_run_one_worker(ctx, config, batch):
        calls.append(batch)
        if len(calls) == 1:
            # First worker defers its whole batch to a fresh worker.
            return [
                ChunkItemResult(
                    file_name=Path(p).name,
                    file_hash=h,
                    status="deferred",
                    file_path=p,
                )
                for p, h in batch
            ]
        return [
            ChunkItemResult(file_name=Path(p).name, file_hash=h, status="ok")
            for p, h in batch
        ]

    monkeypatch.setattr(
        "klea_utils.stores.chunk_worker._run_one_worker", fake_run_one_worker
    )
    items = [(str(tmp_path / f"{i}.md"), f"xxh64:{i}") for i in range(5)]
    results = dispatch_conversion_batches(
        logger, _config(tmp_path), items, batch_size=3
    )

    # First call defers 3 (re-added to the front); second call takes 3
    # and processes them; third call takes the remaining 2.
    assert len(calls) == 3
    assert [r.status for r in results] == ["ok"] * 5


def test_dispatch_marks_dead_worker_batch_failed(tmp_path, monkeypatch):
    """A dead worker batch is retried once, then marked failed (poison)."""
    calls: list[list[tuple[str, str]]] = []

    def fake_run_one_worker(ctx, config, batch):
        calls.append(list(batch))

    monkeypatch.setattr(
        "klea_utils.stores.chunk_worker._run_one_worker",
        fake_run_one_worker,
    )
    items = [(str(tmp_path / "a.md"), "xxh64:a")]
    results = dispatch_conversion_batches(
        logger, _config(tmp_path), items, batch_size=10
    )

    assert len(calls) == 2  # original + one retry
    assert results[0].status == "failed"
    assert results[0].error is not None
    assert "worker died repeatedly" in results[0].error


def test_dispatch_retries_dead_worker_batch_once(tmp_path, monkeypatch):
    """A transient worker death is retried and then succeeds."""
    calls: list[list[tuple[str, str]]] = []

    def fake_run_one_worker(ctx, config, batch):
        calls.append(list(batch))
        if len(calls) == 1:
            return None
        return [
            ChunkItemResult(file_name=Path(p).name, file_hash=h, status="ok")
            for p, h in batch
        ]

    monkeypatch.setattr(
        "klea_utils.stores.chunk_worker._run_one_worker",
        fake_run_one_worker,
    )
    items = [(str(tmp_path / "a.md"), "xxh64:a")]
    results = dispatch_conversion_batches(
        logger, _config(tmp_path), items, batch_size=10
    )

    assert len(calls) == 2
    assert results[0].status == "ok"


@pytest.mark.localonly
def test_worker_real_spawn_converts(tmp_path):
    """A real spawned worker converts an uncached file end-to-end."""
    md = tmp_path / "test.md"
    md.write_text(TEST_MD_CONTENT)
    file_hash = _hash_file(md)

    results = dispatch_conversion_batches(
        logger, _config(tmp_path), [(str(md), file_hash)], batch_size=10
    )

    assert results[0].status == "ok"
    assert (tmp_path / CACHE_DIR_NAME).is_dir()
