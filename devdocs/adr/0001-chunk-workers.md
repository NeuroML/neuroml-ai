---
status: "accepted"
date: 2026-08-26
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Subprocess chunk workers and batched DOI-cache writes for large-corpus ingestion

## Context and Problem Statement

`klea-stores-create chunk` converts every source document with Docling,
chunks it, and caches the result.  On a corpus of thousands of PDFs
(~8000 in production) two problems surfaced:

1. **The process grew without bound and was OOM-killed.**  Docling leaks
   memory per `converter.convert()` call that cannot be freed in-process
   (docling-project/docling#2788 — still open on 2.64.0 through 2.122.0,
   reproducible with/without OCR and with the PyPdfium backend; deleting
   references and `gc.collect()` do not reclaim it, and a fresh
   `DocumentConverter` just adds to the base).  `StoresBuilder.chunk_all`
   held every file's chunked `Document`s in a `results` list, so peak was
   O(corpus) plus the leak.  A live run hit ~22 GiB RSS and was killed;
   the cache write for the file being converted was left torn as a 0-byte
   `.pkl`.

2. **A torn cache entry bricked the next run.**  `_save_to_cache` wrote
   straight to the final `.pkl` path.  A mid-write kill left a truncated
   entry; `_load_from_cache` called `pickle.load` on it outside the
   per-file try/except, so one bad file aborted the whole run.  The same
   non-atomic pattern applied to `doi-cache.json`: `DoiResolver` rewrote
   the entire JSON on every new DOI, so a kill could wipe every cached
   resolution and force re-queries.

We need chunking (and the one-shot `build`) to stay memory-bounded on
corpora of any size, survive kills/crashes without losing the cache, and
remain resumable.

## Decision Drivers

* Must stay memory-bounded on very large corpora (the `chunk` live test
  showed parent RSS ~727 MiB steady while a worker climbed toward its cap;
  total RSS must not be O(corpus)).
* Keep the one-shot `build` UX but make it memory-bounded end-to-end
  (previously it stitched two in-process `chunk_all` runs and held all
  folded documents).
* Preserve resumability: a killed/interrupted run must be re-runnable
  without data loss (cache + manifests are the source of truth).
* Keep `store` strict (missing cache entry → error) while `build`
  tolerates conversion failures (failed files skipped so the rest is
  stored).
* Minimize public API churn pre-1.0, but do not keep tech debt for the
  sake of stability.
* Stay portable (Linux primary, Windows must still bound via batch size).
* Keep `doi-cache.json` human-readable for debugging while fixing its
  torn-write and rewrite-churn problems.

## Considered Options

### Chunking

* **A. In-process, unbounded** — status quo: `chunk_all` accumulates all
  `Document`s in `results`.  Simple, fastest for tiny corpora.
  Rejected: O(corpus) + Docling leak → OOM on large corpora; torn writes
  brick the next run.

* **B. In-process with `collect_results=False`** (Step 1) — release each
  file's `Document`s after caching; only `file_headings` is kept.
  Rejected as a complete fix: removes the O(corpus) list but Docling's
  leak stays in-process and still grows without bound.

* **C. Fresh `spawn` worker per batch (chosen)** — uncached files are
  converted in short-lived subprocess workers that write each file's
  `.pkl` and exit, returning only a lightweight `ChunkItemResult` (file
  name, hash, `file_headings` entry).  The parent stays lean (it never
  loads Docling's models) and reclaims the worker's leaked memory on
  exit.  A per-worker RSS cap (`--worker-mem-limit` GiB, default 4 —
  includes the ~1-1.5 GiB models) and a batch-size cap (`--worker-batch-size`
  default 200) bound the worker: it stops at whichever comes first,
  deferring the tail to a fresh worker.  Fully-cached re-runs have empty
  `pending` and spawn zero workers.

* **D. Persistent worker pool** — one long-lived worker reused across
  batches, restarted only when RSS exceeds the cap.  Fewer model loads,
  but the pool would keep the leak alive and complicate lifecycle
  (per-batch fresh workers give the same amortized cost with simpler
  reasoning, and batching is already cheap — a few extra minutes on a
  multi-hour run).

### DOI cache

* **A. Rewrite whole JSON per new DOI, non-atomic** — status quo.
  Rejected: O(N^2) writes and a torn file wipes the cache.

* **B. Atomic + batched JSON (chosen)** — keep JSON (readable), write
  atomically (`NamedTemporaryFile` + `os.replace`) and persist at most
  once per 25 new DOIs plus once on `close()`.  Sequential phases
  (parent cache-hit handling, then one worker at a time, each reloading
  at start) preserve appends without clobbering.

* **C. Pickle (batched)** — internal-only, faster, smaller.  Rejected:
  JSON's readability for debugging outweighs the marginal speed/size win;
  backward compatibility was explicitly waived but keeping JSON avoids a
  format migration for a cache that is not performance-critical.

### BM25 corpus

* The BM25 corpus was a single `pickle.dump(all_docs)` — O(corpus) on
  write.  Changed to **one `pickle.dump(batch)` per `embed_batch_size`**
  during `store_all`'s single pass (and the same in the standalone
  `write_bm25_store` helper, now removed as a separate public method
  because vector store + BM25 are always written together).  Readers loop
  `pickle.load` until `EOFError`.

## Decision Outcome

Chosen option: "C for chunking, B for DOI cache, batched pickle for BM25", because it bounds memory on very large corpora while keeping the parent lean, preserves resumability and human-readable caches, and reuses the existing streaming store path without a persistent worker pool.

* New module `klea_utils/stores/chunk_worker.py` (`ChunkWorkerConfig`,
  `ChunkItemResult`, `_current_rss_bytes` via `/proc/self/status`,
  `convert_batch_worker`, `_run_one_worker`, `dispatch_conversion_batches`).
* `StoresBuilder.chunk_all` is now always worker-isolated, cache-only,
  returning only `file_headings`.  `DEFAULT_WORKER_MEM_LIMIT = 4 GiB`,
  `DEFAULT_WORKER_BATCH_SIZE = 200`.  The old in-process loop,
  `metadata_map`, and `collect_results` are removed.  `build` composes
  `chunk_all` (workers) → `write_heading_template` → `_load_and_fold_results`
  (streaming generator, `strict=False`) → `store_all` (streaming, BM25
  inline).  `build` gains the same worker flags.  `store` stays strict via
  `strict=True`.
* `cli: chunk` / `build` gain `--worker-mem-limit` (GiB, includes the
  ~1-1.5 GiB models) and `--worker-batch-size`; a worker stops at
  whichever comes first.  Fully-cached re-runs spawn zero workers.  The
  worker configures its own logging via `setup_root_logger` (root at INFO
  so third-party DEBUG stays off, Klea namespaces at the requested level).
* Observability: `convert_batch_worker` puts one `ChunkItemResult` per
  file incrementally on a `multiprocessing.Queue` and updates a shared
  `current_file` `Value(c_char*4096)`; `_run_one_worker` polls the queue,
  logs `Completed {file} (N/M in batch)` and, on worker death, logs
  `pid`/`exitcode`/`signal` + the file from `current_file` and how many
  of the batch were collected.  `build` was restructured to avoid the
  second `chunk_all` materialisation.
* `DoiResolver._save_cache` is now atomic (temp + `os.replace`) and
  batched (every 25 + flush on `close()`).  JSON is kept.
* BM25 corpora are batched pickles; all three readers
  (`retrieval/bm25.py`, `ui/stores_create` store auto-lint, `store-lint`)
  loop `pickle.load` until `EOFError`.

## Consequences

### Positive

* `chunk` (and `build`'s chunk phase) is memory-bounded: parent ~0.7 GiB
  steady, each worker reclaimed on exit.  The live 16 GiB test showed the
  expected turnover (worker 3.2 → 3.4 → fresh worker at the cap).
* `store` (and `build`'s store phase) streams cached chunks one file at a
  time; BM25 batching bounds the write.
* `build` is memory-bounded end-to-end with no second in-memory chunk
  pass.
* Torn cache entries no longer brick the next run: unreadable `.pkl`s are
  moved aside as `*.pkl.corrupt` (bytes preserved) and re-converted;
  `_prune_cache` removes healed artifacts.  Same atomic guarantee for
  `doi-cache.json`.
* Incremental `queue.put` + `current_file` make worker deaths attributable
  to a single file instead of "200 files as failed".
* Fully-cached re-runs stay fast (no workers spawned).

### Negative

* Per-batch model reload cost: each fresh worker re-loads Docling's layout
  model + tokenizer (~10-30 s).  With a 4 GiB cap and large PDFs this
  is a few extra minutes on a multi-hour run — acceptable; users on 64 GiB
  machines can raise the cap to restart less often.
* `chunk_all` no longer returns chunked `Document`s — callers that need
  them read the cache via `_load_and_fold_results` (streaming).  This is a
  pre-1.0 API simplification per the project's versioning.
* `spawn` re-imports modules in the child; Windows falls back to
  batch-size-bounded batching where RSS is unavailable.

### Confirmation

* Existing `pytest -m "not localonly"` suites for `test_stores_ingestion`,
  `test_chunk_worker`, `test_biblio_doi` still pass; new tests cover
  worker batching, RSS-cap deferral, and atomic saves.  `ty` is clean for
  touched files.
* Live test: `klea-stores-create chunk --worker-mem-limit 16` on ~8000 PDFs
  showed parent RSS ~727 MiB steady and worker turnover at the cap
  (`docling#2788` leak reclaimed).

## Pros and Cons of the Options

| Option | Pros | Cons |
|--------|------|------|
| A in-process unbounded | Simple, fastest for tiny corpora | OOM on large corpora; torn writes |
| B `collect_results=False` only | Removes O(corpus) list | Docling leak remains |
| **C fresh spawn per batch (chosen)** | Bounds memory, reclaims leak, simple reasoning, zero workers when cached | Per-batch model reload cost |
| D persistent pool | Fewer reloads | Keeps leak alive, more complex lifecycle |
| DOI A rewrite per DOI | Simple | O(N^2) writes, torn file wipes cache |
| **DOI B atomic+batched JSON (chosen)** | Readable, safe, cheap | Minor: not the fastest binary format |
| DOI C pickle | Faster, smaller | Opaque, format migration |

## More Information

* Related system doc: `../system/store-create.md`.
* Upstream: docling-project/docling#2788 — Docling per-conversion leak.
* Original issue: `klea-stores-create chunk --no-ocr` fails with
  `Ran out of input` on a 0-byte `.pkl` left by a killed run; `chunk`
  OOM-killed at ~22 GiB RSS on ~8000 PDFs.
* Commits: `083e7c6` (atomic cache + corrupt handling), `ce734b2` (`collect_results`),
  `193de33` (store streaming), `54fb119` (BM25 batching), doi-cache batching,
  logging fix, build restructure + `collect_results` removal, worker
  observability.
* Code: `utils_pkg/klea_utils/stores/ingestion.py:256`, `utils_pkg/klea_utils/stores/chunk_worker.py:99`,
  `utils_pkg/klea_utils/biblio/doi.py:329`, `utils_pkg/klea_utils/ui/stores_create.py:132,619`,
  `devdocs/system/store-create.md`
