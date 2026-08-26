---
status: "accepted"
date: 2026-08-27
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Retry worker batches that die with no results instead of marking them failed

## Context and Problem Statement

`dispatch_conversion_batches` (`chunk_worker.py:387`) spawns a fresh
`spawn` worker per batch.  A worker that dies without putting anything
on its `multiprocessing.Queue` (e.g. SIGSEGV from `docling-parse`,
SIGKILL from the OOM killer, or an uncaught exception before the first
`queue.put`) previously caused its whole batch to be marked `failed`
(`chunk_worker.py:434`).  In production this showed as
`Conversion worker #3 produced no results; marking its 200 files as
failed` — 200 innocent files in that batch were skipped for the run and
only retried on the next manual `chunk` invocation.  With incremental
`queue.put` per file the blast radius shrinks to the missing tail, but a
whole-batch death still costs a full batch.

We need the run to make progress without manual re-runs, while still
isolating a true poison file that crashes every worker that touches it.

## Decision Drivers

* A transient worker death (one-off Docling pipeline hiccup, as seen on
  `Zolnik_2026.pdf` — the re-run succeeded) should not cost a whole batch
  for this run.
* A poison file that crashes every worker must not cause an infinite
  retry loop.
* The fix must stay observable: the parent should log which file the
  worker was on and its `pid`/`exitcode`/`signal`.
* Keep the change small and testable via the existing `_run_one_worker`
  seam (`chunk_worker.py:295`).

## Considered Options

* **A. Mark whole batch failed (status quo)** — simple, no retry loop.
  Rejected: a transient kill costs 200 files for this run.

* **B. Re-queue the batch once, then mark failed (chosen)** — on a
  `None` batch (no puts) or a partial batch (`len(batch_results) <
  len(batch)`), re-queue the missing tail for one automatic retry with a
  fresh worker.  Track per-file retry counts in a `retried: set[str]`
  (keyed by file hash); a file that has already been retried and dies
  again is marked `failed` with `error="worker died repeatedly ..."`.
  The re-queued tail is prepended to `pending` so the suspect file
  (first missing, or `current_file` shared `Value` when available) is
  tried first and a second crash confirms it.

* **C. Re-queue indefinitely** — always retry.  Rejected: a poison file
  would loop forever.

## Decision Outcome

Chosen option: "B. Re-queue once, then mark failed", because a single
automatic retry turns the transient `200 files as failed` case into
`200 files retried in the next worker` while a poison file is still
isolated after one retry and surfaced as `failed` with the file name and
exitcode for investigation.

* `dispatch_conversion_batches` now keeps a `retried: set[str]` and, on a
  `None` or partial batch, splits the missing tail into `to_retry` vs
  `to_fail` based on that set, logs `re-queuing N files for retry` vs
  `worker died repeatedly`, and prepends `to_retry` to `pending`.
* `_run_one_worker` already puts per-file results incrementally and
  updates a shared `current_file` `Value(c_char*4096)` before each file,
  so on death the parent can log `Worker pid X died (exitcode=Y signal Z)
  while processing <file>; collected N/M`.

## Consequences

### Positive

* Transient worker deaths (the observed `Zolnik_2026.pdf` case) no longer
  require a manual re-run: the batch is retried immediately in a fresh
  worker and, as the live re-run showed, succeeds.
* Poison files are isolated to one file after one retry, not 200.
* Fully-cached re-runs and `deferred` (RSS-cap) paths are unchanged.

### Negative

* One extra worker spawn per transient death (model reload cost, ~10-30 s).
  Acceptable: it replaces a manual re-run of 200 files.

### Confirmation

* Updated `test_chunk_worker.py`: `test_dispatch_marks_dead_worker_batch_failed`
  now expects two `_run_one_worker` calls and the `worker died repeatedly`
  error; new `test_dispatch_retries_dead_worker_batch_once` covers the
  transient-retry-then-success path.  `pytest -m "not localonly"` 9 passed.

## Pros and Cons of the Options

### A. Mark whole batch failed

* Good, because no retry loop to reason about
* Bad, because a transient kill costs a full batch for this run

### B. Re-queue once, then mark failed (chosen)

* Good, because transient failures are retried immediately in this run
* Good, because poison files are isolated after one retry
* Bad, because one extra spawn on transient failure

### C. Re-queue indefinitely

* Good, because transient failures always eventually succeed
* Bad, because a poison file loops forever

## More Information

* Related: `devdocs/system/store-create.md`, `devdocs/adr/0001-chunk-workers.md`.
* Code: `utils_pkg/klea_utils/stores/chunk_worker.py:387` (`dispatch_conversion_batches`),
  `utils_pkg/klea_utils/stores/chunk_worker.py:295` (`_run_one_worker`),
  `utils_pkg/klea_utils/stores/chunk_worker.py:100` (`convert_batch_worker`).
