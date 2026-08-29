---
status: "accepted"
date: 2026-08-28
decision-makers: Ankur Sinha
consulted: ""
informed: ""
---

# Atomic chunk cache with prune for resumable ingest

## Context and Problem Statement

``klea-stores-create chunk`` writes one ``.pkl`` per source file into
``<source_dir>/.klea-cache/`` (xxh64-hashed name) and a shared
``doi-cache.json``.  A crash, OOM-kill of a ``spawn`` worker, or a
``Ctrl-C`` can leave a torn file (0-byte ``.pkl``, truncated JSON) and
orphaned cache entries whose source file was renamed or deleted.
``store`` then reads the cache; a torn ``.pkl`` bricked the next run
(``pickle.load`` raised outside the per-file try/except per
``ADR-0001``).

How should the ingest cache stay resumable and self-healing across
crashes and source edits?

## Decision Drivers

* Every source file must be resumable: a killed ``chunk`` must be
  re-runnable without manual ``rm .klea-cache``.
* Torn writes must not survive as valid cache entries.
* Renamed or deleted source files must not leave orphaned
  ``.pkl``/``doi-cache`` entries that grow unbounded.
* ``doi-cache.json`` is human-readable JSON (debug), so the fix must
  keep it as JSON.
* ``store`` must remain strict: a source file with no cache entry
  aborts, so the cache is the source of truth for ``store``.

## Considered Options

* **A. Direct writes, no prune (old)** -- ``_save_to_cache`` wrote
  straight to the final ``.pkl`` path; ``DoiResolver._save_cache``
  rewrote the whole JSON on every new DOI; no prune.  Rejected:
  torn file bricks the next run; orphans accumulate; DOI cache has
  torn-JSON risk and O(N^2) writes.
* **B. Atomic writes + periodic prune + corrupt quarantine (chosen)**
  -- every file cache write uses ``NamedTemporaryFile`` + ``os.replace``
  (`` KleaStoresBuilder._save_to_cache``); unreadable ``.pkl``s are
  moved aside as ``*.pkl.corrupt`` (bytes preserved for debugging)
  and re-converted; ``DoiResolver._save_cache`` is also atomic
  (``NamedTemporaryFile`` + ``os.replace``) and batched (every 25 new
  DOIs plus flush on ``close()``).  At the end of **every**
  ``chunk_all`` call (worker or not) ``_prune_cache(current_hashes)``
  removes ``.pkl``s (and ``*.corrupt`` artifacts) whose hash no longer
  matches any source file.  Fully-cached re-runs have empty ``pending``
  and spawn zero workers but still prune.
* **C. Content-addressed cache keyed by file content hash only** --
  variant of B where the key is the raw xxh64 of the file, not a
  stable file-to-hash map.  Rejected: file renames still need
  provenance; ``current_hashes`` from ``_find_files`` + ``_hash_file``
  already gives a stable source->hash map that ``_prune_cache`` can
  consume directly.

## Decision Outcome

Chosen option: "B. Atomic writes + periodic prune + corrupt quarantine".

* ``klea_utils/stores/ingestion.py:836`` ``_save_to_cache`` (atomic
  ``_atomic_write_pickle`` via temp + ``os.replace``); ``ingestion.py:
  256`` ``chunk_all`` try/except moves torn ``.pkl`` to ``*.corrupt``;
  ``ingestion.py:268`` ``_prune_cache`` consumes ``current_hashes`` from
  ``_find_files`` + ``_hash_file`` and removes orphans and healed
  artifacts at ``chunk_all`` exit (``chunk_worker.py`` worker
  ``convert_batch_worker`` also benefits via the parent's prune).
* ``klea_utils/biblio/doi.py:329`` ``DoiResolver._save_cache`` is atomic
  (temp + ``os.replace``) and batched (``25`` + ``close()`` flush);
  JSON is kept for readability; backward compatibility was explicitly
  waived.
* ``chunk`` is always worker-isolated (``ADR-0001``) so the parent's
  prune sees the fresh ``current_hashes`` even after a worker death;
  ``build`` reuses the same ``chunk_all`` path, so its cache hygiene is
  inherited.

### Consequences

* Good, because a torn ``.pkl`` never survives as valid: it is
  quarantined as ``*.corrupt`` and re-converted on the next run; the
  next run does not abort.
* Good, because ``doi-cache.json`` writes are O(1) per 25 DOIs and
  atomically replace the file, so a kill cannot wipe the cache.
* Good, because renames/deletes are reflected immediately: ``_prune_cache``
  at every ``chunk_all`` exit removes orphans, keeping ``.klea-cache``
  proportional to the current source dir.
* Bad, because the ``*.corrupt`` quarantine preserves bytes on disk
  until the next successful re-convert prunes it -- a failed file
  temporarily costs double the ``.pkl`` size.

### Confirmation

* ``utils_pkg/tests/test_stores_ingestion.py``-style coverage for
  ``_save_to_cache`` atomicity and ``_prune_cache`` orphan removal;
  ``test_chunk_worker.py`` ``current_file`` Value observability still
  attributes the file that left a torn entry.
* Manual: kill ``chunk`` mid-``_save_to_cache`` leaves a
  ``*.pkl.corrupt``; re-run converts only that file and prunes the
  artifact.

## Pros and Cons of the Options

### Atomic writes + prune + quarantine (chosen)

* Good, because resumable and self-healing across crashes and renames
* Good, because DOI cache stays human-readable JSON without O(N^2) writes
* Bad, because ``*.corrupt`` briefly duplicates bytes

### Direct writes, no prune

* Good, because minimal code
* Bad, because torn file bricks the next run and orphans accumulate

## More Information

* Code: ``klea_utils/stores/ingestion.py:836`` (``_save_to_cache``),
  ``ingestion.py:256`` (torn ``*.corrupt`` quarantine),
  ``ingestion.py:268`` (``_prune_cache``), ``stores/chunk_worker.py``
  (worker ``convert_batch_worker``), ``biblio/doi.py:329``
  (``DoiResolver`` batch + atomic JSON).
* Related: ``ADR-0001`` (subprocess chunk workers that make the torn-file
  risk load-bearing; this ADR is the cache-hygiene complement),
  ``devdocs/system/store-create.md`` (chunk/store/build cache layout).
* Commits: ``083e7c6`` (atomic cache + ``*.corrupt``), ``ce734b2``
  (``collect_results`` removal), ``54fb119`` (manifest), doi-cache
  batching, ``chunk_all`` prune hook.
* Codified ``2026-08-28``; atomic cache + prune hardened alongside
  ``ADR-0001`` in ``2026-08-26``.
