#!/usr/bin/env python3
"""
Stored-metadata schema for vector store ingestion.

File: klea_utils/stores/metadata.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

#: Metadata keys set on every chunk by the ingestion pipeline itself,
#: rather than supplied by the metadata map.  They are always stored.
MACHINE_SET_METADATA_KEYS = frozenset({"file_name", "file_hash", "headings"})

#: Metadata keys that are always stored in the vector store, together with
#: the bibliographic fields produced by the extraction cascade
#: (``title``, ``authors``, ``keywords``, ``year``, ``journal``, ``doi``).
#: Any ``url*`` key (``url``, ``url_1``, ``url_doi``, ...) is also always
#: stored.  This whitelist documents the guaranteed stored schema; *presence*
#: of the cascade fields is determined by the metadata map (whose
#: researcher-curated keys pass through unmodified).  See
#: :func:`klea_utils.stores.ingestion._apply_store_metadata_policy`.
ALWAYS_STORED_METADATA_KEYS = MACHINE_SET_METADATA_KEYS | frozenset(
    {
        "title",
        "authors",
        "keywords",
        "year",
        "journal",
        "doi",
    }
)

#: Bibliographic fields that every chunk of a source file is expected to
#: share (they come from the file's DEFAULT metadata map entry).  When
#: serializing reference material, these are emitted once per source file
#: rather than repeated on every chunk.
SHARED_DOC_METADATA_KEYS = ALWAYS_STORED_METADATA_KEYS - MACHINE_SET_METADATA_KEYS

#: Metadata keys that are never stored.  Provenance keys from the biblio
#: cascade; keys starting with ``_`` (e.g. ``_metadata_complete``,
#: ``_sources``, ``_source_scores``) are also always dropped.  These guide
#: the researcher reviewing ``metadata-map.template.json`` but carry no
#: meaning in a store.  See
#: :func:`klea_utils.stores.ingestion._apply_store_metadata_policy`.
STORE_DROPPED_METADATA_KEYS = frozenset({"source_path", "source_type", "source_url"})
