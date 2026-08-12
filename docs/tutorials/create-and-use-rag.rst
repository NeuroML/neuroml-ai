Create and use a RAG system
===========================

This tutorial walks through the full lifecycle of a Klea RAG system:
preparing documents, building a vector store, configuring the RAG
pipeline, and querying it.

Overview
--------

By the end you will have:

* A Chroma vector store populated with chunks from your own documents
* A running Klea RAG server backed by that store
* Hands-on experience querying the system via the CLI and web UI

New to RAG?  Read :doc:`../concepts/rag` for an overview of how
retrieval-augmented generation works and why you might use it.

Prerequisites
-------------

* Python 3.12 or later
* Packages installed (see :doc:`install guide <../install>`) with Chroma and ingestion
  extras:

   .. code-block:: bash

      pip install klea_rag[chroma,ollama] klea_utils[ingest]

   .. note::

      ``klea_rag[chroma]`` provides Chroma vector store support.

      ``klea_rag[ollama]`` provides the Ollama inference provider.

      ``klea_utils[ingest]`` pulls in `Docling <https://docling-project.github.io/docling/>`_
      and its PyTorch dependency.  The download is several hundred MB.
      On systems with a CUDA-capable GPU, PyTorch will use the GPU
      automatically for faster document processing.

* A running `Ollama <https://ollama.com/>`_ instance with the required
  models:

  .. code-block:: bash

     ollama pull qwen3:0.6b
     ollama pull llama-guard3:1b
     ollama pull bge-m3:latest

  This tutorial uses Ollama for all inference (chat, guard, and
  embeddings).  Klea supports other providers too -- see
  :doc:`../install` for HuggingFace, OpenAI, Anthropic, and other
  LangChain-compatible options.

Step 1: Prepare source documents
---------------------------------

Place the files you want to index in a single directory.  Docling
handles a wide range of formats: PDF, HTML, Markdown, DOCX, PPTX, XLSX,
images, and more (see `Docling supported formats
<https://docling-project.github.io/docling/usage/supported_formats/>`_
for the full list).

For this tutorial we will refer to this directory as
``<folder-of-files>``.

Step 2: Create a vector store
------------------------------

.. code-block:: bash

   klea-stores-create build <folder-of-files> \\
       --collection my-docs \\
       --store chroma:/path/to/my-store \\
       --bm25-store /path/to/my-bm25-corpus.pkl

The ``build`` command runs the full pipeline:

1. **Convert** -- every supported file is parsed with Docling into a
   structured document.
2. **Chunk** -- documents are split into token-aware chunks (450 tokens
   by default) using Docling's ``HybridChunker``.  Each chunk retains
   its heading hierarchy as metadata.
3. **Embed** -- chunks are embedded using ``bge-m3`` (or whichever
   embedding model you configure).
4. **Store** -- embeddings and text are written to a Chroma vector store
   at the path you specify.

Flags explained:

* ``--collection`` / ``-n`` -- the collection name inside the store
  (e.g. ``my-docs``).  This must match the ``name`` of the store's
  ``vector_stores`` / ``bm25_stores`` entry in the RAG config file
  (e.g. ``klea.json``) -- retrieval looks stores up by name, so a
  mismatch silently returns no results.
* ``--store`` / ``-s`` -- the vector store URI (e.g. ``chroma:/path``).
  For Chroma, point at the store *folder*; the database file inside it
  is always named ``chroma.sqlite3`` (the filename is not
  configurable).  A folder that does not exist yet is created.  Because
  one Chroma store file can hold several collections, the
  ``--collection`` name selects which collection within that file is
  used.
* ``--bm25-store`` -- path to write the combined chunked documents to a
  single pickle file that can be used as a BM25 keyword store.  Defaults
  to ``<collection>.pkl`` in the current directory (always written); you
  can move the file afterwards and point the config at its new location.
* ``--model`` / ``-m`` -- embedding model (default ``ollama:bge-m3:latest``).
* ``--max-tokens`` -- maximum tokens per chunk (default 450).
* ``--ocr`` / ``--no-ocr`` -- whether to perform optical character
  recognition (OCR, see `Wikipedia
  <https://en.wikipedia.org/wiki/Optical_character_recognition>`_)
  during PDF conversion (default: on).  Keep it on for scanned/
  image-based PDFs; pass ``--no-ocr`` for text-based PDFs to speed up
  conversion considerably.
* ``--force`` / ``-f`` -- re-process all files even if previously cached.

Docling's inference accelerator is configured through environment
variables rather than CLI flags.  By default it auto-detects the best
available device, but GPUs whose CUDA capability is below 7.0 (e.g. a
Quadro P1000) fail the Triton compiler used for the layout model.  Force
Docling to the CPU in that case:

.. code-block:: bash

   DOCLING_DEVICE=cpu DOCLING_NUM_THREADS=16 klea-stores-create build \\
       <folder-of-files> --collection my-docs --store chroma:/path/to/my-store

``DOCLING_NUM_THREADS`` (default 4) sets the CPU threads used for model
inference; ``OMP_NUM_THREADS`` is honoured as an alternative.

Re-running ``klea-stores-create build`` on the same directory is safe --
it skips files whose content has not changed and skips chunks whose
hashes already exist in the store (idempotent).  Adding new files to
the source directory and re-running adds only the new content
(incremental ingestion).

The source directory will contain a ``.klea-cache/`` folder after the
first run.  This caches converted chunks so subsequent runs skip the
expensive Docling conversion.

The vector store folder will also have been created, with the
``chroma.sqlite3`` database inside it.  Later runs point ``--store`` at
the same folder; the file is always named ``chroma.sqlite3``.

Step 3: Configure the RAG system
---------------------------------

Create an environment file (e.g. ``my-rag.env``):

.. code-block:: ini

   KLEA_RAG_CHAT_MODEL=ollama:qwen3:0.6b
   KLEA_RAG_GUARD_MODEL=ollama:llama-guard3:1b
   KLEA_RAG_EMBEDDING_MODEL=ollama:bge-m3:latest
   KLEA_RAG_APP_CONFIG_FILE=my-config.json

Create the JSON configuration file (``my-config.json``) that wires the
vector store to a domain:

.. code-block:: json

   {
       "general": {
        "default_k": 5,
        "k_max": 10,
           "non_domain_chat": true,
           "fallback_to_training_data": true
       },
       "domains": {
           "MyDomain": {
               "description": "Documents related to my project",
               "vector_stores": [
                   {
                       "name": "my-docs",
                       "path": "chroma:/path/to/my-store"
                   }
               ],
               "bm25_stores": [
                   {
                       "name": "my-docs-bm25",
                       "path": "/path/to/my-bm25-corpus.pkl"
                   }
               ]
           }
       }
   }

The ``general`` section controls retrieval behaviour:

* ``default_k`` -- number of documents to retrieve per query.  This is the
  graph-wide default; individual vector stores can override it (see below).
* ``k_max`` -- maximum ``k`` when the evaluator requests more context.
* ``k_inc`` -- how much ``k`` is increased by each time the evaluator
  requests more information.
* ``non_domain_chat`` -- whether to fall back to the LLM's training data
  for questions that do not match any domain.
* ``fallback_to_training_data`` -- whether to let the LLM answer from its
  own knowledge when retrieval returns nothing useful.

Each entry under ``domains`` defines a knowledge area with one or more
vector stores.  The ``description`` helps the classifier route queries
to the right domain.

A store's ``name`` must exactly match the ``--collection`` name passed
to ``klea-stores-create``, and its ``path`` must match what was passed
to ``--store`` (for a vector store) or the location of the written BM25
corpus pickle.  Retrieval looks stores up by name, so a mismatch means
the store is never queried.

A domain can also list ``bm25_stores``.  Each ``bm25_stores`` entry's
``path`` points to a combined corpus pickle written by
``klea-stores-create --bm25-store`` (or ``klea-stores-create store
--bm25-store``).  If you did not pass ``--bm25-store``, the corpus was
written to ``<collection>.pkl`` in the directory you ran the command
from; it can be moved anywhere before it is referenced here.  When both
are configured, retrieval queries the vector stores and the BM25 stores
and combines the results with Reciprocal Rank Fusion -- exact
name/symbol matches from BM25 complement the semantic matches from the
vector stores.

Vector stores can override the retrieval settings independently.  Stores
that set their own ``default_k``, ``k_max``, and ``k_inc`` use those
values instead of the ``general`` fallbacks, which is useful when stores
cover corpora of very different sizes:

.. code-block:: json

   {
       "general": {
           "default_k": 5,
           "k_max": 10,
           "k_inc": 1
       },
       "domains": {
           "MyDomain": {
               "description": "Documents related to my project",
               "vector_stores": [
                   {
                       "name": "large-corpus",
                       "path": "chroma:/path/to/large-store",
                       "default_k": 10,
                       "k_max": 25,
                       "k_inc": 5
                   },
                   {
                       "name": "small-corpus",
                       "path": "chroma:/path/to/small-store"
                   }
               ]
           }
       }
   }

Here ``large-corpus`` retrieves up to 10 documents and can grow to 25 in
steps of 5 when the evaluator asks for more context, while ``small-corpus``
inherits the ``general`` settings (5, capped at 10, stepping by 1).  The
dynamic ``k_inc``/``k_max`` adjustments only apply to stores that are
already loaded, so a store only grows once it has been queried once.

.. seealso::

   :doc:`../install` for details on HuggingFace, OpenAI, and other
   provider model naming conventions.

Step 4: Start the RAG server
-----------------------------

For local single-user use this step is optional: the client commands in
Step 5 start a server on the local machine automatically when none is
already running.  Run ``klea-rag-serve serve`` instead when you want a
persistent backend, for example to share one server between several
clients or to run it in a separate terminal:

.. code-block:: bash

   KLEA_RAG_ENV_FILE=my-rag.env klea-rag-serve serve

The server loads the configuration, initialises the embedding model,
and compiles the LangGraph pipeline.  Once ready, check it is alive:

.. code-block:: bash

   curl http://127.0.0.1:8005/health/ready

A ``200 OK`` response means the system is ready to accept queries.

Step 5: Query the RAG
---------------------

The client commands below use ``http://127.0.0.1:8005`` by default.
If no server is running there, they start one on the local machine for
the session and stop it when they exit; if a server is already running
(for example from Step 4) they reuse it.  Pointing ``--server`` at a
remote host connects without starting anything.

Single-query mode is the quickest way to test:

.. code-block:: bash

   klea-rag cli --single-query "What does my collection of documents cover?"

For an interactive session:

.. code-block:: bash

   klea-rag cli

Type your questions at the prompt.  Use ``quit`` to exit.

For a graphical interface, launch the NiceGUI web UI:

.. code-block:: bash

   klea-rag web

The web UI uses NiceGUI and requires the ``[nicegui]`` extra, while the
CLI mode has no extra dependencies.

Both methods use the server at ``http://127.0.0.1:8005`` by default.
Use ``--server`` to point at a different address.

Going further
--------------

Once the basic pipeline works, here are natural next steps:

**Metadata enrichment**
   Add source URLs or other metadata to retrieved chunks.  First run
   ``klea-stores-create chunk`` to generate a ``metadata-map.template.json``.
   Each file's ``DEFAULT`` entry is pre-filled automatically with
   bibliographic metadata (title, authors, keywords, DOI, URL) where it
   could be extracted -- see :doc:`../concepts/rag` for the extraction
   cascade.  Review and correct the values (check the
   ``_metadata_complete`` flag), then ``klea-stores-create store
   --metadata-map <file>``.  See ``klea-stores-create --help`` for
   examples.

**Different embedding models**
   Swap ``ollama:bge-m3:latest`` for a HuggingFace embedding model
   (see :doc:`../install` for model naming conventions).

**Multiple domains**
   Add more ``domains`` entries in the JSON config, each with its own
   vector store and description.  The classifier will route queries
   automatically.

**MCP tools**
   Add ``mcp_servers`` to a domain config to give the LLM access to
   external tools (e.g. a NeuroML validation server).  See the example
   in ``rag_pkg/example-configs/klea_rag.json``.

**Separate chunk-and-store workflow**
   Use ``klea-stores-create chunk`` to convert and cache without writing
   to a store, then ``klea-stores-create store`` later.  This lets you
   inspect the chunks and edit the metadata map before embedding.

**Hybrid keyword retrieval**
   Add a BM25 store alongside a vector store: run
   ``klea-stores-create store --bm25-store /path/to/corpus.pkl`` and add a
   ``bm25_stores`` entry to the domain config.  Retrieval then fuses
   semantic and lexical matches with Reciprocal Rank Fusion, which helps
   with exact names, symbols, and terminology.  See :doc:`../concepts/rag`
   for details.

Troubleshooting
---------------

**Ollama is not running**
   Start it with ``ollama serve`` or run Ollama as a system service.

**Model not found**
   Ensure you have pulled all three models (chat, guard, embedding).
   Run ``ollama list`` to see what is available.

**Server fails to start**
   Check that ``KLEA_RAG_ENV_FILE`` points to a valid env file and that
   the JSON config file path inside it is correct.  Look for JSON syntax
   errors (trailing commas, missing quotes).

**Queries return empty or irrelevant results**
   Increase ``default_k`` in the JSON config.  Verify the vector store
   path and collection name match.  Check that your source files are in
   a format Docling supports.

.. seealso::

   * :doc:`../cli/klea-stores-create` -- full CLI reference for vector store
     creation
   * :doc:`../cli/klea-rag-serve` -- server CLI reference
   * :doc:`../cli/klea-rag` -- client CLI reference
   * :class:`~klea_utils.stores.ingestion.StoresBuilder` -- Python API for
     ingestion
   * :class:`~klea_utils.stores.retrieval.vs.VSRetriever` -- Python API for
      retrieval
