Installation
============

Requirements
------------

* Python 3.12 or later
* A `LangChain-compatible inference provider
  <https://docs.langchain.com/oss/python/integrations/providers/overview>`_
  for LLM access (e.g. OpenAI, Anthropic, Ollama, HuggingFace, etc.)

klea-rag and klea-utils (PyPI)
-------------------------------

The RAG and utilities packages are available on PyPI::

   pip install klea-rag

This installs ``klea_rag`` and its core dependency ``klea_utils``.
Optional extras for vector store backends and document ingestion are
listed below — add them with e.g. ``pip install klea-rag[chroma]``.

If you use `uv <https://github.com/astral-sh/uv>`_, replace ``pip`` with
``uv pip``.

klea-utils extras
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
     - Purpose
   * - ``chroma``
     - ``langchain-chroma``, ``chromadb``
     - `Chroma <https://github.com/chroma-core/chroma>`_ vector store support
   * - ``pgvector``
     - ``langchain-postgres``
     - `pgvector <https://github.com/pgvector/pgvector-python>`_ support
   * - ``qdrant``
     - ``langchain-qdrant``
     - `Qdrant <https://github.com/qdrant/qdrant>`_ vector store support
   * - ``huggingface``
     - ``langchain-huggingface``
     - HuggingFace inference provider
   * - ``ollama``
     - ``langchain-ollama``, ``ollama``
     - Ollama inference provider (local models)
   * - ``ingest``
     - ``docling``, ``typer``, ``xxhash``
     - Document ingestion pipeline
   * - ``nicegui``
     - ``nicegui``
     - NiceGUI web UI frontend
   * - ``full``
     - All of the above
     - All optional extras (vector stores + inference providers + frontends)

Usage::

   pip install klea_utils[chroma]

klea-rag extras
~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1

   * - Extra
     - Installs
     - Purpose
   * - ``chroma``
     - ``klea_utils[chroma]``
     - `Chroma <https://github.com/chroma-core/chroma>`_ support for RAG
   * - ``pgvector``
     - ``klea_utils[pgvector]``
     - `pgvector <https://github.com/pgvector/pgvector-python>`_ support for RAG
   * - ``qdrant``
     - ``klea_utils[qdrant]``
     - `Qdrant <https://github.com/qdrant/qdrant>`_ support for RAG
   * - ``huggingface``
     - ``klea_utils[huggingface]``
     - HuggingFace inference provider for RAG
   * - ``ollama``
     - ``klea_utils[ollama]``
     - Ollama inference provider for RAG
   * - ``nicegui``
     - ``klea_utils[nicegui]``
     - NiceGUI web UI frontend
   * - ``full``
     - All vector store and inference provider extras
     - All RAG optional extras

Usage::

   pip install klea_rag[full]

klea-code (WIP: coming soon) and neuroml-mcp (from source)
----------------------------------------------------------

``klea-code`` is under active development and not yet ready for general use.
``neuroml-mcp`` is also in active development.  Neither is on PyPI.
To install them, clone the repository and follow the
:doc:`development workflow <contributing>`.

Configuration
-------------

Both the RAG and Code packages load configuration from:

1. An env file (``k=v`` format):

   * ``KLEA_RAG_ENV_FILE`` or ``rag.env`` for the RAG system
   * ``KLEA_CODE_ENV_FILE`` or ``klea_code.env`` for the Code system

2. A JSON configuration file referenced inside the env file.

   * ``rag_pkg/example-configs/klea_rag.json`` or a copy you customise
   * ``code_pkg/mcp.json`` for Code MCP server configuration

Example env file::

   KLEA_RAG_CHAT_MODEL=ollama:qwen3:0.6b
   KLEA_RAG_EMBEDDING_MODEL=ollama:bge-m3
   KLEA_RAG_APP_CONFIG_FILE=/path/to/klea_rag.json

Choosing models
~~~~~~~~~~~~~~~

Each model provider requires its corresponding :mod:`klea_utils` extra
to be installed::

   # For Ollama:
   pip install klea-utils[ollama]
   # or via klea-rag:
   pip install klea-rag[ollama]

   # For HuggingFace:
   pip install klea-utils[huggingface]

See the `LangChain provider docs
<https://docs.langchain.com/oss/python/integrations/providers/overview>`_
for other providers and their package names.  The needed extras
(``huggingface``, ``ollama``) are documented in the extras tables above.

Model names are prefixed according to their provider:

* ``ollama:<model_name>:<tag>`` for Ollama models
* ``huggingface:<model_id>`` for HuggingFace inference providers.
  The suffix ``:local`` selects the pipeline backend (runs the model
  locally), while any other suffix (e.g. ``:endpoint``) selects the
  HuggingFace Endpoints API.  HuggingFace models additionally require
  the ``HF_TOKEN`` environment variable to be set (see
  `HuggingFace tokens <https://huggingface.co/docs/hub/security-tokens>`_).
* ``custom:<model_name>:<base_url>`` for OpenAI-compatible endpoints
  (e.g. ``custom:Qwen:https://inf01.example.com/v1/``).  These use the
  ``ChatOpenAI`` provider under the hood and require the
  ``OPENAI_API_KEY`` environment variable.
* Others (e.g. OpenAI, Anthropic) use their standard model names and
  environment variables as supported by LangChain.

Guard models follow the same format.  The guard node is optional --
set ``KLEA_RAG_GUARD_MODEL`` to an empty value to skip safety
screening entirely.

Logging
-------

Each Klea application writes its logs to a rotating file (1 MB per file,
5 backups) inside its platform user-data directory:

* Linux: ``~/.local/share/<app>/<app>.log``
* macOS: ``~/Library/Application Support/<app>/<app>.log``
* Windows: ``%LOCALAPPDATA%\<app>\<app>.log``

The file captures DEBUG output for the Klea packages and third-party
libraries, while the console shows INFO for Klea and INFO-or-above for
third-party libraries.  Each CLI uses its own ``<app>`` name:

.. list-table::
   :header-rows: 1

   * - Application
     - Log file name
   * - ``klea-rag`` (RAG server / graph)
     - ``klea-rag/klea-rag.log``
   * - ``klea-rag`` TUI client
     - ``klea-rag-tui/klea-rag-tui.log``
   * - ``klea-rag`` web client
     - ``klea-rag-web/klea-rag-web.log``
   * - ``klea-code`` (Code server / graph)
     - ``klea-code/klea-code.log``
   * - ``klea-code`` TUI client
     - ``klea-code-tui/klea-code-tui.log``
   * - ``klea-code`` web client
     - ``klea-code-web/klea-code-web.log``
   * - ``nml-mcp`` (MCP server)
     - ``nml_mcp/nml_mcp.log``

``klea-vs-create`` logs to the console only.
