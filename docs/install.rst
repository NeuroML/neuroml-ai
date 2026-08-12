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

klea (WIP: coming soon) and neuroml-mcp (from source)
------------------------------------------------------

``klea`` is under active development and not yet ready for general use.
``neuroml-mcp`` is also in active development.  Neither is on PyPI.
To install them, clone the repository and follow the
:doc:`development workflow <contributing>`.

PyTorch / CUDA (optional)
-------------------------

No Klea package declares PyTorch as a direct dependency, and the correct
torch build depends on your GPU hardware: newer CUDA builds drop kernel
support for older GPUs.  torch is, however, installed transitively by
the document-ingestion extra (``ingest`` -> docling ->
``docling-slim[standard]`` -> ``torch`` + ``torchvision``), which the
``full``/``test``/``dev`` extras all include.  That transitive build is
unpinned and comes from the default index, so it may lack kernels for
your GPU; docling's OCR/layout processing then falls back to the CPU
instead of using the GPU.

If you want a GPU-accelerated build, install it yourself -- see
:file:`requirements-torch.txt` in the repository root for the full
guide, pinned install commands, and a verified example.  Install the
pinned ``torch`` + ``torchvision`` pair (from the same CUDA index)
*before* the Klea requirements: an installed torch that satisfies
docling's ``torch>=2.2.2,<3.0.0`` is left untouched by the installers.
If a wrong build is already present, force-reinstall the pair instead
(see the file).

The short version:

.. list-table::
   :header-rows: 1

   * - GPU compute capability
     - CUDA build to use
   * - 6.0/6.1 (Pascal) and 7.0/7.5 (Volta/Turing)
     - ``cu126`` or ``cu128`` (CUDA 12.8 or earlier)
   * - 8.0+ (Ampere, Ada, Hopper, Blackwell)
     - Any recent build (``cu129``, ``cu130``, ...)

Check your capability with ``torch.cuda.get_device_capability(0)`` and
install the ``torch``/``torchvision`` pair with the full pin so the
package manager cannot pick a different CUDA suffix::

   uv pip install "torch==<version>+cu126" "torchvision==<version>+cu126" \
       --extra-index-url https://download.pytorch.org/whl/cu126

``torchvision`` must come from the same CUDA index as ``torch``; a
mismatch installs silently but breaks ``import torchvision`` at runtime.

PyTorch wheels bundle their own CUDA runtime, so a system CUDA toolkit
is not required to run torch.  It is only needed to compile CUDA
extensions yourself, and it must match the wheel's CUDA version.

Configuration
-------------

Both the RAG and Agent packages load configuration from:

1. An env file (``k=v`` format):

   * ``KLEA_RAG_ENV_FILE`` or ``rag.env`` for the RAG system
   * ``KLEA_AGENT_ENV_FILE`` or ``klea_agent.env`` for the Agent system

2. A JSON configuration file referenced inside the env file.

   * ``rag_pkg/example-configs/klea_rag.json`` or a copy you customise
   * ``agent_pkg/mcp.json`` for Agent MCP server configuration

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
   * - ``klea`` (Agent server / graph)
     - ``klea/klea.log``
   * - ``klea`` TUI client
     - ``klea-tui/klea-tui.log``
   * - ``klea`` web client
     - ``klea-web/klea-web.log``
   * - ``nml-mcp`` (MCP server)
     - ``nml_mcp/nml_mcp.log``

``klea-stores-create`` logs to the console only.

Web client user storage
-----------------------

The NiceGUI web clients (``klea-rag web``, ``klea web``) keep a small
per-browser-session identity file so that a returning browser is linked
back to the same user.  The files are written to a ``.nicegui/`` directory
relative to the working directory the web client is launched from (not the
platform user-data directory) and are named
``storage-user-<session-id>.json``.

Each file stores only a pointer to server-side state::

    {"user_id": "...", "dark_mode": false, "chat_id": "..."}

The chat history itself lives in the server's session store
(``~/.local/share/<app>/sessions.db``) and is not duplicated here.

These files are **never deleted automatically**.  NiceGUI prunes stale
sessions from its in-memory store but leaves the JSON files on disk, so
they accumulate over time and survive server restarts.  Deleting the
``.nicegui/`` directory while no web client is running is safe: the next
page load simply mints a fresh ``user_id`` (existing chats under the old
id are then not shown in that browser).
