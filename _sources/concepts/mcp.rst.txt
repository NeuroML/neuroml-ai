MCP Server Support
==================

Klea uses `Model Context Protocol (MCP)
<https://modelcontextprotocol.io/>`_ servers to give its LLMs access to
external tools (validation, file handling, model lookup, code execution,
etc.).  A domain in a RAG or code config can declare one or more MCP
servers; the tools they expose are fetched at startup and made available
to the graph's tool picker.

Configuring MCP servers
-----------------------

Each :doc:`domain <rag>` lists the MCP servers it can use under
``mcp_servers``.  Each entry maps a server name to the URL of a
streamable HTTP MCP endpoint:

.. code-block:: json

   {
       "domains": {
           "NeuroML": {
               "mcp_servers": {
                   "NeuroML": {
                       "url": "http://127.0.0.1:8542/mcp"
                   }
               }
           }
       }
   }

See :doc:`../tutorials/create-and-use-rag` for a full example and the
:doc:`../cookbook/huggingface` cookbook for a deployed setup.  The
NeuroML package ships with ``nml-mcp``, an MCP server exposing tools for
NeuroML model generation, validation, and lookup.

How Klea uses the tools
-----------------------

At startup, the graph connects to each configured MCP server, lists its
tools, and stores per-domain metadata.  The tool picker node then selects
the tools relevant to the current query, and the selected tools are
called during graph execution.  When several MCP servers are configured,
fastmcp prefixes tool names with the server name (e.g.
``NeuroML_list_files_tool``) so tools from different servers stay
distinct; Klea keeps these prefixed names unchanged.

Writing tools for Klea
----------------------

Klea expects MCP tool functions to follow a *docstring-first* convention,
so that the LLM-facing description stays concise and parameter details are
available in the tool schema:

1. **Tool description** -- the opening text block of the function
   docstring (what the tool does, when to use / not use it, and one short
   example).  Keep this block focused; it is what the tool picker shows
   the LLM.

2. **Parameters** -- describe each parameter in a Google-style ``Args:``
   section.  fastmcp parses these into the tool's input schema, and Klea
   renders them as compact one-line parameter entries (name, type,
   required flag, description).

3. **Do not** set the tool description to the raw full docstring.  Klea
   shows the opening text block as the description and the
   ``Args:``-derived schema as the parameter list, so duplicating the
   ``Args:``/``Returns:`` prose in the description wastes prompt tokens.

Example:

.. code-block:: python

   @tool_meta(ToolInfo(title="Find models on NeuroML-db"))
   async def get_models_from_neuromldb_tool(
       search_query: str, num: int = 3, download: bool = False
   ) -> dict:
       """Search and optionally download cell and ion channel models from
       NeuroML-DB.

       Use this tool when you need example cell or ion channel models, or
       want to download models for local use.

       Use when:
       - Finding example cell and ion channel models.
       - Downloading models for use in your project.

       Do not use for:
       - Creating or editing NeuroML models (use the model template tool
         instead).
       - Running simulations (use the simulation tools instead).

       Example: get_models_from_neuromldb(search_query="cerebellum", download=True)

       Args:
           search_query: search term for querying NeuroML-DB.
           num: number of search results to get (clamped to 1-20).
           download: set to true to also download the models.

       Returns:
           Dictionary of model information with metadata and model content.
       """
       ...

Tool functions must end with ``_tool`` for automatic registration (see
``neuroml_mcp.utils.register_tools``), and carry ``@tool_meta(ToolInfo(
title=..., tags=...))`` metadata.  Validation constraints (e.g.
``Field(min_length=1)``) may be added to parameter annotations and are
preserved in the schema.

Tool description length and style
---------------------------------

The tool description is what the model uses to decide which tool to call,
so it is worth writing carefully.  A good rule of thumb is that a
description should read like a short, structured "how to use this tool"
note: roughly 100-250 tokens per tool.  This keeps the whole tool list
small enough to scale while still giving smaller models the guidance they
need to pick the right tool.  Sources consulted:

* **Anthropic** -- "Define tools"
  (https://docs.claude.com/en/docs/agents-and-tools/tool-use/define-tools)
  calls the description "by far the most important factor in tool
  performance" and recommends *at least 3-4 sentences* covering what the
  tool does, when it should (and should not) be used, what each parameter
  means, and any important caveats or limitations.  Descriptions are input
  tokens, so they count against the context window on every request.

* **OpenAI** -- "Function calling"
  (https://platform.openai.com/docs/guides/function-calling) recommends
  clearly describing the purpose of each function and parameter and
  including examples and edge cases, and suggests shortening descriptions
  when under token pressure and keeping fewer than ~20 tools available at
  once.

* **MCP specification** -- "Tools"
  (https://modelcontextprotocol.io/specification/2025-06-18/server/tools)
  defines ``description`` as a "human-readable description of
  functionality" and gives no length guidance, so the length is up to the
  server author.

* **opencode** (an open-source agentic coding tool,
  https://github.com/sst/opencode) keeps its core tool descriptions terse
  (~50-125 tokens) but ships richer, structured descriptions for its CLI
  tools (~100-600 tokens) that open with a one-sentence summary and use
  "Use when" / "Do NOT use" bullet sections and examples.  Its guidance:
  *"Keep the tool description concise; the full schema documentation
  remains in the signature."*

Klea's docstring-first convention (above) is a middle ground that keeps
descriptions small enough to scale:

* One-sentence summary on the first line.
* A short "Use this tool to ..." sentence.
* "Use when:" and "Do not use for:" bullet sections with cross-tool
  pointers, so smaller models can pick the right tool.
* One concrete "Example:" line.
* Parameter descriptions in a Google-style ``Args:`` section (parsed into
  the schema by fastmcp) rather than in the description.
* No long procedural prose (template structure, next steps, error
  handling, performance notes): that adds tokens without helping tool
  selection.

Reusable template
^^^^^^^^^^^^^^^^^

Copy this template when implementing a new tool; fill in the placeholders
and keep the whole block to roughly 100-250 tokens:

.. code-block:: python

   """<One-sentence summary of what the tool does>.

   Use this tool to <primary purpose>.

   Use when:
   - <case where this tool is the right choice>
   - <another concrete case>

   Do not use for:
   - <case better handled elsewhere> (use the <role> tool instead)

   Example: <tool_name>(<param>=<value>)

   Args:
       <param>: <what the parameter means and how it affects behaviour>.
       <param2>: <description>. Defaults to <default> if not specified.

   Returns:
       <brief description of the data returned>.
   """

Use generic role references ("use the file reading tool") rather than
exact tool names in the "Do not use for" pointers, because fastmcp
prefixes tool names with the server name (``NeuroML_list_files_tool``) and
those prefixes vary between deployments.
