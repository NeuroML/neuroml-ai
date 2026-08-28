Developer info
==============

Klea keeps two documentation layers: this public user guide (``docs/``) and
internal developer notes (``devdocs/``) in the same repository.  ``devdocs/``
is the single source of truth for architecture, component contracts, and
decisions; this page links to it on GitHub (``development`` branch) and avoids
duplicating the diagrams and records here.

Developer docs on GitHub
------------------------

* `devdocs index on GitHub
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/README.md>`__
  -- curated index of all developer notes (system docs and ADRs).
* `devdocs/ on GitHub
  <https://github.com/NeuroML/neuroklea/tree/development/devdocs>`__
  -- browse the full folder (``system/``, ``adr/``, ``.agents/``).

``devdocs/`` is for contributors.  See :doc:`contributing` for workflow,
commands, and the ``AGENTS.md`` instructions for AI assistants.

Architecture (C4 model)
-----------------------

Klea's architecture is documented with the `C4 model
<https://c4model.com/>`_.  The model is maintained as internal developer
documentation in ``devdocs/`` so the diagrams below link to the relevant
files on GitHub.

Level 1 -- System Context
~~~~~~~~~~~~~~~~~~~~~~~~~

The system context diagram shows Klea as a single system, the people who use
it, and the external software systems it depends on.  Klea is developed as a
general-purpose RAG + agentic assistant; neuroscience is the current motivating
domain (via the ``nml-mcp`` server and the curated NeuroML vector stores), but
the agent and RAG are domain-configurable and work for any domain.

- `System Context diagram (Level 1)
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/system/c4-system-context.md>`__

Level 2 -- Container
~~~~~~~~~~~~~~~~~~~~

The container diagram zooms into Klea and shows the containers -- the
independently deployable applications/services/datastores and the shared
library -- plus how they interact and connect to the external systems
from Level 1.

- `Container diagram (Level 2)
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/system/c4-container.md>`__

Levels 3-4 (component, code) and the deployment view are planned and will
be added to ``devdocs/`` as they are written; this page will link to them
when they exist.

Architecture Decision Records (ADRs)
------------------------------------

ADRs are short, numbered records in MADR format in ``devdocs/adr/``
(``NNNN-<slug>.md``).  Each records context, options considered, outcome, and
consequences.  See `adr-template.md
<https://github.com/NeuroML/neuroklea/blob/development/devdocs/adr-template.md>`__
for the template.

- `ADR-0001 -- Subprocess chunk workers and DOI-cache batching
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/adr/0001-chunk-workers.md>`__
- `ADR-0002 -- Retry worker batches that die with no results
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/adr/0002-worker-retry.md>`__
- `ADR-0003 -- Strictly require MCP isError for tool execution failures
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/adr/0003-mcp-iserror-compliance.md>`__

Other system notes
------------------

* `Store creation pipeline -- chunk, store, build, worker isolation and cache
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/system/store-create.md>`__
* `MCP tool permissions -- current in-tool checks and the trust model
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/system/mcp-permissions.md>`__
