Architecture (C4 model)
=======================

Klea's architecture is documented with the `C4 model
<https://c4model.com/>`_.  The model is maintained as internal developer
documentation in ``devdocs/`` (in this same repository), so the diagrams below
link to the relevant files on GitHub (the ``development`` branch).  This keeps a
single source of truth and avoids duplicating the diagrams between the
developer notes and this user guide.

Level 1 -- System Context
-------------------------

The system context diagram shows Klea as a single system, the people who use
it, and the external software systems it depends on.  Klea is developed as a
general-purpose RAG + agentic assistant; neuroscience is the current motivating
domain (via the ``nml-mcp`` server and the curated NeuroML vector stores), but
the agent and RAG are domain-configurable and work for any domain.

- `System Context diagram (Level 1)
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/system/c4-system-context.md>`__

Level 2 -- Container
--------------------

The container diagram zooms into Klea and shows the containers -- the
independently deployable applications/services/datastores and the shared
library -- plus how they interact and connect to the external systems
from Level 1.

- `Container diagram (Level 2)
  <https://github.com/NeuroML/neuroklea/blob/development/devdocs/system/c4-container.md>`__

Levels 3-4 (component, code) and the deployment view are planned and will
be added to ``devdocs/`` as they are written; this page will link to them
when they exist.

.. note::

   The C4 diagrams live in ``devdocs/`` (developer notes), not in this user
   guide, to keep one canonical source.
