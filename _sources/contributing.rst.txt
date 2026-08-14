Contributing
============

Development setup
-----------------

Clone the repository and install in editable mode::

   git clone https://github.com/NeuroML/neuroklea.git
   cd neuroklea
   pip install -r requirements-dev.txt

Workflow
--------

This project uses a standard fork-and-PR workflow:

1. Fork the repository on GitHub.
2. Create a feature branch from ``development``.
3. Make your changes and run verification (lint, typecheck, test).
4. Submit a pull request targeting the ``development`` branch.
5. Address review feedback if any.

Only pick up issues tagged with the ``help wanted`` label
(https://github.com/NeuroML/neuroklea/issues?q=is%3Aissue+state%3Aopen+label%3A%22help+wanted%22);
other issues are planned or internal work.  Before starting on an issue,
comment on it first to check its status, and keep any discussion on the
issue itself so it remains the single place for discussion.

All contributors are acknowledged via the all-contributors specification
(https://allcontributors.org/), so contributions of any kind are welcome
and tracked on the project ``Readme``.

Commands
--------

Lint and format
~~~~~~~~~~~~~~~

.. code-block:: bash

   ruff check . --fix
   ruff format .

Type check
~~~~~~~~~~

.. code-block:: bash

   ty

Run tests
~~~~~~~~~

.. code-block:: bash

   # All tests
   bash scripts/run_tests.sh

   # Single package
   cd utils_pkg && pytest -v

   # Exclude tests that need an LLM
   pytest -m "not localonly"

Pre-commit
~~~~~~~~~~

.. code-block:: bash

   pre-commit run --all-files

CLI verification
~~~~~~~~~~~~~~~~

If you modify a CLI entry point, confirm it starts::

   <cli-name> --help

AI use
------

This repository includes an ``AGENTS.md`` file with instructions for AI
coding assistants.  While it is fine to use AI for development, it is
expected that the human reviews every line of code before submitting PRs.
PRs that are AI only and have not been checked by humans will not be
accepted.  All AI assistance must be clearly noted in the PR
description -- the PR template (``.github/PULL_REQUEST_TEMPLATE.md``)
requires you to confirm manual review of all AI-generated code and to
name the AI agent(s) used.

Pull requests
-------------

* Open PRs targeting the ``development`` branch.
* Keep changes focused -- one logical change per PR.
* Ensure CI passes (lint, typecheck, tests).
* Include a conventional commit message with issue numbers when applicable.
* Follow the PR template (``.github/PULL_REQUEST_TEMPLATE.md``); its
  checklist covers the branch target, tests, CI, commit message
  guidelines (https://chris.beams.io/git-commit) and conventional commits
  (https://www.conventionalcommits.org/en/v1.0.0/).
