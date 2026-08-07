klea-stores-create
==================

Create stores from documents (vector stores and BM25 stores).

The ``build`` and ``store`` commands accept ``--bm25-store <path>``,
which writes the combined chunked documents to a single pickle that can
be configured as a ``bm25_stores`` entry in the RAG config.

.. typer:: klea_utils.ui.stores_create:app
   :prog: klea-stores-create
   :show-nested:
   :width: 70
   :preferred: text
