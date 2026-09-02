#!/usr/bin/env python3
"""
Metadata filter translation for store backends

File: klea_utils/stores/filters.py

Copyright 2026 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

from __future__ import annotations

import logging
from typing import Any, Protocol, cast

from langchain_core.documents import Document

from .config import FilterFieldInfo

logger = logging.getLogger(__name__)

#: Operators valid inside a single field clause of the normalized filter DSL.
#: ``$contains`` is the element-membership operator for list-valued fields
#: (e.g. ``authors``, ``keywords``).
FIELD_OPERATORS = frozenset(
    {"$eq", "$ne", "$gt", "$gte", "$lt", "$lte", "$in", "$nin", "$contains"}
)

#: Logical combinators valid at any level of the normalized filter DSL.
LOGICAL_OPERATORS = frozenset({"$and", "$or"})

#: Scalar value types accepted as operator operands.
_SCALAR_TYPES = (int, float, str, bool)


class _QdrantModels(Protocol):
    """Structural type for the ``qdrant_client.models`` module namespace.

    The Qdrant translator works against the ``models`` module passed in
    (lazily imported), so this Protocol declares the constructors it
    uses without importing ``qdrant_client`` at module load.
    """

    Filter: Any
    FieldCondition: Any
    MatchValue: Any
    MatchAny: Any
    MatchExcept: Any
    Range: Any


def validate_metadata_filter(f: dict[str, Any]) -> dict[str, Any]:
    """Validate a filter and normalize it to the canonical form.

    Filters use a small backend-agnostic DSL: a dict mapping a metadata
    field name to either a bare value (implicit ``$eq``) or an operator
    expression ``{op: value}``, combined with the ``$and``/``$or``
    combinators.  Supported operators are ``$eq``, ``$ne``, ``$gt``,
    ``$gte``, ``$lt``, ``$lte``, ``$in``, ``$nin`` and ``$contains``
    (the last matches a list-valued field, e.g. ``authors``, that
    contains the value).

    The canonical form is the one every backend translator accepts:

    - every field value is an operator expression with exactly one
      operator (bare values are wrapped as ``{"$eq": value}``)
    - several constraints (multiple top-level fields, or several
      operators on one field) are combined with ``$and``
    - the top level has exactly one key (a field, ``$and`` or ``$or``)

    Chroma and ``langchain_postgres`` reject multi-operator field
    expressions and multi-key top levels, so this normalization is what
    makes a single filter usable across all backends.

    Example::

        validate_metadata_filter({"year": {"$gte": 2020, "$lte": 2025}})
        -> {"$and": [{"year": {"$gte": 2020}}, {"year": {"$lte": 2025}}]}

    :param f: Metadata filter in the DSL described above
    :returns: Canonical normalized filter
    :raises ValueError: When the filter is not well-formed (empty,
        unknown operator, malformed operand)
    """
    if not isinstance(f, dict) or not f:
        raise ValueError(f"Metadata filter must be a non-empty dict, got {f!r}")
    if len(f) == 1:
        normalized = _normalize_clause(f)
    else:
        # Several top-level constraints combine into an $and.
        clauses = [_normalize_clause({k: v}) for k, v in f.items()]
        if len(clauses) == 1:
            normalized = clauses[0]
        else:
            normalized = {"$and": clauses}
    logger.debug(f"{f = }\n{normalized = }")
    return normalized


def to_chroma_filter(f: dict[str, Any]) -> dict[str, Any]:
    """Translate a filter to a Chroma ``where`` dict.

    The canonical normalized form is Chroma's native ``where`` syntax
    (single-operator field clauses, ``$and``/``$or`` combinators, and the
    ``$contains`` array-membership operator), so this validates the
    filter and returns the normalized form unchanged.

    Example::

        to_chroma_filter({"authors": {"$contains": "Magee"}})
        -> {"authors": {"$contains": "Magee"}}

    :param f: Metadata filter in the DSL (see
        :func:`validate_metadata_filter`)
    :returns: Chroma ``where`` dict, passable as ``filter=`` to a
        ``langchain_chroma`` store's similarity search
    :raises ValueError: When the filter is not well-formed
    """
    normalized = validate_metadata_filter(f)
    logger.debug(f"{normalized = }")
    return normalized


def to_qdrant_filter(f: dict[str, Any]) -> Any:
    """Translate a filter to a Qdrant ``models.Filter`` object.

    Scalar equality and ``$contains`` become ``MatchValue`` (an array
    element match), ``$in``/``$nin`` become ``MatchAny``/``MatchExcept``,
    and range operators become a ``Range`` condition.  Compound clauses
    are grouped as nested ``Filter`` objects.

    Example::

        to_qdrant_filter({"year": {"$gte": 2020, "$lte": 2025}})
        -> Filter(must=[Filter(must=[FieldCondition(key='year',
                                                    range=Range(gte=2020.0))]),
                        Filter(must=[FieldCondition(key='year',
                                                    range=Range(lte=2025.0))])])

    :param f: Metadata filter in the DSL (see
        :func:`validate_metadata_filter`)
    :returns: Qdrant ``models.Filter`` object, passable as ``filter=``
        to a ``langchain_qdrant`` store's similarity search
    :raises ValueError: When the filter is not well-formed
    """
    normalized = validate_metadata_filter(f)
    # Lazy: importing qdrant_client pulls in the client package; only
    # needed when a Qdrant store is actually configured.
    from qdrant_client import models

    result = _qdrant_clause(normalized, models)
    logger.debug(f"{result = }")
    return result


def to_pgvector_filter(f: dict[str, Any]) -> dict[str, Any]:
    """Translate a filter to a ``langchain_postgres`` filter dict.

    ``langchain_postgres`` accepts the canonical normalized form
    directly.  The one gap is ``$contains``: the backend has no array
    containment operator for metadata fields, so it is approximated with
    ``$like`` over the serialized JSON array text (e.g. a substring match
    against ``["Magee","Smith"]``).  This is a documented approximation
    for the Postgres backend only.

    Example::

        to_pgvector_filter({"authors": {"$contains": "Magee"}})
        -> {"authors": {"$like": "%Magee%"}}

    :param f: Metadata filter in the DSL (see
        :func:`validate_metadata_filter`)
    :returns: ``langchain_postgres`` filter dict, passable as
        ``filter=`` to a ``langchain_postgres`` store's similarity search
    :raises ValueError: When the filter is not well-formed
    """
    normalized = validate_metadata_filter(f)
    result = _pgvector_clause(normalized)
    logger.debug(f"{result = }")
    return result


def translate_metadata_filter(path: str, f: dict[str, Any]) -> Any:
    """Translate a filter for the backend named by a store path.

    Dispatch helper used by the retrievers: reads the URI scheme from a
    store path (e.g. ``chroma:/data/store``) and returns the matching
    backend-native filter.  ``filter_docs_by_metadata`` is *not* returned
    here -- callers that need the in-memory matcher (the BM25 store)
    call it directly.

    Example::

        translate_metadata_filter("chroma:/data/store",
                                  {"authors": {"$contains": "Magee"}})
        -> {"authors": {"$contains": "Magee"}}

    :param path: Store URI (``scheme:location``)
    :param f: Metadata filter in the DSL (see
        :func:`validate_metadata_filter`)
    :returns: Backend-native filter for the store's scheme
    :raises ValueError: When the scheme is missing or unknown
    """
    scheme, sep, _ = path.partition(":")
    if not sep:
        raise ValueError(
            f"Invalid store path '{path}': expected format "
            f"'scheme:location' (e.g. 'chroma:/path/to/store')"
        )
    match scheme.lower():
        case "chroma":
            return to_chroma_filter(f)
        case "qdrant":
            return to_qdrant_filter(f)
        case "pgvector":
            return to_pgvector_filter(f)
        case _:
            raise ValueError(
                f"Unsupported vector store scheme '{scheme}' for filter "
                f"translation. Supported: chroma, qdrant, pgvector"
            )


def normalize_config_filters(
    filters: dict[str, Any],
    allowed_fields: list[FilterFieldInfo],
) -> list[dict[str, Any]]:
    """Validate configured-domain filters into canonical DSL clauses.

    ``filters`` maps a metadata field name (from a deployment's
    ``filter_fields`` configuration) to an operand produced by the
    retrieval query generator: a bare scalar, a list of scalars, or an
    operator expression dict (``{op: value}``).  Each configured field's
    ``value_type`` decides how a bare operand is interpreted:

    - scalar fields (``string``/``int``/``float``): a bare value is
      ``$eq``; a list of values is ``$in``.
    - ``list`` fields (e.g. ``tags``): a bare value requires element
      membership (``$contains``); several values combine with ``$and``
      (every value must be present), mirroring the bibliographic
      ``authors``/``keywords`` handling in
      :meth:`RetrievalQueryOutput.to_metadata_filter`.

    An operator expression dict is validated and normalized through
    :func:`validate_metadata_filter`; an unsupported operator or
    malformed operand raises ``ValueError``.

    Field names not declared in *allowed_fields* are ignored with a
    warning and never reach a backend (the generator must not be able to
    emit a key the deployment did not configure).  An empty operand list
    is likewise ignored.

    The result is a list of single-clause DSL dicts, each directly
    consumable by :func:`validate_metadata_filter` (and hence by every
    backend translator and the in-memory matcher).  An empty input
    returns ``[]``.

    Example::

        fields = [
            FilterFieldInfo(name="repository_type", description="...",
                            value_type="string"),
            FilterFieldInfo(name="tags", description="...", value_type="list"),
        ]
        normalize_config_filters(
            {"repository_type": ["github", "dandi"], "tags": "moose"},
            fields,
        )
        -> [
            {"repository_type": {"$in": ["github", "dandi"]}},
            {"tags": {"$contains": "moose"}},
        ]

    :param filters: Field-name -> operand mapping from the query generator
    :param allowed_fields: Configured filter fields for the domain
    :returns: Canonical DSL single-clause dicts
    :raises ValueError: When an operator expression uses an unsupported
        operator or malformed operand
    """
    allowed: dict[str, FilterFieldInfo] = {f.name: f for f in allowed_fields}
    clauses: list[dict[str, Any]] = []

    for field, operand in filters.items():
        info = allowed.get(field)
        if info is None:
            logger.warning(
                f"Ignoring filter field {field!r}: not declared in the "
                f"domain's filter_fields configuration"
            )
            continue

        if isinstance(operand, dict):
            # Operator expression: validate and normalize to the canonical
            # form (e.g. a ``{$gte, $lte}`` pair becomes an ``$and``).
            clause = {field: operand}
            clauses.append(validate_metadata_filter(clause))
            continue

        if isinstance(operand, list) and not operand:
            logger.warning(f"Ignoring empty filter list for field {field!r}")
            continue

        if info.value_type == "list":
            if not isinstance(operand, list):
                clauses.append({field: {"$contains": operand}})
            elif len(operand) == 1:
                clauses.append({field: {"$contains": operand[0]}})
            else:
                clauses.append(
                    {"$and": [{field: {"$contains": value}} for value in operand]}
                )
        elif isinstance(operand, list):
            clauses.append({field: {"$in": operand}})
        else:
            clauses.append({field: {"$eq": operand}})

    return clauses


def restrict_metadata_filter(
    metadata_filter: dict[str, Any] | None,
    allowed_field_names: set[str],
) -> dict[str, Any] | None:
    """Restrict a metadata filter to clauses on the allowed fields.

    ``metadata_filter`` is a combined DSL filter (as produced by
    :meth:`RetrievalQueryOutput.to_metadata_filter`): a single field
    clause, a top-level ``$and``/``$or`` of clauses, or ``None``.  This
    keeps only the clauses whose referenced metadata field(s) are in
    *allowed_field_names* and recombines the survivors (a single clause
    returned as-is, several wrapped in ``$and``), so a domain that
    declares only e.g. ``journal`` never has a ``username`` clause
    applied to its retrievers.  A filter with nothing left returns
    ``None``.

    A top-level ``$or`` is kept whole only when every field it references
    is allowed (partially splitting an ``$or`` would change its
    semantics); a top-level ``$and`` is decomposed and its sub-clauses
    filtered independently, so the multi-value ``$and`` clauses emitted by
    :func:`normalize_config_filters` are kept or dropped as a unit.

    Example::

        restrict_metadata_filter(
            {"$and": [{"journal": {"$eq": "nature"}},
                      {"username": {"$eq": "padraig"}}]},
            {"journal"},
        )
        -> {"journal": {"$eq": "nature"}}

    :param metadata_filter: Combined metadata filter, or ``None``
    :param allowed_field_names: Metadata field names the caller accepts
    :returns: The restricted filter, or ``None``
    """
    if not metadata_filter:
        return None

    if set(metadata_filter) == {"$and"}:
        items: list[dict[str, Any]] = metadata_filter["$and"]
    else:
        items = [{k: v} for k, v in metadata_filter.items()]

    kept = [clause for clause in items if _clause_fields(clause) <= allowed_field_names]
    if not kept:
        return None
    if len(kept) == 1:
        return kept[0]
    return {"$and": kept}


def filter_docs_by_metadata(docs: list[Document], f: dict[str, Any]) -> list[Document]:
    """Return the subset of *docs* whose metadata matches the filter.

    Python-side matcher for backends without native filter support: the
    BM25 store post-filters its results through this (the ``rank_bm25``
    index has no filter).  Documents without the filter's metadata field
    never match.

    Example::

        docs = [Document(page_content="a", metadata={"authors": ["Magee"]}),
                Document(page_content="b", metadata={"authors": ["Jones"]})]
        filter_docs_by_metadata(docs, {"authors": {"$contains": "Magee"}})
        -> [Document(page_content="a", ...)]

    :param docs: Documents to filter, each carrying ``metadata``
    :param f: Metadata filter in the DSL (see
        :func:`validate_metadata_filter`)
    :returns: Documents whose metadata satisfies the filter
    :raises ValueError: When the filter is not well-formed
    """
    normalized = validate_metadata_filter(f)
    matched = [doc for doc in docs if _clause_matches(normalized, doc)]
    logger.debug(f"Matched {len(matched)} of {len(docs)} with {normalized = }")
    return matched


# ----------------------------------------------------------------------
# Internal helpers
# ----------------------------------------------------------------------


def _normalize_clause(clause: dict[str, Any]) -> dict[str, Any]:
    """Normalize one level of a filter to the canonical form.

    A filter is a tree: at every level a dict with a single key, which
    is either a logical combinator (``$and``/``$or``, recursing into
    each sub-clause) or a metadata field name (a leaf clause).  Leaf
    clauses are normalized so their value is an operator expression with
    exactly one operator: a bare scalar becomes ``{"$eq": value}`` and a
    dict with several operators is split into an ``$and`` of
    single-operator clauses (Chroma and ``langchain_postgres`` reject
    multi-operator expressions).  A one-element ``$and``/``$or``
    collapses to its single sub-clause.

    :param clause: A single-key dict at any level of a filter
    :returns: The canonical normalized form of *clause*
    :raises ValueError: When *clause* is malformed (not a single-key
        dict, empty, or an unknown ``$``-prefixed key)
    """
    if not isinstance(clause, dict) or len(clause) != 1:
        raise ValueError(
            f"Each filter level must be a dict with exactly one key, got {clause!r}"
        )
    key, value = next(iter(clause.items()))

    if key in LOGICAL_OPERATORS:
        if not isinstance(value, list) or not value:
            raise ValueError(
                f"Operator {key} expects a non-empty list of clauses, got {value!r}"
            )
        normalized = [_normalize_clause(sub) for sub in value]
        if len(normalized) == 1:
            return normalized[0]
        return {key: normalized}

    if key.startswith("$"):
        raise ValueError(f"Unsupported filter operator or key: {key!r}")

    # Field clause: a scalar value is a bare equality; a dict is an
    # operator expression.
    if not isinstance(value, dict):
        return {key: {"$eq": _check_operand(key, "$eq", value)}}

    if not value:
        raise ValueError(f"Empty operator expression for field {key!r}")

    value = cast(dict[str, Any], value)

    # Several operators on one field: split into an $and of single-op
    # clauses (Chroma and langchain_postgres reject multi-op expressions).
    if len(value) > 1:
        sub = [{key: {op: operand}} for op, operand in value.items()]
        normalized = [_normalize_clause(s) for s in sub]
        if len(normalized) == 1:
            return normalized[0]
        return {"$and": normalized}

    op, operand = next(iter(value.items()))
    return {key: {op: _check_operand(key, op, operand)}}


def _check_operand(field: str, op: str, operand: Any) -> Any:
    """Validate an operator operand, returning it unchanged.

    Runs during normalization (:func:`_normalize_clause`) so a malformed
    operand is rejected once, up front, instead of surfacing later as a
    confusing backend-specific error.  Type rules: range operators and
    ``$eq``/``$ne``/``$contains`` take a scalar operand; ``$in``/``$nin``
    take a non-empty list of scalars of a single type.

    :param field: Metadata field the operator applies to (used in error
        messages)
    :param op: Operator (``$eq``, ``$in``, ...); must be in
        :data:`FIELD_OPERATORS`
    :param operand: Value to check
    :returns: *operand* unchanged
    :raises ValueError: When *op* is unsupported or a list operand is
        empty
    :raises TypeError: When *operand* has the wrong type for *op*
    """
    if op not in FIELD_OPERATORS:
        raise ValueError(f"Unsupported operator {op!r} on field {field!r}")
    if op in ("$in", "$nin"):
        if not isinstance(operand, list):
            raise TypeError(
                f"Operator {op} on field {field!r} expects a list, got {operand!r}"
            )
        if not operand:
            raise ValueError(
                f"Operator {op} on field {field!r} expects a non-empty list, "
                f"got {operand!r}"
            )
        if not all(isinstance(v, _SCALAR_TYPES) for v in operand):
            raise TypeError(
                f"Operator {op} on field {field!r} expects scalar list values, "
                f"got {operand!r}"
            )
        return operand
    if not isinstance(operand, _SCALAR_TYPES):
        raise TypeError(
            f"Operator {op} on field {field!r} expects a scalar value, got {operand!r}"
        )
    return operand


def _qdrant_clause(clause: dict[str, Any], models: _QdrantModels) -> Any:
    """Build a Qdrant ``Filter``/``FieldCondition`` tree for *clause*.

    Qdrant counterpart of :func:`to_chroma_filter` /
    :func:`to_pgvector_filter`: Qdrant filters are typed objects rather
    than dicts, so this walks the canonical normalized clause and builds
    the matching ``models.Filter`` tree -- ``$and`` becomes ``must``,
    ``$or`` becomes ``should``, and each leaf operator becomes a
    ``FieldCondition`` wrapping a ``MatchValue``/``MatchAny``/
    ``MatchExcept``/``Range``.

    :param clause: Canonical normalized clause (see
        :func:`validate_metadata_filter`)
    :param models: ``qdrant_client.models`` module namespace providing
        the ``Filter``/``FieldCondition`` constructors
    :returns: Qdrant ``models.Filter`` for *clause*
    """
    key, value = next(iter(clause.items()))

    if key == "$and":
        return models.Filter(must=[_qdrant_clause(sub, models) for sub in value])
    if key == "$or":
        return models.Filter(should=[_qdrant_clause(sub, models) for sub in value])

    op, operand = next(iter(value.items()))
    if op == "$eq":
        condition = models.FieldCondition(
            key=key, match=models.MatchValue(value=operand)
        )
    elif op == "$ne":
        condition = models.FieldCondition(
            key=key, match=models.MatchExcept(except_=[operand])
        )
    elif op == "$in":
        condition = models.FieldCondition(key=key, match=models.MatchAny(any=operand))
    elif op == "$nin":
        condition = models.FieldCondition(
            key=key, match=models.MatchExcept(except_=operand)
        )
    elif op in ("$gt", "$gte", "$lt", "$lte"):
        # "$gte" -> Range(gte=...); "$lt" -> Range(lt=...)
        condition = models.FieldCondition(
            key=key, range=models.Range(**{op[1:]: operand})
        )
    elif op == "$contains":
        condition = models.FieldCondition(
            key=key, match=models.MatchValue(value=operand)
        )
    else:  # pragma: no cover - validate_metadata_filter already checked ops
        raise ValueError(f"Unsupported operator {op!r}")
    return models.Filter(must=[condition])


def _pgvector_clause(clause: dict[str, Any]) -> dict[str, Any]:
    """Translate one level of the canonical form to a pgvector filter dict.

    ``langchain_postgres`` accepts the canonical normalized form almost
    unchanged, so this is a thin walk: ``$and``/``$or`` levels pass
    through with their sub-clauses translated, and the only rewrite is
    ``$contains`` -> ``$like`` (the backend has no array-containment
    operator; see :func:`to_pgvector_filter`).

    :param clause: Canonical normalized clause (see
        :func:`validate_metadata_filter`)
    :returns: ``langchain_postgres`` filter dict for *clause*
    """
    key, value = next(iter(clause.items()))
    if key in LOGICAL_OPERATORS:
        return {key: [_pgvector_clause(sub) for sub in value]}
    op, operand = next(iter(value.items()))
    if op == "$contains":
        # langchain_postgres has no array-contains operator; approximate
        # with a substring match over the serialized JSON array text.
        op, operand = "$like", f"%{operand}%"
    return {key: {op: operand}}


def _clause_fields(clause: dict[str, Any]) -> set[str]:
    """Return the metadata fields referenced by a single-clause DSL dict.

    Walks a canonical single-clause filter (from :func:`validate_metadata_filter`)
    and collects the field names it constrains: a leaf field clause yields
    its field; a logical combinator yields the union of its sub-clauses'
    fields.  Used by :func:`restrict_metadata_filter` to decide whether a
    clause may be applied to a domain whose declared field set does not
    include it.

    :param clause: Canonical single-clause DSL dict
    :returns: Set of metadata field names the clause references
    """
    key, value = next(iter(clause.items()))
    if key in LOGICAL_OPERATORS:
        fields: set[str] = set()
        for sub in value:
            fields |= _clause_fields(sub)
        return fields
    return {key}


def _clause_matches(clause: dict[str, Any], doc: Document) -> bool:
    """Evaluate a normalized filter clause against a document.

    Recursive counterpart of :func:`_normalize_clause`: walks a
    canonical normalized filter (from :func:`validate_metadata_filter`)
    top down.  ``$and``/``$or`` levels recurse into their sub-clauses;
    a field clause is delegated to :func:`_operand_matches` with the
    document's value for that field (``None`` when the field is absent).

    Used by :func:`filter_docs_by_metadata` to keep only the documents
    that satisfy every constraint, in-memory where no native backend
    filter exists (the BM25 store).

    :param clause: Canonical normalized clause (see
        :func:`validate_metadata_filter`)
    :param doc: Document whose metadata is tested
    :returns: True when *doc* satisfies *clause*
    """
    key, value = next(iter(clause.items()))
    if key == "$and":
        return all(_clause_matches(sub, doc) for sub in value)
    if key == "$or":
        return any(_clause_matches(sub, doc) for sub in value)
    op, operand = next(iter(value.items()))
    return _operand_matches(op, operand, doc.metadata.get(key))


def _operand_matches(op: str, operand: Any, actual: Any) -> bool:
    """Evaluate a single filter operator against a metadata value.

    This is the leaf evaluator of the Python-side document matcher
    (:func:`filter_docs_by_metadata` / :func:`_clause_matches`): given
    one field clause from a normalized filter, it decides whether a
    document's stored value for that field satisfies the clause.  It is
    the in-memory equivalent of the backends' native ``filter=``
    queries, used where no native filter exists (the BM25 store).

    Operator semantics (``actual`` is the document's metadata value for
    the field, ``operand`` the value taken from the filter):

    - ``$eq`` / ``$ne``: equality.  A list-valued field (e.g.
      ``authors``) matches when the operand is one of its elements; a
      scalar field matches by direct comparison.
    - ``$gt`` / ``$gte`` / ``$lt`` / ``$lte``: numeric comparison
      against a scalar field.  List/dict values never match, and values
      of incomparable types (e.g. an int field vs a str operand) return
      False rather than raising.
    - ``$in`` / ``$nin``: membership.  A scalar field matches when it is
      in the operand list; a list-valued field matches when any of its
      elements is in the operand list.
    - ``$contains``: element-membership of a list-valued field (the
      reverse of ``$in``).  A scalar field falls back to substring
      matching against its string form.

    An absent metadata field (``actual`` is ``None``) never matches any
    operator.

    :param op: Operator from the filter (``$eq``, ``$contains``, ...)
    :param operand: Operand value taken from the filter
    :param actual: The document's metadata value for the field, or
        ``None`` when the field is absent
    :returns: True when *actual* satisfies *op* with *operand*
    """
    if actual is None:
        return False
    if op == "$eq":
        return operand in actual if isinstance(actual, list) else actual == operand
    if op == "$ne":
        return operand not in actual if isinstance(actual, list) else actual != operand
    if op in ("$gt", "$gte", "$lt", "$lte"):
        if isinstance(actual, (list, dict)):
            return False
        try:
            if op == "$gt":
                return actual > operand
            if op == "$gte":
                return actual >= operand
            if op == "$lt":
                return actual < operand
            return actual <= operand
        except TypeError:
            return False
    if op == "$in":
        return (
            any(v in operand for v in actual)
            if isinstance(actual, list)
            else actual in operand
        )
    if op == "$nin":
        return (
            not any(v in operand for v in actual)
            if isinstance(actual, list)
            else actual not in operand
        )
    if op == "$contains":
        return operand in actual if isinstance(actual, list) else operand in str(actual)
    return False  # pragma: no cover - validate_metadata_filter checked ops
