# Ported from langchain-community under the MIT License.
# Copyright (c) 2024 LangChain
# Source: https://github.com/langchain-ai/langchain-community
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files, to deal in the Software
# without restriction, including without limitation the rights to use, copy,
# modify, merge, publish, distribute, sublicense, and/or sell copies.

"""BM25 Retriever ported from langchain-community."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict, Field


def default_preprocessing_func(text: str) -> list[str]:
    return text.split()


class BM25Retriever(BaseRetriever):
    """BM25 retriever using rank_bm25 for keyword-based document scoring.

    This retriever processes a collection of documents into an in-memory Okapi BM25 
    index using the `rank_bm25` package. It tokenizes page contents and ranks 
    documents based on term frequency and inverse document frequency (TF-IDF derivative).

    Attributes:
        vectorizer (Any): The underlying rank_bm25 search index object (BM25Okapi).
        docs (List[Document]): List of underlying LangChain Document objects.
        k (int): Default number of documents to return per query.
        preprocess_func (Callable[[str], List[str]]): Function used to tokenize text.
    """

    vectorizer: Any = None
    """ BM25 vectorizer."""
    docs: list[Document] = Field(repr=False)
    """ List of documents."""
    k: int = 4
    """ Number of documents to return."""
    preprocess_func: Callable[[str], list[str]] = default_preprocessing_func
    """ Preprocessing function to use on the text before BM25 vectorization."""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )

    @classmethod
    def from_texts(
        cls,
        texts: Iterable[str],
        metadatas: Iterable[dict] | None = None,
        ids: Iterable[str] | None = None,
        bm25_params: dict[str, Any] | None = None,
        preprocess_func: Callable[[str], list[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of texts.
        Args:
            texts: A list of texts to vectorize.
            metadatas: A list of metadata dicts to associate with each text.
            ids: A list of ids to associate with each text.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            raise ImportError(
                "Could not import rank_bm25, please install with `pip install "
                "rank_bm25`."
            )

        texts_processed = [preprocess_func(t) for t in texts]
        bm25_params = bm25_params or {}
        vectorizer = BM25Okapi(texts_processed, **bm25_params)
        metadatas = metadatas or ({} for _ in texts)
        if ids:
            docs = [
                Document(page_content=t, metadata=m, id=i)
                for t, m, i in zip(texts, metadatas, ids)
            ]
        else:
            docs = [
                Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)
            ]
        return cls(
            vectorizer=vectorizer, docs=docs, preprocess_func=preprocess_func, **kwargs
        )

    @classmethod
    def from_documents(
        cls,
        documents: Iterable[Document],
        *,
        bm25_params: dict[str, Any] | None = None,
        preprocess_func: Callable[[str], list[str]] = default_preprocessing_func,
        **kwargs: Any,
    ) -> BM25Retriever:
        """
        Create a BM25Retriever from a list of Documents.
        Args:
            documents: A list of Documents to vectorize.
            bm25_params: Parameters to pass to the BM25 vectorizer.
            preprocess_func: A function to preprocess each text before vectorization.
            **kwargs: Any other arguments to pass to the retriever.

        Returns:
            A BM25Retriever instance.
        """
        texts, metadatas, ids = zip(
            *((d.page_content, d.metadata, d.id) for d in documents)
        )
        return cls.from_texts(
            texts=texts,
            bm25_params=bm25_params,
            metadatas=metadatas,
            ids=ids,
            preprocess_func=preprocess_func,
            **kwargs,
        )

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        processed_query = self.preprocess_func(query)
        return_docs = self.vectorizer.get_top_n(processed_query, self.docs, n=self.k)
        return return_docs