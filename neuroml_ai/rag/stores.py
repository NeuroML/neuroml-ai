#!/usr/bin/env python3
"""
Vector stores

File: neuroml_ai/stores.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import mimetypes
import shutil
import sys
from glob import glob
from hashlib import sha256
from pathlib import Path
from typing import Any

import chromadb
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from .utils import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    logger_formatter_info,
    logger_formatter_other,
    setup_llm
)

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class NML_Stores(object):
    """Vector stores"""

    md_headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    def __init__(
        self,
        embedding_model: str,
        logging_level: int = logging.DEBUG,
    ):
        """Init"""
        # per store
        self.default_k = 5
        self.k_max = 10
        self.k = self.default_k
        self.sim_thresh = 0.15
        self.chunk_size = 600
        self.chunk_overlap = 60

        self.embedding_model = embedding_model
        self.embeddings = None

        # we prefer markdown because the one page PDF that is available for the
        # documentation does not work too well with embeddings
        my_path = Path(__file__).parent
        self.data_dir = f"{my_path}/data/"
        self.stores_path = f"{self.data_dir}/vector-stores"
        self.stores_sources_path = f"{self.stores_path}/sources"

        self.text_vector_stores: dict[str, Chroma] = {}
        self.image_vector_stores: dict[str, Any] = {}

        self.logger = logging.getLogger("NeuroML-AI")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(logger_formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(logger_formatter_other)
        self.logger.addHandler(stderr_handler)

    def setup(self):
        """Setup stores"""
        self.embeddings = setup_llm(self.embedding_model, self.logger, True)


    def inc_k(self, inc: int = 1):
        """Increase k by inc

        :param inc: int to increase k by
        :returns: True if k was increased, False otherwise

        """
        if (self.k + inc) <= self.k_max:
            self.k += inc
            self.logger.debug(f"k increased to {self.k =}")
            return True

        return False

    def reset_k(self):
        """Reset k to default value"""
        self.k = self.default_k
        self.logger.debug(f"k reset to {self.k =}")

    def remove(self):
        """Remove all vector stores.
        Usually needed when they need to be regenerated
        """
        sure = input("NeuroML-AI >>> Delete all vector stores, are you sure? [Y/N] ")
        if sure.lower() == "y":
            vec_stores = glob(f"{self.stores_path}/*.db", recursive=False)
            for store in vec_stores:
                self.logger.info(f"Deleting vector {store}")
                shutil.rmtree(store)
        else:
            self.logger.info("Did not delete any vector stores. Continuing.")

    def load(self):
        """Create/load the vector store"""
        assert self.embeddings

        self.logger.debug("Setting up/loading Chroma vector store")

        self.logger.debug(f"{self.stores_sources_path =}")
        vec_store_sources = glob(f"{self.stores_sources_path}/*", recursive=False)
        self.logger.debug(f"{vec_store_sources =}")

        assert len(vec_store_sources)

        for src in vec_store_sources:
            self.logger.debug(f"Setting up vector store: {src}")
            src_path = Path(src)

            assert src_path.is_dir()

            vs_persist_dir = f"{self.stores_path}/{src_path.name}_{self.embedding_model.replace(':', '_')}.db"
            self.logger.debug(f"{vs_persist_dir =}")

            chroma_client_settings_text = chromadb.config.Settings(
                is_persistent=True,
                persist_directory=vs_persist_dir,
                anonymized_telemetry=False,
            )
            store = Chroma(
                collection_name=src_path.name,
                embedding_function=self.embeddings,
                client_settings=chroma_client_settings_text,
            )

            self.text_vector_stores[src_path.name] = store

            info_files = glob(f"{src}/*", recursive=True)
            self.logger.debug(f"Loaded {len(info_files)} files from {src}")

            for info_file in info_files:
                try:
                    file_type = mimetypes.guess_file_type(info_file)[0]
                except AttributeError:
                    # for py<3.13
                    file_type = mimetypes.guess_type(info_file)[0]

                if file_type:
                    if "markdown" in file_type:
                        self.add_md(store, info_file)
                    else:
                        self.logger.warning(
                            f"File {info_file} is of type {file_type} which is not currently supported. Skipping"
                        )
                else:
                    self.logger.warning(
                        f"Could not guess file type for file {info_file}. Skipping"
                    )

    def add_md(self, store, file):
        """Add a markdown file to the vector store

        We add the file hash as extra metadata so that we can filter on it
        later.

        TODO: Handle images referenced in the markdown file.

        For this, we need to use the same metadata for the chunks and for the
        images in those chunks when they're added to the text and image stores.
        The text chunks need to have an id each, and a list of figures too. The
        images being added will need to have the document/file id, and the
        figure ids.

        For retrieval, we will first run the similarity search on both the text
        and images. For text results, we will retrieve any linked images.

        Note that for text only LLMs, only the associated metadata of the
        obtained images (captions and so on) can be used in the context. To use
        the images too, we need to use multi-modal LLMs.
        """
        file_path = Path(file)
        file_hash = sha256(file_path.name.encode("utf-8")).hexdigest()
        already_added = store.get(where={"file_hash": file_hash})

        if already_added and already_added["ids"]:
            self.logger.debug(f"File already exists in vector store: {file_path}")
            return

        self.logger.debug(f"Adding markdown file to text vector store: {file_path}")
        with open(file, "r") as f:
            md_doc = f.read()
            self.logger.debug(f"Length of loaded file: {len(md_doc.split())}")
            md_splitter = MarkdownHeaderTextSplitter(
                self.md_headers_to_split_on, strip_headers=False
            )
            md_splits = md_splitter.split_text(md_doc)
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
            )
            splits = text_splitter.split_documents(md_splits)
            for split in splits:
                split.metadata.update(
                    {
                        "file_hash": file_hash,
                        "file_name": file_path.name,
                        "file_path": str(file_path),
                    }
                )

            self.logger.debug(f"Length of split docs: {len(splits)}")
            _ = store.add_documents(documents=splits)

    def retrieve(self, query: str) -> list[tuple[Document, float]]:
        """Retrieve embeddings from documentation to answer a query

        :param query: user query
        :returns: list of tuples (document, score)

        """
        self.load()

        assert len(self.text_vector_stores)

        res = []

        for sname, store in self.text_vector_stores.items():
            data = store.similarity_search_with_relevance_scores(
                query, k=self.k, score_threshold=self.sim_thresh
            )
            self.logger.debug(f"{data =}")
            res.extend(data)

        return res
