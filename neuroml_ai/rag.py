#!/usr/bin/env python3
"""
NeuroML RAG implementation

File: rag.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
import getpass
import logging
from langchain.chat_models import init_chat_model
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class NML_RAG(object):
    """NeuroML RAG implementation"""

    nml_doc_pdf_path = "../data/neuroml-documentation.pdf"

    def __init__(self, llm: str = "gemini"):
        """Initialise"""
        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass(
                "Enter API key for Google Gemini: "
            )

        self.llm = llm
        self.model = None
        self.embeddings = None

        self.logger = logging.getLogger("NML_RAG")
        self.logger.setLevel(logging.DEBUG)

    def setup(self):
        """Set up basics."""

        if self.llm.lower() == "gemini":
            self.__setup_gemini()
        if self.llm.lower() == "ollama_qwen":
            self.__setup_ollama_qwen()

        assert self.model
        assert self.embeddings

        self.__load_doc()


    def __setup_gemini(self):
        """Set up Gemini"""
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        self.model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

    def __setup_ollama_qwen(self):
        """Set up Qwen via Ollama """
        from langchain_ollama import OllamaEmbeddings

        self.model = init_chat_model("qwen3:4b")
        self.embeddings = OllamaEmbeddings(
            model="models/gemini-embedding-001"
        )


    def __load_doc(self):
        """Load NeuroML documentation
        :returns: TODO

        """
        self.loader = PyPDFLoader(self.nml_doc_pdf_path)
        self.nml_doc = self.loader.load()

        self.logger.debug(f"Length of loaded nml_docs: {len(self.nml_doc)}")

        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                            chunk_overlap=200,
                                                            add_start_index=True)

        self.splits = self.text_splitter.split_documents(self.nml_doc)
        self.logger.debug(f"Length of split docs: {len(self.splits)}")

        self.vector_store = InMemoryVectorStore(self.embeddings)
        self.index = self.vector_store.add_documents(documents=self.splits)


    def run(self):
        """Main runner method"""
        results = self.vector_store.similarity_search("Describe how I can create a new NeuroML model from scratch?")
        print(results)


if __name__ == "__main__":
    nml_ai = NML_RAG()
    nml_ai.setup()
    nml_ai.run()
