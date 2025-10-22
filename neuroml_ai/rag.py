#!/usr/bin/env python3
"""
NeuroML RAG implementation

File: rag.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import os
import sys
import getpass
import logging
from langchain.chat_models import init_chat_model
from langchain_chroma import Chroma
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent


logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class NML_RAG(object):
    """NeuroML RAG implementation"""

    nml_doc_pdf_path = "../data/neuroml-documentation.pdf"

    def __init__(self, chat_model: str = "ollama:qwen3:1.7b", embedding_model: str = "bge-m3"):
        """Initialise"""
        self.chat_model = chat_model
        self.model = None
        self.embedding_model = embedding_model
        self.embeddings = None

        self.logger = logging.getLogger("NML_RAG")
        self.logger.setLevel(logging.DEBUG)

        self.system_prompt = """
        You are a fact-based research assistant. Your primary and only goal is
        to provide accurate and current answers to user queries. Your
        speciality is NeuroML, the standard and software ecosystem for
        biophysically detailed modelling and related tools.

        Core Directives:
        1. Top priority: use the Tools. For any user query that requires a
        factual answer, use the tool. Never answer a knowledge based question
        directly from your general training data without using the tools. The
        only exceptions are general conversational greetings, where you may
        skip using the tool. Do not use synonyms to replace observations
        obtained from the tools.

        2. When using the tools, generate precise queries. Do not use stop
        words.

        3. After obtaining the data from the tool, only use the obtained
        information to craft your answer.

        Available tools:
        You have access to the following tools:

        1. "retrieve_docs": use this tool to search the provided documentation.

        Your thought process (ReAct):

        You must always structure your response using the
        Thought/Action/Observation/Final Answer pattern:

        - Thought: Reason about the user's request. Always conclude that a
          factual query requires an `Action: retrieve`.
        - ActionGenerate the tool call (e.g., `retrieve({"query": "focused
          search term"})`).
        - Observation: This is the result of the tool execution (the
          documents).
        - Final Answer: Generate the final response based only on the
          Observation.

        """

    def setup(self):
        """Set up basics."""

        if self.chat_model.lower() == "gemini":
            self.__setup_gemini()
        elif self.chat_model.lower().startswith("ollama:"):
            self.__setup_ollama()
        else:
            self.logger.error(f"Unknown LLM model given: {self.chat_model}. Exiting.")
            sys.exit(-1)

        assert self.model

        self.__load_doc()
        self.__setup_agent()

    def __setup_agent(self):
        """Set up the chat agent"""

        # langchain tool decorator does not work with class methods because
        # Python expects `self` as the first argument which is not provided
        # when the tool is called. So, we can either bind manually as below, or
        # we can refactor the code to make the tool an external function that
        # is not a class method
        retrieve_docs_tool = tool("retrieve_docs", response_format="content_and_artifact")(self.__retrieve_docs)
        self.tools = [retrieve_docs_tool]
        # prompt = "Ask a question about NeuroML"

        self.agent = create_agent(self.model, self.tools, system_prompt=self.system_prompt)

    def __setup_gemini(self):
        """Set up Gemini"""
        self.logger.info("Setting up Gemini")

        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        if not os.environ.get("GOOGLE_API_KEY"):
            os.environ["GOOGLE_API_KEY"] = getpass.getpass(
                "Enter API key for Google Gemini: "
            )

        self.model = init_chat_model(
            "gemini-2.5-flash", model_provider="google_genai", temperature=0
        )
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001"
        )

    def __setup_ollama(self):
        """Set up Ollama model"""


        from langchain_ollama import OllamaEmbeddings

        self.logger.info(f"Setting up chat model: {self.chat_model}")
        self.model = init_chat_model(self.chat_model.replace("ollama:", ""), model_provider="ollama")

        self.logger.info(f"Setting up embedding model: {self.embedding_model}")
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

    def __load_doc(self):
        """Load NeuroML documentation
        :returns: TODO

        """
        assert self.embeddings

        self.loader = PyPDFLoader(self.nml_doc_pdf_path)
        self.nml_doc = self.loader.load()

        self.logger.debug(f"Length of loaded nml_docs: {len(self.nml_doc)}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, add_start_index=True
        )

        self.splits = self.text_splitter.split_documents(self.nml_doc)
        self.logger.debug(f"Length of split docs: {len(self.splits)}")

        chroma_client_settings = chromadb.config.Settings(
            is_persistent=True,
            persist_directory=f"../data/neuroml_docs_{self.embedding_model.replace(':', '_')}.db",
            anonymized_telemetry=False,
        )
        self.vector_store = Chroma(
            collection_name="nml_docs",
            embedding_function=self.embeddings,
            client_settings=chroma_client_settings,
        )

        if self.vector_store._collection.count() == 0:
            self.logger.info(
                "Vector store appears empty. Generating embeddings and adding documents. "
                "This may take a while, depending on your hardware."
            )
            self.index = self.vector_store.add_documents(documents=self.splits)
        else:
            self.logger.info("Vector store already set up.")

    def __retrieve_docs(self, query: str):
        """Retrieve embeddings from documentation to answer a query

        :param query: user query
        :returns: serialised metadata/page content and vector_store look up result

        """
        res = self.vector_store.similarity_search(query)
        serialized = "\n\n".join(
            (f"Source: {r.metadata}\nContent:{r.page_content}") for r in res
        )
        return serialized, res

    def run(self):
        """Main runner method"""
        assert self.agent

        query = "List the standard NeuroML component types "

        for event in self.agent.stream(
            {
                "messages": [
                    {"role": "user", "content": query},
                    # {"role": "system", "content": self.system_prompt},
                ]
            },
            stream_mode="values",
        ):
            event["messages"][-1].pretty_print()


if __name__ == "__main__":
    nml_ai = NML_RAG()
    nml_ai.setup()
    nml_ai.run()
