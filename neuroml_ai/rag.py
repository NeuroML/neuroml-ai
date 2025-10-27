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
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.tools import tool
from langchain.agents import create_agent
from typing_extensions import TypedDict, Literal, List
from pydantic import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from textwrap import dedent
from langgraph.prebuilt import ToolNode, tools_condition


logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class QueryTypeState(TypedDict):
    """Classification of user query"""

    category: Literal["question", "code_generation", "unknown"]


class AgentState(TypedDict):
    """The state of the graph"""

    query: str
    query_type: QueryTypeState
    messages: List[BaseMessage]


class QueryTypeSchema(BaseModel):
    """Docstring for QueryTypeSchema."""

    query_type: Literal["question", "code_generation"] = Field(
        description="'question' if user is asking for information, 'code_generation', if the user is asking for code"
    )


class NML_RAG(object):
    """NeuroML RAG implementation"""

    # This is generated using the script in the data/scripts folder.
    # Effectively:
    # - we download the documentation sources
    # - we use the jupyterbook system to generate a one page html
    # - we use pandoc to convert the one page html to a one page markdown

    # we prefer markdown because the one page PDF that is available for the
    # documentation does not work too well with embeddings
    nml_doc_md_path = "../data/single-markdown.md"
    def __init__(
        self,
        chat_model: str = "ollama:qwen3:1.7b",
        embedding_model: str = "bge-m3",
        logging_level: int = logging.DEBUG,
    ):
        """Initialise"""
        self.chat_model = chat_model
        self.model = None
        self.embedding_model = embedding_model
        self.embeddings = None

        self.vector_store = None

        self.logger = logging.getLogger("NeuroML-AI")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False
        formatter = logging.Formatter("%(name)s (%(levelname)s) >>> %(message)s")
        handler = logging.StreamHandler()
        handler.setLevel(logging_level)
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

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

        # self.__setup_agent()

    def __question_or_code_node(self, state: AgentState) -> dict:
        """LLM decides what type the user query is"""
        assert self.model

        system_prompt = """You are an expert query classifier. Analyse the user's request
            and determine its intent. Is it a 'quertion' or a 'code_generation'
            request? Provide your answer as a JSON object matching the
            requested schema."""

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.model.with_structured_output(QueryTypeSchema)
        prompt = prompt_template.invoke({"query": state["query"]})

        output = query_node_llm.invoke(prompt)
        self.logger.debug(f"{output =}")

        return {"query_type": output.query_type}

    def __generate_code_node(self, state: AgentState) -> dict:
        """Generate code"""
        messages = state["messages"]
        messages.append(AIMessage("I can generate code for you"))

        return {"messages": messages}

    def __answer_question_node(self, state: AgentState) -> dict:
        """Answer the question"""

        system_prompt = dedent("""
        You are a fact-based research assistant. Your primary and only goal is
        to provide accurate and current answers to user queries. Your
        speciality is NeuroML, the standard and software ecosystem for
        biophysically detailed modelling and related tools.

        # Core Directives:
        1. Top priority: use the Tools. For any user query that requires a
          factual answer, use the tool. Never answer a knowledge based question
          directly from your general training data without using the tools. The
          only exceptions are general conversational greetings, where you may
          skip using the tool.

        2. When using the tools, generate precise queries. Do not use stop
          words.

        3. After obtaining the data from the tool, only use the obtained
          information to craft your answer.

        4. If you are unable to find the answer in the documentation, let the
          user know and ask them to modify their query.

        ## Available tools:

        You have access to the following tools:

        1. `__retrieve_docs`: use this tool to search the NeuroML documentation

        ## Your thought process (ReAct):

        You must always structure your response using the
        Thought, Action, Observation, Final Answer pattern in that order:

        1. Thought: Reason about the user's request. Always conclude that a
          factual query requires an `Action: retrieve`.
        2. Action: Generate the tool call (e.g., `retrieve({{"query": "focused
          search term"}})`).
        3. Observation: This is the result of the tool execution (the
          documents).
        4. Final Answer: Generate the final response based only on the
          Observation.

        """)

        assert self.model

        question_prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
            # [("human", "User query: {query}")]
        )
        self.logger.debug(f"{question_prompt_template =}")
        prompt = question_prompt_template.invoke({"query": state["query"]})

        self.__load_doc()
        retrieve_docs_tool = tool(
            "__retrieve_docs",
            description="Retrieve information from NeuroML documentation",
            response_format="content_and_artifact",
        )(self.__retrieve_docs)

        self.logger.debug(f"{retrieve_docs_tool =}")

        output = self.model.bind_tools([retrieve_docs_tool]).invoke(prompt)
        # self.logger.debug(f"{output =}")
        self.logger.debug(output.pretty_print())

        messages = state["messages"]
        messages.append(output)
        return {"messages": messages}

    def __generate_answer_node(self, state: AgentState) -> dict:
        """Generate the answer"""
        assert self.model

        system_prompt = dedent("""
        You are a fact-based research assistant. Your primary and only goal is
        to provide accurate and current answers to user queries. Your
        speciality is NeuroML, the standard and software ecosystem for
        biophysically detailed modelling and related tools.

        # Core Directives:

        - Limit yourself to facts from the provided context only, avoid using
          knowledge from your general training. If you cannot find an answer in
          the context, tell the user and suggest a better question.
        - Use concise, formal language.

        ## Your thought process (ReAct):

        You must always structure your response using the
        Thought, Action, Observation, Final Answer pattern in that order:

        1. Thought: Reason about the user's request. Always conclude that a
          factual query requires an `Action: retrieve`.
        2. Action: Generate the tool call (e.g., `retrieve({{"query": "focused
          search term"}})`).
        3. Observation: This is the result of the tool execution (the
          documents).
        4. Final Answer: Generate the final response based only on the
          Observation.

        """)

        generate_answer_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "Question: {question}\nContext:{context}")]
            # [("human", "User query: {query}")]
        )
        question = state["query"]
        context = state["messages"][-1].content
        prompt = generate_answer_template.invoke({"question": question, "context": context})
        self.logger.debug(f"{prompt =}")
        output = self.model.invoke(prompt)
        self.logger.debug(output.pretty_print())

        messages = state["messages"]
        messages.append(output)
        return {"messages": messages}


    def __route_query_node(self, state: AgentState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        query_type = state["query_type"]

        if query_type == "question":
            return "answer_question_node"
        elif query_type == "code_generation":
            return "generate_code_node"
        else:
            return "handle_unknown_node"

    def __create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("classify_query", self.__question_or_code_node)
        self.workflow.add_node("answer_question", self.__answer_question_node)
        self.workflow.add_node("retrieve_docs", ToolNode([self.__retrieve_docs]))
        self.workflow.add_node("generate_code", self.__generate_code_node)
        self.workflow.add_node("generate_answer", self.__generate_answer_node)

        self.workflow.add_edge(START, "classify_query")

        self.workflow.add_conditional_edges(
            "classify_query",
            self.__route_query_node,
            {
                "answer_question_node": "answer_question",
                "generate_code_node": "generate_code",
                "unknown": END,
            },
        )

        self.workflow.add_conditional_edges(
            "answer_question",
            tools_condition,
            {
                "tools": "retrieve_docs",
                END: END,
            }
        )

        self.workflow.add_edge("retrieve_docs", "generate_answer")
        self.workflow.add_edge("generate_answer", END)
        self.workflow.add_edge("generate_code", END)

        self.graph = self.workflow.compile()
        self.graph.get_graph().draw_mermaid_png(output_file_path="nml-ai-lang-graph.png")

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
        self.model = init_chat_model(
            self.chat_model.replace("ollama:", ""), model_provider="ollama"
        )

        self.logger.info(f"Setting up embedding model: {self.embedding_model}")
        self.embeddings = OllamaEmbeddings(model=self.embedding_model)

    def __load_doc(self):
        """Load NeuroML documentation into the vector store"""

        # if a populated vector store already exists and is loaded, don't do
        # anything
        if self.vector_store and self.vector_store._collection.count() != 0:
            self.logger.info("Vector store loaded")
            return

        assert self.embeddings

        self.logger.debug("Setting up/loading Chroma vector store")
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

        # if a vector store exists, but it empty, load documents
        if self.vector_store._collection.count() == 0:
            self.logger.info(
                "Vector store appears empty. Generating embeddings and adding documents. "
                "This may take a while, depending on your hardware."
            )

            # update to use MD
            headers_to_split_on = [
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
                ("####", "Header 4"),

            ]
            with open(self.nml_doc_md_path, 'r') as f:
                md_doc = f.read()
                self.logger.debug(f"Length of loaded nml_docs: {len(md_doc.split())}")
                md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on, strip_headers=False)
                md_splits = md_splitter.split_text(md_doc)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(md_splits)

                self.logger.debug(f"Length of split docs: {len(splits)}")
                self.index = self.vector_store.add_documents(documents=splits)
        else:
            self.logger.info("Vector store loaded.")

    def __retrieve_docs(self, query: str):
        """Retrieve embeddings from documentation to answer a query

        :param query: user query
        :returns: serialised metadata/page content and vector_store look up result

        """
        assert self.vector_store

        res = self.vector_store.similarity_search(query, k=5)

        serialized = "\n\n".join(
            (f"Source: {r.metadata}\nContent:{r.page_content}") for r in res
        )
        self.logger.debug(res)
        return serialized, res

    def test_retrieval(self):
        """Test the retrieval system"""
        self.__load_doc()
        self.__retrieve_docs("NeuroML community")

    def run_graph(self, query: str):
        """Run the graph"""
        self.__create_graph()

        initial_state = AgentState(
            query=query,
            query_type="",
            messages=[],
        )

        # output = self.__answer_question_node(initial_state)
        # output["messages"][-1].pretty_print()

        for chunk in self.graph.stream(initial_state):
            for node, state in chunk.items():
                print()
                print(f"Update from node '{node}'")
                try:
                    state["messages"][-1].pretty_print()
                except KeyError:
                    print(state)
                print()


if __name__ == "__main__":
    nml_ai = NML_RAG(
        chat_model="ollama:qwen3:1.7b",
        embedding_model="bge-m3",
        logging_level=logging.DEBUG,
    )
    nml_ai.setup()
    # nml_ai.test_retrieval()
    nml_ai.run_graph("Summarise why one should use NeuroML")
