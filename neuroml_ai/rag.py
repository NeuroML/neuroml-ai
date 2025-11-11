#!/usr/bin/env python3
"""
NeuroML RAG implementation

File: rag.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import mimetypes
import sys
from glob import glob
from hashlib import sha256
from pathlib import Path
from textwrap import dedent

import chromadb
import ollama
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain.tools import tool
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from .schemas import QueryTypeSchema, EvaluateAnswerSchema, AgentState

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class LoggerNotInfoFilter(logging.Filter):
    """Allow only non INFO messages"""

    def filter(self, record):
        return record.levelno != logging.INFO


class LoggerInfoFilter(logging.Filter):
    """Allow only INFO messages"""

    def filter(self, record):
        return record.levelno == logging.INFO


class NML_RAG(object):
    """NeuroML RAG implementation"""

    # This is generated using the script in the data/scripts folder.
    # Effectively:
    # - we download the documentation sources
    # - we use the jupyterbook system to generate a one page html
    # - we use pandoc to convert the one page html to a one page markdown

    # update to use MD
    md_headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
        ("####", "Header 4"),
    ]

    checkpointer = InMemorySaver()

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

        self.text_vector_store = None
        self.image_vector_store = None

        self.default_k = 2
        self.k = self.default_k

        # we prefer markdown because the one page PDF that is available for the
        # documentation does not work too well with embeddings
        my_path = Path(__file__).parent
        self.data_dir = f"{my_path}/data/"
        self.data_files_path = f"{self.data_dir}/files/"

        self.logger = logging.getLogger("NeuroML-AI")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        formatter = logging.Formatter("%(name)s (%(levelname)s) >>> %(message)s")

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(formatter)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(formatter)
        self.logger.addHandler(stderr_handler)

    def setup(self):
        """Set up basics."""

        if self.chat_model.lower().startswith("ollama:"):
            self.__check_ollama_model(self.chat_model.lower().replace("ollama:", ""))

        self.model = init_chat_model(
            self.chat_model,
            temperature=0.3,
        )

        if self.embedding_model.lower().startswith("ollama:"):
            self.__check_ollama_model(
                self.embedding_model.lower().replace("ollama:", "")
            )

        self.embeddings = init_embeddings(self.embedding_model)

        assert self.model
        assert self.embeddings

        self._create_graph()

    def _classify_query_node(self, state: AgentState) -> dict:
        """LLM decides what type the user query is"""
        assert self.model

        system_prompt = """You are an expert query classifier. Analyse the user's request
            and determine its intent. Is it a 'general question', a 'NeuroML
            question', or a 'neuroml code generation' request? Provide your
            answer as a JSON object matching the
            requested schema."""

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.model.with_structured_output(QueryTypeSchema)
        prompt = prompt_template.invoke({"query": state.query})

        output = query_node_llm.invoke(prompt)
        self.logger.debug(f"{output = }")

        return {"query_type": QueryTypeSchema(query_type=output.query_type)}

    def _generate_neuroml_code_node(self, state: AgentState) -> dict:
        """Generate code"""
        messages = state.messages
        messages.append(AIMessage("I can generate code for you"))

        return {"messages": messages}

    def _answer_general_question_node(self, state: AgentState) -> dict:
        """Answer a general question"""

        system_prompt = dedent("""
        You are an AI assistant. Answer questions to the best of your
        knowledge.

        ## Core directives

        1. Only provide information you are confident about. If you are
        unsuare, clearly say so.
        2. Avoid inventing facts. If a fact is not known or uncertain, respond
        with "I was unable to find factual information about this query".
        3. Keep answers clear, concise, formal, and user-friendly.
        4. Always prefix all your answers with this warning "This information
        has not been retrieved from any provided documentation and may contain
        errors. It is generated from general LLM knowledge."

        """)

        assert self.model

        question_prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )
        self.logger.debug(f"{question_prompt_template =}")
        prompt = question_prompt_template.invoke({"query": state.query})

        output = self.model.invoke(prompt)
        # self.logger.debug(f"{output =}")
        self.logger.debug(output)

        messages = state.messages
        messages.append(output)
        return {"messages": messages, "user_message": output.content}

    def _answer_neuroml_question_node(self, state: AgentState) -> dict:
        """Answer a NeuroML question"""

        system_prompt = dedent("""
        You are a fact-based research assistant. Your only goal is to provide
        accurate and current answers to user queries. Your speciality is
        NeuroML, the standard and software ecosystem for biophysically detailed
        modelling of neurons and neuronal circuits and related software tools.

        # Core Directives:
        1. Top priority: use the Tools. For any user query that requires a
          factual answer, use the tool. Never answer a knowledge based question
          directly from your general training data without using the tools.
          Only skip using the tool for queries that are not NeuroML related.

        2. When using the tools, generate precise queries. Do not use stop
          words.

        3. After obtaining the data from the tool, only use the obtained
          information to craft your answer.

        4. Prefer Python over other programming languages that may be mentioned
          in the documentation.

        5. If you are unable to find the answer in the documentation, let the
          user know.

        ## Available tools:

        You have access to the following tools:

        1. `_retrieve_docs`: use this tool to search the NeuroML documentation

        ## Your thought process (ReAct):

        You must always structure your response using the
        Thought, Action, Observation, Final Answer pattern in that order:

        1. Thought: Reason about the user's request. Always conclude that a
          factual query about NeuroML requires an `Action: retrieve`.
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
        prompt = question_prompt_template.invoke({"query": state.query})

        self._load_vector_stores()
        retrieve_docs_tool = tool(
            "_retrieve_docs",
            description="Retrieve information from NeuroML documentation",
            response_format="content_and_artifact",
        )(self._retrieve_docs)

        self.logger.debug(f"{retrieve_docs_tool =}")

        output = self.model.bind_tools([retrieve_docs_tool]).invoke(prompt)
        # self.logger.debug(f"{output =}")
        self.logger.debug(output)

        messages = state.messages
        messages.append(output)
        return {"messages": messages}

    def _generate_answer_from_context_node(self, state: AgentState) -> dict:
        """Generate the answer"""
        assert self.model

        system_prompt = dedent("""
        You are an expert summariser. Use only the provided context to generate
        an answer to the provided question.

        # Core Directives:

        - Limit yourself to facts from the provided context only, avoid using
          knowledge from your general training.
        - Use concise, formal language.

        """)

        generate_answer_template = ChatPromptTemplate(
            [
                ("system", system_prompt),
                ("human", "Question: {question}\nContext:{context}"),
            ]
            # [("human", "User query: {query}")]
        )
        question = state.query
        context = state.messages[-1].content
        prompt = generate_answer_template.invoke(
            {"question": question, "context": context}
        )
        self.logger.debug(f"{prompt =}")
        output = self.model.invoke(prompt)
        self.logger.debug(output.pretty_repr())

        messages = state.messages
        messages.append(output)
        return {"messages": messages}

    def _evaluate_answer_node(self, state: AgentState) -> dict:
        """Evaluate the answer"""
        evaluator_model = init_chat_model(
            self.chat_model.replace("ollama:", ""),
            model_provider="ollama",
            temperature=0,
        )

        evaluator_prompt = dedent("""
            You are a critical grader evaluating an answer produced by a retrieval-augmented generation (RAG) system.

            You are given:
            1. The user's question.
            2. The retrieved context used to generate the answer.
            3. The system's answer.

            Your job:
            - Judge the answer strictly based on the context.
            - DO NOT use external knowledge.
            - Provide your answer as a JSON object matching the provided
              schema, with these values:

            {{
              "relevance": 0-1,         // How well the answer addresses the question
              "groundedness": 0-1,      // How well it sticks to the provided context
              "completeness": 0-1,      // Whether it covers all necessary info from context
              "coherence": 0-1,         // Logical flow and clarity
              "conciseness": 0-1,       // Avoids fluff or repetition
              "confidence": 0-1,        // Overall sufficiency of context to support answer
              "next_step": "continue", "retrieve_more_info", "modify_query", "ask_user"// next actions
              "summary": "Brief natural-language justification for the grades"
            }}

            Guidelines for 'next_step':
            - Set to 'continue' if the answer is clear and should be passed to the user
            - Set to 'retrieve_more_info' if the answer is incomplete but grounded and needs more context
            - Set to 'modify_query' if the answer is ungrounded or irrelevant and the query needs to be reformulated to improve retrieval precision
            - Set to 'ask_user' if the query cannot be answered from the corpus and we need to ask the user for clarification or additional information
            """)

        question = state.query

        context = state.messages[-2].content
        answer = state.messages[-1].content
        assert len(question)
        assert len(context)
        assert len(answer)

        prompt_template = ChatPromptTemplate(
            [
                ("system", evaluator_prompt),
                (
                    "human",
                    dedent("""
                        Question:
                        {question}

                        -----
                        Context:
                        {context}

                        -----
                        Answer:
                        {answer}

                 """),
                ),
            ]
        )

        # can use | to merge these lines
        query_node_llm = evaluator_model.with_structured_output(EvaluateAnswerSchema)
        prompt = prompt_template.invoke(
            {"question": question, "context": context, "answer": answer}
        )

        output = query_node_llm.invoke(prompt)
        self.logger.debug(f"{output =}")

        # add the summary as a message also, to keep the message chain going
        messages = state.messages
        messages.append(AIMessage(content=output.summary))
        return {"messages": messages, "text_response_eval": output}

    def _route_answer_evaluator_node(self, state: AgentState) -> str:
        """Route depending on evaluation of answer"""
        # next_step: Literal["continue", "retrieve_more_info", "modify_query", "ask_user", "undefined"] = Field(default="undefined")
        text_response_eval = state.text_response_eval.next_step

        if text_response_eval == "continue":
            self.logger.debug(state.messages[-1].pretty_repr())
            self.k = self.default_k
            return "continue"
        elif text_response_eval == "retrieve_more_info":
            self.k += 1
            return "retrieve_more_info"
        elif text_response_eval == "modify_query":
            return "modify_query"
        else:
            return "undefined"

    def _route_query_node(self, state: AgentState) -> str:
        """Route the query depending on LLM's result"""
        self.logger.debug(f"{state =}")
        query_type = state.query_type.query_type

        return query_type

    def _give_neuroml_answer_to_user_node(self, state: AgentState) -> dict:
        """Return the answer message to the user"""
        messages = state.messages
        answer = messages[-2]

        self.logger.info(f"Returning final answer to user: {answer}")

        return {"user_message": answer.content}

    def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("classify_query", self._classify_query_node)
        self.workflow.add_node(
            "answer_neuroml_question", self._answer_neuroml_question_node
        )
        self.workflow.add_node(
            "answer_general_question", self._answer_general_question_node
        )
        self.workflow.add_node("retrieve_docs", ToolNode([self._retrieve_docs]))
        self.workflow.add_node(
            "generate_neuroml_code", self._generate_neuroml_code_node
        )
        self.workflow.add_node(
            "generate_answer_from_context", self._generate_answer_from_context_node
        )
        self.workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        self.workflow.add_node(
            "give_neuroml_answer_to_user", self._give_neuroml_answer_to_user_node
        )

        self.workflow.add_edge(START, "classify_query")

        self.workflow.add_conditional_edges(
            "classify_query",
            self._route_query_node,
            {
                "general question": "answer_general_question",
                "neuroml question": "answer_neuroml_question",
                "neuroml code generation": "generate_neuroml_code",
                "unknown": END,
            },
        )

        self.workflow.add_conditional_edges(
            "answer_neuroml_question",
            tools_condition,
            {
                "tools": "retrieve_docs",
                END: END,
            },
        )

        self.workflow.add_conditional_edges(
            "evaluate_answer",
            self._route_answer_evaluator_node,
            {
                "continue": "give_neuroml_answer_to_user",
                "retrieve_more_info": "give_neuroml_answer_to_user",
                # "retrieve_more_info": "answer_neuroml_question",
                "unknown": "give_neuroml_answer_to_user",
            },
        )

        self.workflow.add_edge("retrieve_docs", "generate_answer_from_context")
        self.workflow.add_edge("generate_answer_from_context", "evaluate_answer")
        self.workflow.add_edge("give_neuroml_answer_to_user", END)
        self.workflow.add_edge("answer_general_question", END)
        self.workflow.add_edge("generate_neuroml_code", END)

        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        self.graph.get_graph().draw_mermaid_png(
            output_file_path="nml-ai-lang-graph.png"
        )

    def __check_ollama_model(self, model):
        """Check if ollama model is available

        :param model: ollama model name
        :type model: str
        :returns: None

        :throws ollama.ResponseError: if `model` is not available
        :throws ConnectionError: if cannot connect to an Ollama server

        """
        try:
            _ = ollama.show(model)
        except ollama.ResponseError:
            self.logger.error(f"Could not find ollama model: {model}")
            self.logger.error("Please ensure you have pulled the model")
            sys.exit(-1)
        except ConnectionError:
            self.logger.error("Could not connect to Ollama.")
            sys.exit(-1)

    def _load_vector_stores(self):
        """Create/load the vector store"""
        self.logger.debug("Setting up/loading Chroma vector store")

        chroma_client_settings_text = chromadb.config.Settings(
            is_persistent=True,
            persist_directory=f"{self.data_dir}/neuroml_docs_text_{self.embedding_model.replace(':', '_')}.db",
            anonymized_telemetry=False,
        )
        self.text_vector_store = Chroma(
            collection_name="nml_docs",
            embedding_function=self.embeddings,
            client_settings=chroma_client_settings_text,
        )

        info_files = glob(f"{self.data_files_path}/*", recursive=True)
        self.logger.debug(f"Loaded {len(info_files)} files from {self.data_files_path}")

        for info_file in info_files:
            file_type = mimetypes.guess_file_type(info_file)[0]

            if "markdown" in file_type:
                self._add_md_file_to_store(info_file)
            else:
                self.logger.warning(
                    f"File {info_file} is of type {file_type} which is not currently supported. Skipping"
                )

    def _add_md_file_to_store(self, file):
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
        already_added = self.text_vector_store.get(where={"file_hash": file_hash})

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
                chunk_size=1000, chunk_overlap=200
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
            self.index = self.text_vector_store.add_documents(documents=splits)

    def _retrieve_docs(self, query: str):
        """Retrieve embeddings from documentation to answer a query

        :param query: user query
        :returns: serialised metadata/page content and vector_store look up result

        """
        assert self.text_vector_store

        res = self.text_vector_store.similarity_search(query, k=self.k)

        serialized = "\n\n".join(
            (f"Source: {r.metadata}\nContent:{r.page_content}") for r in res
        )
        self.logger.debug(res)
        return serialized, res

    def test_retrieval(self):
        """Test the retrieval system"""
        self._load_vector_stores()
        self._retrieve_docs("NeuroML community")

    def run_graph_invoke(self, query: str, thread_id: str = "default_thread"):
        """Run the graph"""

        config = {"configurable": {"thread_id": thread_id}}
        final_state = self.graph.invoke({"query": query}, config=config)
        self.logger.debug(final_state)
        if message := final_state.get("user_message", None):
            return message
        else:
            return "I was unable to answer"

    def run_graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.graph.stream({"query": query}, config=config):
            for node, state in chunk.items():
                self.logger.debug(f"{node}: {repr(state)}")
                # all nodes must return dicts
                if message := state.get("user_message", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    def graph_stream(self, query: str, thread_id: str = "default_threaD"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        return self.graph.stream({"query": query}, config=config)


if __name__ == "__main__":
    nml_ai = NML_RAG(
        chat_model="ollama:qwen3:1.7b",
        embedding_model="ollama:bge-m3",
        logging_level=logging.DEBUG,
    )
    nml_ai.setup()
    # nml_ai.test_retrieval()
    nml_ai.run_graph_invoke(
        "Give me a summary of the NeuroML project's primary goals and also detail the exact steps required to install the core Python library"
    )
