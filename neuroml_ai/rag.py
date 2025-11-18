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
from typing import Optional, Any

import chromadb
import ollama
from langchain.chat_models import init_chat_model
from langchain.embeddings import init_embeddings
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import (MarkdownHeaderTextSplitter,
                                      RecursiveCharacterTextSplitter)
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from .schemas import AgentState, EvaluateAnswerSchema, QueryTypeSchema

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
        embedding_model: str = "ollama:bge-m3",
        logging_level: int = logging.DEBUG,
    ):
        """Initialise"""
        self.chat_model = chat_model
        self.model = None
        self.embedding_model = embedding_model
        self.embeddings = None

        self.text_vector_stores: dict[str, Any] = {}
        self.image_vector_stores: dict[str, Any] = {}

        # per vector store
        self.default_k = 2
        self.k_max = 5
        self.k = self.default_k

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        # we prefer markdown because the one page PDF that is available for the
        # documentation does not work too well with embeddings
        my_path = Path(__file__).parent
        self.data_dir = f"{my_path}/data/"
        self.vector_stores_path = f"{self.data_dir}/vector-stores"
        self.vector_stores_sources_path = f"{self.vector_stores_path}/sources"

        self.logger = logging.getLogger("NeuroML-AI")
        self.logger.setLevel(logging_level)
        self.logger.propagate = False

        formatter_info = logging.Formatter(
            "%(name)s (%(levelname)s) >>> %(message)s\n\n"
        )
        formatter_other = logging.Formatter(
            "%(name)s (%(levelname)s) in '%(funcName)s' >>> %(message)s\n\n"
        )

        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setLevel(logging.INFO)
        stdout_handler.addFilter(LoggerInfoFilter())
        stdout_handler.setFormatter(formatter_info)
        self.logger.addHandler(stdout_handler)

        stderr_handler = logging.StreamHandler(sys.stderr)
        stderr_handler.setLevel(logging_level)
        stderr_handler.addFilter(LoggerNotInfoFilter())
        stderr_handler.setFormatter(formatter_other)
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

    def _summarise_history_node(self, state: AgentState) -> dict:
        """Clean ups after every round of conversation"""
        assert self.model

        conversation, human_messages, ai_messages = self._get_recent_conversation(
            state.messages, state.summarised_till, None
        )
        conversations_num = len(human_messages) + len(ai_messages)

        if conversations_num < self.num_recent_messages:
            self.logger.debug(
                f"Not enough conversations to summarise yet: {conversations_num}/{self.num_recent_messages}"
            )
            return {}

        # Summarise history
        system_prompt = dedent("""You are a memory/conversation summarisation
        assistant. Your job is to maintain a concise, factual memory of an
        ongoing conversation between a user and an AI assistant. This history
        will help the AI assistant in future conversations with the user.

        Guidelines:

        1. Preserve key facts, user intentions, user requirements, and user
        constraints.
        2. Remove filler, greetings, and irrelevant small talk.
        3. Keep the summary coherent and readable as a standalone record.
        4. Exclude reasoning steps, or internal thought processes. Do not add
        explanations or commentary. Exclude requests to summarise the
        conversation in the summary.
        5. Limit the summary to 5-10 sentences
        unless the conversation is very complex.
        6. Make it self-contained. Clearly note what the user said, and what the assistant's reply was.

        """)

        user_prompt = dedent("""
        Please create a summary of the conversation between the user and the AI
        assistant.

        ------

        Here is the current summary of the conversation so far:

        {old_summary}

        ------

        Here are the exchanges between the user and the assistant since the
        last summarisation:

        {conversation}

        """)

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", user_prompt)]
        )

        self.logger.debug(f"{conversation =}")

        prompt = prompt_template.invoke(
            {
                "old_summary": state.context_summary,
                "conversation": conversation,
            }
        )

        self.logger.debug(f"{prompt =}")

        output = self.model.invoke(prompt)
        self.logger.debug(f"Current history summary is:\n{output.content}")

        # Do not update messages here, since we don't want this to be noted as
        # an AI response to a user query
        return {
            "context_summary": output.content,
            "summarised_till": len(state.messages),
        }

    def _add_memory_to_prompt(self, state: AgentState) -> str:
        """Add memory to system prompt.

        Adds the context summary and recent conversation

        :param state: agent state
        :returns: "memory" string to add to the system prompt

        """
        ret_string = ""

        directive = dedent(
            """
            IMPORTANT:

            - Consider both the latest user message AND the conversation history.
            - If the latest query is contextually about NeuroML due to prior discussion, treat it as a NeuroML related query even if the word does not appear.

            """
        )

        if len(state.context_summary):
            ret_string += dedent(
                f"""
            -----

            Here is a concise summary of the previous conversation to maintain
            continuity:

            {state.context_summary}

            -----
            """
            )

        conversation, _, _ = self._get_recent_conversation(
            state.messages, (-1 * self.num_recent_messages), None
        )
        if len(conversation):
            ret_string += dedent(
                f"""
            -----

            Here are the recent messages between the user and the assistant:

            {conversation}

            -----
            """
            )

        if len(ret_string):
            ret_string += directive

        return ret_string

    def _get_recent_conversation(
        self, all_messages, start: int = 0, stop: Optional[int] = None
    ) -> tuple[str, list[HumanMessage], list[AIMessage]]:
        """Get recent converstations between start and stop indices

        :param all_messages: all the messages
        :param start: start index
        :param stop: stop index
        :returns: (conversation, list of human messages, list of ai messages)

        """
        conv_messages = list(
            filter(
                lambda x: isinstance(x, (HumanMessage, AIMessage)),
                all_messages[start:stop],
            )
        )
        human_messages = []
        ai_messages = []
        conversation = ""
        for msg in conv_messages:
            if isinstance(msg, HumanMessage):
                conversation += f"{msg.pretty_repr()}"
                human_messages.append(msg)
            else:
                conversation += f": {msg.pretty_repr()}"
                ai_messages.append(msg)

        return (
            conversation.replace("{", "{{").replace("}", "}}"),
            human_messages,
            ai_messages,
        )

    def _init_rag_state_node(self, state: AgentState) -> dict:
        """Initialise, reset state before next iteration"""
        return {
            "query_type": QueryTypeSchema(),
            "text_response_eval": EvaluateAnswerSchema(),
            "message_for_user": "",
        }

    def _classify_query_node(self, state: AgentState) -> dict:
        """LLM decides what type the user query is"""
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
            You are an expert query classifier. Analyse the user's request and
            determine its intent.

            # Classification rules:

            Reason about the user's request. Go step by step. Take past
            conversation history and context into account to identify the
            objective of the query.

            - If the query is a NeuroML code generation request, respond 'neuroml_code_generation'
            - If the query mentions NeuroML at all in its text, respond 'neuroml_question'.
            - If the query is asking for information related to NeuroML, respond 'neuroml_question'

            - If the query is unrelated to NeuroML, only then respond "general_question".
            - If it is general conversation unrelated to NeuroML, respond "undefined".

            Examples:
            - "How do I get learn NeuroML?": {{"query_type": "neuroml_question"}}
            - "How do I get started with NeuroML?": {{"query_type": "neuroml_question"}}
            - "How do I define ion channels in NeuroML?": {{"query_type": "neuroml_question"}}
            - "Generate NeuroML code for a neuron": {{"query_type": "neuroml_code_generation"}}
            - "What is the capital of France?": {{"query_type": "general_question"}}
            - "What are we talking about?": {{"query_type": "general_question"}}

            Provide your answer as a JSON object matching the requested schema.
            Do not add any explanation or text.
            """)

        system_prompt += self._add_memory_to_prompt(state)

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.model.with_structured_output(QueryTypeSchema)
        prompt = prompt_template.invoke({"query": state.query})

        self.logger.debug(f"{prompt = }")

        output = query_node_llm.invoke(prompt)

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        return {
            "query_type": QueryTypeSchema(query_type=output.query_type),
            "messages": messages,
        }

    def _generate_neuroml_code_node(self, state: AgentState) -> dict:
        """Generate code"""
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(AIMessage("I can generate code for you"))

        return {"messages": messages}

    def _answer_general_question_node(self, state: AgentState) -> dict:
        """Answer a general question"""
        assert self.model
        self.logger.debug(f"{state =}")

        if state.query_type == "general_question":
            system_prompt = dedent("""
            You are an AI assistant. Answer questions to the best of your knowledge.
            This is a general query, not related to NeuroML.

            ## Core directives

            - Do not assume this question is related to NeuroML or other technical domains.
            - Only provide information you are confident about. If you are unsuare, clearly say so.
            - Avoid inventing facts. If a fact is not known or uncertain, respond with "I was unable to find factual information about this query".
            - Keep answers clear, concise, formal, and user-friendly.

            Examples:
            User: Thank you.
            Assistant: You are welcome.
            User: I like cats.
            Assistant: That's great, I like cats too. I also like dogs.

            """)
        else:
            system_prompt = dedent(
                """
            You are a warm, easy-going conversational assistant.
            Engage with the user even if they are simply talking rather than asking questions.
            Reflect their tone, acknowledge what they say, and continue the conversation naturally.

            """
            )

        system_prompt += self._add_memory_to_prompt(state)

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

        return {"messages": messages, "message_for_user": output.content}

    def _generate_clean_query_node(self, state: AgentState) -> dict:
        """Answer a NeuroML question"""
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        Rewrite the user's query into a concise retrieval query for NeuroML
        documentation. Think about the user's intent step by step.

        - remove stop words
        - include relevant NeuroML concepts
        - convert natural language questions into keyword like terms
        - 3-8 words
        - no senteces
        - no explanations
        - use the past conversation history to understand the context of the
          user's query

        Only return the rewritten query
        """)

        system_prompt += self._add_memory_to_prompt(state)

        question_prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )
        prompt = question_prompt_template.invoke({"query": state.query})
        self.logger.debug(f"{prompt =}")

        output = self.model.invoke(prompt)

        self.logger.debug(f"{output =}")

        messages = state.messages
        messages.append(output)

        return {"messages": messages}

    def _generate_answer_from_context_node(self, state: AgentState) -> dict:
        """Generate the answer"""
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        You are a NeuroML expert and experienced modeller in computational
        neuroscience. You specialise in NeuroML, LEMS, and data-driven
        modelling of detailed neurons, neuronal circuits and its
        components---ion channels, active and passive conductances, detailed
        cells with morphologies, synapse models, networks including these
        components. Your goal is to provide clear, accurate guidance to users
        based strictly on the information available in the retrieved context.

        # Core Directives:

        - Limit yourself to facts from the provided context only, avoid using
          knowledge from your general training.
        - Use concise, formal language appropriate for neurosience and
          computational modelling.
        - Write the answer as a self contained explanation that does not assume
          access to the context.
        - Do not mention "context", "reference material", "documents" or
          "retrieval".
        - Do not refer to documents indirectly (e.g., "as described above",
          "follow the tutorial"). Instead, restate the necessary information
          directly to the user in clear natural language.

        # Context (reference material not visible to the user):

        {reference_material}

        """)

        system_prompt += self._add_memory_to_prompt(state)

        generate_answer_template = ChatPromptTemplate(
            [
                ("system", system_prompt),
                (
                    "human",
                    "Question: {question}",
                ),
            ]
        )
        question = state.query

        self.logger.debug(f"retrieval query: {state.messages[-1].content}")

        res = self._retrieve_docs(state.messages[-1].content)
        serialized = ""

        for rs in res:
            for r in rs:
                metadata = [f"{key}: {val}" for key, val in r.metadata.items() if "header" in key.lower()]
                metadata_str = "Document: " + " | ".join(metadata)
                serialized += "\n\n" + f"{metadata_str}\n\n:{r.page_content}"
        self.logger.debug(f"{serialized =}")

        reference_material = serialized
        prompt = generate_answer_template.invoke(
            {
                "question": question,
                "reference_material": reference_material.replace("{", "{{").replace(
                    "}", "}}"
                ),
            }
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
              "next_step": "continue", "retrieve_more_info", "modify_query", "undefined"// next actions
              "summary": "Brief natural-language justification for the grades"
            }}

            Guidelines for 'next_step':
            - Set to 'continue' if the answer is clear and should be passed to the user
            - Set to 'retrieve_more_info' if the answer is incomplete but grounded and needs more context
            - Set to 'modify_query' if the answer is ungrounded or irrelevant and the query needs to be reformulated to improve retrieval precision
            - Set to 'undefined' if the query cannot be answered from the corpus and we need to ask the user for clarification or additional information
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
            {
                "question": question,
                "context": context.replace("{", "{{").replace("}", "}}"),
                "answer": answer,
            }
        )

        output = query_node_llm.invoke(prompt)
        self.logger.debug(f"{output =}")

        # do not store the evaluation message in state
        return {"text_response_eval": output, "messages": state.messages}

    def _route_answer_evaluator_node(self, state: AgentState) -> str:
        """Route depending on evaluation of answer"""
        text_response_eval = state.text_response_eval.next_step

        if text_response_eval == "continue":
            self.logger.debug(state.messages[-1].pretty_repr())
            self.k = self.default_k
            return "continue"
        elif text_response_eval == "retrieve_more_info":
            # limit what max k we can have, otherwise, we end up pulling the
            # whole store..
            if self.k < self.k_max:
                self.k += 1
                return "retrieve_more_info"
            else:
                # we are already at max context, so we need to modify the query
                # to get a better result
                return "modify_query"
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
        self.logger.debug(f"{state =}")

        messages = state.messages
        answer = messages[-1]

        self.logger.info(f"Returning final answer to user: {answer}")

        return {"message_for_user": answer.content}

    def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("init_rag_state", self._init_rag_state_node)
        self.workflow.add_node("classify_query", self._classify_query_node)

        self.workflow.add_node("generate_clean_query", self._generate_clean_query_node)
        self.workflow.add_node(
            "answer_general_question", self._answer_general_question_node
        )
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
        self.workflow.add_node("summarise_history", self._summarise_history_node)

        self.workflow.add_edge(START, "init_rag_state")
        self.workflow.add_edge("init_rag_state", "classify_query")

        self.workflow.add_conditional_edges(
            "classify_query",
            self._route_query_node,
            {
                "general_question": "answer_general_question",
                "neuroml_question": "generate_clean_query",
                "neuroml_code_generation": "generate_neuroml_code",
                "undefined": "answer_general_question",
            },
        )

        self.workflow.add_conditional_edges(
            "evaluate_answer",
            self._route_answer_evaluator_node,
            {
                "continue": "give_neuroml_answer_to_user",
                "retrieve_more_info": "give_neuroml_answer_to_user",
                # "retrieve_more_info": "generate_clean_query",
                "undefined": "give_neuroml_answer_to_user",
            },
        )

        self.workflow.add_edge("generate_clean_query", "generate_answer_from_context")
        self.workflow.add_edge("generate_answer_from_context", "evaluate_answer")
        self.workflow.add_edge("give_neuroml_answer_to_user", "summarise_history")
        self.workflow.add_edge("answer_general_question", "summarise_history")
        self.workflow.add_edge("generate_neuroml_code", "summarise_history")
        self.workflow.add_edge("summarise_history", END)

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

        self.logger.debug(f"{self.vector_stores_sources_path =}")
        vec_store_sources = glob(f"{self.vector_stores_sources_path}/*", recursive=False)
        self.logger.debug(f"{vec_store_sources =}")

        assert len(vec_store_sources)

        for src in vec_store_sources:
            self.logger.debug(f"Setting up vector store: {src}")
            src_path = Path(src)

            assert src_path.is_dir()

            vs_persist_dir = f"{self.vector_stores_path}/{src_path.name}_{self.embedding_model.replace(':', '_')}.db"
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
                file_type = mimetypes.guess_file_type(info_file)[0]

                if "markdown" in file_type:
                    self._add_md_file_to_store(store, info_file)
                else:
                    self.logger.warning(
                        f"File {info_file} is of type {file_type} which is not currently supported. Skipping"
                    )

    def _add_md_file_to_store(self, store, file):
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
            _ = store.add_documents(documents=splits)

    def _retrieve_docs(self, query: str) -> list:
        """Retrieve embeddings from documentation to answer a query

        :param query: user query
        :returns: serialised metadata/page content and vector_store look up result

        """
        self._load_vector_stores()

        assert self.text_vector_stores

        res = []

        for sname, store in self.text_vector_stores.items():
            res.append(store.similarity_search(query, k=self.k))

        return res

    def run_graph_invoke_state(self, state: dict, thread_id: str = "default_thread"):
        """Run the graph but accept and return states"""

        config = {"configurable": {"thread_id": thread_id}}

        if "query" not in state:
            self.logger.error(f"Provided state should include the key 'query': {state}")
            sys.exit(-1)

        final_state = self.graph.invoke(state, config=config)
        self.logger.debug(final_state)
        return final_state

    def run_graph_invoke(self, query: str, thread_id: str = "default_thread"):
        """Run the graph by using and returning string input"""

        config = {"configurable": {"thread_id": thread_id}}

        final_state = self.graph.invoke({"query": query}, config=config)
        self.logger.debug(f"{final_state =}")
        if message := final_state.get("message_for_user", None):
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
                if message := state.get("message_for_user", None):
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
