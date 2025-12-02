#!/usr/bin/env python3
"""
NeuroML RAG implementation

File: rag.py

Copyright 2025 Ankur Sinha
Author: Ankur Sinha <sanjay DOT ankur AT gmail DOT com>
"""

import logging
import sys
from textwrap import dedent
from typing import Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from typing_extensions import List, Tuple, Dict

from .schemas import AgentState, EvaluateAnswerSchema, QueryTypeSchema, ToolCallSchema
from .stores import NML_Stores
from .utils import (
    LoggerInfoFilter,
    LoggerNotInfoFilter,
    check_ollama_model,
    logger_formatter_info,
    logger_formatter_other,
    parse_output_with_thought,
    split_thought_and_output,
    check_model_works,
)

from fastmcp import Client

logging.basicConfig()
logging.root.setLevel(logging.WARNING)


class NML_RAG(object):
    """NeuroML RAG implementation"""

    checkpointer = InMemorySaver()

    def __init__(
        self,
        mcp_client: Client,
        chat_model: str = "ollama:qwen3:1.7b",
        embedding_model: str = "ollama:bge-m3",
        logging_level: int = logging.DEBUG,
    ):
        """Initialise"""
        self.chat_model = chat_model
        self.model = None
        self.stores = NML_Stores(embedding_model)
        # total number of reference documents
        self.num_refs_max = 10

        # toggle for answer generator
        self.modify_query = False

        # number of conversations after which to summarise
        # no need to summarise after each
        # 5 rounds: 10 messages
        self.num_recent_messages = 10

        self.mcp_client: Client = mcp_client
        self.mcp_tools = None
        self.tool_description = ""

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

    async def setup(self):
        """Set up basics."""

        self._setup_chat_model()
        self.stores.setup()

        async with self.mcp_client:
            self.mcp_tools = await self.mcp_client.list_tools()
        # Persists because it's only holding the return value
        # To make the call, though, we will need the context again
        self.logger.debug(f"{self.mcp_tools =}")

        # Generate tool description
        # Note: we remove the param/type information from the description and
        # use the type hints directly instead
        ctr = 0
        for t in self.mcp_tools:
            ctr += 1
            self.tool_description += (
                f"## {ctr}. {t.name}\n{t.description.split(':param')[0].strip()}"
            )
            args = t.inputSchema.get("properties", [])
            if len(args):
                self.tool_description += " Inputs:\n"
                for arg, arginfo in args.items():
                    self.tool_description += f"- {arg}: {arginfo.get('type')}"
                self.tool_description += "\n"

        self.logger.debug(f"{self.tool_description =}")
        await self._create_graph()

    def _setup_chat_model(self):
        """Set up a chat model"""
        if self.chat_model.lower().startswith("huggingface:"):
            from langchain_huggingface import HuggingFaceEndpoint
            # from huggingface_hub import login
            # login()

            _, model_name, provider = self.chat_model.split(":")
            self.logger.debug(f"Using huggingface model: {model_name}")

            llm = HuggingFaceEndpoint(
                repo_id=f"{model_name}",
                provider="auto",
                task="text-generation",
                max_new_tokens=512,
                do_sample=False,
                repetition_penalty=1.03,
            )

            self.model = init_chat_model(
                model_name,
                model_provider="huggingface",
                llm=llm,
                configurable_fields=("temperature"),
            )
        else:
            if self.chat_model.lower().startswith("ollama:"):
                check_ollama_model(
                    self.logger, self.chat_model.lower().replace("ollama:", "")
                )

            self.model = init_chat_model(
                self.chat_model, configurable_fields=("temperature")
            )

        assert self.model

        state, msg = check_model_works(self.model)
        assert state

        self.logger.info(f"Using chat model: {self.chat_model}")

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

        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        self.logger.debug(f"Current history summary is:\n{output.content}")

        thought, answer = split_thought_and_output(output)

        # Do not update messages here, since we don't want this to be noted as
        # an AI response to a user query
        return {
            "context_summary": answer,
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

        if len(state.code):
            ret_string += dedent(
                f"""
            -----

            This is the generated code so far:

            {state.code}

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
            "reference_material": {},
        }

    def _classify_query_node(self, state: AgentState) -> dict:
        """LLM decides what type the user query is"""
        assert self.model
        self.logger.debug(f"{state =}")

        messages = state.messages
        messages.append(HumanMessage(content=state.query))

        system_prompt = dedent("""
            You are an expert query classifier. Analyse the user's request and
            determine its intent.

            Provide your answer ONLY as a JSON object matching the requested
            schema. Do not give any explanations or note your thinking.

            # Classification rules:

            Reason about the user's request. Go step by step. Take past
            conversation history and context into account to identify the
            objective of the query.

            - If the query is a NeuroML code generation request, respond 'neuroml_code_generation'
            - If the query mentions NeuroML at all in its text, always respond 'neuroml_question'.
            - If the query is asking for information related to NeuroML, always respond 'neuroml_question'
            """)

        # only if neuroml is not mentioned, do we even bother with non neuroml
        # classifications
        if "neuroml" not in state.query.lower():
            system_prompt += dedent("""
                - If the query is unrelated to NeuroML, only then respond "general_question".
                """)

        system_prompt += dedent("""
                Examples:
                - "How do I get learn NeuroML?": {{"query_type": "neuroml_question"}}
                - "How do I get started with NeuroML?": {{"query_type": "neuroml_question"}}
                - "How do I define ion channels in NeuroML?": {{"query_type": "neuroml_question"}}
                - "Generate NeuroML code for a neuron": {{"query_type": "neuroml_code_generation"}}
                """)

        if "neuroml" not in state.query.lower():
            system_prompt += dedent("""
                - "What is the capital of France?": {{"query_type": "general_question"}}
                - "What are we talking about?": {{"query_type": "general_question"}}
                """)

        system_prompt += self._add_memory_to_prompt(state)

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.model.with_structured_output(
            QueryTypeSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke({"query": state.query})

        self.logger.debug(f"{prompt = }")

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        if output["parsing_error"]:
            query_type_result = parse_output_with_thought(
                output["raw"], QueryTypeSchema
            )
        else:
            query_type_result = output["parsed"]

        self.logger.debug(f"{query_type_result =}")
        return {
            "query_type": QueryTypeSchema(query_type=query_type_result["query_type"]),
            "messages": messages,
        }

    def _neuroml_code_tool_decider_node(self, state: AgentState) -> dict:
        """Generate code"""
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        You are a NeuroML code generation assistant. You are proficient in
        Python, and in developing biophysically detailed computational models
        of the brain. Think step by step about how to generate the code
        requested by the user.

        You have access to these tools:

        # Tools

        {tool_description}


        # Output format:

        {{
            action: "tool_call" if a tool is to be called, "update_code" if the
            code needs to be updated, "final_answer" if the code is ready to be
            returned to the user.
            tool: name of the tool to call
            args: arguments to be given to the tool
            reason: a short concise text string explaining your decision
        }}

        """)

        system_prompt += self._add_memory_to_prompt(state)
        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        query_node_llm = self.model.with_structured_output(
            ToolCallSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {"query": state.query, "tool_description": self.tool_description}
        )

        self.logger.debug(f"{prompt = }")

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )

        if output["parsing_error"]:
            result = parse_output_with_thought(output["raw"], ToolCallSchema)
        else:
            result = output["parsed"]

        return {
            "tool_call": ToolCallSchema(
                action=result.action,
                tool=result.tool,
                args=result.args,
                reason=result.reason,
            ),
        }

    async def _neuroml_code_tools_node(self, state: AgentState) -> dict:
        """Node that does the tool calling"""
        tool = state.tool_call.tool
        tool_args = state.tool_call.args

        async with self.mcp_client:
            result = await self.mcp_client.call_tool(tool, tool_args)

        tool_call = state.tool_call
        tool_call.output = result

        return {"tool_call": tool_call}

    def _neuroml_code_tool_router(self, state: AgentState) -> str:
        """Tool router"""
        tool_call = state.tool_call
        return tool_call.action

    def _neuroml_code_generator_node(self, state: AgentState) -> dict:
        """Code generator node"""
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        You are a NeuroML code generation assistant. You are proficient in
        Python, and in developing biophysically detailed computational models
        of the brain.

        Use the existing code and the output from the tool call to generate
        code to answer the user's query.

        # Existing code:

        {current_code}


        # Tool output:

        {tool_output}

        """)

        system_prompt += self._add_memory_to_prompt(state)

        prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )

        # can use | to merge these lines
        prompt = prompt_template.invoke(
            {
                "query": state.query,
                "current_code": state.code,
                "tool_output": state.tool_call.output,
            }
        )

        self.logger.debug(f"{prompt = }")
        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )

        thought, answer = split_thought_and_output(output)

        return {
            "code": answer,
        }

    def _give_neuroml_code_to_user_node(self, state: AgentState) -> dict:
        """Return the answer message to the user"""
        self.logger.debug(f"{state =}")
        self.logger.info(f"Returning code to user: {state.code}")

        return {"message_for_user": state.code}

    def _answer_general_question_node(self, state: AgentState) -> dict:
        """Answer a general question"""
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        You are a warm, easy-going conversational assistant.
        The user has asked something unrelated to NeuroML.
        Engage with the user and answer questions to the best of your ability.
        Reflect their tone, acknowledge what they say, and continue the conversation naturally.

        ## Core directives

        - Do not assume this question is related to NeuroML or other technical domains.
        - Only provide information you are confident about. If you are unsuare, clearly say so.
        - Avoid inventing facts. If a fact is not known or uncertain, respond with "I was unable to find factual information about this query".
        - Keep answers clear, concise, and user-friendly.
        - Respond in a formal, academic style.
        - Note that you can only provide certain information or carry out certain tasks if the user is asking about NeuroML, or about generating NeuroML code.

        Examples:
        User: Thank you.
        Assistant: You are welcome.
        User: I like cats.
        Assistant: That's great, I like cats too. I also like dogs.

        """)

        system_prompt += self._add_memory_to_prompt(state)

        question_prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )
        self.logger.debug(f"{question_prompt_template =}")
        prompt = question_prompt_template.invoke({"query": state.query})

        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        # self.logger.debug(f"{output =}")
        self.logger.debug(output)
        thought, answer = split_thought_and_output(output)

        messages = state.messages
        output.content = answer
        messages.append(output)

        return {"messages": messages, "message_for_user": output.content}

    def _generate_retrieval_query_node(self, state: AgentState) -> dict:
        """Generate a retrieval query"""
        assert self.model
        self.logger.debug(f"{state =}")

        system_prompt = dedent("""
        Generate a concise retrieval query from the user's question.  Think
        about the user's intent step by step.

        Directives:
        - a concept is a single technical entity or noun phrase
        - extract all concepts from the query
        - split multiple concepts that are joined by 'and', commas and other
          conjunctions into separate, individual concepts
        - generate a query for EXACTLY one concept

        For the rewritten query:
        - only include content words (nouns, verbs, adjectives)
        - do NOT include stop words: a, an, the, in, of, for, on, at, and
        - limit yourself to 3-8 words
        - no sentences
        - no explanations
        - ignore sentency fluency, only use keywords

        Only return the rewritten query.
        """)

        system_prompt += self._add_memory_to_prompt(state)

        if self.modify_query:
            # toggle off
            system_prompt += dedent("""
            Generate a new query on EXACTLY one of concepts that the
            evaluator's feedback says is missing.

            Evaluator feedback:
            {feedback}
            """)

        question_prompt_template = ChatPromptTemplate(
            [("system", system_prompt), ("human", "User query: {query}")]
        )
        prompt = question_prompt_template.invoke(
            {"query": state.query, "feedback": state.text_response_eval.summary}
        )
        self.logger.debug(f"{prompt =}")

        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )

        self.logger.debug(f"{output =}")
        thought, answer = split_thought_and_output(output)

        messages = state.messages
        output.content = answer
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
        components: ion channels, active and passive conductances, detailed
        cells with morphologies, synapse models, networks including these
        components. Your goal is to provide clear, accurate guidance to users
        based strictly on the information provided in the retrieved context.

        If the context is missing some information to answer the question,
        say so.

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
        - Do not include your thinking in your response.

        # Context (reference material not visible to the user, ordered from most relevant to least relevant):

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

        # current reference material
        reference_material = state.reference_material

        cleaned_query = state.messages[-1].content

        # new references, or more references for an existing query from all
        # stores
        res = self.stores.retrieve(cleaned_query)
        # rank info from all stores, keep top N
        # remember that when asking for more ks from the vector store, they'll
        # still return the initial ones, so we don't need to do any manual
        # merging here for more refs for a particular query
        sorted_res = sorted(res, key=lambda tup: tup[1], reverse=True)
        new_ref = {cleaned_query: sorted_res[: self.num_refs_max]}

        reference_material.update(new_ref)
        self.logger.debug(f"{reference_material =}")

        reference_material_text = self._serialize_reference(reference_material)
        prompt = generate_answer_template.invoke(
            {"question": question, "reference_material": reference_material_text}
        )
        self.logger.debug(f"{prompt =}")
        output = self.model.invoke(
            prompt, config={"configurable": {"temperature": 0.3}}
        )
        thought, answer = split_thought_and_output(output)
        output.content = answer
        self.logger.debug(output.pretty_repr())

        messages = state.messages
        messages.append(output)

        return {"messages": messages, "reference_material": reference_material}

    def _serialize_reference(self, reference_material: Dict[str, List[Tuple]]) -> str:
        """serialize references into text for use in context

        We also sort the documents based on their relevance scores.

        :param reference_material:  Dict {cleaned query: list of tuples (doc, relevance score)}
        :returns: str representation of reference

        """

        serialized = ""
        for q, sorted_refs in reference_material.items():
            ctr = 1
            serialized += f"## {q}\n"
            for r, score in sorted_refs:
                metadata = [
                    f"{key}: {val}"
                    for key, val in r.metadata.items()
                    if "header" in key.lower()
                ]
                metadata_str = f"### Document {ctr}/{len(sorted_refs)}: " + " | ".join(
                    metadata
                )
                serialized += "\n\n" + f"{metadata_str}\n\n:{r.page_content}"
                ctr += 1

        reference_material_text = serialized.replace("{", "{{").replace("}", "}}")
        self.logger.debug(f"{reference_material_text =}")

        return reference_material_text

    def _evaluate_answer_node(self, state: AgentState) -> dict:
        """Evaluate the answer"""
        assert self.model

        evaluator_prompt = dedent("""
            You are a critical grader evaluating an answer produced by a retrieval-augmented generation (RAG) system.

            You are given:
            1. The user's question.
            2. The retrieved context used to generate the answer.
            3. The system's answer.

            Your job:
            - Judge the answer strictly based on the context.
            - DO NOT use external knowledge.
            - ALWAYS provide your answer as a JSON object matching the provided
              schema, with these values:

            {{
              "confidence": 0-1,        // How sufficent the context is to answer the question
              "coverage": 0-1,          // How well the context includes information about all parts/sub-questions in the query
              "relevance": 0-1,         // How well the answer addresses the question
              "groundedness": 0-1,      // How well it sticks to the provided context
              "coherence": 0-1,         // Logical flow and clarity
              "conciseness": 0-1,       // Avoids fluff or repetition
              "next_step": "continue", "retrieve_more_info", "modify_query", "rewrite_answer", "undefined"
              "summary": "A brief natural-language justification for the grades."
            }}

            Guidance for values:

            "coverage" and "confidence" are based ONLY on the context, NOT on
            the generated answer. Relevance, groundedness, coherence,
            conciseness are related only to the generated answer.

            coverage:
            * 0.8 - 1.0: all sub-questions/topics/concepts present in context
            * 0.4 - 0.7: some covered, some missing
            * 0.0 - 0.3: most sub-questions missing

            confidence:
            * 0.8 - 1.0: clear, explicit, unambiguous context
            * 0.4 - 0.7: usable but incomplete
            * 0.0 - 0.3: vague/insufficient context

            relevance:
            * 0.8 - 1.0: fully addresses question
            * 0.4 - 0.7: partially addresses question
            * 0.0 - 0.3: barely addresses question

            groundedness:
            * 0.8 - 1.0: entirely based on context
            * 0.4 - 0.7: mixed grounded + inferred
            * 0.0 - 0.3: mostly hallucinated

            coherence:
            * 0.8 - 1.0: clear and logically structured
            * 0.4 - 0.7: understandable but uneven
            * 0.0 - 0.3: disorganised

            conciseness:
            * 0.8 - 1.0: mimimal + efficient
            * 0.4 - 0.7: moderately concise
            * 0.0 - 0.3: verbose

            Guidelines for 'next_step':
            1. high coverage, confident, relevant, grounded, with acceptable
            coherence and conciseness: return "continue".
            2. low coverage: return "modify_query".
            3. high coverage,  low confidence: return "retrieve_more_info".
            4. high coverage and confidence, low relevance, groundedness, coherence, or conciseness: return "rewrite_answer"
            5. all coverage, confidence, relevance and groundedness are low: return "undefined".

            Always return a brief summary.
            """)

        question = state.query

        context = self._serialize_reference(state.reference_material)
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

                        Context
                        {context}

                        -----

                        Answer:
                        {answer}

                 """),
                ),
            ]
        )

        # can use | to merge these lines
        query_node_llm = self.model.with_structured_output(
            EvaluateAnswerSchema, method="json_schema", include_raw=True
        )
        prompt = prompt_template.invoke(
            {
                "question": question,
                "context": context,
                "answer": answer,
            }
        )

        output = query_node_llm.invoke(
            prompt, config={"configurable": {"temperature": 0.0}}
        )
        self.logger.debug(f"{output =}")

        if output["parsing_error"]:
            result = parse_output_with_thought(output["raw"], EvaluateAnswerSchema)
        else:
            result = output["parsed"]

        # do not store the evaluation message in state
        return {"text_response_eval": result, "messages": state.messages}

    def _route_answer_evaluator_node(self, state: AgentState) -> str:
        """Route depending on evaluation of answer"""
        self.logger.debug(f"{state =}")
        resp = state.text_response_eval
        next_step = resp.next_step

        if next_step == "continue" and (
            resp.coverage >= 0.5
            and resp.confidence >= 0.5
            and resp.relevance >= 0.5
            and resp.groundedness >= 0.5
            and resp.coherence >= 0.5
            and resp.conciseness >= 0.5
        ):
            self.stores.reset_k()
            return "continue"
        elif next_step == "modify_query" or resp.coverage < 0.3:
            self.modify_query = True
            return "modify_query"
        elif next_step == "retrieve_more_info" or (
            resp.coverage >= 0.5 and resp.confidence < 0.5
        ):
            # limit what max k we can have, otherwise, we end up pulling the
            # whole store..
            if self.stores.inc_k():
                return "retrieve_more_info"
            else:
                # we are already at max context, so we need to modify the query
                # to get a better result
                self.modify_query = True
                return "modify_query"
        elif next_step == "rewrite_answer" or (
            resp.coverage >= 0.5
            and resp.confidence >= 0.5
            and (
                resp.relevance < 0.5
                and resp.groundedness < 0.5
                and resp.coherence < 0.5
                and resp.conciseness < 0.5
            )
        ):
            return "rewrite_answer"
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

    def _ask_user_for_clarification_node(self, state: AgentState) -> dict:
        """Ask the user for clarification or a different question"""
        self.logger.debug(f"{state =}")

        answer = AIMessage(
            "Apologies. I could not answer that question. Can you please ask another one or try to reword it and I will try again?"
        )

        self.logger.info(f"Asking user for clarification: {answer.content}")

        return {"message_for_user": answer.content}

    async def _create_graph(self):
        """Create the LangGraph"""
        self.workflow = StateGraph(AgentState)
        self.workflow.add_node("init_rag_state", self._init_rag_state_node)
        self.workflow.add_node("classify_query", self._classify_query_node)

        self.workflow.add_node(
            "generate_retrieval_query", self._generate_retrieval_query_node
        )
        self.workflow.add_node(
            "answer_general_question", self._answer_general_question_node
        )
        self.workflow.add_node(
            "neuroml_code_tool_decider", self._neuroml_code_tool_decider_node
        )
        self.workflow.add_node("neuroml_code_tools", self._neuroml_code_tools_node)
        self.workflow.add_node(
            "neuroml_code_generator", self._neuroml_code_generator_node
        )
        self.workflow.add_node(
            "give_neuroml_code_to_user", self._give_neuroml_code_to_user_node
        )
        self.workflow.add_node(
            "generate_answer_from_context", self._generate_answer_from_context_node
        )
        self.workflow.add_node("evaluate_answer", self._evaluate_answer_node)
        self.workflow.add_node(
            "give_neuroml_answer_to_user", self._give_neuroml_answer_to_user_node
        )
        self.workflow.add_node(
            "ask_user_for_clarification", self._ask_user_for_clarification_node
        )
        self.workflow.add_node("summarise_history", self._summarise_history_node)

        self.workflow.add_edge(START, "init_rag_state")
        self.workflow.add_edge("init_rag_state", "classify_query")

        self.workflow.add_conditional_edges(
            "classify_query",
            self._route_query_node,
            {
                "general_question": "answer_general_question",
                "neuroml_question": "generate_retrieval_query",
                "neuroml_code_generation": "neuroml_code_tool_decider",
                # for completeness: the classifier should rarely return
                # undefined
                "undefined": "answer_general_question",
            },
        )
        self.workflow.add_conditional_edges(
            "neuroml_code_tool_decider",
            self._neuroml_code_tool_router,
            {
                "tool_call": "neuroml_code_tools",
                "update_code": "neuroml_code_generator",
                "final_answer": "give_neuroml_code_to_user",
            },
        )
        self.workflow.add_edge("neuroml_code_tools", "neuroml_code_generator")
        self.workflow.add_edge("neuroml_code_generator", "neuroml_code_tool_decider")

        self.workflow.add_conditional_edges(
            "evaluate_answer",
            self._route_answer_evaluator_node,
            {
                "continue": "give_neuroml_answer_to_user",
                "retrieve_more_info": "generate_answer_from_context",
                "rewrite_answer": "generate_answer_from_context",
                "modify_query": "generate_retrieval_query",
                "undefined": "ask_user_for_clarification",
            },
        )

        self.workflow.add_edge(
            "generate_retrieval_query", "generate_answer_from_context"
        )
        self.workflow.add_edge("generate_answer_from_context", "evaluate_answer")
        self.workflow.add_edge("give_neuroml_answer_to_user", "summarise_history")
        self.workflow.add_edge("ask_user_for_clarification", "summarise_history")
        self.workflow.add_edge("answer_general_question", "summarise_history")
        self.workflow.add_edge("summarise_history", END)

        self.graph = self.workflow.compile(checkpointer=self.checkpointer)
        self.graph.get_graph().draw_mermaid_png(
            output_file_path="nml-ai-lang-graph.png"
        )

    async def run_graph_invoke_state(
        self, state: dict, thread_id: str = "default_thread"
    ):
        """Run the graph but accept and return states"""

        config = {"configurable": {"thread_id": thread_id}}

        if "query" not in state:
            self.logger.error(f"Provided state should include the key 'query': {state}")
            sys.exit(-1)

        final_state = await self.graph.ainvoke(state, config=config)
        self.logger.debug(final_state)
        return final_state

    async def run_graph_invoke(self, query: str, thread_id: str = "default_thread"):
        """Run the graph by using and returning string input"""

        config = {"configurable": {"thread_id": thread_id}}

        final_state = await self.graph.ainvoke({"query": query}, config=config)

        self.logger.debug(f"{final_state =}")
        if message := final_state.get("message_for_user", None):
            return message
        else:
            return "I was unable to answer"

    async def run_graph_stream(self, query: str, thread_id: str = "default_thread"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        for chunk in self.graph.astream({"query": query}, config=config):
            for node, state in chunk.items():
                self.logger.debug(f"{node}: {repr(state)}")
                # all nodes must return dicts
                if message := state.get("message_for_user", None):
                    self.logger.info(f"User message: {message}")
                    yield message
                else:
                    self.logger.debug(f"Working in node: {node}")

    async def graph_stream(self, query: str, thread_id: str = "default_threaD"):
        """Run the graph but return the stream"""
        config = {"configurable": {"thread_id": thread_id}}

        res = await self.graph.astream({"query": query}, config=config)
        return res
