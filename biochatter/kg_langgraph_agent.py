from abc import ABC
from datetime import datetime
from typing import Callable, Dict, List, Optional
from langchain_core.messages import (
    HumanMessage, 
    BaseMessage, 
    ToolMessage, 
    AIMessage,
)
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.pydantic_v1 import (
    BaseModel,
    Field,
    ValidationError
)
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langsmith import traceable
from langgraph.graph import MessageGraph, END

import json
import os
import neo4j_utils as nu
import logging

logger = logging.getLogger(__name__)

from biochatter.llm_connect import GptConversation

class Reflection(BaseModel):
    improving: str = Field(description="Critique of what to improve.")

SEARCH_QUERIES = "search_queries"
SEARCH_QUERIES_DESCRIPTION = "query for genomicKB graph database"
REVISED_QUERY = "revised_query"
REVISED_QUERY_DESCRIPTION = "Revised query"

class GenerateQuery(BaseModel):
    """Generate the query."""
    answer: str = Field(description="Cypher query according to user's question.")
    reflection: str = Field(description="Your reflection on the initial answer, critique of what to improve")
    search_queries: List[str] = Field(
        description=SEARCH_QUERIES_DESCRIPTION
    )

class ResponderWithRetries:
    def __init__(self, runnable, validator):
        self.runnable = runnable
        self.validator = validator

    @traceable
    def respond(self, state: List[BaseMessage]):
        response = []
        for attempt in range(3):
            try:
                response = self.runnable.invoke({"messages": state})
                self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response
    
class ReviseQuery(GenerateQuery):
    """Revise your previous query according to your question."""

    revised_query: str = Field(
        description="Revised query"
    )

def _get_num_iterations(state: List[BaseMessage]):
    i = 0
    for m in state[::-1]:
        if not isinstance(m, (ToolMessage, AIMessage)):
            break
        i += 1
    return i

def _get_last_tool_results_num(state: List[BaseMessage]):
    i = 0
    for m in state[::-1]:
        if not isinstance(m, ToolMessage):
            continue
        message: ToolMessage = m
        logger.info(f"query result: {message.content}")
        results = json.loads(message.content)
        return len(results["result"]) if results["result"] is not None else 0
    
    return 0

MAX_ITERATIONS=20
def should_continue(state: List[BaseMessage]) -> str:
    # in our case, we'll just stop after N plans
    num_iterations = _get_num_iterations(state)
    last_tool_result_num = _get_last_tool_results_num(state)
    if num_iterations > MAX_ITERATIONS or last_tool_result_num > 0:
        return END
    return "query_tools"

class KGLangGraphAgent(ABC):
    def __init__(
        self,
        connection_args: dict,
        conversation_factory: Callable[[], GptConversation]
    ) -> None:
        super().__init__()
        self.conversation_factory = (
            conversation_factory 
            if conversation_factory is not None 
            else self._get_conversation
        )
        self.model_name = "gpt-3.5-turbo"
        self.neodriver: Optional[nu.Driver] = None
        self.connection_args = connection_args
        
        self.parser = JsonOutputToolsParser(return_id=True)
        self._logs: str = ""
    
    def generate_query(self, prompt: str, question: str, query_language: Optional[str]="Cypher"):
        
        def query_tools(state: List[BaseMessage]) -> List[BaseMessage]:
            tool_invocation: AIMessage = state[-1]
            parsed_tool_calls = self.parser.invoke(tool_invocation)
            results = []
            for parsed_call in parsed_tool_calls:
                try:
                    query = ( parsed_call["args"]["revised_query"] 
                             if REVISED_QUERY in parsed_call["args"] 
                             else (parsed_call[REVISED_QUERY_DESCRIPTION] 
                                   if REVISED_QUERY_DESCRIPTION in parsed_call["args"] 
                                   else None) )
                    if query is not None:
                        result = self._query_graph_database(query)
                        results.append({"query": query, "result": result[0]})
                        continue
                    queries = ( parsed_call["args"][SEARCH_QUERIES] 
                               if SEARCH_QUERIES in parsed_call["args"] 
                               else parsed_call["args"][SEARCH_QUERIES_DESCRIPTION] )
                    for query in queries:
                        result = self._query_graph_database(query)
                        results.append({"query": query, "result": result[0]})
                except Exception as e:
                    str_obj = json.dumps(parsed_call["args"])
                    logger.error("Print parsed_call")
                    logger.error(str_obj)
                    logger.error(f"Error occurred: {str(e)}")
                    print(f"Error occurred: {str(e)}")
                    continue
            if len(results) > 1:
                # if there are multiple results, we only return the first no-empty result
                for res in results:
                    if len(res["result"]) > 0:
                        return ToolMessage(
                            content=json.dumps(res),
                            tool_call_id=parsed_call["id"]
                        )
            return ToolMessage(
                content=json.dumps(results[0]),
                tool_call_id=parsed_call["id"]
            )
        
        actor_prompt_template = ChatPromptTemplate.from_messages(
            [(
                "system",
                ("As a senior biomedical researcher and graph database expert, "
                 f"your task is to generate '{query_language}' queries to extract data from our genomicKB graph database based on the user's question. " 
                 """The database utilizes a knowledge graph to consolidate genomic datasets and annotations from over 30 consortia, representing genomic 
entities as nodes and relationships as edges with properties.
Current time {time}
{first_instruction}""")
            ), 
            MessagesPlaceholder(variable_name="messages"),
            ("system", "Only generate query according to the user's question above."),]
        ).partial(
            time=lambda: datetime.now().isoformat()
        )
        self._clear_logs()
        conversation = self.conversation_factory()
        llm = conversation.chat
        initial_answer_chain = actor_prompt_template.partial(
            first_instruction=prompt,
        ) | llm.bind_tools(
            tools=[GenerateQuery],
            tool_choice="GenerateQuery",
        )
        validator = PydanticToolsParser(tools=[GenerateQuery])
        first_responder = ResponderWithRetries(
            runnable=initial_answer_chain,
            validator=validator,
        )
        first_responder.respond([HumanMessage(
            content=question
        )])

        revise_instruction = """
Revise you previous query using the query result and follow the guidelines:
1. if you consistently obtain empty result, please consider removing constraints, like relationship constraint to try to obtain some results.
2. you should use previous critique to improve your query.
"""
        revision_chain = actor_prompt_template.partial(
        first_instruction=revise_instruction,
        ) | llm.bind_tools(tools=[ReviseQuery], tool_choice="ReviseQuery")
        revision_validator = PydanticToolsParser(tools=[ReviseQuery])
        revisor = ResponderWithRetries(
            runnable=revision_chain,
            validator=revision_validator,
        )
        
        builder = MessageGraph()
        builder.add_node("draft", first_responder.respond)
        builder.add_node("query_tools", query_tools)
        builder.add_node("revise", revisor.respond)
        builder.add_edge("draft", "query_tools")
        builder.add_edge("query_tools", "revise")
    
        builder.add_conditional_edges("revise", should_continue)
        builder.set_entry_point("draft")
        graph = builder.compile()
    
        events = graph.stream(
            [HumanMessage(content=question)], {
                "recursion_limit": 30
            }
        )
        for i, step in enumerate(events):
            node, output = next(iter(step.items()))
            self._log_message(f"## {i+1}. {node}")
            try:
                self._log_message(
                    f'Answer: {self.parser.invoke(output)[0]["args"]["answer"]}')
                self._log_message(
                    f'Reflection | Improving: {self.parser.invoke(output)[0]["args"]["reflection"]}')
                self._log_message('Reflection | Search Queries:')
                
                parsed_output = self.parser.invoke(output)[0]["args"]
                for y, sq in enumerate(parsed_output["search_queries"]):
                    self._log_message(f"{y+1}: {sq}")
                if "revised_query" in parsed_output:
                    self._log_message('Reflection | Revised Query:')
                    self._log_message(parsed_output["revised_query"])
                self._log_message("✦✧✦✧✦✧✧✦✧✦✧ Node Output ✦✧✦✧✦✧✧✦✧✦✧")

                continue
        
            except Exception as e:
                logger.error(str(output)[:100] + " ...")
                self._log_message(str(output)[:100] + " ...")
        
        self._log_message("\n\n✦✧✦✧✦✧✧✦✧✦✧ Final Generated Response ✦✧✦✧✦✧✧✦✧✦✧\n\n")
        final_result = self.parser.invoke(step[END][-1])[0]["args"]["answer"]
        self._log_message(final_result)
        return final_result
    

    def _get_conversation(
        self,
        model_name: Optional[str] = None,
    ) -> GptConversation:
        """
        Create a conversation object given a model name.

        Args:
            model_name: The name of the model to use for the conversation.

        Returns:
            A BioChatter Conversation object for connecting to the LLM.

        Todo:
            Genericise to models outside of OpenAI.
        """

        conversation = GptConversation(
            model_name=model_name or self.model_name,
            prompts={},
            correct=False,
        )
        conversation.set_api_key(
            api_key=os.getenv("OPENAI_API_KEY"), user="test_user"
        )
        return conversation
    def _connect_db(self):
        if self.neodriver is not None:
            return
        try:
            db_uri = "bolt://" + self.connection_args.get("host") + \
                ":" + self.connection_args.get("port")
            self.neodriver = nu.Driver(
                db_name=self.connection_args.get("db_name") or "neo4j",
                db_uri=db_uri,
            )
        except Exception as e:
            logger.error(e)

    def _query_graph_database(self, query: str):
        try:
            if self.neodriver is None:
                self._connect_db()
            result = self.neodriver.query(query)
            return result
        except Exception as e:
            logger.error(str(e))
            return str(e)

    @property
    def logs(self):
        return self._logs
    
    @logs.setter
    def logs(self, value: str):
        self._logs = value

    def _log_message(self, msg: str):
        self.logs = self.logs + f"{datetime.now().isoformat()} - {msg}\n"

    def _clear_logs(self):
        self.logs = ""
