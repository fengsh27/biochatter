from collections import defaultdict
from datetime import datetime
from typing import List
import logging

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.pydantic_v1 import BaseModel, Field, ValidationError
from langchain.output_parsers.openai_tools import (
    JsonOutputToolsParser,
    PydanticToolsParser,
)

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langsmith import traceable
from langgraph.graph import MessageGraph, END

import json
import os
from dotenv import load_dotenv
import neo4j_utils as nu
import pytest

from biochatter.llm_connect import AzureGptConversation, GptConversation
from biochatter.prompts import BioCypherPromptEngine

logger = logging.getLogger(__name__)

connection_args = {
    "host": "10.95.224.94",
    'port': "47687",
}
try:
    db_uri = "bolt://" + connection_args.get("host") + \
        ":" + connection_args.get("port")
    neodriver = nu.Driver(
        db_name=connection_args.get("db_name") or "neo4j",
        db_uri=db_uri,
    )
except Exception as e:
    print(e)

def output_msg(msg: str):
    with open("./temp2.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()} - {msg}\n")

print=output_msg

parser = JsonOutputToolsParser(return_id=True)

 
# Initial responder
actor_prompt_template = ChatPromptTemplate.from_messages(
    [(
        "system",
        """As a senior biomedical researcher and graph database expert, your task is to generate 
Neo4j queries to extract data from our genomicKB graph database based on the user's question. The database utilizes 
a knowledge graph to consolidate genomic datasets and annotations from over 30 consortia, representing genomic 
entities as nodes and relationships as edges with properties.
Current time {time}
{first_instruction}"""
    ), 
    MessagesPlaceholder(variable_name="messages"),
    ("system", "Answer the user's question above using the required format."),]
).partial(
    time=lambda: datetime.now().isoformat()
)

class Reflection(BaseModel):
    improving: str = Field(description="Critique of what to improve.")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    answer: str = Field(description="Cypher query according to user's question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer")
    search_queries: List[str] = Field(
        description="Cypher query for genomicKB graph database."
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
                validated_res = self.validator.invoke(response)
                return response
            except ValidationError as e:
                state = state + [HumanMessage(content=repr(e))]
        return response

class ReviseAnswer(AnswerQuestion):
    """Revise your original query according to your question."""

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
        output_msg(f"query result: {message.content}")
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

def generate_query_with_agents(
    prompts: str, 
    question: str, 
    connection_args: dict,
    conversation_factory: callable
):
    neodriver = None
    def _connect_db():
        nonlocal neodriver
        if neodriver is not None:
            return
        try:
            db_uri = "bolt://" + connection_args["host"] + \
                ":" + connection_args["port"]
            neodriver = nu.Driver(
                db_name=(
                    connection_args["db_name"] 
                    if "db_name" in connection_args else "neo4j"
                ),
                db_uri=db_uri,
            )
        except Exception as e:
            logger.error(e)   
    def query_graph_database(query_str: str):
        _connect_db()
        nonlocal neodriver
        try:
            output_msg(f"query: {query_str}\n")
            result = neodriver.query(query_str)
            output_msg(f"results: {result}\n")
            return result
        except Exception as e:
            return str(e)
        
    def query_tools(state: List[BaseMessage]) -> List[BaseMessage]:
        tool_invocation: AIMessage = state[-1]
        parsed_tool_calls = parser.invoke(tool_invocation)
        ids = []
        results = []
        for parsed_call in parsed_tool_calls:
            query = parsed_call["args"]["revised_query"] if "revised_query" in parsed_call["args"] else None
            if query is not None:
                result = query_graph_database(query)
                results.append({"query": query, "result": result[0]})
                continue
            queries = parsed_call["args"]["search_queries"]
            for query in queries:
                result = query_graph_database(query)
                results.append({"query": query, "result": result[0]})
        if len(results) > 1:
            for res in results:
                if len(res["result"]) > 0:
                    return ToolMessage(
                        content=json.dumps(res), tool_call_id=parsed_call["id"]
                    )
        return ToolMessage(
            content=json.dumps(results[0] if len(results) > 0 else []), 
            tool_call_id=parsed_call["id"]
        )
 

    OPENAI_API_TYPE="OPENAI_API_TYPE"
    OPENAI_DEPLOYMENT_NAME="OPENAI_DEPLOYMENT_NAME"
    OPENAI_MODEL="OPENAI_MODEL"
    OPENAI_API_VERSION="OPENAI_API_VERSION"
    AZURE_OPENAI_ENDPOINT="AZURE_OPENAI_ENDPOINT"
    OPENAI_API_KEY="OPENAI_API_KEY"
    
    llm = AzureChatOpenAI(
        openai_api_version=os.environ.get(OPENAI_API_VERSION),
        azure_deployment=os.environ[OPENAI_DEPLOYMENT_NAME],
        azure_endpoint=os.environ.get(AZURE_OPENAI_ENDPOINT) ,
        api_key=os.environ.get(OPENAI_API_KEY)
    )

    initial_answer_chain = actor_prompt_template.partial(
        first_instruction=prompts,
    ) | llm.bind_tools(tools=[AnswerQuestion], tool_choice="AnswerQuestion")
    validator = PydanticToolsParser(tools=[AnswerQuestion])
    first_responder = ResponderWithRetries(
        runnable=initial_answer_chain,
        validator=validator,
    )
    initial = first_responder.respond([HumanMessage(content=question)])

    revise_instruction = """
Revise you previous query using the query result and follow the guidelines:
1. if you consistently obtain empty result, please consider removing constraints, like relationship constraint to try to obtain some results.
2. you should use previous critique to improve your query.
"""
    revision_chain = actor_prompt_template.partial(
        first_instruction=revise_instruction,
    ) | llm.bind_tools(tools=[ReviseAnswer], tool_choice="ReviseAnswer")
    revision_validator = PydanticToolsParser(tools=[ReviseAnswer])
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
        print(f"## {i+1}. {node}")
        try:
            print(f'Answer: {parser.invoke(output)[0]["args"]["answer"]}')
            print(
                f'Reflection | Improving: {parser.invoke(output)[0]["args"]["reflection"]["improving"]}')
            print('Reflection | Search Queries:')
            
            for y, sq in enumerate(parser.invoke(output)[0]["args"]["search_queries"]):
                print(f"{y+1}: {sq}")
            print("✦✧✦✧✦✧✧✦✧✦✧ Node Output ✦✧✦✧✦✧✧✦✧✦✧")
            continue
    
        except Exception as e:
            print("Error: " + str(output)[:100] + " ...")
            logger.error("Error: " + str(output)[:100] + " ...")
    
    print("\n\n✦✧✦✧✦✧✧✦✧✦✧ Final Generated Response ✦✧✦✧✦✧✧✦✧✦✧\n\n")
    res = (parser.invoke(step[END][-1])[0]["args"]["answer"])
    return res
