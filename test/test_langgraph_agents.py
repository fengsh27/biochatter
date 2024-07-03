from collections import defaultdict
from datetime import datetime
from typing import List

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

load_dotenv()

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

def query_graph_database(query_str: str):
    try:
        output_msg(f"query: {query_str}\n")
        result = neodriver.query(query_str)
        output_msg(f"results: {result}\n")
        return result
    except Exception as e:
        return str(e)
    
parser = JsonOutputToolsParser(return_id=True)

def query_tools(state: List[BaseMessage]) -> List[BaseMessage]:
    tool_invocation: AIMessage = state[-1]
    parsed_tool_calls = parser.invoke(tool_invocation)
    ids = []
    results = []
    for parsed_call in parsed_tool_calls:
        for query in parsed_call["args"]["search_queries"]:
            q = query
            result = query_graph_database(q)

            results.append({"query": q, "result": result[0]})
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
    superfluous: str = Field(description="Critique of what is made up.")

class AnswerQuestion(BaseModel):
    """Answer the question."""
    answer: str = Field(description="Cypher query according to user's question.")
    reflection: Reflection = Field(description="Your reflection on the initial answer")
    search_queries: List[str] = Field(
        description="query for genomicKB graph database."
    )

# prompts_generated = """Generate a database query in Cypher that answers the user's question.
# You can use the following entities: ['gene', 'enhancer'], relationships: ['regulate'],
# and properties: {'gene': {'name': '', 'id': '', 'type': '', 'data_source': ''},
# 'Regulate': {'tissue_id': '', 'tissue_name': '', 'weak_experiments': '', 'strong_experiments': '', 'data_source': ''}}.
# Given the following valid combinations of source, relationship, and target: '(g:gene)-(r:regulate)->(c:gene)', '(g:gene)-(r:regulate)->(c:enhancer)',
# '(g:enhancer)-(r:regulate)->(c:gene)', '(g:enhancer)-(r:regulate)->(c:enhancer)', generate a Cypher query using one of these combinations.
# Note: Please limit distinct result to 5. Only return the query, without any additional text."""
# question = "What transcription factor likely regulate the expression of gene LRRC32"
# prompts_generated = """Generate a database query in Cypher that answers the user's question. 
# You can use the following entities: ['gene', 'enhancer'], relationships: ['regulate'], 
# and properties: {'gene': ['name', 'description', 'id'], 'enhancer': ['tissue_name', 'disease'], 'Regulate': ['tissue_name', 'weak_experiments', 'strong_experiments']}. 
# Given the following valid combinations of source, relationship, and target: '(g:gene)-(r:regulate)->(c:gene)', '(g:gene)-(r:regulate)->(c:enhancer)', 
# '(g:enhancer)-(r:regulate)->(c:gene)', '(g:enhancer)-(r:regulate)->(c:enhancer)', generate a Cypher query using one of these combinations. 
# Note: Please limit distinct result to 5. Only return the query, without any additional text."""
# question="Does EOMES regulate different genes under different contexts?"
# prompts_generated="""Generate a database query in Cypher that answers the user's question. 
# You can use the following entities: ['gene', 'enhancer'], relationships: ['regulate'], 
# and properties: {'gene': ['name', 'id']}. Given the following valid combinations of source, relationship, 
# and target: '(g:gene)-(r:regulate)->(c:gene)', '(g:gene)-(r:regulate)->(c:enhancer)', '(g:enhancer)-(r:regulate)->(c:gene)', 
# '(g:enhancer)-(r:regulate)->(c:enhancer)', generate a Cypher query using one of these combinations. Note: Please limit distinct result to 5. 
# Only return the query, without any additional text."""
# question="What genes does EOMES primarily regulate?"

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

def create_conversation():
    if OPENAI_API_TYPE in os.environ and os.environ[OPENAI_API_TYPE] == "azure":
        chatter = AzureGptConversation(
            deployment_name=os.environ[OPENAI_DEPLOYMENT_NAME],
            model_name=os.environ[OPENAI_MODEL],
            prompts={"rag_agent_prompts": ""},
            version=os.environ[OPENAI_API_VERSION],
            base_url=os.environ[AZURE_OPENAI_ENDPOINT],
        )
        chatter.set_api_key(os.environ[OPENAI_API_KEY])
    else:
        chatter = GptConversation(
            "gpt-3.5-turbo", prompts={"rag_agent_prompts": ""}
        )
        temp_api_key = os.environ.get("OPENAI_API_KEY", None)
        if temp_api_key is not None:
            chatter.set_api_key(temp_api_key, "test")
    return chatter

@pytest.fixture
def prompt_engine():
    return BioCypherPromptEngine(
        schema_config_or_info_path="./test/genomicKB_schema_config.yaml",
        snake_name=True,
        conversation_factory=create_conversation
    )

def test_langgraph_agent(prompt_engine):
    # question="What genes does EOMES primarily regulate?"
    question = "What transcription factor likely regulate the expression of gene LRRC32?"
    # question = "It’s odd for a transcription factor to be regulating both stemness of a cell and cytotoxic potential. Which of these is the primary function of EOMES and which is secondary?"
    prompt_engine._select_entities(question, prompt_engine.conversation_factory())
    prompt_engine._select_relationships(prompt_engine.conversation_factory())
    prompt_engine._select_properties(prompt_engine.conversation_factory())
    msg = prompt_engine._generate_query(
        question,
        prompt_engine.selected_entities,
        prompt_engine.selected_relationship_labels,
        prompt_engine.selected_properties,
        "Cypher",
        prompt_engine.conversation_factory,
        True
    )
    tmp = actor_prompt_template.partial(first_instruction=msg,)
    msgs = tmp.messages
    initial_answer_chain = actor_prompt_template.partial(
        first_instruction=msg,
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
2. you should use previous critique to remove superfluous information and improve your query.
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
            print(
                f'Reflection | Superfluous: {parser.invoke(output)[0]["args"]["reflection"]["superfluous"]}')
            print('Reflection | Search Queries:')
            
            parsed_output = parser.invoke(output)[0]["args"]
            for y, sq in enumerate(parsed_output["search_queries"]):
                print(f"{y+1}: {sq}")
            if "revised_query" in parsed_output:
                print('Reflection | Revised Query:')
                print(parsed_output["revised_query"])
            print("✦✧✦✧✦✧✧✦✧✦✧ Node Output ✦✧✦✧✦✧✧✦✧✦✧")
            continue
    
        except Exception as e:
            print("Error: " + str(output)[:100] + " ...")
    
    print("\n\n✦✧✦✧✦✧✧✦✧✦✧ Final Generated Response ✦✧✦✧✦✧✧✦✧✦✧\n\n")
    print(parser.invoke(step[END][-1])[0]["args"]["answer"])

