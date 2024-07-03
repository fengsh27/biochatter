import os
from typing import Any, Coroutine, Type
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI, AzureChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage

import neo4j_utils as nu

import pytest

from biochatter.llm_connect import AzureGptConversation, GptConversation
from biochatter.prompts import BioCypherPromptEngine

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

def query_graph_database(query_str: str):
    try:
        with open("./temp1.log", "+a") as fobj:
            fobj.write(f"query: {query_str}\n")
            result = neodriver.query(query_str)
            fobj.write(f"results: {result}\n")
            return result
    except Exception as e:
        return str(e)
    
class QueryDatabaseInput(BaseModel):
    target: str = Field(
        description="graph data base query string"
    )
    
class QueryDatabaseTool(BaseTool):
    name = "query_graph_database"
    description = "Search for data in genomicKB graph database (neo4j) to get results"
    args_schema: Type[BaseModel] = QueryDatabaseInput

    def _run(self, target: str):
        return query_graph_database(target)
    
    def _arun(self, *args: PromptTemplate, **kwargs: PromptTemplate) -> Coroutine[Any, Any, Any]:
        raise NotImplemented("Error")
    
OPENAI_API_TYPE="OPENAI_API_TYPE"
OPENAI_DEPLOYMENT_NAME="OPENAI_DEPLOYMENT_NAME"
OPENAI_MODEL="OPENAI_MODEL"
OPENAI_API_VERSION="OPENAI_API_VERSION"
AZURE_OPENAI_ENDPOINT="AZURE_OPENAI_ENDPOINT"
OPENAI_API_KEY="OPENAI_API_KEY"
os.environ[OPENAI_API_TYPE]="azure"
os.environ[AZURE_OPENAI_ENDPOINT]="https://copper-gpt4-1.openai.azure.com/"
os.environ[OPENAI_DEPLOYMENT_NAME]="gpt4o-deploy-1"
os.environ[OPENAI_API_VERSION]="2024-02-01"
os.environ[OPENAI_API_KEY]="57e00b999398482fba73192631335857"
os.environ[OPENAI_MODEL]="gpt-4o"
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

def test_agent(prompt_engine):
    system_content = """As a senior biomedical researcher and graph database expert, your task is to generate 
Neo4j queries to extract data from our genomicKB graph database based on the user's question. The database utilizes 
a knowledge graph to consolidate genomic datasets and annotations from over 30 consortia, representing genomic 
entities as nodes and relationships as edges with properties.
Ensure the following guidelines are followed:
1. If the query result is empty or an error occurs, modify the query string and retry, up to 10 times.
2. Evaluate the query results after each attempt, and if you keep receiving empty result, please consider removing relationship constraint,
like "(g:{{node_type}})-[r]->(c:{{node_type}})"
3. Provide factual data and information without fabricating details.
4. Include all query statements and results in the final output to support your research.
5. Please only use relationship provided, you should not make up relationship. If empty relationship is provided, please
just leave relationship empty in query
        """
    # question = "What genes does EOMES primarily regulate"
    # question = "What genes does EOMES primarily regulate or be regulated?"
    question = "What transcription factor likely regulate the expression of gene LRRC32?"
    # question = "In chromosome 1 of GM12878 cell line, how many loops have both anchors locating in the same TAD?"
    # question = "Which of the enhancer near PVT1 long non-coding RNA?"

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
    msg = msg[69:]
    msg_length = len(msg)
    msg = msg[:(msg_length - 51)]
    system_content += "\n Now,"
    system_content += msg
    with open("./temp1.log", "+a") as fobj:
        fobj.write(f"prompts: {system_content}\n")

    system_message = SystemMessage(
        content=system_content
    )

    agent_kwargs = {
        "extra_prompt_messages": [MessagesPlaceholder(variable_name="memory")],
        "system_message": system_message,
    }
    # llm = ChatOpenAI(
    #     temperature=0, 
    #     model="gpt-3.5-turbo-16k-0613",
    #     api_key=os.environ.get("OPENAI_API_KEY")
    # )
    
    llm = AzureChatOpenAI(
        openai_api_version="2024-02-01",
        azure_deployment="gpt4o-deploy-1", # os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint="https://copper-gpt4-1.openai.azure.com/",
    )
    memory = ConversationSummaryBufferMemory(
        memory_key="memory",
        return_messages=True,
        llm=llm,
        max_token_limit=10000,
    )
    agent = initialize_agent(
        tools=[QueryDatabaseTool()], 
        llm=llm, 
        agent=AgentType.OPENAI_FUNCTIONS,
        verbose=True,
        agent_kwargs=agent_kwargs,
        memory=memory
    )
    print("start generating query")
    result = agent({"input": question})
    # result = agent({"input": "Does EOMES regulate different genes under different contexts?"})
    # result = agent({"input": "What transcription factor likely regulate the expression of gene LRRC32"})
    print("end generating query")
    print(result)
    





