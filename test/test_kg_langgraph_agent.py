
import pytest
import os
from datetime import datetime
import logging

from biochatter.kg_langgraph_agent import (
    KGLangGraphAgent,
)

from biochatter.llm_connect import AzureGptConversation, GptConversation
from biochatter.prompts import BioCypherPromptEngine

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
file_handler = logging.FileHandler("./test_logs.log")
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter(
    "%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)
file_handler.setFormatter(formatter)
root_logger = logging.getLogger()
root_logger.addHandler(file_handler)

def output_msg(msg: str):
    with open("./temp2.log", "a+") as fobj:
        fobj.write(f"{datetime.now().isoformat()} - {msg}\n")

OPENAI_API_TYPE="OPENAI_API_TYPE"
OPENAI_DEPLOYMENT_NAME="OPENAI_DEPLOYMENT_NAME"
OPENAI_MODEL="OPENAI_MODEL"
OPENAI_API_VERSION="OPENAI_API_VERSION"
AZURE_OPENAI_ENDPOINT="AZURE_OPENAI_ENDPOINT"
OPENAI_API_KEY="OPENAI_API_KEY"
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

def test_KGLangGraphAgent(prompt_engine):
    question = "It’s odd for a transcription factor to be regulating both stemness of a cell and cytotoxic potential. Which of these is the primary function \
        of EOMES and which is secondary?"
    # question = "What transcription factor likely regulate the expression of gene LRRC32?"
    # question="What genes does EOMES primarily regulate?"
    connection_args= dict()
    connection_args["host"] = os.environ.get("GRAPH_DB_HOST")
    connection_args["port"] = os.environ.get("GRAPH_DB_PORT")
    kg_agent = KGLangGraphAgent(
        connection_args=connection_args,
        conversation_factory=create_conversation,
    )
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
    try:
        query = kg_agent.generate_query(msg, question)
        output_msg("Final result: " + query + "\n")
        assert "(g:gene {name: 'EOMES'})" in query
    except Exception as e:
        logger.error(str(e))
    finally:
        output_msg(kg_agent.logs)
            

