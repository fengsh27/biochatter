import json
from typing import Dict, List
import pytest
from unittest.mock import patch, MagicMock
import os
from dotenv import load_dotenv
import neo4j_utils as nu
import logging
from langchain_core.messages import (
    BaseMessage,
    AIMessage,
)
from langchain.output_parsers.openai_tools import (
    PydanticToolsParser,
)
import shortuuid

from biochatter.selector_agent import (
    RagAgentChoiceModel,
    RagAgentSelector,
    RagAgentRevisionModel,
)
from biochatter.langgraph_agent_base import ResponderWithRetries
from biochatter.llm_connect import AzureGptConversation
from biochatter.rag_agent import RagAgent, RagAgentModeEnum

load_dotenv()

logger = logging.getLogger(__name__)


def find_schema_info_node(connection_args: dict):
    try:
        """
        Look for a schema info node in the connected BioCypher graph and load the
        schema info if present.
        """
        db_uri = (
            "bolt://"
            + connection_args.get("host")
            + ":"
            + connection_args.get("port")
        )
        neodriver = nu.Driver(
            db_name=connection_args.get("db_name") or "neo4j",
            db_uri=db_uri,
        )
        result = neodriver.query("MATCH (n:Schema_info) RETURN n LIMIT 1")

        if result[0]:
            schema_info_node = result[0][0]["n"]
            schema_dict = json.loads(schema_info_node["schema_info"])
            return schema_dict

        return None
    except Exception as e:
        logger.error(e)
        return None


def create_conversation():
    chatter = AzureGptConversation(
        deployment_name=os.environ["OPENAI_DEPLOYMENT_NAME"],
        model_name=os.environ["OPENAI_MODEL"],
        prompts={"rag_agent_prompts": ""},
        version=os.environ["OPENAI_API_VERSION"],
        base_url=os.environ["AZURE_OPENAI_ENDPOINT"],
    )
    chatter.set_api_key(os.environ["OPENAI_API_KEY"])
    return chatter


class ChatOpenAIMock:
    def __init__(self) -> None:
        self.chat = None


class InitialResponder:
    def __init__(self, rag_agent: str):
        self.rag_agent = rag_agent

    def invoke(self, msg_obj: Dict[str, List[BaseMessage]]) -> BaseMessage:
        msg = AIMessage(content="initial test")
        id = "call_" + shortuuid.uuid()
        msg.additional_kwargs = {
            "tool_calls": [
                {
                    "id": id,
                    "function": {
                        "arguments": '{"answer":"'
                        + f"{self.rag_agent}"
                        + '","reflection":"balahbalah"}',
                        "name": "RagAgentChoiceModel",
                    },
                    "type": "function",
                }
            ]
        }
        msg.id = id
        msg.tool_calls = [
            {
                "name": "RagAgentChoiceModel",
                "args": {
                    "answer": f"{self.rag_agent}",
                    "reflection": "balahbalah",
                },
                "id": id,
            }
        ]
        return msg


class ReviseResponder:
    def __init__(self, rag_agent: str):
        self.rag_agent = rag_agent

    def invoke(self, msg_obj: Dict[str, List[BaseMessage]]) -> BaseMessage:
        return AIMessage(
            content="",
            additional_kwargs={
                "tool_calls": [
                    {
                        "id": "call_wTO40b9",
                        "function": {
                            "arguments": '{"answer":"'
                            + f"{self.rag_agent}"
                            + '","reflection":"balahbalah.","revised_answer":"'
                            + f"{self.rag_agent}"
                            + '","score":"10","tool_result":"balahbalah"}',
                            "name": "RagAgentRevisionModel",
                        },
                        "type": "function",
                    }
                ]
            },
            response_metadata={
                "token_usage": {
                    "completion_tokens": 146,
                    "prompt_tokens": 419,
                    "total_tokens": 565,
                },
                "model_name": "gpt-4o-2024-05-13",
                "system_fingerprint": "fp_abc28019ad",
                "prompt_filter_results": [
                    {
                        "prompt_index": 0,
                        "content_filter_results": {
                            "hate": {"filtered": False, "severity": "safe"},
                            "self_harm": {
                                "filtered": False,
                                "severity": "safe",
                            },
                            "sexual": {"filtered": False, "severity": "safe"},
                            "violence": {"filtered": False, "severity": "safe"},
                        },
                    }
                ],
                "finish_reason": "stop",
                "logprobs": None,
                "content_filter_results": {},
            },
            id="run-ad95b309-4f99-4886-b70f-bcb8b59f537f-0",
            tool_calls=[
                {
                    "name": "RagAgentRevisionModel",
                    "args": {
                        "answer": f"{self.rag_agent}",
                        "reflection": "balahbalah",
                        "revised_answer": f"{self.rag_agent}",
                        "score": "10",
                        "tool_result": "balahbalah",
                    },
                    "id": "call_wTO40b9",
                }
            ],
        )


class AgentSelectorMock(RagAgentSelector):
    def __init__(self, rag_agents, conversation_factory, expected: str):
        super().__init__(rag_agents, conversation_factory)
        self.expected_ragagent = expected

    def _create_initial_responder(
        self, prompt: str | None = None
    ) -> ResponderWithRetries:
        runnable = InitialResponder(self.expected_ragagent)
        validator = PydanticToolsParser(tools=[RagAgentChoiceModel])
        return ResponderWithRetries(
            runnable=runnable,
            validator=validator,
        )

    def _create_revise_responder(
        self, prompt: str | None = None
    ) -> ResponderWithRetries:
        runnable = ReviseResponder(self.expected_ragagent)
        validator = PydanticToolsParser(tools=[RagAgentRevisionModel])
        return ResponderWithRetries(runnable=runnable, validator=validator)


def create_agent_selector(expected_rag_agent: str):
    dbAgent = MagicMock()
    dbAgent.mode = RagAgentModeEnum.KG
    dbAgent.get_description = MagicMock(return_value="mock database agent")
    dbAgent.generate_responses = MagicMock(
        return_value=[
            ('{"bp.name": "balahbalah"}', {"cypher_query": "balahbalah"}),
            ('{"bp.name": "balahbalah"}', {"cypher_query": "balahbalah"}),
        ]
    )
    vectorstoreAgent = MagicMock()
    vectorstoreAgent.mode = RagAgentModeEnum.VectorStore
    vectorstoreAgent.get_description = MagicMock(
        return_value="mock vector store"
    )
    vectorstoreAgent.generate_responses = MagicMock(
        return_value=["mock content 1", "mock content 2"]
    )
    blastAgent = MagicMock()
    blastAgent.mode = RagAgentModeEnum.API_BLAST
    blastAgent.get_description = MagicMock(return_value="mock blast api")
    blastAgent.generate_responses = MagicMock(
        return_value=[
            "blast search result 1",
            "blast search result 2",
        ]
    )
    oncokbAgent = MagicMock()
    oncokbAgent.mode = RagAgentModeEnum.API_ONCOKB
    oncokbAgent.get_description = MagicMock(return_value="mock oncokb api")
    oncokbAgent.generate_responses = MagicMock(
        return_value=[
            "oncokb search result 1",
            "oncokb search result 2",
        ]
    )
    return AgentSelectorMock(
        rag_agents=[dbAgent, vectorstoreAgent, blastAgent, oncokbAgent],
        conversation_factory=create_conversation,
        expected=expected_rag_agent,
    )


def test_agent_selector_oncokb():
    agent_selector = create_agent_selector(str(RagAgentModeEnum.API_ONCOKB))
    result = agent_selector.execute(
        "What is the oncogenic potential of BRAF V600E mutation?"
    )
    assert result.answer == "api_oncokb"


def test_agent_selector_blast():
    agent_selector = create_agent_selector(str(RagAgentModeEnum.API_BLAST))
    result = agent_selector.execute(
        "What is the oncogenic potential of BRAF V600E mutation?"
    )
    assert result.answer == "api_blast"


def test_agent_selector_kg():
    agent_selector = create_agent_selector(str(RagAgentModeEnum.KG))
    result = agent_selector.execute(
        "What is the oncogenic potential of BRAF V600E mutation?"
    )
    assert result.answer == "kg"


def test_agent_selector_vectorstore():
    agent_selector = create_agent_selector(str(RagAgentModeEnum.VectorStore))
    result = agent_selector.execute(
        "What is the oncogenic potential of BRAF V600E mutation?"
    )
    assert result.answer == "vectorstore"