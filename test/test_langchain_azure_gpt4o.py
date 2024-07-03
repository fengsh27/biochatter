import os
from typing import Any, Coroutine, Type
from langchain import PromptTemplate
from langchain.agents import initialize_agent, Tool, AgentType
from langchain.chat_models import ChatOpenAI
from langchain.prompts import MessagesPlaceholder
from langchain.memory import ConversationSummaryBufferMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from langchain.schema import SystemMessage
from langchain.chat_models import AzureChatOpenAI
from langchain_core.messages import HumanMessage

openai_type = "azure"
if openai_type == "azure":
    llm = AzureChatOpenAI(
        openai_api_version="2024-02-01",
        azure_deployment="gpt4o-deploy-1", # os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"],
        azure_endpoint="https://copper-gpt4-1.openai.azure.com/",
        openai_api_key="57e00b999398482fba73192631335857"
    )
    # model = os.environ.get("OPENAI_DEPLOYMENT_NAME", None)
else:
    llm = None


def test_gpt_4o():
    message = HumanMessage(
        content="Translate this sentence from English to French. I love programming."
    )
    res = llm.invoke([message])
    print(res)
