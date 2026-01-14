import asyncio
import os

import dotenv
from langchain.agents import create_agent

from core.mcp_clients.docs_mcp import docs_mcp_client
from core.models.models import agent_model
from core.tools.local_retriever import search_local_docs

dotenv.load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("SI_API_KEY")
os.environ['OPENAI_BASE_URL'] = os.getenv("SI_BASE_URL")

SYS_PROMPT = """

"""


async def build_agent():
    tools = []
    mcp_tools = await docs_mcp_client.get_tools()
    tools.extend(mcp_tools)
    tools.append(search_local_docs)

    RAG_agent = create_agent(
        agent_model,
        tools=tools,
        system_prompt=SYS_PROMPT,
    )

    return RAG_agent

agent = asyncio.run(build_agent())
