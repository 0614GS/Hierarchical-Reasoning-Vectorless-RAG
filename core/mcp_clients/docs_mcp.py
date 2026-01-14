from langchain_mcp_adapters.client import MultiServerMCPClient

docs_mcp_client = MultiServerMCPClient(
        {
            "langchain_doc_server": {
                "transport": "http",
                "url": "https://docs.langchain.com/mcp"
            }
        }
    )
