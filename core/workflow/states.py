import operator
from typing import TypedDict, List, Annotated
from langchain_core.documents import Document


class State(TypedDict):
    query: str
    doc_ids: List[str]
    node_ids: List[str]
    final_node_ids: List[str]
    final_content: List[str]
