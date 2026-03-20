from typing import TypedDict


class DocChatState(TypedDict, total=False):
    """State for the DocChat RAG pipeline."""
    query: str
    search_keywords: list[str]
    retrieved_chunks: list[dict]
    response: str
    sources: list[dict]
