from unittest.mock import patch, MagicMock
from graph.nodes import planner, retriever, generator
from graph.state import DocChatState


@patch("graph.nodes.ChatOpenAI")
def test_planner_extracts_keywords(mock_llm_cls):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="LangGraph, agents, framework")
    mock_llm_cls.return_value = mock_llm

    state: DocChatState = {"query": "What is LangGraph used for?"}
    result = planner(state)
    assert "search_keywords" in result
    assert isinstance(result["search_keywords"], list)
    assert len(result["search_keywords"]) > 0


def test_retriever_searches_and_returns_chunks():
    mock_search = MagicMock()
    mock_search.query.return_value = [
        {"text": "LangGraph builds agents.", "page": 1, "source": "doc.pdf", "score": 0.9}
    ]
    state: DocChatState = {
        "query": "What is LangGraph?",
        "search_keywords": ["LangGraph", "agents"],
    }
    result = retriever(state, search_engine=mock_search)
    assert "retrieved_chunks" in result
    assert len(result["retrieved_chunks"]) > 0


@patch("graph.nodes.ChatOpenAI")
def test_generator_produces_response_with_sources(mock_llm_cls):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(
        content="LangGraph is a framework for building agents. [Source: doc.pdf, p.1]"
    )
    mock_llm_cls.return_value = mock_llm

    state: DocChatState = {
        "query": "What is LangGraph?",
        "retrieved_chunks": [
            {"text": "LangGraph builds agents.", "page": 1, "source": "doc.pdf", "score": 0.9}
        ],
    }
    result = generator(state)
    assert "response" in result
    assert len(result["response"]) > 0
    assert "sources" in result
