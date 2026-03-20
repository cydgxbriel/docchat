from unittest.mock import patch, MagicMock
from graph.builder import build_graph


def test_build_graph_returns_compiled_graph():
    graph = build_graph()
    assert graph is not None
    assert hasattr(graph, "invoke")


@patch("graph.nodes.ChatOpenAI")
def test_graph_invoke_with_mocked_llm(mock_llm_cls):
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = MagicMock(content="keyword1, keyword2")
    mock_llm_cls.return_value = mock_llm

    mock_search = MagicMock()
    mock_search.query.return_value = [
        {"text": "Test content.", "page": 1, "source": "test.pdf", "score": 0.9}
    ]

    graph = build_graph(search_engine=mock_search)
    result = graph.invoke({"query": "What is this about?"})

    assert "response" in result
    assert "sources" in result
