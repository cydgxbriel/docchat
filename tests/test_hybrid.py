# /home/cydgxbriel/portfolio/docchat/tests/test_hybrid.py
from unittest.mock import patch, MagicMock
from search.hybrid import HybridSearch


def _make_chunks():
    return [
        {"text": "LangGraph is a framework for building AI agents.", "page": 1, "source": "doc.pdf", "chunk_id": "c0"},
        {"text": "Python is a popular programming language.", "page": 2, "source": "doc.pdf", "chunk_id": "c1"},
        {"text": "RAG combines retrieval with generation.", "page": 3, "source": "doc.pdf", "chunk_id": "c2"},
    ]


@patch("search.hybrid.OpenAIEmbeddings")
def test_index_and_search_returns_results(mock_emb_cls):
    mock_emb = MagicMock()
    mock_emb.embed_documents.return_value = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    mock_emb.embed_query.return_value = [1, 0, 0]
    mock_emb_cls.return_value = mock_emb

    search = HybridSearch()
    chunks = _make_chunks()
    search.index(chunks)

    results = search.query("LangGraph agents", k=2)
    assert len(results) <= 2
    assert all("text" in r for r in results)
    assert all("page" in r for r in results)
    assert all("score" in r for r in results)


@patch("search.hybrid.OpenAIEmbeddings")
def test_query_before_index_returns_empty(mock_emb_cls):
    mock_emb_cls.return_value = MagicMock()
    search = HybridSearch()
    results = search.query("anything", k=3)
    assert results == []
