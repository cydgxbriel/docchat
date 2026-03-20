from ingestion.chunker import chunk_pages


def test_chunk_pages_splits_long_text():
    pages = [{"text": "word " * 500, "page": 1, "source": "test.pdf"}]
    chunks = chunk_pages(pages, chunk_size=200, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert "text" in chunk
        assert "page" in chunk
        assert "source" in chunk
        assert "chunk_id" in chunk


def test_chunk_pages_preserves_metadata():
    pages = [
        {"text": "Short text.", "page": 3, "source": "report.pdf"},
    ]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["page"] == 3
    assert chunks[0]["source"] == "report.pdf"


def test_chunk_pages_handles_multiple_pages():
    pages = [
        {"text": "Page one text.", "page": 1, "source": "doc.pdf"},
        {"text": "Page two text.", "page": 2, "source": "doc.pdf"},
    ]
    chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=0)
    assert len(chunks) == 2
