# DocChat — RAG Document Assistant

Chatbot that answers questions over your PDF documents with hybrid search
(FAISS semantic + BM25 keyword) and page-level source citations — a three-node
LangGraph pipeline behind a streaming Streamlit chat.

## The problem it solves

Answering questions over real documents fails in two opposite ways: pure
semantic search misses exact terms (names, codes, legal wording), and pure
keyword search misses paraphrases. DocChat combines both retrieval modes and
cites the source page for every answer, so responses are verifiable instead of
plausible-sounding.

## Architecture

```
┌──────────┐     ┌───────────┐     ┌───────────┐
│  Planner │────▶│ Retriever │────▶│ Generator │
│          │     │           │     │           │
│ Extract  │     │  Hybrid   │     │ Answer +  │
│ keywords │     │  search   │     │  sources  │
└──────────┘     └───────────┘     └───────────┘
                       │
              ┌────────┴────────┐
              │  FAISS + BM25   │
              │ Semantic+Keyword│
              └─────────────────┘
```

**Pipeline:** LangGraph StateGraph with 3 nodes
- **Planner** — extracts search keywords from the user query
- **Retriever** — hybrid search (FAISS semantic + BM25 keyword) over indexed chunks
- **Generator** — produces the answer with source citations from retrieved context

## Features

- Upload and index multiple PDFs
- Token-based chunking (800–1200 tokens) preserving page metadata
- Hybrid search combining semantic similarity with keyword matching
- Source citations with page numbers in every response
- Streaming chat interface

## Technical decisions

- **Hybrid retrieval (FAISS + BM25)** — the two retrieval modes fail on
  different queries; merging them raises recall on documents that mix prose
  with exact identifiers.
- **A planner node before retrieval** — retrieval quality depends on the query,
  so the pipeline extracts search keywords first instead of embedding the raw
  user message.
- **Token-based chunks (800–1200) with page metadata** — sized for the
  embedding model, and page numbers survive all the way into citations.
- **Deterministic tests, no API key needed** — chunker, loader, hybrid search,
  nodes and graph wiring are covered by 5 pytest modules that run offline.

## Tech stack

- **Orchestration:** LangGraph ≥0.2 + LangChain ≥0.3
- **LLM:** OpenAI GPT-4o (langchain-openai ≥0.2)
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector store:** FAISS ≥1.8 (local) · **Keyword search:** rank-bm25 ≥0.2.2
- **PDF processing:** pdfplumber ≥0.11 · **Chunking:** tiktoken ≥0.7
- **Frontend:** Streamlit ≥1.38 · **Tests:** pytest ≥8

## Run locally

```bash
git clone https://github.com/cydgxbriel/docchat.git && cd docchat
python -m venv venv && source venv/bin/activate && pip install -r requirements.txt
cp .env.example .env && streamlit run app.py    # add OPENAI_API_KEY to .env first
```

## Tests

```bash
pytest    # 5 modules: chunker, loader, hybrid search, nodes, graph — no API key required
```

## License

MIT
