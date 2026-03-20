# DocChat — RAG Document Assistant

An intelligent chatbot that answers questions over your PDF documents using a production-grade RAG pipeline.

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
- **Generator** — produces answer with source citations from retrieved context

## Features

- Upload and index multiple PDFs
- Token-based chunking (800-1200 tokens) preserving page metadata
- Hybrid search combining semantic similarity with keyword matching
- Source citations with page numbers in every response
- Streaming chat interface

## Tech Stack

- **Orchestration:** LangGraph + LangChain
- **LLM:** OpenAI GPT-4o
- **Embeddings:** OpenAI text-embedding-3-small
- **Vector Store:** FAISS (local)
- **Keyword Search:** BM25 (rank_bm25)
- **Frontend:** Streamlit
- **PDF Processing:** pdfplumber

## Quick Start

```bash
# Clone
git clone https://github.com/cydgxbriel/docchat.git
cd docchat

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# Run
streamlit run app.py
```

## Demo

<!-- TODO: Add GIF here after recording -->
![DocChat Demo](demo.gif)

## License

MIT
