from langchain_openai import ChatOpenAI
from graph.state import DocChatState

_DEFAULT_MODEL = "gpt-4o"


def planner(state: DocChatState) -> dict:
    """Extract search keywords from the user query."""
    llm = ChatOpenAI(model=_DEFAULT_MODEL, temperature=0)
    prompt = (
        "Extract 3-5 search keywords from the following question. "
        "Return only the keywords separated by commas, nothing else.\n\n"
        f"Question: {state['query']}"
    )
    result = llm.invoke(prompt)
    keywords = [kw.strip() for kw in result.content.split(",") if kw.strip()]
    return {"search_keywords": keywords}


def retriever(state: DocChatState, search_engine=None) -> dict:
    """Retrieve relevant chunks using hybrid search."""
    if search_engine is None:
        return {"retrieved_chunks": []}

    combined_query = state["query"] + " " + " ".join(state.get("search_keywords", []))
    chunks = search_engine.query(combined_query, k=5)
    return {"retrieved_chunks": chunks}


def generator(state: DocChatState) -> dict:
    """Generate a response with source citations."""
    llm = ChatOpenAI(model=_DEFAULT_MODEL, temperature=0)

    chunks = state.get("retrieved_chunks", [])
    context = "\n\n".join(
        f"[Source: {c['source']}, page {c['page']}]\n{c['text']}"
        for c in chunks
    )

    prompt = (
        "You are a helpful document assistant. Answer the question based ONLY on the "
        "provided context. Cite sources using [Source: filename, page X] format. "
        "If the context doesn't contain the answer, say so.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['query']}\n\n"
        "Answer:"
    )

    result = llm.invoke(prompt)
    sources = [
        {"source": c["source"], "page": c["page"]}
        for c in chunks
    ]

    return {"response": result.content, "sources": sources}
