import streamlit as st
from dotenv import load_dotenv
from ingestion.loader import load_pdf
from ingestion.chunker import chunk_pages
from search.hybrid import HybridSearch
from graph.builder import build_graph

load_dotenv()

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")
st.title("📄 DocChat — RAG Document Assistant")
st.caption("Upload PDFs and ask questions. Powered by LangGraph + Hybrid Search.")

if "search_engine" not in st.session_state:
    st.session_state.search_engine = HybridSearch()
    st.session_state.indexed = False
    st.session_state.messages = []

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    if uploaded_files and st.button("Index Documents"):
        all_chunks = []
        for uploaded in uploaded_files:
            import tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                f.write(uploaded.read())
                tmp_path = f.name

            pages = load_pdf(tmp_path)
            chunks = chunk_pages(pages, chunk_size=1000, chunk_overlap=200)
            all_chunks.extend(chunks)
            os.unlink(tmp_path)

        with st.spinner(f"Indexing {len(all_chunks)} chunks..."):
            st.session_state.search_engine.index(all_chunks)
            st.session_state.indexed = True

        st.success(f"Indexed {len(all_chunks)} chunks from {len(uploaded_files)} file(s).")

    if st.session_state.indexed:
        st.info("Documents indexed and ready.")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.indexed:
        st.warning("Please upload and index documents first.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            graph = build_graph(search_engine=st.session_state.search_engine)

            with st.spinner("Searching documents..."):
                result = {}
                for event in graph.stream({"query": prompt}):
                    for node_name, node_output in event.items():
                        result.update(node_output)

            st.markdown(result.get("response", "No response generated."))

            if result.get("sources"):
                with st.expander("Sources"):
                    seen = set()
                    for s in result["sources"]:
                        key = f"{s['source']}:p{s['page']}"
                        if key not in seen:
                            st.write(f"- **{s['source']}**, page {s['page']}")
                            seen.add(key)

        st.session_state.messages.append(
            {"role": "assistant", "content": result["response"]}
        )
