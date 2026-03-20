import tiktoken


def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[dict]:
    """Split pages into token-sized chunks preserving metadata."""
    enc = tiktoken.encoding_for_model("gpt-4o")
    chunks = []

    for page in pages:
        tokens = enc.encode(page["text"])

        if len(tokens) <= chunk_size:
            chunks.append({
                "text": page["text"],
                "page": page["page"],
                "source": page["source"],
                "chunk_id": f"{page['source']}:p{page['page']}:c0",
            })
            continue

        start = 0
        chunk_idx = 0
        while start < len(tokens):
            end = min(start + chunk_size, len(tokens))
            chunk_text = enc.decode(tokens[start:end])
            chunks.append({
                "text": chunk_text,
                "page": page["page"],
                "source": page["source"],
                "chunk_id": f"{page['source']}:p{page['page']}:c{chunk_idx}",
            })
            start += chunk_size - chunk_overlap
            chunk_idx += 1

    return chunks
