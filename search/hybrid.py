# /home/cydgxbriel/portfolio/docchat/search/hybrid.py
import numpy as np
from rank_bm25 import BM25Okapi
from langchain_openai import OpenAIEmbeddings
import faiss


class HybridSearch:
    """Hybrid search combining FAISS semantic search with BM25 keyword matching."""

    def __init__(self, semantic_weight: float = 0.7):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight
        self._chunks: list[dict] = []
        self._bm25: BM25Okapi | None = None
        self._faiss_index: faiss.IndexFlatIP | None = None

    def index(self, chunks: list[dict]) -> None:
        """Index chunks for both semantic and keyword search."""
        self._chunks = chunks
        texts = [c["text"] for c in chunks]

        tokenized = [t.lower().split() for t in texts]
        self._bm25 = BM25Okapi(tokenized)

        vectors = self.embeddings.embed_documents(texts)
        matrix = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(matrix)
        self._faiss_index = faiss.IndexFlatIP(matrix.shape[1])
        self._faiss_index.add(matrix)

    def query(self, query_text: str, k: int = 5) -> list[dict]:
        """Search using hybrid semantic + keyword scoring."""
        if not self._chunks or self._faiss_index is None:
            return []

        n = len(self._chunks)

        q_vec = np.array(
            [self.embeddings.embed_query(query_text)], dtype=np.float32
        )
        faiss.normalize_L2(q_vec)
        sem_scores, sem_indices = self._faiss_index.search(q_vec, n)
        sem_score_map = {
            int(idx): float(score)
            for idx, score in zip(sem_indices[0], sem_scores[0])
            if idx >= 0
        }

        bm25_scores = self._bm25.get_scores(query_text.lower().split())
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1.0
        bm25_norm = bm25_scores / max_bm25

        combined = []
        for i in range(n):
            sem = sem_score_map.get(i, 0.0)
            kw = float(bm25_norm[i])
            score = self.semantic_weight * sem + self.keyword_weight * kw
            combined.append((i, score))

        combined.sort(key=lambda x: x[1], reverse=True)

        results = []
        for idx, score in combined[:k]:
            result = {**self._chunks[idx], "score": round(score, 4)}
            results.append(result)

        return results
