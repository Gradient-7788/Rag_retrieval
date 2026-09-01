import numpy as np
from typing import List, Dict
import re
from rank_bm25 import BM25Okapi
from rag_engine.chunking.base import Chunk
from rag_engine.embeddings.base import BaseEmbeddingModel
from rag_engine.retrieval.base import BaseRetriever, RetrievalResult

class DenseRetriever(BaseRetriever):
    def __init__(self, embedding_model: BaseEmbeddingModel):
        self.embedding_model = embedding_model
        self.chunks: List[Chunk] = []
        self.chunk_embeddings: np.ndarray = None

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            self.chunk_embeddings = np.empty((0, self.embedding_model.dimension))
            return
        texts = [c.text for c in chunks]
        self.chunk_embeddings = self.embedding_model.embed_texts(texts)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.chunks or self.chunk_embeddings is None or len(self.chunks) == 0:
            return []

        query_vec = self.embedding_model.embed_query(query)
        # Cosine similarity (since normalized)
        scores = np.dot(self.chunk_embeddings, query_vec)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return results

class BM25Retriever(BaseRetriever):
    def __init__(self):
        self.chunks: List[Chunk] = []
        self.bm25: BM25Okapi = None

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        if not chunks:
            self.bm25 = None
            return
        corpus = [self._tokenize(c.text) for c in chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.chunks or self.bm25 is None:
            return []

        query_tokens = self._tokenize(query)
        scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(
                RetrievalResult(
                    chunk=self.chunks[idx],
                    score=float(scores[idx]),
                    rank=rank,
                )
            )
        return results

class HybridRetriever(BaseRetriever):
    """
    Hybrid Retriever supporting:
    1. Reciprocal Rank Fusion (RRF) with constant k (default k=60).
    2. Weighted Convex Combination of normalized dense and BM25 scores.
    """
    def __init__(
        self,
        dense_retriever: DenseRetriever,
        bm25_retriever: BM25Retriever,
        fusion_mode: str = "rrf",  # 'rrf' or 'weighted'
        alpha: float = 0.5,         # Weight for dense in 'weighted' mode (0.0 to 1.0)
        rrf_k: int = 60,
    ):
        self.dense_retriever = dense_retriever
        self.bm25_retriever = bm25_retriever
        self.fusion_mode = fusion_mode
        self.alpha = alpha
        self.rrf_k = rrf_k
        self.chunks: List[Chunk] = []

    def index(self, chunks: List[Chunk]):
        self.chunks = chunks
        self.dense_retriever.index(chunks)
        self.bm25_retriever.index(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        if not self.chunks:
            return []

        fetch_k = max(top_k * 3, 20)
        dense_res = self.dense_retriever.retrieve(query, top_k=fetch_k)
        bm25_res = self.bm25_retriever.retrieve(query, top_k=fetch_k)

        chunk_map: Dict[str, Chunk] = {c.chunk_id: c for c in self.chunks}
        fused_scores: Dict[str, float] = {}

        if self.fusion_mode == "rrf":
            for rank, r in enumerate(dense_res, start=1):
                cid = r.chunk.chunk_id
                fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
            for rank, r in enumerate(bm25_res, start=1):
                cid = r.chunk.chunk_id
                fused_scores[cid] = fused_scores.get(cid, 0.0) + (1.0 / (self.rrf_k + rank))
        else:
            # Weighted min-max score normalization
            def normalize_scores(res_list):
                if not res_list:
                    return {}
                scs = [r.score for r in res_list]
                min_s, max_s = min(scs), max(scs)
                rng = (max_s - min_s) if (max_s - min_s) > 1e-9 else 1.0
                return {r.chunk.chunk_id: (r.score - min_s) / rng for r in res_list}

            dense_norm = normalize_scores(dense_res)
            bm25_norm = normalize_scores(bm25_res)

            all_ids = set(dense_norm.keys()) | set(bm25_norm.keys())
            for cid in all_ids:
                d_score = dense_norm.get(cid, 0.0)
                b_score = bm25_norm.get(cid, 0.0)
                fused_scores[cid] = self.alpha * d_score + (1.0 - self.alpha) * b_score

        # Sort combined chunks by score descending
        sorted_items = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for rank, (cid, score) in enumerate(sorted_items, start=1):
            results.append(
                RetrievalResult(
                    chunk=chunk_map[cid],
                    score=float(score),
                    rank=rank,
                )
            )
        return results

class Reranker:
    """
    Reranker using cross-encoder scoring logic or fine-grained word/semantic overlap reranking.
    """
    def __init__(self, reranker_model_name: str = "ms-marco-MiniLM-L-6-v2"):
        self.model_name = reranker_model_name

    def rerank(self, query: str, results: List[RetrievalResult], top_n: int = 5) -> List[RetrievalResult]:
        if not results:
            return []

        query_words = set(re.findall(r'\w+', query.lower()))
        reranked = []

        for r in results:
            text_words = set(re.findall(r'\w+', r.chunk.text.lower()))
            overlap = len(query_words.intersection(text_words)) / max(1, len(query_words))
            # Blend initial retrieval score with lexical exact-match overlap score
            new_score = 0.6 * r.score + 0.4 * overlap
            reranked.append((r.chunk, new_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        final_n = reranked[:top_n]

        output = []
        for rank, (chunk, score) in enumerate(final_n, start=1):
            output.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank,
                )
            )
        return output
