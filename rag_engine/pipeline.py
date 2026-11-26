import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from rag_engine.chunking.base import BaseChunker, Chunk
from rag_engine.chunking import FixedWindowChunker, SentenceChunker, RecursiveChunker, ParentChildChunker, SemanticChunker
from rag_engine.embeddings import BaseEmbeddingModel, SimulatedEmbeddingModel, TFIDFEmbeddingModel, SentenceTransformersEmbeddingModel
from rag_engine.retrieval import BaseRetriever, DenseRetriever, BM25Retriever, HybridRetriever, LexicalOverlapReranker, CrossEncoderReranker, RetrievalResult
from rag_engine.query_rewriting import BaseQueryRewriter, DirectQueryRewriter, HyDEQueryRewriter, MultiQueryRewriter, StepBackQueryRewriter, KeywordExpansionRewriter
from rag_engine.generation import BaseLLMGenerator, MockLLMGenerator, OpenAICompatibleLLMGenerator, LLMResponse

class PipelineConfig(BaseModel):
    name: str = "Standard RAG Config"
    chunker_type: str = "fixed_window"  # fixed_window, sentence, recursive, parent_child, semantic
    chunk_size: int = 500
    chunk_overlap: int = 50
    # "tfidf" (TF-IDF + truncated SVD, offline), "st-<hf model>" (sentence-transformers, e.g.
    # "st-BAAI/bge-small-en-v1.5"), or "hashed" (bag-of-words feature hashing, smoke test only)
    embedding_model_name: str = "tfidf"
    retriever_type: str = "hybrid"  # dense, bm25, hybrid
    fusion_mode: str = "rrf"  # rrf, weighted
    hybrid_alpha: float = 0.5
    top_k: int = 5            # passages returned to the LLM (and the K in NDCG@K / Recall@K)
    fetch_k: int = 10         # candidates retrieved per (rewritten) query before merging / reranking
    enable_reranker: bool = True
    reranker_type: str = "lexical"   # "lexical" (offline) or "cross_encoder" (ms-marco-MiniLM-L-6-v2)
    query_rewriter_type: str = "none"  # none, hyde, multi_query, step_back, keyword
    llm_backend: str = "mock"        # "mock" (offline template) or "openai" (OPENAI_API_KEY / OPENAI_BASE_URL)
    llm_model_name: str = "gpt-4o-mini"

class RAGPipelineResult(BaseModel):
    query: str
    rewritten_queries: List[str]
    retrieved_chunks: List[RetrievalResult]
    llm_response: LLMResponse
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    pipeline_config: PipelineConfig

class RAGPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config

        # Instantiate Chunker
        if config.chunker_type == "sentence":
            self.chunker = SentenceChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        elif config.chunker_type == "recursive":
            self.chunker = RecursiveChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        elif config.chunker_type == "parent_child":
            self.chunker = ParentChildChunker(parent_chunk_size=config.chunk_size * 2, child_chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        elif config.chunker_type == "semantic":
            self.chunker = SemanticChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        else:
            self.chunker = FixedWindowChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)

        # Instantiate Embedding Model
        if config.embedding_model_name == "tfidf":
            self.embedding_model = TFIDFEmbeddingModel()
        elif config.embedding_model_name.startswith("st-"):
            self.embedding_model = SentenceTransformersEmbeddingModel(config.embedding_model_name[3:])
        elif config.embedding_model_name == "hashed":
            self.embedding_model = SimulatedEmbeddingModel(model_name="hashed-bow-384")
        else:
            raise ValueError(f"Unknown embedding_model_name: {config.embedding_model_name}")

        # Instantiate Retriever
        self.dense_retriever = DenseRetriever(self.embedding_model)
        self.bm25_retriever = BM25Retriever()

        if config.retriever_type == "dense":
            self.retriever = self.dense_retriever
        elif config.retriever_type == "bm25":
            self.retriever = self.bm25_retriever
        else:
            self.retriever = HybridRetriever(
                dense_retriever=self.dense_retriever,
                bm25_retriever=self.bm25_retriever,
                fusion_mode=config.fusion_mode,
                alpha=config.hybrid_alpha,
            )

        # Reranker
        self.reranker = None
        if config.enable_reranker:
            self.reranker = CrossEncoderReranker() if config.reranker_type == "cross_encoder" else LexicalOverlapReranker()

        # LLM Generator
        if config.llm_backend == "openai":
            import os
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                raise ValueError("llm_backend='openai' needs OPENAI_API_KEY")
            self.llm_generator = OpenAICompatibleLLMGenerator(model_name=config.llm_model_name, api_key=key,
                                                              base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
        else:
            self.llm_generator = MockLLMGenerator(model_name="mock-llm")

        # Query Rewriter
        if config.query_rewriter_type == "hyde":
            self.query_rewriter = HyDEQueryRewriter(llm_generator=self.llm_generator)
        elif config.query_rewriter_type == "multi_query":
            self.query_rewriter = MultiQueryRewriter(llm_generator=self.llm_generator,
                                                     use_llm=config.llm_backend != "mock")
        elif config.query_rewriter_type == "step_back":
            self.query_rewriter = StepBackQueryRewriter()
        elif config.query_rewriter_type == "keyword":
            self.query_rewriter = KeywordExpansionRewriter()
        else:
            self.query_rewriter = DirectQueryRewriter()

        self.indexed_chunks: List[Chunk] = []

    def index_documents(self, documents: List[Dict[str, str]]):
        """Documents format: [{'doc_id': 'd1', 'text': '...'}]"""
        all_chunks = []
        for doc in documents:
            doc_id = doc.get("doc_id", "doc_0")
            text = doc.get("text", "")
            chunks = self.chunker.chunk(text, doc_id=doc_id)
            all_chunks.extend(chunks)

        self.indexed_chunks = all_chunks
        self.retriever.index(all_chunks)

    def query(self, query_text: str) -> RAGPipelineResult:
        t_start = time.perf_counter()

        # 1. Query Rewriting
        rewritten_queries = self.query_rewriter.rewrite(query_text)

        # 2. Retrieval: fetch_k candidates per rewritten query, merged with Reciprocal Rank Fusion
        #    across query variants (rank-based, so per-variant score scales don't matter).
        t_ret_start = time.perf_counter()
        fused: Dict[str, float] = {}
        chunk_by_id: Dict[str, Chunk] = {}
        for q in rewritten_queries:
            for r in self.retriever.retrieve(q, top_k=self.config.fetch_k):
                cid = r.chunk.chunk_id
                chunk_by_id[cid] = r.chunk
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (60 + r.rank)
        ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)
        candidates = [RetrievalResult(chunk=chunk_by_id[c], score=sc, rank=i)
                      for i, (c, sc) in enumerate(ranked, start=1)]

        # 3. Reranking (scores the candidates against the ORIGINAL query), then keep top_k
        if self.reranker and candidates:
            top_retrieved = self.reranker.rerank(query_text, candidates, top_n=self.config.top_k)
        else:
            top_retrieved = candidates[: self.config.top_k]

        t_ret_end = time.perf_counter()
        retrieval_latency_ms = (t_ret_end - t_ret_start) * 1000.0

        # 4. Generation
        t_gen_start = time.perf_counter()
        llm_resp = self.llm_generator.generate_with_context(query_text, top_retrieved)
        t_gen_end = time.perf_counter()
        generation_latency_ms = (t_gen_end - t_gen_start) * 1000.0

        t_end = time.perf_counter()
        total_latency_ms = (t_end - t_start) * 1000.0

        return RAGPipelineResult(
            query=query_text,
            rewritten_queries=rewritten_queries,
            retrieved_chunks=top_retrieved,
            llm_response=llm_resp,
            retrieval_latency_ms=round(retrieval_latency_ms, 2),
            generation_latency_ms=round(generation_latency_ms, 2),
            total_latency_ms=round(total_latency_ms, 2),
            pipeline_config=self.config,
        )
