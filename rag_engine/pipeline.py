import time
from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from rag_engine.chunking.base import BaseChunker, Chunk
from rag_engine.chunking import FixedWindowChunker, SentenceChunker, RecursiveChunker, ParentChildChunker, SemanticChunker
from rag_engine.embeddings import BaseEmbeddingModel, SimulatedEmbeddingModel, TFIDFEmbeddingModel, SentenceTransformersEmbeddingModel
from rag_engine.retrieval import BaseRetriever, DenseRetriever, BM25Retriever, HybridRetriever, Reranker, RetrievalResult
from rag_engine.query_rewriting import BaseQueryRewriter, DirectQueryRewriter, HyDEQueryRewriter, MultiQueryRewriter, StepBackQueryRewriter, KeywordExpansionRewriter
from rag_engine.generation import BaseLLMGenerator, MockLLMGenerator, OpenAICompatibleLLMGenerator, LLMResponse

class PipelineConfig(BaseModel):
    name: str = "Standard RAG Config"
    chunker_type: str = "fixed_window"  # fixed_window, sentence, recursive, parent_child, semantic
    chunk_size: int = 500
    chunk_overlap: int = 50
    embedding_model_name: str = "bge-small-en-v1.5"
    retriever_type: str = "hybrid"  # dense, bm25, hybrid
    fusion_mode: str = "rrf"  # rrf, weighted
    hybrid_alpha: float = 0.5
    top_k: int = 5
    enable_reranker: bool = True
    rerank_top_n: int = 3
    query_rewriter_type: str = "none"  # none, hyde, multi_query, step_back, keyword
    llm_model_name: str = "mock-gpt-4o-mini"

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
            self.embedding_model = SentenceTransformersEmbeddingModel(config.embedding_model_name.replace("st-", ""))
        else:
            self.embedding_model = SimulatedEmbeddingModel(model_name=config.embedding_model_name)

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
        self.reranker = Reranker() if config.enable_reranker else None

        # LLM Generator
        self.llm_generator = MockLLMGenerator(model_name=config.llm_model_name)

        # Query Rewriter
        if config.query_rewriter_type == "hyde":
            self.query_rewriter = HyDEQueryRewriter(llm_generator=self.llm_generator)
        elif config.query_rewriter_type == "multi_query":
            self.query_rewriter = MultiQueryRewriter()
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

        # 2. Retrieval
        t_ret_start = time.perf_counter()
        retrieved_map: Dict[str, RetrievalResult] = {}

        for q in rewritten_queries:
            res_list = self.retriever.retrieve(q, top_k=self.config.top_k)
            for r in res_list:
                cid = r.chunk.chunk_id
                if cid not in retrieved_map or r.score > retrieved_map[cid].score:
                    retrieved_map[cid] = r

        combined_retrieved = list(retrieved_map.values())
        combined_retrieved.sort(key=lambda x: x.score, reverse=True)
        top_retrieved = combined_retrieved[: self.config.top_k]

        # 3. Reranking
        if self.reranker and top_retrieved:
            top_retrieved = self.reranker.rerank(query_text, top_retrieved, top_n=self.config.rerank_top_n)

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
