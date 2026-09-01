from typing import List, Dict, Tuple
from rag_engine.evaluator import BenchmarkQuery

class BenchmarkDatasetLoader:
    @staticmethod
    def get_sample_tech_corpus() -> Tuple[List[Dict[str, str]], List[BenchmarkQuery]]:
        docs = [
            {
                "doc_id": "doc_rag_architecture",
                "text": "Retrieval-Augmented Generation (RAG) optimizes LLM outputs by retrieving relevant external context before generating responses. It uses dense vector representations, BM25 keyword matching, or hybrid search combining both using Reciprocal Rank Fusion (RRF). Reranking models like Cross-Encoders re-score retrieved candidates for optimal context precision."
            },
            {
                "doc_id": "doc_vector_embeddings",
                "text": "Vector embeddings map high-dimensional textual tokens into dense numerical vectors. Sentence-transformer models like bge-small-en-v1.5 and all-MiniLM-L6-v2 capture semantic similarity through cosine distance. HNSW (Hierarchical Navigable Small World) graphs accelerate approximate nearest neighbor search."
            },
            {
                "doc_id": "doc_chunking_strategies",
                "text": "Chunking strategies slice documents into readable text blocks. Fixed-size chunking uses fixed character windows with overlap. Sentence chunking respects sentence boundaries. Parent-Child chunking retrieves small child chunks while injecting surrounding parent chunk context into the LLM prompt."
            },
            {
                "doc_id": "doc_query_transformation",
                "text": "Query rewriting techniques transform user queries to improve retrieval recall. HyDE (Hypothetical Document Embeddings) generates a pseudo-answer to search vector space. Step-Back prompting abstracts specific questions to high-level concept queries."
            },
            {
                "doc_id": "doc_llm_benchmarking",
                "text": "RAG evaluation measures retrieval metrics (Recall@K, Precision@K, MRR, NDCG@K), generation metrics (Faithfulness, Answer Relevance, Context Relevance, Citation Accuracy), and system metrics (Latency p50/p90/p99, Token Cost, QPS, Memory RSS)."
            }
        ]

        queries = [
            BenchmarkQuery(
                query_id="q1",
                query="How does hybrid search combine dense and sparse retrieval in RAG?",
                ground_truth_doc_ids=["doc_rag_architecture"],
                target_answer="Hybrid search combines dense vector search and BM25 keyword matching using Reciprocal Rank Fusion (RRF)."
            ),
            BenchmarkQuery(
                query_id="q2",
                query="What is the purpose of Parent-Child chunking in document processing?",
                ground_truth_doc_ids=["doc_chunking_strategies"],
                target_answer="Parent-Child chunking retrieves fine-grained child chunks while providing large parent context to LLMs."
            ),
            BenchmarkQuery(
                query_id="q3",
                query="What is HyDE query rewriting and how does it work?",
                ground_truth_doc_ids=["doc_query_transformation"],
                target_answer="HyDE generates a hypothetical document answer and uses its embedding to retrieve relevant passages."
            ),
            BenchmarkQuery(
                query_id="q4",
                query="Which metrics are used for generation quality and system performance in RAG evaluation?",
                ground_truth_doc_ids=["doc_llm_benchmarking"],
                target_answer="Generation quality uses Faithfulness, Answer Relevance, Context Relevance, Citation Accuracy. System performance tracks Latency, Token Cost, QPS, Memory."
            ),
            BenchmarkQuery(
                query_id="q5",
                query="How do HNSW graphs accelerate vector similarity search for embedding models?",
                ground_truth_doc_ids=["doc_vector_embeddings"],
                target_answer="HNSW graphs accelerate approximate nearest neighbor search for high-dimensional vector embeddings."
            )
        ]

        return docs, queries
