from typing import List, Dict, Tuple
from rag_engine.evaluator import BenchmarkQuery


class BenchmarkDatasetLoader:
    """Hand-built retrieval benchmark.

    14 documents: 10 are ground truth for at least one query and 4 are pure distractors
    (graph databases / small-world networks, relational parent-child tables, query autocomplete,
    network latency). Distractors deliberately share vocabulary with an answer document (HNSW,
    parent-child, query rewriting, p50/p99 latency) without answering the question. Two answer
    documents also act as distractors for other queries: MRR evaluation vs Reciprocal Rank Fusion,
    and product quantization vs chunking.

    12 queries: 5 single-aspect and 7 multi-aspect. Multi-aspect queries need two documents,
    so a retriever that only covers one aspect is penalised by Recall@5 and NDCG@5.
    Relevance is binary at document level.
    """

    @staticmethod
    def get_sample_tech_corpus() -> Tuple[List[Dict[str, str]], List[BenchmarkQuery]]:
        docs = [
            # ---------------- answer documents
            {"doc_id": "doc_rag_architecture",
             "text": "Retrieval-Augmented Generation (RAG) optimizes LLM outputs by retrieving relevant external context before generating responses. Hybrid search runs a dense vector retriever and a BM25 keyword retriever in parallel and merges the two ranked lists with Reciprocal Rank Fusion (RRF), which adds 1/(k + rank) for every list a passage appears in, with k usually set to 60. Because RRF only uses ranks, the very different score scales of BM25 and cosine similarity do not need to be normalized."},
            {"doc_id": "doc_vector_embeddings",
             "text": "Vector embeddings map text into dense numerical vectors. Bi-encoder sentence-transformer models such as bge-small-en-v1.5 and all-MiniLM-L6-v2 encode the query and each passage independently, so passage vectors can be precomputed and compared with cosine similarity. For large collections, HNSW (Hierarchical Navigable Small World) graphs provide approximate nearest neighbor search over these embeddings in logarithmic time instead of a brute-force scan."},
            {"doc_id": "doc_chunking_strategies",
             "text": "Chunking strategies slice documents into retrievable text blocks. Fixed-size chunking uses fixed character windows with overlap, while sentence chunking respects sentence boundaries. Parent-Child chunking indexes small child chunks for precise retrieval but injects the surrounding parent chunk into the LLM prompt for context. Smaller chunks improve precision but multiply the number of vectors that must be stored and searched."},
            {"doc_id": "doc_query_transformation",
             "text": "Query rewriting techniques transform user queries to improve retrieval recall. HyDE (Hypothetical Document Embeddings) asks an LLM to write a hypothetical answer passage and retrieves with the embedding of that passage, because answers look more like documents than questions do. Step-Back prompting instead abstracts a specific question into a more general concept question and retrieves background material for it."},
            {"doc_id": "doc_llm_benchmarking",
             "text": "RAG evaluation separates retrieval metrics such as Recall@K, Precision@K, MRR and NDCG@K from generation metrics such as faithfulness, answer relevance and citation accuracy, and from system metrics such as p50 and p99 latency, token cost and queries per second. Low retrieval scores point to a retriever problem, while high retrieval scores with low faithfulness point to a generation problem."},
            {"doc_id": "doc_bm25_scoring",
             "text": "BM25 scores a document by summing, over query terms, the inverse document frequency of the term multiplied by a saturated term frequency. The parameter k1, usually between 1.2 and 2.0, controls how quickly repeated occurrences of a term stop adding score, and b, usually 0.75, normalizes for document length relative to the average document length. BM25 is strong for exact identifiers that embeddings tend to blur."},
            {"doc_id": "doc_reranking_cross_encoders",
             "text": "A cross-encoder reranker feeds the query and a candidate passage through the transformer together, so attention can compare every query token with every passage token. This joint encoding is more accurate than comparing two independently computed bi-encoder vectors, but it requires one forward pass per candidate and cannot be precomputed, so cross-encoders only rerank the top 20 to 100 candidates and add tens of milliseconds per query."},
            {"doc_id": "doc_llm_hallucination",
             "text": "Hallucination occurs when a language model states facts that are not supported by its context. Faithfulness metrics split the generated answer into individual claims and check whether each claim is entailed by the retrieved passages, often with a natural language inference model or an LLM judge. Grounding the answer in cited passages reduces hallucination but only when retrieval actually surfaced the supporting evidence."},
            # ---------------- distractor documents
            {"doc_id": "doc_embedding_compression",
             "text": "Product quantization compresses embedding vectors by splitting each vector into sub-vectors and replacing every sub-vector with the id of its nearest centroid in a small codebook. A 384-dimensional float vector that needs 1,536 bytes can be stored in 48 bytes, which lets billions of vectors fit in memory at the cost of some recall. Reducing embedding dimensionality has a similar memory effect."},
            {"doc_id": "doc_hnsw_graph_databases",
             "text": "Graph databases store nodes and edges and answer traversal queries such as friends of friends. Some graph databases describe their adjacency layout as hierarchical, and navigable small world networks are studied in social network analysis because most people are connected through a handful of intermediaries. These graph structures model relationships between entities rather than embedding similarity search."},
            {"doc_id": "doc_parent_child_relational",
             "text": "In relational databases a parent-child relationship links a child table to a parent table through a foreign key. Deleting a parent row can cascade to its child rows, and an ORM may load child records lazily in chunks to limit memory. Parent-child hierarchies also appear in XML documents and organisational charts."},
            {"doc_id": "doc_mrr_evaluation",
             "text": "Mean Reciprocal Rank evaluates a ranked list by taking the reciprocal of the rank of the first relevant result and averaging it over queries, so a relevant document at rank 1 scores 1.0 and at rank 4 scores 0.25. Reciprocal rank is an evaluation metric that measures a single ranking; it does not combine or fuse several rankings."},
            {"doc_id": "doc_query_autocomplete",
             "text": "Search engines rewrite queries for spelling correction and autocomplete: a misspelled query is rewritten to the most likely intended query using edit distance and query logs, and popular completions are suggested as the user types. This kind of query rewriting targets typing errors and does not generate hypothetical documents."},
            {"doc_id": "doc_network_latency",
             "text": "Network engineers report latency percentiles such as p50, p95 and p99 because tail latency, not the average, determines user experience. Queries per second, packet loss and jitter are tracked on dashboards, and a p99 regression often points to garbage collection pauses or overloaded load balancers rather than to application logic."},
        ]

        q = BenchmarkQuery
        queries = [
            # single-aspect
            q(query_id="q1", query="How does hybrid search combine dense and sparse retrieval in RAG?",
              ground_truth_doc_ids=["doc_rag_architecture"]),
            q(query_id="q2", query="What is the purpose of parent-child chunking when building a RAG index?",
              ground_truth_doc_ids=["doc_chunking_strategies"]),
            q(query_id="q3", query="What is HyDE query rewriting and how does it work?",
              ground_truth_doc_ids=["doc_query_transformation"]),
            q(query_id="q4", query="How do HNSW graphs speed up nearest neighbor search over sentence embeddings?",
              ground_truth_doc_ids=["doc_vector_embeddings"]),
            q(query_id="q5", query="How is Mean Reciprocal Rank computed when evaluating a retriever?",
              ground_truth_doc_ids=["doc_mrr_evaluation"]),
            # multi-aspect (two relevant documents)
            q(query_id="q6", query="How does BM25 weight term frequency and document length, and how is it fused with dense results?",
              ground_truth_doc_ids=["doc_bm25_scoring", "doc_rag_architecture"]),
            q(query_id="q7", query="Why is a cross-encoder reranker more accurate than bi-encoder embeddings, and what latency does it add?",
              ground_truth_doc_ids=["doc_reranking_cross_encoders", "doc_vector_embeddings"]),
            q(query_id="q8", query="How do faithfulness metrics detect hallucination, and which retrieval metrics complement them?",
              ground_truth_doc_ids=["doc_llm_hallucination", "doc_llm_benchmarking"]),
            q(query_id="q9", query="How does chunk size affect the number of stored vectors, and how can vectors be compressed to save memory?",
              ground_truth_doc_ids=["doc_chunking_strategies", "doc_embedding_compression"]),
            q(query_id="q10", query="How does step-back prompting differ from HyDE, and which evaluation metric shows whether rewriting improved ranking?",
              ground_truth_doc_ids=["doc_query_transformation", "doc_llm_benchmarking"]),
            q(query_id="q11", query="Which index gives approximate nearest neighbor search over embeddings, and how does product quantization reduce memory?",
              ground_truth_doc_ids=["doc_vector_embeddings", "doc_embedding_compression"]),
            q(query_id="q12", query="Which latency percentiles and cost metrics should a RAG system track, and what does low faithfulness indicate?",
              ground_truth_doc_ids=["doc_llm_benchmarking", "doc_llm_hallucination"]),
        ]
        return docs, queries
