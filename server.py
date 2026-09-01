from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import os

from rag_engine import PipelineConfig, RAGPipeline, RAGEvaluator, BenchmarkQuery, RAGOptimizer, SweepSpace
from rag_engine.datasets import BenchmarkDatasetLoader, SyntheticDatasetGenerator

app = FastAPI(title="RAG Benchmarking & Optimization Engine API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global in-memory dataset state
active_documents, active_queries = BenchmarkDatasetLoader.get_sample_tech_corpus()

class SingleQueryRequest(BaseModel):
    query: str
    pipeline_config: PipelineConfig

class EvaluateRequest(BaseModel):
    pipeline_config: PipelineConfig
    custom_documents: Optional[List[Dict[str, str]]] = None
    custom_queries: Optional[List[BenchmarkQuery]] = None

class SweepRequest(BaseModel):
    sweep_space: Optional[SweepSpace] = None
    max_evals: int = 8
    custom_documents: Optional[List[Dict[str, str]]] = None
    custom_queries: Optional[List[BenchmarkQuery]] = None

class UploadDatasetRequest(BaseModel):
    documents: List[Dict[str, str]]
    generate_synthetic_queries: bool = True

@app.get("/api/health")
def health_check():
    return {"status": "ok", "documents_loaded": len(active_documents), "queries_loaded": len(active_queries)}

@app.get("/api/dataset")
def get_dataset():
    return {
        "documents": active_documents,
        "benchmark_queries": [q.model_dump() for q in active_queries]
    }

@app.post("/api/dataset/upload")
def upload_dataset(req: UploadDatasetRequest):
    global active_documents, active_queries
    active_documents = req.documents
    if req.generate_synthetic_queries:
        generator = SyntheticDatasetGenerator()
        active_queries = generator.generate_from_documents(active_documents)
    return {"status": "success", "documents_count": len(active_documents), "queries_count": len(active_queries)}

@app.post("/api/query")
def run_single_query(req: SingleQueryRequest):
    pipeline = RAGPipeline(req.pipeline_config)
    pipeline.index_documents(active_documents)
    result = pipeline.query(req.query)
    return result.model_dump()

@app.post("/api/evaluate")
def evaluate_pipeline(req: EvaluateRequest):
    docs = req.custom_documents if req.custom_documents else active_documents
    queries = req.custom_queries if req.custom_queries else active_queries

    pipeline = RAGPipeline(req.pipeline_config)
    pipeline.index_documents(docs)

    evaluator = RAGEvaluator(pipeline)
    report = evaluator.evaluate(queries)
    return report.model_dump()

@app.get("/api/comparison")
def compare_baseline_strategies():
    """Runs out-of-the-box comparative benchmark across key RAG architecture choices."""
    docs, queries = active_documents, active_queries

    configs = [
        PipelineConfig(name="1. Dense Vector (bge-small)", retriever_type="dense", enable_reranker=False, query_rewriter_type="none"),
        PipelineConfig(name="2. Sparse Keyword (BM25)", retriever_type="bm25", enable_reranker=False, query_rewriter_type="none"),
        PipelineConfig(name="3. Hybrid RRF (Dense + BM25)", retriever_type="hybrid", fusion_mode="rrf", enable_reranker=False, query_rewriter_type="none"),
        PipelineConfig(name="4. Hybrid + Reranker", retriever_type="hybrid", fusion_mode="rrf", enable_reranker=True, rerank_top_n=3, query_rewriter_type="none"),
        PipelineConfig(name="5. Hybrid + Rerank + HyDE", retriever_type="hybrid", fusion_mode="rrf", enable_reranker=True, rerank_top_n=3, query_rewriter_type="hyde"),
        PipelineConfig(name="6. Parent-Child + Hybrid + MultiQuery", chunker_type="parent_child", retriever_type="hybrid", enable_reranker=True, query_rewriter_type="multi_query"),
    ]

    reports = []
    for cfg in configs:
        pipeline = RAGPipeline(cfg)
        pipeline.index_documents(docs)
        evaluator = RAGEvaluator(pipeline)
        reports.append(evaluator.evaluate(queries).model_dump())

    return {"comparison_reports": reports}

@app.post("/api/sweep")
def run_hyperparameter_sweep(req: SweepRequest):
    docs = req.custom_documents if req.custom_documents else active_documents
    queries = req.custom_queries if req.custom_queries else active_queries

    sweep = req.sweep_space if req.sweep_space else SweepSpace()
    optimizer = RAGOptimizer(docs, queries)
    opt_report = optimizer.run_sweep(sweep, max_evals=req.max_evals)
    return opt_report.model_dump()

# Serve Frontend static assets if available
dist_dir = os.path.join(os.path.dirname(__file__), "frontend", "dist")
if os.path.exists(dist_dir):
    app.mount("/", StaticFiles(directory=dist_dir, html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
