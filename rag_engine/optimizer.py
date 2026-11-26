from typing import List, Dict, Any
import itertools
from pydantic import BaseModel
from rag_engine.pipeline import PipelineConfig, RAGPipeline
from rag_engine.evaluator import RAGEvaluator, EvaluationReport, BenchmarkQuery

class SweepSpace(BaseModel):
    """Grid of PipelineConfig fields to sweep. The default is the 18-config retrieval-tuning grid:
    2 retrievers x 3 query-expansion strategies x 3 retrieval depths (fetch_k), no reranker."""
    retriever_types: List[str] = ["dense", "hybrid"]
    query_rewriter_types: List[str] = ["none", "multi_query", "keyword"]
    fetch_k_options: List[int] = [5, 10, 20]
    enable_reranker_options: List[bool] = [False]
    chunk_sizes: List[int] = [500]
    top_k_options: List[int] = [5]
    embedding_model_name: str = "tfidf"
    reranker_type: str = "lexical"

class OptimizationReport(BaseModel):
    total_configs_evaluated: int
    reports: List[EvaluationReport]
    best_config_by_ndcg: EvaluationReport
    best_config_by_composite: EvaluationReport
    pareto_frontier_configs: List[EvaluationReport]

class RAGOptimizer:
    def __init__(self, documents: List[Dict[str, str]], benchmark_queries: List[BenchmarkQuery]):
        self.documents = documents
        self.benchmark_queries = benchmark_queries

    def run_sweep(self, sweep_space: SweepSpace, max_evals: int = None, seed: int = 42) -> OptimizationReport:
        keys = ["retriever_type", "query_rewriter_type", "fetch_k", "enable_reranker", "chunk_size", "top_k"]
        grid = list(itertools.product(sweep_space.retriever_types, sweep_space.query_rewriter_types,
                                      sweep_space.fetch_k_options, sweep_space.enable_reranker_options,
                                      sweep_space.chunk_sizes, sweep_space.top_k_options))
        if max_evals is not None and max_evals < len(grid):
            # Seeded random subsample instead of truncating the grid (truncation would only ever
            # explore the first values of the outer loops).
            import random
            grid = random.Random(seed).sample(grid, max_evals)

        reports: List[EvaluationReport] = []
        for idx, combo in enumerate(grid, start=1):
            params = dict(zip(keys, combo))
            cfg = PipelineConfig(
                name=f"#{idx} {params['retriever_type']}|{params['query_rewriter_type']}|fetch{params['fetch_k']}"
                     f"|rerank={params['enable_reranker']}",
                embedding_model_name=sweep_space.embedding_model_name,
                reranker_type=sweep_space.reranker_type,
                **params,
            )
            pipeline = RAGPipeline(cfg)
            pipeline.index_documents(self.documents)
            reports.append(RAGEvaluator(pipeline).evaluate(self.benchmark_queries))

        # Find best by NDCG
        best_ndcg = max(reports, key=lambda x: x.ndcg_at_k)
        best_composite = max(reports, key=lambda x: x.composite_quality_score)

        # Compute Pareto Frontier (Quality vs Latency vs Cost)
        pareto = self._compute_pareto_frontier(reports)

        return OptimizationReport(
            total_configs_evaluated=len(reports),
            reports=reports,
            best_config_by_ndcg=best_ndcg,
            best_config_by_composite=best_composite,
            pareto_frontier_configs=pareto,
        )

    def _compute_pareto_frontier(self, reports: List[EvaluationReport]) -> List[EvaluationReport]:
        """
        Pareto non-dominated sorting maximizing Composite Quality Score and minimizing Latency p50.
        """
        pareto = []
        for r1 in reports:
            dominated = False
            for r2 in reports:
                if r1 == r2:
                    continue
                # r2 dominates r1 if r2 has >= quality AND <= latency, with at least one strictly better
                if (r2.composite_quality_score >= r1.composite_quality_score and
                    r2.system_telemetry.latency_p50_ms <= r1.system_telemetry.latency_p50_ms and
                    (r2.composite_quality_score > r1.composite_quality_score or r2.system_telemetry.latency_p50_ms < r1.system_telemetry.latency_p50_ms)):
                    dominated = True
                    break
            if not dominated:
                pareto.append(r1)
        return pareto
