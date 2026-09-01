from typing import List, Dict, Any
import itertools
from pydantic import BaseModel
from rag_engine.pipeline import PipelineConfig, RAGPipeline
from rag_engine.evaluator import RAGEvaluator, EvaluationReport, BenchmarkQuery

class SweepSpace(BaseModel):
    chunk_sizes: List[int] = [250, 500, 1000]
    retriever_types: List[str] = ["dense", "bm25", "hybrid"]
    hybrid_alphas: List[float] = [0.3, 0.5, 0.7]
    query_rewriter_types: List[str] = ["none", "hyde", "multi_query"]
    enable_reranker_options: List[bool] = [True, False]
    top_k_options: List[int] = [3, 5]

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

    def run_sweep(self, sweep_space: SweepSpace, max_evals: int = 10) -> OptimizationReport:
        # Generate grid combinations
        keys = ["chunk_size", "retriever_type", "hybrid_alpha", "query_rewriter_type", "enable_reranker", "top_k"]
        values = [
            sweep_space.chunk_sizes,
            sweep_space.retriever_types,
            sweep_space.hybrid_alphas,
            sweep_space.query_rewriter_types,
            sweep_space.enable_reranker_options,
            sweep_space.top_k_options,
        ]

        all_combinations = list(itertools.product(*values))
        # Limit max evals if combination space is huge
        selected_combinations = all_combinations[:max_evals]

        reports: List[EvaluationReport] = []

        for idx, (cs, rt, ha, qr, er, tk) in enumerate(selected_combinations, start=1):
            cfg = PipelineConfig(
                name=f"Config #{idx} ({rt}-cs{cs}-ha{ha}-qr_{qr}-rerank_{er})",
                chunk_size=cs,
                retriever_type=rt,
                hybrid_alpha=ha,
                query_rewriter_type=qr,
                enable_reranker=er,
                top_k=tk,
            )

            pipeline = RAGPipeline(cfg)
            pipeline.index_documents(self.documents)

            evaluator = RAGEvaluator(pipeline)
            report = evaluator.evaluate(self.benchmark_queries)
            reports.append(report)

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
