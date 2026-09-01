from typing import List, Dict, Any
from pydantic import BaseModel

from rag_engine.pipeline import RAGPipeline, PipelineConfig, RAGPipelineResult
from rag_engine.metrics.retrieval_metrics import recall_at_k, precision_at_k, mrr, hit_rate_at_k, ndcg_at_k
from rag_engine.metrics.generation_metrics import compute_faithfulness, compute_answer_relevance, compute_context_relevance, compute_citation_accuracy
from rag_engine.metrics.system_metrics import SystemTracker, SystemTelemetry

class BenchmarkQuery(BaseModel):
    query_id: str
    query: str
    ground_truth_doc_ids: List[str]
    ground_truth_chunk_ids: List[str] = []
    target_answer: str = None

class EvaluationReport(BaseModel):
    pipeline_name: str
    pipeline_config: PipelineConfig
    total_queries: int

    # Retrieval Metrics
    recall_at_k: float
    precision_at_k: float
    mrr: float
    ndcg_at_k: float
    hit_rate: float

    # Generation Metrics
    faithfulness: float
    answer_relevance: float
    context_relevance: float
    citation_accuracy: float

    # Overall Quality Index (composite 0.0 - 1.0 score)
    composite_quality_score: float

    # System Metrics
    system_telemetry: SystemTelemetry

class RAGEvaluator:
    def __init__(self, pipeline: RAGPipeline):
        self.pipeline = pipeline
        self.system_tracker = SystemTracker()

    def evaluate(self, benchmark_queries: List[BenchmarkQuery]) -> EvaluationReport:
        if not benchmark_queries:
            raise ValueError("No benchmark queries provided for evaluation.")

        recalls = []
        precisions = []
        mrrs = []
        ndcgs = []
        hit_rates = []

        faithfulness_scores = []
        answer_rel_scores = []
        context_rel_scores = []
        citation_acc_scores = []

        latencies_ms = []
        retrieval_latencies = []
        generation_latencies = []
        total_prompt_tok = 0
        total_comp_tok = 0
        total_cost = 0.0

        for b_q in benchmark_queries:
            res: RAGPipelineResult = self.pipeline.query(b_q.query)

            # Retrieved doc/chunk IDs
            retrieved_chunk_ids = [r.chunk.chunk_id for r in res.retrieved_chunks]
            retrieved_doc_ids = [r.chunk.doc_id for r in res.retrieved_chunks]

            gt_doc_ids = b_q.ground_truth_doc_ids
            # Use chunk ground truth if available, otherwise doc ground truth
            gt_ids = b_q.ground_truth_chunk_ids if b_q.ground_truth_chunk_ids else gt_doc_ids
            ret_ids = retrieved_chunk_ids if b_q.ground_truth_chunk_ids else retrieved_doc_ids

            k = self.pipeline.config.top_k

            # Calculate retrieval metrics
            recalls.append(recall_at_k(ret_ids, gt_ids, k))
            precisions.append(precision_at_k(ret_ids, gt_ids, k))
            mrrs.append(mrr(ret_ids, gt_ids))
            ndcgs.append(ndcg_at_k(ret_ids, gt_ids, k))
            hit_rates.append(hit_rate_at_k(ret_ids, gt_ids, k))

            # Calculate generation metrics
            faithfulness_scores.append(compute_faithfulness(res.llm_response.text, res.retrieved_chunks))
            answer_rel_scores.append(compute_answer_relevance(res.llm_response.text, b_q.query))
            context_rel_scores.append(compute_context_relevance(b_q.query, res.retrieved_chunks))
            citation_acc_scores.append(compute_citation_accuracy(res.llm_response.text, res.retrieved_chunks))

            # Latency and token accounting
            latencies_ms.append(res.total_latency_ms)
            retrieval_latencies.append(res.retrieval_latency_ms)
            generation_latencies.append(res.generation_latency_ms)
            total_prompt_tok += res.llm_response.prompt_tokens
            total_comp_tok += res.llm_response.completion_tokens
            total_cost += self.pipeline.llm_generator.compute_cost(res.llm_response.prompt_tokens, res.llm_response.completion_tokens)

        # Average metrics
        avg_recall = float(sum(recalls) / len(recalls))
        avg_precision = float(sum(precisions) / len(precisions))
        avg_mrr = float(sum(mrrs) / len(mrrs))
        avg_ndcg = float(sum(ndcgs) / len(ndcgs))
        avg_hit_rate = float(sum(hit_rates) / len(hit_rates))

        avg_faithfulness = float(sum(faithfulness_scores) / len(faithfulness_scores))
        avg_answer_rel = float(sum(answer_rel_scores) / len(answer_rel_scores))
        avg_context_rel = float(sum(context_rel_scores) / len(context_rel_scores))
        avg_citation_acc = float(sum(citation_acc_scores) / len(citation_acc_scores))

        # Composite score calculation (weighted harmonic mean)
        quality_score = round(
            0.35 * avg_ndcg + 0.25 * avg_recall + 0.20 * avg_faithfulness + 0.10 * avg_answer_rel + 0.10 * avg_citation_acc, 4
        )

        telemetry = self.system_tracker.profile_execution(
            latencies_ms=latencies_ms,
            retrieval_latencies_ms=retrieval_latencies,
            generation_latencies_ms=generation_latencies,
            total_prompt_tokens=total_prompt_tok,
            total_completion_tokens=total_comp_tok,
            estimated_cost=total_cost,
            num_queries=len(benchmark_queries),
        )

        return EvaluationReport(
            pipeline_name=self.pipeline.config.name,
            pipeline_config=self.pipeline.config,
            total_queries=len(benchmark_queries),
            recall_at_k=round(avg_recall, 4),
            precision_at_k=round(avg_precision, 4),
            mrr=round(avg_mrr, 4),
            ndcg_at_k=round(avg_ndcg, 4),
            hit_rate=round(avg_hit_rate, 4),
            faithfulness=round(avg_faithfulness, 4),
            answer_relevance=round(avg_answer_rel, 4),
            context_relevance=round(avg_context_rel, 4),
            citation_accuracy=round(avg_citation_acc, 4),
            composite_quality_score=quality_score,
            system_telemetry=telemetry,
        )
