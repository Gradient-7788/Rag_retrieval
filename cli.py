import argparse
import json

from rag_engine import PipelineConfig, RAGPipeline, RAGEvaluator, RAGOptimizer, SweepSpace
from rag_engine.datasets import BenchmarkDatasetLoader

def main():
    parser = argparse.ArgumentParser(description="RAG Benchmarking & Optimization Engine CLI")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Run benchmark evaluation")
    eval_parser.add_argument("--retriever", type=str, default="hybrid", choices=["dense", "bm25", "hybrid"])
    eval_parser.add_argument("--chunker", type=str, default="fixed_window", choices=["fixed_window", "sentence", "recursive", "parent_child", "semantic"])
    eval_parser.add_argument("--rewriter", type=str, default="none", choices=["none", "hyde", "multi_query", "step_back", "keyword"])
    eval_parser.add_argument("--reranker", action="store_true", help="Enable reranking stage")

    # Compare command
    subparsers.add_parser("compare", help="Compare all baseline architectures side-by-side")

    # Sweep command
    sweep_parser = subparsers.add_parser("sweep", help="Run hyperparameter optimization sweep")
    sweep_parser.add_argument("--evals", type=int, default=8, help="Number of evaluations to run")

    args = parser.parse_args()

    docs, queries = BenchmarkDatasetLoader.get_sample_tech_corpus()

    if args.command == "evaluate" or args.command is None:
        cfg = PipelineConfig(
            name="CLI Pipeline Run",
            chunker_type=args.chunker if hasattr(args, 'chunker') else "fixed_window",
            retriever_type=args.retriever if hasattr(args, 'retriever') else "hybrid",
            query_rewriter_type=args.rewriter if hasattr(args, 'rewriter') else "none",
            enable_reranker=args.reranker if hasattr(args, 'reranker') else True,
        )
        print(f"\n=======================================================")
        print(f" RAG BENCHMARKING ENGINE: EVALUATING '{cfg.name}'")
        print(f"=======================================================\n")
        pipeline = RAGPipeline(cfg)
        pipeline.index_documents(docs)
        evaluator = RAGEvaluator(pipeline)
        report = evaluator.evaluate(queries)

        print(json.dumps(report.model_dump(), indent=2))

    elif args.command == "compare":
        print(f"\n=======================================================")
        print(f" RAG BENCHMARKING ENGINE: COMPARATIVE BASELINE SWEEP")
        print(f"=======================================================\n")

        configs = [
            PipelineConfig(name="Dense Vector (bge-small)", retriever_type="dense", enable_reranker=False),
            PipelineConfig(name="Sparse Keyword (BM25)", retriever_type="bm25", enable_reranker=False),
            PipelineConfig(name="Hybrid RRF (Dense + BM25)", retriever_type="hybrid", enable_reranker=False),
            PipelineConfig(name="Hybrid + Reranker", retriever_type="hybrid", enable_reranker=True),
            PipelineConfig(name="Hybrid + Reranker + HyDE", retriever_type="hybrid", enable_reranker=True, query_rewriter_type="hyde"),
        ]

        print(f"{'Strategy Name':<32} | {'NDCG@5':<8} | {'Recall@5':<8} | {'Faithful':<8} | {'P50 (ms)':<8} | {'Score':<6}")
        print("-" * 82)

        for cfg in configs:
            pipeline = RAGPipeline(cfg)
            pipeline.index_documents(docs)
            evaluator = RAGEvaluator(pipeline)
            rep = evaluator.evaluate(queries)
            print(f"{rep.pipeline_name:<32} | {rep.ndcg_at_k:<8.4f} | {rep.recall_at_k:<8.4f} | {rep.faithfulness:<8.4f} | {rep.system_telemetry.latency_p50_ms:<8.2f} | {rep.composite_quality_score:<6.4f}")

        print("-" * 82)

    elif args.command == "sweep":
        print(f"\n=======================================================")
        print(f" RAG BENCHMARKING ENGINE: HYPERPARAMETER SWEEP")
        print(f"=======================================================\n")
        optimizer = RAGOptimizer(docs, queries)
        sweep_report = optimizer.run_sweep(SweepSpace(), max_evals=args.evals)
        print(f"Evaluated {sweep_report.total_configs_evaluated} configurations.")
        print(f"Best Config by NDCG: {sweep_report.best_config_by_ndcg.pipeline_name} (NDCG: {sweep_report.best_config_by_ndcg.ndcg_at_k})")
        print(f"Best Config by Composite Score: {sweep_report.best_config_by_composite.pipeline_name} (Score: {sweep_report.best_config_by_composite.composite_quality_score})")
        print(f"Pareto Frontier count: {len(sweep_report.pareto_frontier_configs)} configs.")

if __name__ == "__main__":
    main()
