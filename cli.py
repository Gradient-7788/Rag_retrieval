import argparse
import json

from rag_engine import PipelineConfig, RAGPipeline, RAGEvaluator, RAGOptimizer, SweepSpace
from rag_engine.datasets import BenchmarkDatasetLoader

COMPARE = [
    ("Sparse Keyword (BM25)", dict(retriever_type="bm25", enable_reranker=False)),
    ("Dense Vector", dict(retriever_type="dense", enable_reranker=False)),
    ("Hybrid RRF (Dense + BM25)", dict(retriever_type="hybrid", enable_reranker=False)),
    ("Hybrid + Reranker", dict(retriever_type="hybrid", enable_reranker=True)),
    ("Hybrid + Reranker + HyDE", dict(retriever_type="hybrid", enable_reranker=True, query_rewriter_type="hyde")),
]


def run(cfg, docs, queries):
    p = RAGPipeline(cfg)
    p.index_documents(docs)
    return RAGEvaluator(p).evaluate(queries)


def main():
    parser = argparse.ArgumentParser(description="RAG Benchmarking & Optimization Engine CLI")
    parser.add_argument("--embedding", default="tfidf",
                        help="tfidf (offline) | st-BAAI/bge-small-en-v1.5 (sentence-transformers) | hashed")
    parser.add_argument("--reranker-type", default="lexical", choices=["lexical", "cross_encoder"])
    parser.add_argument("--llm", default="mock", choices=["mock", "openai"], help="openai needs OPENAI_API_KEY")
    parser.add_argument("--fetch-k", type=int, default=10)
    parser.add_argument("--out", default=None, help="write JSON results to this file")
    sub = parser.add_subparsers(dest="command")

    ev = sub.add_parser("evaluate", help="Evaluate one pipeline")
    ev.add_argument("--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    ev.add_argument("--chunker", default="fixed_window", choices=["fixed_window", "sentence", "recursive", "parent_child", "semantic"])
    ev.add_argument("--rewriter", default="none", choices=["none", "hyde", "multi_query", "step_back", "keyword"])
    ev.add_argument("--reranker", action="store_true")
    sub.add_parser("compare", help="Compare the 5 baseline architectures")
    sw = sub.add_parser("sweep", help="Run the retrieval-tuning sweep (default 18 configs)")
    sw.add_argument("--evals", type=int, default=None, help="random subsample of the grid")
    args = parser.parse_args()

    docs, queries = BenchmarkDatasetLoader.get_sample_tech_corpus()
    common = dict(embedding_model_name=args.embedding, reranker_type=args.reranker_type,
                  llm_backend=args.llm, fetch_k=args.fetch_k)
    print(f"Benchmark: {len(docs)} documents, {len(queries)} queries "
          f"({sum(len(q.ground_truth_doc_ids) > 1 for q in queries)} multi-aspect) | embedding={args.embedding} "
          f"| reranker={args.reranker_type} | llm={args.llm}\n")
    out = None

    if args.command == "compare":
        rows = []
        print(f"{'Strategy':<28} | {'NDCG@5':>7} | {'Recall@5':>8} | {'MRR':>6} | {'P50 ms':>7} | {'P90 ms':>7}")
        print("-" * 78)
        for name, kw in COMPARE:
            rep = run(PipelineConfig(name=name, **common, **kw), docs, queries)
            t = rep.system_telemetry
            print(f"{name:<28} | {rep.ndcg_at_k:>7.4f} | {rep.recall_at_k:>8.4f} | {rep.mrr:>6.3f} | "
                  f"{t.latency_p50_ms:>7.2f} | {t.latency_p90_ms:>7.2f}")
            rows.append({"strategy": name, "ndcg@5": rep.ndcg_at_k, "recall@5": rep.recall_at_k, "mrr": rep.mrr,
                         "p50_ms": t.latency_p50_ms, "p90_ms": t.latency_p90_ms})
        out = {"compare": rows}

    elif args.command == "sweep":
        space = SweepSpace(embedding_model_name=args.embedding, reranker_type=args.reranker_type)
        rep = RAGOptimizer(docs, queries).run_sweep(space, max_evals=args.evals)
        rows = sorted(rep.reports, key=lambda r: -r.ndcg_at_k)
        for r in rows:
            print(f"{r.pipeline_name:<48} NDCG@5 {r.ndcg_at_k:.4f} | Recall@5 {r.recall_at_k:.4f} | "
                  f"P50 {r.system_telemetry.latency_p50_ms:.2f} ms")
        print(f"\nEvaluated {rep.total_configs_evaluated} configs. Best by NDCG@5: "
              f"{rep.best_config_by_ndcg.pipeline_name} ({rep.best_config_by_ndcg.ndcg_at_k}). "
              f"Pareto frontier: {len(rep.pareto_frontier_configs)} configs.")
        out = {"sweep": [{"config": r.pipeline_name, "ndcg@5": r.ndcg_at_k, "recall@5": r.recall_at_k,
                          "p50_ms": r.system_telemetry.latency_p50_ms} for r in rows]}

    else:
        cfg = PipelineConfig(name="CLI Pipeline Run", **common,
                             chunker_type=getattr(args, "chunker", "fixed_window"),
                             retriever_type=getattr(args, "retriever", "hybrid"),
                             query_rewriter_type=getattr(args, "rewriter", "none"),
                             enable_reranker=getattr(args, "reranker", False))
        rep = run(cfg, docs, queries)
        out = rep.model_dump()
        print(json.dumps(out, indent=2))

    if args.out and out is not None:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
