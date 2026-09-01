import React, { useState } from 'react';
import { Play, Send, Cpu, Layers, Search, Sparkles, FileText, CheckCircle2, Clock, DollarSign } from 'lucide-react';

export default function WorkbenchView() {
  const [query, setQuery] = useState("How does hybrid search combine dense and sparse retrieval in RAG?");
  const [config, setConfig] = useState({
    name: "Workbench Custom Pipeline",
    chunker_type: "fixed_window",
    chunk_size: 500,
    chunk_overlap: 50,
    embedding_model_name: "bge-small-en-v1.5",
    retriever_type: "hybrid",
    fusion_mode: "rrf",
    hybrid_alpha: 0.5,
    top_k: 5,
    enable_reranker: true,
    rerank_top_n: 3,
    query_rewriter_type: "none",
    llm_model_name: "mock-gpt-4o-mini",
  });

  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const sampleQueries = [
    "How does hybrid search combine dense and sparse retrieval in RAG?",
    "What is the purpose of Parent-Child chunking in document processing?",
    "What is HyDE query rewriting and how does it work?",
    "Which metrics are used for generation quality and system performance in RAG evaluation?",
  ];

  const handleRunQuery = async () => {
    if (!query.strip && !query) return;
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, pipeline_config: config }),
      });
      if (!res.ok) throw new Error("Query execution failed");
      const data = await res.json();
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">Interactive RAG Pipeline Workbench</h2>
        <p className="text-xs text-slate-400 font-mono">Configure custom pipeline components and inspect intermediate chunk retrieval, query expansion & LLM outputs</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Panel: Configuration controls */}
        <div className="lg:col-span-4 space-y-4">
          <div className="glass-card p-5 rounded-xl border border-slate-800 space-y-4">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Layers className="w-4 h-4 text-cyan-400" />
              <span>Pipeline Architecture Controls</span>
            </h3>

            {/* Chunking Strategy */}
            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1">Chunking Strategy</label>
              <select
                value={config.chunker_type}
                onChange={(e) => setConfig({ ...config, chunker_type: e.target.value })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
              >
                <option value="fixed_window">Fixed Window (Character Overlap)</option>
                <option value="sentence">Sentence Boundary Chunker</option>
                <option value="recursive">Recursive Character Splitting</option>
                <option value="parent_child">Parent-Child Context Chunker</option>
                <option value="semantic">Semantic Topic Boundary Chunker</option>
              </select>
            </div>

            {/* Retrieval Strategy */}
            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1">Retrieval Engine</label>
              <select
                value={config.retriever_type}
                onChange={(e) => setConfig({ ...config, retriever_type: e.target.value })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
              >
                <option value="hybrid">Hybrid Search (RRF Vector + BM25)</option>
                <option value="dense">Dense Vector Search (Cosine)</option>
                <option value="bm25">Sparse BM25 Keyword Search</option>
              </select>
            </div>

            {/* Query Rewriting */}
            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1">Query Rewriting Strategy</label>
              <select
                value={config.query_rewriter_type}
                onChange={(e) => setConfig({ ...config, query_rewriter_type: e.target.value })}
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
              >
                <option value="none">Direct Query (No rewrite)</option>
                <option value="hyde">HyDE (Hypothetical Doc Embedding)</option>
                <option value="multi_query">Multi-Query Expansion (3 perspectives)</option>
                <option value="step_back">Step-Back Concept Abstraction</option>
                <option value="keyword">Keyword & Synonym Expansion</option>
              </select>
            </div>

            {/* Reranker Toggle */}
            <div className="flex items-center justify-between pt-2 border-t border-slate-800">
              <span className="text-xs font-mono text-slate-300">Cross-Encoder Reranking</span>
              <button
                onClick={() => setConfig({ ...config, enable_reranker: !config.enable_reranker })}
                className={`w-12 h-6 rounded-full transition-colors p-1 ${config.enable_reranker ? 'bg-cyan-500' : 'bg-slate-800'}`}
              >
                <div className={`w-4 h-4 rounded-full bg-white transition-transform ${config.enable_reranker ? 'translate-x-6' : 'translate-x-0'}`}></div>
              </button>
            </div>
          </div>

          {/* Quick Benchmark Queries */}
          <div className="glass-card p-4 rounded-xl border border-slate-800 space-y-2">
            <label className="text-xs font-mono text-slate-400 block">Sample Benchmark Queries</label>
            <div className="space-y-1.5">
              {sampleQueries.map((sq, idx) => (
                <button
                  key={idx}
                  onClick={() => setQuery(sq)}
                  className="w-full text-left text-xs text-slate-300 bg-slate-900/60 hover:bg-slate-800 p-2 rounded-lg border border-slate-800/80 transition-colors truncate"
                >
                  {sq}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right Panel: Query execution input & Deep-dive results */}
        <div className="lg:col-span-8 space-y-6">
          {/* Query Bar */}
          <div className="glass-card p-3 rounded-xl border border-slate-800 flex items-center space-x-3">
            <Search className="w-5 h-5 text-cyan-400 ml-2" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Enter test RAG query..."
              className="flex-1 bg-transparent border-none text-slate-100 text-sm focus:outline-none placeholder-slate-500 font-mono"
            />
            <button
              onClick={handleRunQuery}
              disabled={loading}
              className="flex items-center space-x-2 px-5 py-2.5 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-semibold text-xs transition-all shadow-lg shadow-cyan-500/20 disabled:opacity-50"
            >
              {loading ? (
                <span>Executing...</span>
              ) : (
                <>
                  <Send className="w-3.5 h-3.5" />
                  <span>Run Pipeline</span>
                </>
              )}
            </button>
          </div>

          {error && (
            <div className="p-4 bg-red-950/40 border border-red-800 rounded-xl text-red-300 font-mono text-xs">
              {error}
            </div>
          )}

          {/* Results display */}
          {result && (
            <div className="space-y-6">
              {/* Telemetry Bar */}
              <div className="grid grid-cols-4 gap-3">
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-mono">Total Latency</div>
                  <div className="text-base font-bold text-cyan-400 font-mono">{result.total_latency_ms} ms</div>
                </div>
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-mono">Retrieval Time</div>
                  <div className="text-base font-bold text-indigo-400 font-mono">{result.retrieval_latency_ms} ms</div>
                </div>
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-mono">Total Tokens</div>
                  <div className="text-base font-bold text-emerald-400 font-mono">{result.llm_response.total_tokens}</div>
                </div>
                <div className="bg-slate-900/80 p-3 rounded-lg border border-slate-800">
                  <div className="text-[10px] text-slate-400 font-mono">Citations Included</div>
                  <div className="text-base font-bold text-amber-400 font-mono">{result.llm_response.citations.length}</div>
                </div>
              </div>

              {/* Query Expansion Output */}
              {result.rewritten_queries && result.rewritten_queries.length > 1 && (
                <div className="glass-card p-4 rounded-xl border border-slate-800 space-y-2">
                  <h4 className="text-xs font-bold text-cyan-400 font-mono flex items-center space-x-2">
                    <Sparkles className="w-3.5 h-3.5" />
                    <span>Query Rewriting Output ({result.pipeline_config.query_rewriter_type})</span>
                  </h4>
                  <div className="space-y-1">
                    {result.rewritten_queries.map((rq, idx) => (
                      <div key={idx} className="text-xs text-slate-300 font-mono bg-slate-900/60 p-2 rounded border border-slate-800">
                        <span className="text-slate-500">[{idx + 1}]</span> {rq}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* LLM Generated Answer */}
              <div className="glass-card p-5 rounded-xl border border-slate-800 space-y-3">
                <h4 className="text-xs font-bold text-emerald-400 font-mono flex items-center space-x-2">
                  <CheckCircle2 className="w-4 h-4" />
                  <span>Generated Answer ({result.pipeline_config.llm_model_name})</span>
                </h4>
                <div className="text-sm text-slate-200 leading-relaxed font-sans bg-slate-900/80 p-4 rounded-lg border border-slate-800/80 whitespace-pre-wrap">
                  {result.llm_response.text}
                </div>
              </div>

              {/* Retrieved Chunks list */}
              <div className="glass-card p-5 rounded-xl border border-slate-800 space-y-4">
                <h4 className="text-xs font-bold text-white font-mono flex items-center justify-between">
                  <span>Top Retrieved Chunks (Top {result.retrieved_chunks.length})</span>
                  <span className="text-slate-400 text-[10px]">Ranked by similarity score</span>
                </h4>

                <div className="space-y-3">
                  {result.retrieved_chunks.map((item, idx) => (
                    <div key={idx} className="bg-slate-900/80 p-3.5 rounded-lg border border-slate-800 space-y-2">
                      <div className="flex items-center justify-between text-xs font-mono">
                        <div className="flex items-center space-x-2">
                          <span className="px-2 py-0.5 rounded bg-cyan-950 text-cyan-400 border border-cyan-800 font-bold">
                            Rank #{item.rank}
                          </span>
                          <span className="text-slate-300 font-semibold">{item.chunk.chunk_id}</span>
                          <span className="text-slate-500">({item.chunk.doc_id})</span>
                        </div>
                        <span className="text-emerald-400 font-bold">Score: {item.score.toFixed(4)}</span>
                      </div>
                      <p className="text-xs text-slate-300 font-mono bg-slate-950/60 p-2.5 rounded border border-slate-900">
                        {item.chunk.text}
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
