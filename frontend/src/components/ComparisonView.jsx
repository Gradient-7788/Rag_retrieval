import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend } from 'recharts';
import { Play, RefreshCw, Trophy, Zap, ShieldCheck, DollarSign, Activity } from 'lucide-react';

export default function ComparisonView() {
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const fetchComparison = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/comparison');
      if (!res.ok) throw new Error('Failed to fetch benchmark comparison');
      const data = await res.json();
      setReports(data.comparison_reports || []);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchComparison();
  }, []);

  if (loading) {
    return (
      <div className="flex flex-col items-center justify-center py-24 space-y-4">
        <RefreshCw className="w-10 h-10 text-cyan-400 animate-spin" />
        <p className="text-slate-400 font-mono text-sm">Running Benchmark Evaluation Suite across candidate pipelines...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6 bg-red-950/40 border border-red-800 rounded-xl text-red-300 font-mono text-sm max-w-2xl mx-auto my-12">
        Error running comparison benchmark: {error}
      </div>
    );
  }

  if (!reports.length) return null;

  // Chart data formatting
  const chartData = reports.map((r) => ({
    name: r.pipeline_name.split(' (')[0],
    NDCG: r.ndcg_at_k,
    Recall: r.recall_at_k,
    Faithfulness: r.faithfulness,
    AnswerRel: r.answer_relevance,
    CitationAcc: r.citation_accuracy,
    LatencyMs: r.system_telemetry.latency_p50_ms,
    CostPer1k: r.system_telemetry.cost_per_1000_queries_usd,
    CompositeScore: r.composite_quality_score,
  }));

  const bestNDCG = [...reports].sort((a, b) => b.ndcg_at_k - a.ndcg_at_k)[0];
  const bestFaithful = [...reports].sort((a, b) => b.faithfulness - a.faithfulness)[0];
  const lowestLatency = [...reports].sort((a, b) => a.system_telemetry.latency_p50_ms - b.system_telemetry.latency_p50_ms)[0];

  return (
    <div className="space-y-8">
      {/* Top Controls & Banner */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Comparative RAG Architecture Sweep</h2>
          <p className="text-xs text-slate-400 font-mono">Evaluate Dense, BM25, Hybrid RRF, Reranking, and HyDE Query Expansion side-by-side</p>
        </div>
        <button
          onClick={fetchComparison}
          className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-semibold text-sm transition-all shadow-md shadow-cyan-500/20"
        >
          <RefreshCw className="w-4 h-4" />
          <span>Re-run Benchmark Suite</span>
        </button>
      </div>

      {/* KPI Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-mono">Top Retrieval NDCG@5</span>
            <Trophy className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-400 mt-2">{bestNDCG?.ndcg_at_k}</div>
          <p className="text-xs text-slate-400 mt-1 truncate">{bestNDCG?.pipeline_name}</p>
        </div>

        <div className="glass-card p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-mono">Top Faithfulness</span>
            <ShieldCheck className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-2xl font-bold text-emerald-400 mt-2">{bestFaithful?.faithfulness}</div>
          <p className="text-xs text-slate-400 mt-1 truncate">{bestFaithful?.pipeline_name}</p>
        </div>

        <div className="glass-card p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-mono">Lowest Latency (p50)</span>
            <Zap className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-cyan-400 mt-2">{lowestLatency?.system_telemetry?.latency_p50_ms} ms</div>
          <p className="text-xs text-slate-400 mt-1 truncate">{lowestLatency?.pipeline_name}</p>
        </div>

        <div className="glass-card p-4 rounded-xl border border-slate-800">
          <div className="flex items-center justify-between">
            <span className="text-xs text-slate-400 font-mono">Avg Throughput (QPS)</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-indigo-400 mt-2">{reports[0]?.system_telemetry?.throughput_qps} QPS</div>
          <p className="text-xs text-slate-400 mt-1">Queries per second</p>
        </div>
      </div>

      {/* Charts Section */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Quality Score Bar Chart */}
        <div className="glass-card p-5 rounded-xl border border-slate-800">
          <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center justify-between">
            <span>Retrieval & Generation Quality Metrics</span>
            <span className="text-xs text-slate-500 font-mono">Score range: 0.0 - 1.0</span>
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 25 }}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={10} interval={0} angle={-15} textAnchor="end" />
                <YAxis stroke="#64748b" fontSize={11} domain={[0, 1.0]} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="NDCG" fill="#38bdf8" name="NDCG@5" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Recall" fill="#818cf8" name="Recall@5" radius={[4, 4, 0, 0]} />
                <Bar dataKey="Faithfulness" fill="#34d399" name="Faithfulness" radius={[4, 4, 0, 0]} />
                <Bar dataKey="CompositeScore" fill="#f59e0b" name="Composite Score" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Latency Comparison Chart */}
        <div className="glass-card p-5 rounded-xl border border-slate-800">
          <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center justify-between">
            <span>System Latency Breakdown (p50 ms)</span>
            <span className="text-xs text-slate-500 font-mono">Lower is faster</span>
          </h3>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartData} margin={{ top: 10, right: 10, left: -10, bottom: 25 }}>
                <XAxis dataKey="name" stroke="#64748b" fontSize={10} interval={0} angle={-15} textAnchor="end" />
                <YAxis stroke="#64748b" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                />
                <Bar dataKey="LatencyMs" fill="#f43f5e" name="Latency p50 (ms)" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Comprehensive Multi-Dimensional Metric Matrix Table */}
      <div className="glass-card rounded-xl border border-slate-800 overflow-hidden">
        <div className="p-4 border-b border-slate-800 bg-slate-900/50 flex items-center justify-between">
          <h3 className="text-sm font-bold text-white">Full Multi-Dimensional Evaluation Matrix</h3>
          <span className="text-xs text-slate-400 font-mono">13 Key Metrics tracked across Retrieval, Generation & Telemetry</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead className="bg-slate-900/80 text-slate-400 border-b border-slate-800">
              <tr>
                <th className="px-4 py-3">Pipeline Strategy</th>
                <th className="px-3 py-3 text-cyan-400">Recall@5</th>
                <th className="px-3 py-3 text-cyan-400">Prec@5</th>
                <th className="px-3 py-3 text-cyan-400">MRR</th>
                <th className="px-3 py-3 text-cyan-400 font-bold">NDCG@5</th>
                <th className="px-3 py-3 text-emerald-400">Faithful</th>
                <th className="px-3 py-3 text-emerald-400">AnswerRel</th>
                <th className="px-3 py-3 text-emerald-400">ContextRel</th>
                <th className="px-3 py-3 text-emerald-400">CiteAcc</th>
                <th className="px-3 py-3 text-rose-400">P50 (ms)</th>
                <th className="px-3 py-3 text-rose-400">P99 (ms)</th>
                <th className="px-3 py-3 text-amber-400">Cost/1k ($)</th>
                <th className="px-3 py-3 text-purple-400">Mem (MB)</th>
                <th className="px-4 py-3 text-white font-bold bg-slate-800/40">Composite</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800 text-slate-300">
              {reports.map((r, idx) => (
                <tr key={idx} className="hover:bg-slate-800/40 transition-colors">
                  <td className="px-4 py-3 font-semibold text-slate-100 flex items-center space-x-2">
                    <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                    <span>{r.pipeline_name}</span>
                  </td>
                  <td className="px-3 py-3">{r.recall_at_k}</td>
                  <td className="px-3 py-3">{r.precision_at_k}</td>
                  <td className="px-3 py-3">{r.mrr}</td>
                  <td className="px-3 py-3 font-bold text-cyan-400">{r.ndcg_at_k}</td>
                  <td className="px-3 py-3">{r.faithfulness}</td>
                  <td className="px-3 py-3">{r.answer_relevance}</td>
                  <td className="px-3 py-3">{r.context_relevance}</td>
                  <td className="px-3 py-3">{r.citation_accuracy}</td>
                  <td className="px-3 py-3 text-rose-300">{r.system_telemetry.latency_p50_ms}</td>
                  <td className="px-3 py-3 text-rose-300">{r.system_telemetry.latency_p99_ms}</td>
                  <td className="px-3 py-3 text-amber-300">${r.system_telemetry.cost_per_1000_queries_usd}</td>
                  <td className="px-3 py-3 text-purple-300">{r.system_telemetry.memory_rss_mb}</td>
                  <td className="px-4 py-3 font-bold text-amber-400 bg-slate-800/20">{r.composite_quality_score}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
