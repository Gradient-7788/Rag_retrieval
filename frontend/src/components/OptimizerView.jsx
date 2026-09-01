import React, { useState } from 'react';
import { ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Sparkles, Play, Trophy, CheckCircle, RefreshCw, Layers } from 'lucide-react';

export default function OptimizerView() {
  const [maxEvals, setMaxEvals] = useState(8);
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleRunSweep = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch('/api/sweep', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ max_evals: maxEvals }),
      });
      if (!res.ok) throw new Error("Sweep optimization failed");
      const data = await res.json();
      setReport(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const scatterData = report?.reports?.map((r) => ({
    name: r.pipeline_name,
    latency: r.system_telemetry.latency_p50_ms,
    quality: r.composite_quality_score,
    ndcg: r.ndcg_at_k,
    cost: r.system_telemetry.cost_per_1000_queries_usd,
    isPareto: report.pareto_frontier_configs.some((p) => p.pipeline_name === r.pipeline_name),
  })) || [];

  return (
    <div className="space-y-8">
      {/* Top Banner */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">Automated Hyperparameter Sweep & Pareto Optimizer</h2>
          <p className="text-xs text-slate-400 font-mono">Perform multi-objective grid search across chunk sizes, retrieval modes, alpha weights, and rerankers</p>
        </div>

        <div className="flex items-center space-x-3">
          <div className="flex items-center space-x-2 bg-slate-900 px-3 py-1.5 rounded-lg border border-slate-800 text-xs font-mono">
            <span className="text-slate-400">Sweep Evals:</span>
            <input
              type="number"
              min="2"
              max="20"
              value={maxEvals}
              onChange={(e) => setMaxEvals(parseInt(e.target.value) || 5)}
              className="w-12 bg-slate-800 text-cyan-400 font-bold px-1 py-0.5 rounded text-center focus:outline-none"
            />
          </div>
          <button
            onClick={handleRunSweep}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 rounded-lg bg-gradient-to-r from-cyan-500 to-indigo-600 hover:from-cyan-400 hover:to-indigo-500 text-white font-semibold text-xs transition-all shadow-md shadow-cyan-500/20 disabled:opacity-50"
          >
            {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            <span>Run Sweep</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-950/40 border border-red-800 rounded-xl text-red-300 font-mono text-xs">
          {error}
        </div>
      )}

      {/* Sweep summary KPIs */}
      {report && (
        <div className="space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400 font-mono">Evaluated Configurations</div>
              <div className="text-2xl font-bold text-cyan-400 mt-1">{report.total_configs_evaluated} Pipelines</div>
              <p className="text-xs text-slate-400 mt-1">Full Grid Search executed</p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400 font-mono">Best Config by NDCG</div>
              <div className="text-sm font-bold text-emerald-400 mt-1 truncate">{report.best_config_by_ndcg.pipeline_name}</div>
              <p className="text-xs text-slate-400 mt-1 font-mono">NDCG: {report.best_config_by_ndcg.ndcg_at_k}</p>
            </div>

            <div className="glass-card p-4 rounded-xl border border-slate-800">
              <div className="text-xs text-slate-400 font-mono">Pareto Frontier Count</div>
              <div className="text-2xl font-bold text-amber-400 mt-1">{report.pareto_frontier_configs.length} Optimal Trade-offs</div>
              <p className="text-xs text-slate-400 mt-1 font-mono">Non-dominated configurations</p>
            </div>
          </div>

          {/* Scatter Chart: Pareto Frontier */}
          <div className="glass-card p-5 rounded-xl border border-slate-800">
            <h3 className="text-sm font-semibold text-slate-200 mb-4 flex items-center justify-between">
              <span>Pareto Frontier Scatter Plot (Quality vs Latency)</span>
              <div className="flex items-center space-x-4 text-xs font-mono">
                <span className="flex items-center space-x-1"><span className="w-2.5 h-2.5 rounded-full bg-amber-400"></span><span className="text-slate-300">Pareto Optimal</span></span>
                <span className="flex items-center space-x-1"><span className="w-2.5 h-2.5 rounded-full bg-cyan-500"></span><span className="text-slate-400">Standard Run</span></span>
              </div>
            </h3>

            <div className="h-72">
              <ResponsiveContainer width="100%" height="100%">
                <ScatterChart margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
                  <XAxis type="number" dataKey="latency" name="Latency p50" unit="ms" stroke="#64748b" fontSize={11} label={{ value: 'Latency p50 (ms)', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 11 }} />
                  <YAxis type="number" dataKey="quality" name="Quality Score" domain={[0, 1.0]} stroke="#64748b" fontSize={11} label={{ value: 'Composite Quality Score', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }} />
                  <Tooltip
                    cursor={{ strokeDasharray: '3 3' }}
                    contentStyle={{ backgroundColor: '#0f172a', borderColor: '#334155', borderRadius: '8px', fontSize: '12px' }}
                  />
                  <Scatter name="Pipelines" data={scatterData}>
                    {scatterData.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.isPareto ? '#f59e0b' : '#06b6d4'} />
                    ))}
                  </Scatter>
                </ScatterChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Pareto Optimal Configs List */}
          <div className="glass-card p-5 rounded-xl border border-slate-800 space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Trophy className="w-4 h-4 text-amber-400" />
              <span>Recommended Pareto Optimal Architecture Candidates</span>
            </h3>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
              {report.pareto_frontier_configs.map((p, idx) => (
                <div key={idx} className="bg-slate-900/80 p-3.5 rounded-lg border border-amber-500/30 space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-bold text-amber-400 font-mono">Option #{idx + 1}</span>
                    <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-amber-950 text-amber-300 border border-amber-800">Pareto Efficient</span>
                  </div>
                  <div className="text-xs font-bold text-slate-100 truncate">{p.pipeline_name}</div>
                  <div className="grid grid-cols-3 gap-2 text-[11px] font-mono text-slate-300 pt-1 border-t border-slate-800">
                    <div>Score: <span className="text-amber-400 font-bold">{p.composite_quality_score}</span></div>
                    <div>NDCG: <span className="text-cyan-400 font-bold">{p.ndcg_at_k}</span></div>
                    <div>P50: <span className="text-rose-400 font-bold">{p.system_telemetry.latency_p50_ms}ms</span></div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      {!report && !loading && (
        <div className="glass-card p-12 text-center rounded-xl border border-slate-800 space-y-4">
          <Sparkles className="w-12 h-12 text-cyan-400 mx-auto" />
          <h3 className="text-lg font-bold text-white">Ready for Multi-Objective Hyperparameter Sweep</h3>
          <p className="text-xs text-slate-400 max-w-md mx-auto font-mono">
            Click 'Run Sweep' above to evaluate pipeline variations across chunking window size, hybrid alpha weights, query expansion, and reranking.
          </p>
        </div>
      )}
    </div>
  );
}
