import React, { useState, useEffect } from 'react';
import { Database, Plus, Sparkles, FileText, CheckCircle2, RefreshCw } from 'lucide-react';

export default function DatasetView({ setDocCount, setQueryCount }) {
  const [dataset, setDataset] = useState({ documents: [], benchmark_queries: [] });
  const [newDocId, setNewDocId] = useState('');
  const [newDocText, setNewDocText] = useState('');
  const [loading, setLoading] = useState(false);

  const fetchDataset = async () => {
    try {
      const res = await fetch('/api/dataset');
      if (res.ok) {
        const data = await res.json();
        setDataset(data);
        if (setDocCount) setDocCount(data.documents.length);
        if (setQueryCount) setQueryCount(data.benchmark_queries.length);
      }
    } catch (err) {
      console.error(err);
    }
  };

  useEffect(() => {
    fetchDataset();
  }, []);

  const handleAddDocument = async () => {
    if (!newDocText.trim()) return;
    setLoading(true);
    const docId = newDocId.trim() || `custom_doc_${dataset.documents.length + 1}`;
    const updatedDocs = [...dataset.documents, { doc_id: docId, text: newDocText.trim() }];

    try {
      const res = await fetch('/api/dataset/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ documents: updatedDocs, generate_synthetic_queries: true }),
      });
      if (res.ok) {
        setNewDocId('');
        setNewDocText('');
        await fetchDataset();
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-xl font-bold text-white">Corpus & Synthetic Benchmark Dataset Manager</h2>
        <p className="text-xs text-slate-400 font-mono">Manage test document collections and generate ground-truth QA evaluation benchmarks automatically</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Add Document Form */}
        <div className="lg:col-span-5 space-y-4">
          <div className="glass-card p-5 rounded-xl border border-slate-800 space-y-4">
            <h3 className="text-sm font-bold text-white flex items-center space-x-2">
              <Plus className="w-4 h-4 text-cyan-400" />
              <span>Add Custom Document to Corpus</span>
            </h3>

            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1">Document Identifier (ID)</label>
              <input
                type="text"
                value={newDocId}
                onChange={(e) => setNewDocId(e.target.value)}
                placeholder="e.g. doc_system_architecture"
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
              />
            </div>

            <div>
              <label className="text-xs font-mono text-slate-400 block mb-1">Document Content / Technical Passage</label>
              <textarea
                rows={5}
                value={newDocText}
                onChange={(e) => setNewDocText(e.target.value)}
                placeholder="Paste technical text, API docs, specification or articles..."
                className="w-full bg-slate-900 border border-slate-700 rounded-lg px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-cyan-500 font-mono"
              ></textarea>
            </div>

            <button
              onClick={handleAddDocument}
              disabled={loading || !newDocText.trim()}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2.5 rounded-lg bg-cyan-500 hover:bg-cyan-400 text-slate-950 font-bold text-xs transition-all shadow-md shadow-cyan-500/20 disabled:opacity-50"
            >
              {loading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
              <span>Add Document & Generate Synthetic QA</span>
            </button>
          </div>
        </div>

        {/* Right Column: Active Documents & Ground Truth Queries */}
        <div className="lg:col-span-7 space-y-6">
          {/* Document list */}
          <div className="glass-card p-5 rounded-xl border border-slate-800 space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center justify-between">
              <span>Active Corpus Documents ({dataset.documents.length})</span>
              <span className="text-xs text-slate-400 font-mono">Indexed for retrieval</span>
            </h3>

            <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
              {dataset.documents.map((doc, idx) => (
                <div key={idx} className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
                  <div className="text-xs font-bold text-cyan-400 font-mono flex items-center space-x-2">
                    <FileText className="w-3.5 h-3.5" />
                    <span>{doc.doc_id}</span>
                  </div>
                  <p className="text-xs text-slate-300 font-mono line-clamp-2">{doc.text}</p>
                </div>
              ))}
            </div>
          </div>

          {/* Benchmark Queries List */}
          <div className="glass-card p-5 rounded-xl border border-slate-800 space-y-3">
            <h3 className="text-sm font-bold text-white flex items-center justify-between">
              <span>Ground Truth Benchmark QA Pairs ({dataset.benchmark_queries.length})</span>
              <span className="text-xs text-slate-400 font-mono">Used for NDCG & Faithfulness evaluation</span>
            </h3>

            <div className="space-y-2 max-h-72 overflow-y-auto pr-1">
              {dataset.benchmark_queries.map((q, idx) => (
                <div key={idx} className="bg-slate-900/80 p-3 rounded-lg border border-slate-800 space-y-1">
                  <div className="flex items-center justify-between text-xs font-mono">
                    <span className="font-bold text-indigo-400">Query #{idx + 1}: {q.query_id}</span>
                    <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-slate-400">Target Doc: {q.ground_truth_doc_ids?.join(', ')}</span>
                  </div>
                  <div className="text-xs font-semibold text-slate-200">{q.query}</div>
                  {q.target_answer && (
                    <p className="text-[11px] text-slate-400 font-mono bg-slate-950/50 p-1.5 rounded">
                      Gold Target Answer: {q.target_answer}
                    </p>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
