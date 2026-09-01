import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import ComparisonView from './components/ComparisonView';
import WorkbenchView from './components/WorkbenchView';
import OptimizerView from './components/OptimizerView';
import DatasetView from './components/DatasetView';

export default function App() {
  const [activeTab, setActiveTab] = useState('comparison');
  const [docCount, setDocCount] = useState(5);
  const [queryCount, setQueryCount] = useState(5);

  useEffect(() => {
    fetch('/api/dataset')
      .then((res) => res.json())
      .then((data) => {
        if (data.documents) setDocCount(data.documents.length);
        if (data.benchmark_queries) setQueryCount(data.benchmark_queries.length);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen bg-[#0b0f19] text-slate-100 flex flex-col font-['Inter',sans-serif]">
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        docCount={docCount}
        queryCount={queryCount}
      />

      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'comparison' && <ComparisonView />}
        {activeTab === 'workbench' && <WorkbenchView />}
        {activeTab === 'optimizer' && <OptimizerView />}
        {activeTab === 'dataset' && (
          <DatasetView setDocCount={setDocCount} setQueryCount={setQueryCount} />
        )}
      </main>

      <footer className="border-t border-slate-800/80 py-6 text-center text-xs text-slate-500 font-mono">
        RAG Benchmarking & Optimization Engine &bull; Comparative Retrieval & Generation Research Platform
      </footer>
    </div>
  );
}
