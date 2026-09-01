import React from 'react';
import { Cpu, Database, Sparkles, BarChart3, Sliders, Layers } from 'lucide-react';

export default function Header({ activeTab, setActiveTab, docCount, queryCount }) {
  const tabs = [
    { id: 'comparison', label: 'Comparative Benchmark', icon: BarChart3 },
    { id: 'workbench', label: 'Pipeline Workbench', icon: Sliders },
    { id: 'optimizer', label: 'Hyperparameter Sweep', icon: Sparkles },
    { id: 'dataset', label: 'Corpus & Benchmark QA', icon: Database },
  ];

  return (
    <header className="border-b border-slate-800 bg-[#0f172a]/80 backdrop-blur-md sticky top-0 z-50">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16">
          {/* Logo & Title */}
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-500 to-indigo-600 shadow-lg shadow-cyan-500/20">
              <Layers className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="text-lg font-bold text-gradient">
                RAG Benchmarking Engine
              </h1>
              <p className="text-xs text-slate-400 font-mono">
                Modular RAG Evaluation & Optimization Platform
              </p>
            </div>
          </div>

          {/* Navigation Tabs */}
          <nav className="flex space-x-1 bg-slate-900/60 p-1 rounded-xl border border-slate-800">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              const isActive = activeTab === tab.id;
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center space-x-2 px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                    isActive
                      ? 'bg-gradient-to-r from-cyan-500 to-indigo-600 text-white shadow-md shadow-cyan-500/10'
                      : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800/50'
                  }`}
                >
                  <Icon className="w-4 h-4" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </nav>

          {/* Status Badge */}
          <div className="hidden md:flex items-center space-x-3 text-xs font-mono">
            <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-900 border border-slate-800">
              <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
              <span className="text-slate-300">{docCount} Docs</span>
              <span className="text-slate-600">|</span>
              <span className="text-slate-300">{queryCount} Benchmark QA</span>
            </div>
          </div>
        </div>
      </div>
    </header>
  );
}
