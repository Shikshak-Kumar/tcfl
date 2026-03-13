import React from 'react';
import { Shield, Zap, TrendingUp, ArrowRight, Github, Activity } from 'lucide-react';

export default function LandingPage({ onGetStarted }) {
  return (
    <div className="min-h-screen bg-[#020617] text-slate-100 overflow-x-hidden selection:bg-indigo-500/30">
      {/* Decorative Background Elements */}
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute top-[-10%] left-[-10%] w-[40%] h-[40%] bg-indigo-500/10 rounded-full blur-[120px]" />
        <div className="absolute bottom-[5%] right-[-5%] w-[30%] h-[30%] bg-blue-600/10 rounded-full blur-[100px]" />
        <div className="absolute top-[20%] right-[10%] w-[20%] h-[20%] bg-purple-500/5 rounded-full blur-[80px]" />
      </div>

      {/* Navigation */}
      <nav className="relative z-10 flex items-center justify-between px-6 py-6 max-w-7xl mx-auto">
        <div className="flex items-center gap-2">
          <div className="bg-gradient-to-br from-indigo-500 to-blue-600 p-2 rounded-xl shadow-lg shadow-indigo-500/20">
            <Activity className="w-6 h-6 text-white" />
          </div>
          <span className="text-xl font-bold tracking-tight bg-clip-text text-transparent bg-gradient-to-r from-white to-slate-400">
            AdaptFlow
          </span>
        </div>
        <div className="flex items-center gap-6">
          <a href="https://github.com/shikshak-Kumar/tcfl" target="_blank" rel="noreferrer" className="text-slate-400 hover:text-white transition-colors">
            <Github className="w-5 h-5" />
          </a>
          <button
            onClick={onGetStarted}
            className="px-5 py-2.5 rounded-full bg-white text-black text-sm font-bold hover:bg-slate-200 transition-all active:scale-95 shadow-xl shadow-white/5"
          >
            Get Started
          </button>
        </div>
      </nav>

      {/* Hero Section */}
      <main className="relative z-10 max-w-7xl mx-auto px-6 pt-20 pb-32">
        <div className="text-center">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 text-xs font-bold mb-8 animate-fade-in">
            <Zap className="w-3 h-3" />
            <span>AI-Powered Traffic Orchestration</span>
          </div>

          <h1 className="text-5xl lg:text-8xl font-black mb-6 leading-tight tracking-tight">
            The Future of <br />
            <span className="bg-clip-text text-transparent bg-gradient-to-r from-indigo-400 via-blue-400 to-cyan-400">
              Smart Cities
            </span>
          </h1>

          <p className="max-w-2xl mx-auto text-slate-400 text-lg lg:text-xl mb-12 leading-relaxed">
            Eliminate congestion with multi-objective federated learning. Intelligent, spatio-temporal autonomous traffic control that prioritizes what matters.
          </p>

          <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
            <button
              onClick={onGetStarted}
              className="group relative px-8 py-4 rounded-2xl bg-indigo-600 font-bold overflow-hidden transition-all hover:bg-indigo-500 active:scale-95 shadow-2xl shadow-indigo-600/20"
            >
              <div className="relative z-10 flex items-center gap-2">
                Launch Dashboard
                <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
              </div>
            </button>
            <button className="px-8 py-4 rounded-2xl bg-white/5 border border-white/10 font-bold hover:bg-white/10 transition-all">
              Research Theory
            </button>
          </div>
        </div>

        {/* Features / Novelties */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mt-32">
          <FeatureCard
            icon={<Shield className="w-6 h-6 text-indigo-400" />}
            title="Priority Clustering"
            description="Intelligent POI tiering (Hospitals, Schools) ensures critical infrastructure never hits a red light."
          />
          <FeatureCard
            icon={<Zap className="w-6 h-6 text-blue-400" />}
            title="Spatio-Temporal GAT"
            description="Graph Attention Networks that learn behavioral patterns across the city both in space and time."
          />
          <FeatureCard
            icon={<TrendingUp className="w-6 h-6 text-cyan-400" />}
            title="Pareto Optimization"
            description="Balance competing objectives: Queue Length, Latency, Safety, and Stability in real-time."
          />
        </div>
      </main>

      {/* Footer */}
      <footer className="relative z-10 border-t border-white/5 py-12">
        <div className="max-w-7xl mx-auto px-6 text-center text-slate-500 text-sm">
          <p>© 2026 AdaptFlow AI. Built for Advanced Traffic Analytics.</p>
        </div>
      </footer>
    </div>
  );
}

function FeatureCard({ icon, title, description }) {
  return (
    <div className="glass-panel p-8 rounded-3xl group hover:border-indigo-500/30 transition-all hover:-translate-y-2">
      <div className="bg-slate-900/50 w-14 h-14 rounded-2xl flex items-center justify-center mb-6 border border-white/5 group-hover:scale-110 transition-transform">
        {icon}
      </div>
      <h3 className="text-xl font-bold mb-3 group-hover:text-indigo-400 transition-colors">{title}</h3>
      <p className="text-slate-400 leading-relaxed text-sm">
        {description}
      </p>
    </div>
  );
}
