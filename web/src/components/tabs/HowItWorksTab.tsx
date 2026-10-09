import React from "react";
import { HelpCircle, Layers, CheckCircle2, BookOpen, ShieldAlert, Cpu, Sparkles } from "lucide-react";

export const HowItWorksTab: React.FC = () => {
  const tradeOffs = [
    {
      question: "Why HistGradientBoosting / Tree Regressor instead of a huge Deep Transformer?",
      answer: "Turbofan sensor telemetry is tabular time-series where tree models offer superior sample efficiency, deterministic convergence, fast training (< 15 seconds), and native quantile pinball loss for p10/p50/p90 intervals. It also allows direct SHAP attribution down to specific engine subsystems."
    },
    {
      question: "How do you avoid data leakage across train and validation splits?",
      answer: "We split strictly by engine unit, NEVER by row. Group-level validation ensures telemetry from an engine never appears in both training and evaluation, preventing the model from memorizing individual engine identities."
    },
    {
      question: "Why clip the RUL target at 125 cycles?",
      answer: "In early operating cycles, turbofans run in nominal steady-state where sensor degradation is imperceptible from background noise. Capping at 125 cycles prevents the regression head from overfitting to uninformative flat steady-state cycles."
    },
    {
      question: "Why is the NASA Score asymmetric, and what does a late prediction cost?",
      answer: "In aerospace fleet management, predicting late (overestimating remaining life) means an engine may suffer an in-flight catastrophe before scheduled maintenance. The NASA score penalizes late predictions exponentially harder (exp(d/10)) than early predictions (exp(-d/13))."
    },
    {
      question: "Why is the Triage Agent strictly Zero-LLM?",
      answer: "In safety-critical aviation operations, maintenance dispatch decisions must be 100% deterministic, audit-traceable, and idempotent. Generative LLMs hallucinate and lack formal latency guarantees. A rule-based state machine provides transparent decision traces for every generated work order."
    }
  ];

  return (
    <div className="space-y-6">
      {/* Architecture Overview */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center gap-2">
          <Layers className="w-5 h-5 text-indigo-400" />
          <h3 className="font-bold text-base text-white">Closed-Loop Architecture Flow</h3>
        </div>
        <p className="text-xs text-slate-400">
          From streaming telemetry to automated work order generation and human exception routing:
        </p>

        <div className="bg-slate-950 p-5 rounded-2xl border border-slate-800 font-mono text-xs text-slate-300 space-y-3">
          <div className="flex flex-col md:flex-row items-center gap-2 text-center md:text-left justify-between">
            <div className="p-3 bg-slate-900 rounded-xl border border-indigo-500/20 text-indigo-300">
              <span className="block font-bold">1. Raw Telemetry</span>
              <span className="text-[10px] text-slate-400">21 Sensors + 3 Settings</span>
            </div>
            <span className="text-cyan-400 font-bold">→</span>
            <div className="p-3 bg-slate-900 rounded-xl border border-purple-500/20 text-purple-300">
              <span className="block font-bold">2. Data Cleaning & K-Means</span>
              <span className="text-[10px] text-slate-400">Regime Scaling & EWMA</span>
            </div>
            <span className="text-cyan-400 font-bold">→</span>
            <div className="p-3 bg-slate-900 rounded-xl border border-cyan-500/20 text-cyan-300">
              <span className="block font-bold">3. Multi-Head ML</span>
              <span className="text-[10px] text-slate-400">RUL Quantiles + Anomaly</span>
            </div>
            <span className="text-cyan-400 font-bold">→</span>
            <div className="p-3 bg-slate-900 rounded-xl border border-emerald-500/20 text-emerald-300">
              <span className="block font-bold">4. Rule-Based Agent</span>
              <span className="text-[10px] text-slate-400">Deterministic Policy</span>
            </div>
            <span className="text-cyan-400 font-bold">→</span>
            <div className="p-3 bg-slate-900 rounded-xl border border-rose-500/20 text-rose-300">
              <span className="block font-bold">5. Work Order Store</span>
              <span className="text-[10px] text-slate-400">Auto-Dispatched Ticket</span>
            </div>
          </div>
        </div>
      </div>

      {/* Design Decisions & Review FAQs */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center gap-2">
          <BookOpen className="w-5 h-5 text-cyan-400" />
          <h3 className="font-bold text-base text-white">Design Decisions & Defensible Trade-Offs</h3>
        </div>
        <p className="text-xs text-slate-400">
          Core technical rationales prepared for technical review and defense:
        </p>

        <div className="space-y-3">
          {tradeOffs.map((item, idx) => (
            <div key={idx} className="bg-slate-900/70 p-4 rounded-2xl border border-slate-800 space-y-1.5">
              <h4 className="text-xs font-bold text-indigo-300 flex items-center gap-2">
                <CheckCircle2 className="w-3.5 h-3.5 text-indigo-400" />
                <span>{item.question}</span>
              </h4>
              <p className="text-xs text-slate-300 leading-relaxed pl-5">
                {item.answer}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset Citation & Acknowledgement */}
      <div className="glass-panel p-5 rounded-2xl text-xs text-slate-400 space-y-2 border-indigo-500/15">
        <div className="font-semibold text-slate-300">Dataset Benchmark Citation:</div>
        <p className="font-mono text-[11px] bg-slate-950 p-3 rounded-xl border border-slate-800 text-slate-300">
          A. Saxena and K. Goebel (2008), "Turbofan Engine Degradation Simulation Data Set", NASA Ames Prognostics Data Repository.<br />
          Background Paper: Saxena et al., "Damage Propagation Modeling for Aircraft Engine Run-to-Failure Simulation", PHM08.
        </p>
      </div>
    </div>
  );
};
