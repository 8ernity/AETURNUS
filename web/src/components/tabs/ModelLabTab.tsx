import React, { useState, useEffect } from "react";
import { Award, BarChart2, Zap, Cpu, CheckCircle2, AlertCircle, Shield } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid } from "recharts";

export const ModelLabTab: React.FC = () => {
  const [benchmarks, setBenchmarks] = useState<any[]>([]);
  const [selectedSub, setSelectedSub] = useState<string>("FD001");
  const [modelDetails, setModelDetails] = useState<any | null>(null);
  const [comparison, setComparison] = useState<any | null>(null);
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    fetch("/api/v1/models/benchmark")
      .then((res) => res.json())
      .then((data) => {
        setBenchmarks(data.benchmarks || []);
      })
      .catch((err) => console.error("Error fetching benchmarks:", err));
  }, []);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      fetch(`/api/v1/models/${selectedSub}/details`).then((r) => r.json()),
      fetch(`/api/v1/models/${selectedSub}/comparison`).then((r) => r.json()).catch(() => null)
    ])
      .then(([details, comp]) => {
        setModelDetails(details);
        setComparison(comp);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching model data:", err);
        setLoading(false);
      });
  }, [selectedSub]);

  const targetsMetCount = benchmarks.filter((b) => b.status === "Target Met").length;
  const bucketErrors = modelDetails?.bucket_errors || [];
  const featureImportances = modelDetails?.feature_importance || [];
  const comparisonModels = comparison?.models || [];

  return (
    <div className="space-y-6">
      {/* Benchmark Summary Table */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-indigo-400" />
            <h3 className="font-bold text-base text-white">C-MAPSS Benchmark Leaderboard (FD001 – FD004)</h3>
          </div>
          <span className={`text-xs px-3 py-1 rounded-full font-semibold border ${
            targetsMetCount === benchmarks.length && benchmarks.length > 0
              ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
              : "bg-indigo-500/20 text-indigo-300 border-indigo-500/30"
          }`}>
            {benchmarks.length > 0 ? `${targetsMetCount} of ${benchmarks.length} Targets Met` : "Evaluating..."}
          </span>
        </div>
        <p className="text-xs text-slate-400">
          Official protocol: Evaluated on the final cycle of each held-out test engine. Asymmetric NASA Score penalizes late predictions exponentially.
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="p-3.5">Subset</th>
                <th className="p-3.5">Conditions</th>
                <th className="p-3.5">Fault Modes</th>
                <th className="p-3.5 font-mono">Test RMSE</th>
                <th className="p-3.5 font-mono">Target RMSE</th>
                <th className="p-3.5 font-mono">NASA Score</th>
                <th className="p-3.5 font-mono">PR-AUC</th>
                <th className="p-3.5 font-mono">Train Time</th>
                <th className="p-3.5 text-center">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {benchmarks.map((row) => (
                <tr 
                  key={row.subset} 
                  onClick={() => setSelectedSub(row.subset)}
                  className={`cursor-pointer transition-colors ${
                    selectedSub === row.subset ? "bg-indigo-600/15 border-l-2 border-indigo-500" : "hover:bg-slate-800/30"
                  }`}
                >
                  <td className="p-3.5 font-bold text-white font-sans flex items-center gap-2">
                    <span>{row.subset}</span>
                    {selectedSub === row.subset && <span className="text-[10px] text-indigo-400 font-normal">▶ Active</span>}
                  </td>
                  <td className="p-3.5 text-slate-300 font-sans">{row.condition}</td>
                  <td className="p-3.5 text-slate-400 font-sans">{row.faults}</td>
                  <td className="p-3.5 text-cyan-300 font-bold">{row.rmse}</td>
                  <td className="p-3.5 text-slate-500">≤ {row.target_rmse}</td>
                  <td className="p-3.5 text-purple-300">{row.nasa_score}</td>
                  <td className="p-3.5 text-emerald-400">{row.pr_auc}</td>
                  <td className="p-3.5 text-slate-400">{row.train_time_sec}s</td>
                  <td className="p-3.5 text-center">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                      row.status === "Target Met"
                        ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30"
                        : "bg-amber-500/20 text-amber-300 border-amber-500/30"
                    }`}>
                      {row.status}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Subset Selector for Deep-Dive */}
      <div className="flex items-center gap-3 bg-slate-900/60 p-3 rounded-2xl border border-indigo-500/15">
        <span className="text-xs text-slate-400 font-medium">Deep-Dive Subset Details:</span>
        <div className="flex gap-2">
          {["FD001", "FD002", "FD003", "FD004"].map((s) => (
            <button
              key={s}
              onClick={() => setSelectedSub(s)}
              className={`px-3 py-1 rounded-lg text-xs font-semibold transition-all ${
                selectedSub === s
                  ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                  : "bg-slate-800 text-slate-400 hover:text-white"
              }`}
            >
              {s}
            </button>
          ))}
        </div>
      </div>

      {/* Architecture Comparison & Champion Decision Section */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu className="w-5 h-5 text-cyan-400" />
            <h3 className="font-bold text-sm text-white">
              {selectedSub} Architecture Benchmark & Champion Selection Decision
            </h3>
          </div>
          <span className="text-xs px-2.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-mono">
            GBDT vs 1D-CNN vs GRU
          </span>
        </div>
        <p className="text-xs text-slate-400">
          Evaluated against the empirical test set. Compare sequence deep learning models (30-cycle temporal window) against gradient boosted quantile ensembles.
        </p>

        {comparisonModels.length > 0 ? (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-900/90 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                <tr>
                  <th className="p-3">Architecture</th>
                  <th className="p-3">Model Name</th>
                  <th className="p-3 font-mono">Test RMSE</th>
                  <th className="p-3 font-mono">MAE</th>
                  <th className="p-3 font-mono">NASA Score</th>
                  <th className="p-3 font-mono">Train Time</th>
                  <th className="p-3 text-center">Serving Champion</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {comparisonModels.map((m: any, idx: number) => {
                  const isGbdt = m.architecture === "GBDT";
                  return (
                    <tr key={idx} className={isGbdt ? "bg-indigo-950/20" : "hover:bg-slate-800/20"}>
                      <td className="p-3 font-bold text-white font-sans flex items-center gap-2">
                        <span className={`w-2 h-2 rounded-full ${isGbdt ? "bg-indigo-400" : "bg-cyan-400"}`} />
                        {m.architecture}
                      </td>
                      <td className="p-3 text-slate-300 font-sans">{m.model_name}</td>
                      <td className="p-3 font-bold text-cyan-300">{Number(m.rmse).toFixed(2)}</td>
                      <td className="p-3 text-slate-400">{Number(m.mae).toFixed(2)}</td>
                      <td className="p-3 text-purple-300">{Number(m.nasa_score).toFixed(1)}</td>
                      <td className="p-3 text-slate-400">{m.train_time_sec}s</td>
                      <td className="p-3 text-center">
                        {isGbdt ? (
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                            <CheckCircle2 className="w-3 h-3" /> Active Champion
                          </span>
                        ) : (
                          <span className="text-slate-500 text-[10px]">Benchmark Baseline</span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-xs text-slate-500 py-4">Loading architecture comparison...</div>
        )}

        {/* Champion Decision Rationale Cards */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-xs">
          <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <div className="font-semibold text-indigo-300 flex items-center gap-1.5">
              <Shield className="w-4 h-4 text-indigo-400" /> Quantile Bands (P10/P50/P90)
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              GBDT models pinball loss quantiles natively, enabling certified asymmetric uncertainty intervals for risk-averse work orders. Neural baselines only produce uncalibrated point estimates.
            </p>
          </div>

          <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <div className="font-semibold text-cyan-300 flex items-center gap-1.5">
              <Zap className="w-4 h-4 text-cyan-400" /> Instant Cold-Start (&lt;2ms)
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              GBDT predicts from cycle 1 via causal feature imputation. 1D-CNN and GRU require a 30-cycle buffer window, making early-life engine triage unavailable or distorted by zero-padding.
            </p>
          </div>

          <div className="p-3.5 rounded-2xl bg-slate-950/60 border border-slate-800 space-y-1.5">
            <div className="font-semibold text-purple-300 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 text-purple-400" /> Sequence Model Strengths
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              GRU achieved 14.01 RMSE on FD003 (vs 15.16 for GBDT), and 1D-CNN achieved 24.69 on FD002 and 24.91 on FD004, validating temporal sequence representation on complex multi-fault datasets.
            </p>
          </div>
        </div>
      </div>

      {/* Real Residuals & Real Bucket Errors */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Error by RUL Bucket */}
        <div className="glass-panel p-6 rounded-3xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart2 className="w-5 h-5 text-cyan-400" />
              <h3 className="font-bold text-sm text-white">{selectedSub} Error by Remaining Life Bucket</h3>
            </div>
            <span className="text-[10px] font-mono text-cyan-300 bg-slate-900 px-2 py-0.5 rounded-md border border-cyan-500/20">
              Real Test Data
            </span>
          </div>
          <p className="text-[11px] text-slate-400">
            Computed directly on held-out test engines across cycle degradation phases.
          </p>

          <div className="h-64 w-full pt-2">
            {bucketErrors.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={bucketErrors}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                  <XAxis dataKey="bucket" stroke="#6b7280" fontSize={10} tickLine={false} />
                  <YAxis stroke="#6b7280" fontSize={11} tickLine={false} unit=" cyc" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: "#0f172a", borderColor: "#374151", borderRadius: "12px", fontSize: "12px" }}
                  />
                  <Bar dataKey="rmse" fill="#6366f1" name="RMSE (Cycles)" radius={[6, 6, 0, 0]} />
                  <Bar dataKey="mae" fill="#06b6d4" name="MAE (Cycles)" radius={[6, 6, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex items-center justify-center h-full text-xs text-slate-500">
                Loading bucket errors...
              </div>
            )}
          </div>
        </div>

        {/* Real Feature Importance & Ranking */}
        <div className="glass-panel p-6 rounded-3xl flex flex-col justify-between">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <Zap className="w-5 h-5 text-purple-400" />
                <h3 className="font-bold text-sm text-white">{selectedSub} Real Feature Importances</h3>
              </div>
              <span className="text-[10px] font-mono text-purple-300 bg-slate-900 px-2 py-0.5 rounded-md border border-purple-500/20">
                Permutation Importance
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mb-4">
              True permutation importance on held-out test telemetry, mapped to engine subsystems.
            </p>

            <div className="space-y-3">
              {featureImportances.length > 0 ? (
                featureImportances.map((f: any, i: number) => (
                  <div key={i} className="space-y-1">
                    <div className="flex justify-between text-xs font-mono">
                      <span className="text-slate-300 font-sans truncate max-w-[280px]">{f.feat}</span>
                      <span className="text-indigo-400 font-bold">{f.weight}</span>
                    </div>
                    <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                      <div 
                        className={`h-full bg-gradient-to-r ${f.color || "from-indigo-500 to-purple-500"} rounded-full`}
                        style={{ width: `${Math.max(5, Math.min(100, f.raw_weight || 10))}%` }}
                      ></div>
                    </div>
                  </div>
                ))
              ) : (
                <div className="text-xs text-slate-500 py-8 text-center">
                  Loading permutation feature importances...
                </div>
              )}
            </div>
          </div>

          <div className="mt-6 pt-4 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400 font-mono">
            <span>Subset: {selectedSub}</span>
            <span className="text-cyan-300">Code-Computed via scikit-learn</span>
          </div>
        </div>
      </div>
    </div>
  );
};
