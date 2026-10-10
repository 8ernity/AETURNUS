import React, { useState, useEffect } from "react";
import { GitBranch, RefreshCw, AlertTriangle, CheckCircle2, ShieldCheck, Flame, ArrowRight, XCircle } from "lucide-react";

interface MLOpsTabProps {
  currentSubset: string;
}

export const MLOpsTab: React.FC<MLOpsTabProps> = ({ currentSubset }) => {
  const [registry, setRegistry] = useState<any | null>(null);
  const [driftReport, setDriftReport] = useState<any | null>(null);
  const [isRetraining, setIsRetraining] = useState<boolean>(false);
  const [retrainResult, setRetrainResult] = useState<any | null>(null);
  const [scenario, setScenario] = useState<string>("B"); // Scenario B shows drift & promotion gate

  const fetchMLOps = () => {
    fetch(`/api/v1/mlops/registry?subset=${currentSubset}`)
      .then((res) => res.json())
      .then((data) => setRegistry(data))
      .catch((err) => console.error(err));

    fetch(`/api/v1/monitoring/drift?subset=${currentSubset}`)
      .then((res) => res.json())
      .then((data) => setDriftReport(data))
      .catch((err) => console.error(err));
  };

  useEffect(() => {
    fetchMLOps();
  }, [currentSubset]);

  const handleRetrain = async () => {
    setIsRetraining(true);
    setRetrainResult(null);
    try {
      const res = await fetch(`/api/v1/mlops/retrain?scenario=${scenario}&subset=${currentSubset}`, {
        method: "POST"
      });
      const data = await res.json();
      setRetrainResult(data);
      fetchMLOps();
    } catch (err) {
      console.error(err);
    } finally {
      setIsRetraining(false);
    }
  };

  const models = registry?.models || [];
  const sensorDrifts = driftReport?.sensor_metrics || {};

  const [healingHistory, setHealingHistory] = useState<any[]>([]);
  const [isHealing, setIsHealing] = useState<boolean>(false);
  const [healingFeedback, setHealingFeedback] = useState<any | null>(null);

  const fetchHealing = () => {
    fetch("/api/v1/self-healing/history")
      .then((res) => res.json())
      .then((data) => setHealingHistory(data.history || []))
      .catch((err) => console.error(err));
  };

  const handleExecuteHealing = async (actionType: string) => {
    setIsHealing(true);
    setHealingFeedback(null);
    try {
      const res = await fetch(`/api/v1/self-healing/execute?engine_id=unit_1&action_type=${actionType}&subset=${currentSubset}`, {
        method: "POST"
      });
      const data = await res.json();
      setHealingFeedback(data.action);
      fetchHealing();
    } catch (err) {
      console.error(err);
    } finally {
      setIsHealing(false);
    }
  };

  useEffect(() => {
    fetchHealing();
  }, [currentSubset]);

  return (
    <div className="space-y-6">
      {/* Retraining Trigger Card */}
      <div className="glass-panel p-6 rounded-3xl border-indigo-500/30 flex flex-col md:flex-row md:items-center justify-between gap-6">
        <div>
          <div className="flex items-center gap-2 mb-1.5">
            <RefreshCw className={`w-5 h-5 text-indigo-400 ${isRetraining ? "animate-spin" : ""}`} />
            <h3 className="font-bold text-base text-white">Automated Retraining & Challenger Gate</h3>
          </div>
          <p className="text-xs text-slate-400 max-w-xl">
            Simulate incoming telemetry batches. When drift or performance degradation is detected, an automated retraining flow fits a challenger and evaluates the strict 3-criteria promotion gate.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <select 
            value={scenario} 
            onChange={(e) => setScenario(e.target.value)}
            className="text-xs bg-slate-900 text-white px-3 py-2.5 rounded-xl border border-slate-700 focus:outline-none"
          >
            <option value="B">Scenario B · Severe Distribution Shift (Drift Alert)</option>
            <option value="A">Scenario A · Incremental Batch Arrival</option>
          </select>

          <button
            onClick={handleRetrain}
            disabled={isRetraining}
            className="glass-button text-white text-xs px-5 py-2.5 rounded-xl font-bold flex items-center gap-2 shadow-lg whitespace-nowrap"
          >
            {isRetraining ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Flame className="w-4 h-4 text-cyan-300" />}
            <span>{isRetraining ? "Evaluating Gate..." : "Simulate Retrain"}</span>
          </button>
        </div>
      </div>

      {/* Retraining Gate Verdict Banner (if run) */}
      {retrainResult && (
        <div className={`glass-panel p-6 rounded-3xl border ${retrainResult.gate_passed ? "border-emerald-500/40 bg-emerald-500/5" : "border-rose-500/40 bg-rose-500/5"} space-y-3`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {retrainResult.gate_passed ? (
                <CheckCircle2 className="w-5 h-5 text-emerald-400" />
              ) : (
                <XCircle className="w-5 h-5 text-rose-400" />
              )}
              <h4 className="font-bold text-sm text-white">
                Promotion Gate Verdict: {retrainResult.gate_passed ? "CHALLENGER PROMOTED TO CHAMPION" : "CHALLENGER REJECTED"}
              </h4>
            </div>
            {retrainResult.promoted_version && (
              <span className="font-mono text-xs px-3 py-1 rounded-full font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                New Champion: {retrainResult.promoted_version}
              </span>
            )}
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono pt-2">
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800 space-y-1">
              <span className="text-slate-400 block text-[10px] uppercase">Champion Baseline</span>
              <div>RMSE: <b className="text-white">{retrainResult.champion_metrics?.rmse}</b></div>
              <div>NASA Score: <b className="text-white">{retrainResult.champion_metrics?.nasa_score}</b></div>
            </div>
            <div className="bg-slate-900/80 p-3 rounded-xl border border-slate-800 space-y-1">
              <span className="text-slate-400 block text-[10px] uppercase">Challenger Candidate</span>
              <div>RMSE: <b className="text-cyan-300">{retrainResult.challenger_metrics?.rmse}</b></div>
              <div>NASA Score: <b className="text-purple-300">{retrainResult.challenger_metrics?.nasa_score}</b></div>
            </div>
          </div>

          {/* Gate Criteria Checklist */}
          <div className="space-y-1 pt-2 font-mono text-[11px]">
            {retrainResult.gate_reasons?.map((reason: string, i: number) => (
              <div key={i} className={`flex items-center gap-2 ${reason.startsWith("PASS") ? "text-emerald-300" : "text-rose-300"}`}>
                <span>{reason.startsWith("PASS") ? "✓" : "✗"}</span>
                <span>{reason}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Autonomous Self-Healing Closed-Loop Panel */}
      <div className="glass-panel p-6 rounded-3xl space-y-4 border-cyan-500/30">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <ShieldCheck className="w-5 h-5 text-cyan-400" />
            <div>
              <h3 className="font-bold text-sm text-white">Autonomous Self-Healing Closed-Loop Control</h3>
              <p className="text-xs text-slate-400">
                Executes dynamic thrust derates, active cooling bleed modulation, and zero-AOG logistics reservations.
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExecuteHealing("DYNAMIC_DERATE")}
              disabled={isHealing}
              className="text-xs bg-cyan-600/20 hover:bg-cyan-600/40 text-cyan-300 border border-cyan-500/40 px-3.5 py-2 rounded-xl font-bold flex items-center gap-1.5 transition-colors"
            >
              <Flame className="w-3.5 h-3.5" />
              <span>Apply Dynamic Derate</span>
            </button>
            <button
              onClick={() => handleExecuteHealing("COOLING_BLEED_MODULATION")}
              disabled={isHealing}
              className="text-xs bg-indigo-600/20 hover:bg-indigo-600/40 text-indigo-300 border border-indigo-500/40 px-3.5 py-2 rounded-xl font-bold flex items-center gap-1.5 transition-colors"
            >
              <RefreshCw className="w-3.5 h-3.5" />
              <span>Modulate Bleed Vane</span>
            </button>
          </div>
        </div>

        {/* Live Feedback Banner */}
        {healingFeedback && (
          <div className="bg-cyan-500/10 border border-cyan-500/30 p-4 rounded-2xl flex items-center justify-between text-xs font-mono">
            <div>
              <span className="text-cyan-400 font-bold block">{healingFeedback.title} Executed!</span>
              <span className="text-slate-300">
                Engine {healingFeedback.engine_id} · Subsystem {healingFeedback.subsystem} · RUL extended from {healingFeedback.original_rul} to <b className="text-emerald-300">{healingFeedback.extended_rul} cycles</b> (+{healingFeedback.life_extension_pct}%).
              </span>
            </div>
            <span className="px-2.5 py-1 rounded-full text-[10px] font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/40">
              LEDGER SYNCED
            </span>
          </div>
        )}

        {/* Self-Healing Ledger History */}
        {healingHistory.length > 0 && (
          <div className="overflow-x-auto pt-2">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                <tr>
                  <th className="p-3">Action ID</th>
                  <th className="p-3">Engine</th>
                  <th className="p-3">Remediation Type</th>
                  <th className="p-3">Subsystem</th>
                  <th className="p-3 font-mono">Derate</th>
                  <th className="p-3 font-mono">Life Gain</th>
                  <th className="p-3 font-mono">Extended RUL</th>
                  <th className="p-3">Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {healingHistory.slice(0, 5).map((act: any, idx: number) => (
                  <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                    <td className="p-3 font-bold text-white text-[10px] truncate max-w-[100px]">{act.action_id}</td>
                    <td className="p-3 text-cyan-300">{act.engine_id}</td>
                    <td className="p-3 font-sans text-slate-300">{act.action_type.replace(/_/g, " ")}</td>
                    <td className="p-3 text-purple-300">{act.subsystem}</td>
                    <td className="p-3 text-amber-300">-{act.derate_pct}%</td>
                    <td className="p-3 text-emerald-300 font-bold">+{act.life_extension_pct}%</td>
                    <td className="p-3 text-emerald-400 font-bold">{act.extended_rul} cyc</td>
                    <td className="p-3">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                        {act.status}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Model Registry Table */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <GitBranch className="w-5 h-5 text-indigo-400" />
            <h3 className="font-bold text-sm text-white">Model Registry & Version Aliases</h3>
          </div>
          <span className="text-xs text-slate-400">Serving from local model repository</span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="p-3.5">Version</th>
                <th className="p-3.5">Alias</th>
                <th className="p-3.5">Algorithm</th>
                <th className="p-3.5 font-mono">Test RMSE</th>
                <th className="p-3.5 font-mono">NASA Score</th>
                <th className="p-3.5 font-mono">Registered At</th>
                <th className="p-3.5 font-mono">Data Hash</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {models.map((m: any, idx: number) => (
                <tr key={idx} className="hover:bg-slate-800/30 transition-colors">
                  <td className="p-3.5 font-bold text-white">{m.version}</td>
                  <td className="p-3.5">
                    <span className={`px-2.5 py-0.5 rounded-full text-[10px] font-bold ${
                      m.alias === "champion" 
                        ? "bg-emerald-500/20 text-emerald-300 border border-emerald-500/30"
                        : m.alias.includes("rejected")
                        ? "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                        : "bg-slate-800 text-slate-400"
                    }`}>
                      {m.alias}
                    </span>
                  </td>
                  <td className="p-3.5 font-sans text-slate-300">{m.algorithm}</td>
                  <td className="p-3.5 text-cyan-300 font-bold">{m.metrics?.rmse}</td>
                  <td className="p-3.5 text-purple-300">{m.metrics?.nasa_score}</td>
                  <td className="p-3.5 text-slate-400">{m.created_at}</td>
                  <td className="p-3.5 text-slate-500 text-[10px] truncate max-w-[120px]">{m.data_hash}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Sensor Distribution Drift Heatmap */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-400" />
            <h3 className="font-bold text-sm text-white">Sensor Distribution Drift (PSI & KS-Test)</h3>
          </div>
          <span className={`text-xs px-2.5 py-1 rounded-full font-bold ${
            driftReport?.drift_alert ? "badge-critical" : "badge-healthy"
          }`}>
            {driftReport?.drift_alert ? "DRIFT ALERT ACTIVE" : "STABLE DISTRIBUTIONS"}
          </span>
        </div>

        <p className="text-xs text-slate-400">
          Monitors Population Stability Index (PSI &gt; 0.20 triggers alert) and Kolmogorov-Smirnov 2-sample p-values on incoming telemetry.
        </p>

        <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-7 gap-3">
          {Object.entries(sensorDrifts).map(([sensor, val]: [string, any]) => (
            <div 
              key={sensor} 
              className={`p-3 rounded-2xl border text-xs font-mono space-y-1 transition-all ${
                val.is_drifting 
                  ? "bg-rose-500/10 border-rose-500/40 text-rose-300"
                  : "bg-slate-900/60 border-slate-800 text-slate-300"
              }`}
            >
              <div className="font-bold flex items-center justify-between">
                <span>{sensor.replace("_ewma", "")}</span>
                {val.is_drifting && <span className="text-[10px] text-rose-400 font-sans font-bold">DRIFT</span>}
              </div>
              <div className="text-[10px] text-slate-400">PSI: <b className="text-white">{val.psi}</b></div>
              <div className="text-[10px] text-slate-400">p-val: <b className="text-white">{val.ks_pvalue}</b></div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};
