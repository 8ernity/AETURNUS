import React, { useState, useEffect } from "react";
import { 
  Cpu, 
  TrendingDown, 
  Activity, 
  ShieldCheck, 
  AlertTriangle, 
  Clock, 
  Layers, 
  BarChart3,
  CheckCircle2,
  FileText,
  FileDown,
  Plane,
  PackageCheck,
  Wrench
} from "lucide-react";
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  Tooltip, 
  Legend, 
  CartesianGrid 
} from "recharts";

interface EngineDetailTabProps {
  selectedEngineId: string;
  currentSubset: string;
  onBackToFleet: () => void;
}

export const EngineDetailTab: React.FC<EngineDetailTabProps> = ({
  selectedEngineId,
  currentSubset,
  onBackToFleet
}) => {
  const [engineData, setEngineData] = useState<any>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedSensor, setSelectedSensor] = useState<string>("s3"); // s3 is T30 (HPC temp)

  useEffect(() => {
    if (!selectedEngineId) return;
    setLoading(true);
    fetch(`/api/v1/engines/${selectedEngineId}?subset=${currentSubset}`)
      .then((res) => res.json())
      .then((data) => {
        setEngineData(data);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Error fetching engine detail:", err);
        setLoading(false);
      });
  }, [selectedEngineId, currentSubset]);

  if (loading || !engineData) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex items-center gap-3 text-slate-400">
          <Activity className="w-6 h-6 animate-spin text-indigo-400" />
          <span>Loading telemetry curves for {selectedEngineId}...</span>
        </div>
      </div>
    );
  }

  const pred = engineData.prediction || {};
  const trajectory = engineData.trajectory || [];
  const shapBars = engineData.shapBars || engineData.shap_bars || [];
  const workOrder = engineData.work_order;
  const isCritical = (pred.conservative_rul || 999) <= 15;

  return (
    <div className="space-y-6">
      {/* Top Banner & Quick Controls */}
      <div className="glass-panel p-6 rounded-3xl flex flex-col md:flex-row md:items-center justify-between gap-4 border-indigo-500/25">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-indigo-600 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-600/30">
            <Cpu className="w-7 h-7 text-white" />
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h2 className="text-2xl font-black font-mono text-white tracking-tight">
                {engineData.engine_id}
              </h2>
              <span className="text-xs px-2.5 py-1 rounded-full font-mono bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                Regime #{engineData.regime}
              </span>
              <span className={`text-xs px-3 py-1 rounded-full font-bold uppercase tracking-wider ${
                pred.conservative_rul <= 15 ? "badge-critical" : pred.conservative_rul <= 30 ? "badge-high" : "badge-healthy"
              }`}>
                {pred.conservative_rul <= 15 ? "CRITICAL" : pred.conservative_rul <= 30 ? "HIGH" : "HEALTHY"}
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Turbofan Unit {engineData.unit} · Current Replay Cycle: <span className="text-cyan-300 font-mono font-bold">{engineData.current_cycle}</span> · Suspected Subsystem: <span className="text-purple-300 font-semibold">{shapBars[0]?.subsystem || "HPC"}</span>
            </p>
          </div>
        </div>

        {/* Prediction Summary & One-Click PDF Download */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-4 bg-slate-900/80 p-3 rounded-2xl border border-slate-800">
            <div className="text-right">
              <div className="text-[11px] text-slate-400 uppercase tracking-wider">Conservative RUL (p10)</div>
              <div className="text-2xl font-black font-mono text-white flex items-baseline justify-end gap-1">
                <span>{pred.conservative_rul}</span>
                <span className="text-xs text-slate-400 font-normal">cycles</span>
              </div>
            </div>
            <div className="h-10 w-px bg-slate-800"></div>
            <div className="text-right">
              <div className="text-[11px] text-slate-400 uppercase tracking-wider">Uncertainty Envelope</div>
              <div className="text-xs font-mono text-cyan-300">
                [{pred.rul_p10} - {pred.rul_p90}]
              </div>
              <div className="text-[10px] text-slate-500">p10 to p90 interval</div>
            </div>
          </div>

          <a
            href={`/api/v1/engines/${engineData.engine_id}/maintenance-report.pdf?subset=${currentSubset}`}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center gap-2 px-4 py-3 rounded-2xl bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-xs font-bold transition-all shadow-md group"
          >
            <FileDown className="w-4 h-4 text-indigo-300 group-hover:scale-110 transition-transform" />
            <span className="hidden sm:inline">Maintenance Report PDF</span>
          </a>
        </div>
      </div>

      {/* Main Charts: 1. RUL Prediction vs Ground Truth & 2. Health Index Degradation */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* RUL Prediction Curve with Confidence Envelope */}
        <div className="glass-panel p-6 rounded-3xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="w-5 h-5 text-indigo-400" />
              <h3 className="font-bold text-sm text-white">Remaining Useful Life (RUL) Trajectory</h3>
            </div>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1.5 text-indigo-400">
                <span className="w-3 h-0.5 bg-indigo-400 inline-block"></span> p50 Median
              </span>
              <span className="flex items-center gap-1.5 text-emerald-400">
                <span className="w-3 h-0.5 bg-emerald-400 inline-block"></span> Ground Truth
              </span>
            </div>
          </div>
          <p className="text-[11px] text-slate-400">
            Shaded blue envelope represents 80% confidence interval [p10–p90]. Model tracks linear degradation.
          </p>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={trajectory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="cycle" stroke="#6b7280" fontSize={11} tickLine={false} />
                <YAxis stroke="#6b7280" fontSize={11} tickLine={false} domain={[0, 'auto']} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#374151", borderRadius: "12px", fontSize: "12px" }}
                />
                <Area 
                  type="monotone" 
                  dataKey="rul_p90" 
                  stroke="none" 
                  fill="#6366f1" 
                  fillOpacity={0.15} 
                  name="Uncertainty Band (p90)"
                />
                <Area 
                  type="monotone" 
                  dataKey="rul_p10" 
                  stroke="none" 
                  fill="#090a10" 
                  fillOpacity={1.0} 
                  name="Uncertainty Floor (p10)"
                />
                <Line 
                  type="monotone" 
                  dataKey="rul_p50" 
                  stroke="#818cf8" 
                  strokeWidth={2.5} 
                  dot={false} 
                  name="Predicted RUL (p50)"
                />
                <Line 
                  type="monotone" 
                  dataKey="true_rul" 
                  stroke="#34d399" 
                  strokeWidth={2} 
                  strokeDasharray="4 4" 
                  dot={false} 
                  name="Ground Truth RUL"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* 2. Health Index (0-1) & Anomaly Onset */}
        <div className="glass-panel p-6 rounded-3xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-400" />
              <h3 className="font-bold text-sm text-white">Continuous Health Index (HI) &amp; Anomaly Onset</h3>
            </div>
            <div className="flex items-center gap-3 text-[11px]">
              <span className="flex items-center gap-1.5 text-cyan-400">
                <span className="w-3 h-0.5 bg-cyan-400 inline-block"></span> Health Index (0-1)
              </span>
              <span className="flex items-center gap-1.5 text-rose-400">
                <span className="w-3 h-0.5 bg-rose-400 inline-block"></span> Anomaly Score
              </span>
            </div>
          </div>
          <p className="text-[11px] text-slate-400">
            PCA projection smoothed monotonically. Degradation onset flagged when Health Index deviates &gt; 3-sigma.
          </p>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="cycle" stroke="#6b7280" fontSize={11} tickLine={false} />
                <YAxis stroke="#6b7280" fontSize={11} tickLine={false} domain={[0, 1]} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#374151", borderRadius: "12px", fontSize: "12px" }}
                />
                <Line 
                  type="monotone" 
                  dataKey="health_index" 
                  stroke="#06b6d4" 
                  strokeWidth={2.5} 
                  dot={false} 
                  name="Health Index (HI)"
                />
                <Line 
                  type="monotone" 
                  dataKey="anomaly_score" 
                  stroke="#fb7185" 
                  strokeWidth={2} 
                  dot={false} 
                  name="Isolation Forest Anomaly"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Sensor Overlay (Raw vs Cleaned Causal EWMA) & Explainability (SHAP Subsystem Attribution) */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Sensor Telemetry Curve (2 Cols) */}
        <div className="lg:col-span-2 glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="flex items-center gap-2">
              <Layers className="w-5 h-5 text-purple-400" />
              <h3 className="font-bold text-sm text-white">Sensor Telemetry (Raw vs Causal EWMA Filter)</h3>
            </div>

            {/* Sensor selector */}
            <div className="flex items-center gap-1.5 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
              {[
                { id: "s2", name: "s2 (T24 · LPC)" },
                { id: "s3", name: "s3 (T30 · HPC)" },
                { id: "s4", name: "s4 (T50 · LPT)" },
                { id: "s7", name: "s7 (P30 · HPC)" }
              ].map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSelectedSensor(s.id)}
                  className={`px-2.5 py-1 text-xs rounded-lg transition-all font-medium ${
                    selectedSensor === s.id
                      ? "bg-purple-600 text-white font-semibold"
                      : "text-slate-400 hover:text-slate-200"
                  }`}
                >
                  {s.name}
                </button>
              ))}
            </div>
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" vertical={false} />
                <XAxis dataKey="cycle" stroke="#6b7280" fontSize={11} tickLine={false} />
                <YAxis stroke="#6b7280" fontSize={11} tickLine={false} domain={['auto', 'auto']} />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#374151", borderRadius: "12px", fontSize: "12px" }}
                />
                <Line 
                  type="monotone" 
                  dataKey={`${selectedSensor}_raw`} 
                  stroke="#94a3b8" 
                  strokeWidth={1} 
                  dot={false} 
                  opacity={0.6}
                  name="Raw Sensor"
                />
                <Line 
                  type="monotone" 
                  dataKey={`${selectedSensor}_ewma`} 
                  stroke="#a855f7" 
                  strokeWidth={2.5} 
                  dot={false} 
                  name="Causal EWMA Filter"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Explainability / SHAP Sensor Attribution & Parts BOM preview (1 Col) */}
        <div className="glass-panel p-6 rounded-3xl flex flex-col justify-between">
          <div>
            <div className="flex items-center gap-2 mb-3">
              <BarChart3 className="w-5 h-5 text-indigo-400" />
              <h3 className="font-bold text-sm text-white">Top Contributing Sensors</h3>
            </div>
            <p className="text-[11px] text-slate-400 mb-4">
              Sensor attribution deviation from healthy baseline mapped directly to turbofan component bill-of-materials.
            </p>

            <div className="space-y-3">
              {shapBars.map((item: any, idx: number) => (
                <div key={idx} className="space-y-1">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-mono text-white font-semibold">{item.sensor} ({item.name})</span>
                    <span className="text-purple-300 font-semibold">{item.subsystem}</span>
                  </div>
                  <div className="h-2 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                      style={{ width: `${Math.min(100, Math.max(10, item.contribution * 100))}%` }}
                    ></div>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Linked Work Order Snippet */}
          {workOrder && (
            <div className="mt-6 pt-4 border-t border-slate-800 space-y-2">
              <div className="flex items-center justify-between text-xs">
                <span className="text-slate-400">Linked Work Order:</span>
                <span className="font-mono text-indigo-300 font-bold">{workOrder.work_order_id}</span>
              </div>
              <p className="text-xs text-slate-300 bg-slate-900/80 p-2.5 rounded-xl border border-slate-800">
                {workOrder.recommended_action}
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
