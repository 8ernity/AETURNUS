import React, { useState, useEffect, useMemo } from "react";
import toast from "react-hot-toast";
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
  Wrench,
  Compass,
  Loader2
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
  CartesianGrid,
  ReferenceLine,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis
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
  const [radarMode, setRadarMode] = useState<"degradation" | "health">("degradation");
  const [multiHorizonMode, setMultiHorizonMode] = useState<"stacked" | "cumulative">("stacked");
  const [isDownloadingPdf, setIsDownloadingPdf] = useState<boolean>(false);

  const handleDownloadPdf = async () => {
    if (isDownloadingPdf || !engineData?.engine_id) return;
    setIsDownloadingPdf(true);
    const toastId = toast.loading(`Generating Maintenance Report PDF for ${engineData.engine_id}...`);
    try {
      const res = await fetch(`/api/v1/engines/${engineData.engine_id}/maintenance-report.pdf?subset=${currentSubset}`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to generate report`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `maintenance_report_${engineData.engine_id}_${currentSubset}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success(`Maintenance Report PDF downloaded for ${engineData.engine_id}`, { id: toastId });
    } catch (err: any) {
      console.error("PDF download failed:", err);
      toast.error(`Download failed: ${err.message || "Error generating report"}`, { id: toastId });
    } finally {
      setIsDownloadingPdf(false);
    }
  };

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

  const pred = engineData?.prediction || {};
  const trajectory = engineData?.trajectory || [];
  const shapBars = engineData?.shapBars || engineData?.shap_bars || [];
  const workOrder = engineData?.work_order;
  const isCritical = (pred.conservative_rul || 999) <= 15;

  const rawSubsystemHealth = engineData?.subsystem_health;
  const subsystemHealthData = useMemo(() => {
    if (rawSubsystemHealth && rawSubsystemHealth.length > 0) {
      return rawSubsystemHealth.map((s: any) => ({
        ...s,
        attribution_share: s.attribution_share !== undefined ? s.attribution_share : s.degradation,
        nominal_baseline: 20.0
      }));
    }
    const stations = ["Fan / LPC", "HPC", "Combustor", "HPT", "LPT / Exhaust"];
    return stations.map((sub) => ({
      subsystem: sub,
      attribution_share: 20.0,
      share: 20.0,
      degradation: 20.0,
      health: 80.0,
      baseline: 20.0,
      nominal_baseline: 20.0,
      status: "NOMINAL"
    }));
  }, [rawSubsystemHealth]);

  // Multi-Horizon Failure Probabilities Evolution (H=10, 30, 60)
  const multiHorizonData = useMemo(() => {
    return trajectory.map((pt: any) => {
      let p10 = pt.fail_prob_10;
      let p30 = pt.fail_prob_30;
      let p60 = pt.fail_prob_60;

      // Fallback calculation if not in telemetry point
      if (p10 === undefined || p30 === undefined || p60 === undefined) {
        const rul = pt.rul_p50 || 100;
        p10 = Number((1.0 / (1.0 + Math.exp((rul - 10) / 4.0))).toFixed(3));
        p30 = Number((1.0 / (1.0 + Math.exp((rul - 30) / 7.0))).toFixed(3));
        p60 = Number((1.0 / (1.0 + Math.exp((rul - 60) / 12.0))).toFixed(3));
      }

      // Enforce monotonicity: p10 <= p30 <= p60
      p30 = Math.max(p10, p30);
      p60 = Math.max(p30, p60);

      const p30_inc = Math.max(0, p30 - p10);
      const p60_inc = Math.max(0, p60 - p30);
      const safe_margin = Math.max(0, 1.0 - p60);

      return {
        cycle: pt.cycle,
        p10_pct: Number((p10 * 100).toFixed(1)),
        p30_pct: Number((p30 * 100).toFixed(1)),
        p60_pct: Number((p60 * 100).toFixed(1)),
        p10_inc_pct: Number((p10 * 100).toFixed(1)),
        p30_inc_pct: Number((p30_inc * 100).toFixed(1)),
        p60_inc_pct: Number((p60_inc * 100).toFixed(1)),
        safe_pct: Number((safe_margin * 100).toFixed(1)),
        true_rul: pt.true_rul
      };
    });
  }, [trajectory]);

  const latestP10 = pred.fail_prob_10 !== undefined ? (pred.fail_prob_10 * 100).toFixed(1) : "0.0";
  const latestP30 = pred.fail_prob_30 !== undefined ? (pred.fail_prob_30 * 100).toFixed(1) : "0.0";
  const latestP60 = pred.fail_prob_60 !== undefined ? (pred.fail_prob_60 * 100).toFixed(1) : "0.0";
  const threshold30 = pred.classifier_threshold_30 !== undefined ? (pred.classifier_threshold_30 * 100).toFixed(0) : "70";
  const isRuleM3Fired = Number(latestP30) >= Number(threshold30);

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

          <button
            onClick={handleDownloadPdf}
            disabled={isDownloadingPdf}
            className="flex items-center gap-2 px-4 py-3 rounded-2xl bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-xs font-bold transition-all shadow-md group disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isDownloadingPdf ? (
              <Loader2 className="w-4 h-4 animate-spin text-indigo-300" />
            ) : (
              <FileDown className="w-4 h-4 text-indigo-300 group-hover:scale-110 transition-transform" />
            )}
            <span className="hidden sm:inline">
              {isDownloadingPdf ? "Generating PDF..." : "Maintenance Report PDF"}
            </span>
          </button>
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

      {/* Multi-Horizon Failure Probability Stacked Area Chart (H=10, 30, 60) */}
      <div className="glass-panel p-6 rounded-3xl space-y-4 border-indigo-500/20">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-indigo-500/10 border border-indigo-500/20 text-indigo-400">
              <Layers className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-sm text-white">Multi-Horizon Failure Probability Evolution</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-indigo-500/10 text-indigo-300 border border-indigo-500/20 uppercase font-bold">
                  H = 10, 30, 60 Cycles
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Probability of uncommanded failure across distinct dispatch windows as telemetry degrades toward end-of-life.
              </p>
            </div>
          </div>

          {/* Mode Switcher & M3 Rule Badge */}
          <div className="flex items-center gap-2 self-start sm:self-auto">
            {isRuleM3Fired && (
              <span className="px-2.5 py-1 rounded-lg text-[10px] font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30 flex items-center gap-1 animate-pulse">
                <AlertTriangle className="w-3 h-3 text-amber-400" />
                Rule M3 Active (P₃₀ ≥ τ₃₀)
              </span>
            )}
            <div className="flex items-center gap-1 bg-slate-900/80 p-1 rounded-xl border border-slate-800">
              <button
                onClick={() => setMultiHorizonMode("stacked")}
                className={`px-3 py-1 text-xs rounded-lg transition-all font-medium ${
                  multiHorizonMode === "stacked"
                    ? "bg-indigo-600 text-white font-semibold shadow-md shadow-indigo-600/30"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Stacked Risk Windows
              </button>
              <button
                onClick={() => setMultiHorizonMode("cumulative")}
                className={`px-3 py-1 text-xs rounded-lg transition-all font-medium ${
                  multiHorizonMode === "cumulative"
                    ? "bg-purple-600 text-white font-semibold shadow-md shadow-purple-600/30"
                    : "text-slate-400 hover:text-slate-200"
                }`}
              >
                Cumulative Horizons
              </button>
            </div>
          </div>
        </div>

        {/* Multi-Horizon KPI Strip */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {/* H = 10 */}
          <div className="bg-slate-900/60 p-3 rounded-2xl border border-rose-500/20">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-rose-300 font-semibold uppercase tracking-wider">H = 10 Cycles</span>
              <span className="w-2 h-2 rounded-full bg-rose-500"></span>
            </div>
            <div className="text-xl font-bold text-rose-400 font-mono mt-0.5">{latestP10}%</div>
            <div className="text-[10px] text-slate-500 font-mono">Immediate Critical Hazard</div>
          </div>

          {/* H = 30 */}
          <div className="bg-slate-900/60 p-3 rounded-2xl border border-amber-500/20">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-amber-300 font-semibold uppercase tracking-wider">H = 30 Cycles</span>
              <span className="w-2 h-2 rounded-full bg-amber-500"></span>
            </div>
            <div className="text-xl font-bold text-amber-400 font-mono mt-0.5">{latestP30}%</div>
            <div className="text-[10px] text-slate-500 font-mono">Triage Threshold: {threshold30}%</div>
          </div>

          {/* H = 60 */}
          <div className="bg-slate-900/60 p-3 rounded-2xl border border-indigo-500/20">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-indigo-300 font-semibold uppercase tracking-wider">H = 60 Cycles</span>
              <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
            </div>
            <div className="text-xl font-bold text-indigo-300 font-mono mt-0.5">{latestP60}%</div>
            <div className="text-[10px] text-slate-500 font-mono">Early Advisory Horizon</div>
          </div>

          {/* Action Trigger */}
          <div className="bg-slate-900/60 p-3 rounded-2xl border border-slate-800">
            <div className="flex items-center justify-between">
              <span className="text-[10px] text-slate-400 font-semibold uppercase tracking-wider">Triage Trigger</span>
              <span className={`w-2 h-2 rounded-full ${isRuleM3Fired ? "bg-amber-400 animate-ping" : "bg-emerald-400"}`}></span>
            </div>
            <div className={`text-sm font-bold font-mono mt-1 ${isRuleM3Fired ? "text-amber-300" : "text-emerald-400"}`}>
              {isRuleM3Fired ? "Escalated: Order Dispatched" : "Nominal: Standard Watch"}
            </div>
            <div className="text-[10px] text-slate-500 font-mono truncate">Rule M3: P(fail ≤ 30) ≥ τ₃₀</div>
          </div>
        </div>

        {/* Stacked / Cumulative Area Chart */}
        <div className="h-72 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={multiHorizonData} margin={{ top: 10, right: 20, bottom: 20, left: 0 }}>
              <defs>
                <linearGradient id="p10Grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.7} />
                  <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.2} />
                </linearGradient>
                <linearGradient id="p30Grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#f97316" stopOpacity={0.65} />
                  <stop offset="95%" stopColor="#f97316" stopOpacity={0.15} />
                </linearGradient>
                <linearGradient id="p60Grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#818cf8" stopOpacity={0.55} />
                  <stop offset="95%" stopColor="#818cf8" stopOpacity={0.1} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis 
                dataKey="cycle" 
                stroke="#64748b" 
                fontSize={11} 
                tickLine={false}
                label={{ value: "Operating Engine Flight Cycle", position: "insideBottom", offset: -12, fill: "#94a3b8", fontSize: 11 }}
              />
              <YAxis 
                domain={[0, 100]} 
                stroke="#64748b" 
                fontSize={11} 
                tickLine={false} 
                unit="%" 
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (active && payload && payload.length) {
                    const d = payload[0].payload;
                    return (
                      <div className="bg-slate-900/95 backdrop-blur-md border border-slate-700/80 p-3.5 rounded-2xl shadow-xl text-xs space-y-2 min-w-[220px]">
                        <div className="font-bold text-white border-b border-slate-800 pb-1.5 flex items-center justify-between">
                          <span>Flight Cycle {d.cycle}</span>
                          <span className="text-[10px] font-mono text-slate-400">RUL: {d.true_rul} cyc</span>
                        </div>
                        <div className="space-y-1 font-mono text-[11px]">
                          <div className="flex justify-between items-center text-rose-300">
                            <span className="flex items-center gap-1.5 font-sans">
                              <span className="w-2 h-2 rounded-full bg-rose-500"></span>
                              H = 10 cyc:
                            </span>
                            <span className="font-bold">{d.p10_pct}%</span>
                          </div>
                          <div className="flex justify-between items-center text-amber-300">
                            <span className="flex items-center gap-1.5 font-sans">
                              <span className="w-2 h-2 rounded-full bg-amber-500"></span>
                              H = 30 cyc:
                            </span>
                            <span className="font-bold">{d.p30_pct}%</span>
                          </div>
                          <div className="flex justify-between items-center text-indigo-300">
                            <span className="flex items-center gap-1.5 font-sans">
                              <span className="w-2 h-2 rounded-full bg-indigo-400"></span>
                              H = 60 cyc:
                            </span>
                            <span className="font-bold">{d.p60_pct}%</span>
                          </div>
                          {multiHorizonMode === "stacked" && (
                            <div className="flex justify-between items-center text-slate-400 pt-1 border-t border-slate-800/80 text-[10px]">
                              <span className="font-sans">Safe Operating Reserve (&gt;60):</span>
                              <span>{d.safe_pct}%</span>
                            </div>
                          )}
                        </div>
                      </div>
                    );
                  }
                  return null;
                }}
              />
              {/* Validation tuned threshold reference line */}
              <ReferenceLine 
                y={Number(threshold30)} 
                stroke="#f59e0b" 
                strokeDasharray="4 4" 
                strokeWidth={1.5}
                label={{ 
                  value: `τ₃₀ = ${threshold30}% Triage Threshold`, 
                  fill: "#fbbf24", 
                  fontSize: 10, 
                  position: "insideTopRight",
                  offset: 8 
                }} 
              />

              {multiHorizonMode === "stacked" ? (
                <>
                  <Area 
                    type="monotone" 
                    dataKey="p10_inc_pct" 
                    stackId="risk"
                    stroke="#f43f5e" 
                    strokeWidth={2} 
                    fill="url(#p10Grad)" 
                    name="Immediate Hazard (0-10 cyc)" 
                  />
                  <Area 
                    type="monotone" 
                    dataKey="p30_inc_pct" 
                    stackId="risk"
                    stroke="#f97316" 
                    strokeWidth={2} 
                    fill="url(#p30Grad)" 
                    name="Overhaul Window (10-30 cyc)" 
                  />
                  <Area 
                    type="monotone" 
                    dataKey="p60_inc_pct" 
                    stackId="risk"
                    stroke="#818cf8" 
                    strokeWidth={2} 
                    fill="url(#p60Grad)" 
                    name="Advisory Window (30-60 cyc)" 
                  />
                </>
              ) : (
                <>
                  <Area 
                    type="monotone" 
                    dataKey="p60_pct" 
                    stroke="#818cf8" 
                    strokeWidth={2} 
                    fill="url(#p60Grad)" 
                    name="P(fail ≤ 60 cyc)" 
                  />
                  <Area 
                    type="monotone" 
                    dataKey="p30_pct" 
                    stroke="#f97316" 
                    strokeWidth={2.5} 
                    fill="url(#p30Grad)" 
                    name="P(fail ≤ 30 cyc)" 
                  />
                  <Area 
                    type="monotone" 
                    dataKey="p10_pct" 
                    stroke="#f43f5e" 
                    strokeWidth={2.5} 
                    fill="url(#p10Grad)" 
                    name="P(fail ≤ 10 cyc)" 
                  />
                </>
              )}
            </AreaChart>
          </ResponsiveContainer>
        </div>

        {/* Legend Footnote */}
        <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 pt-2 border-t border-slate-800/80 gap-2">
          <div className="flex flex-wrap items-center gap-4">
            <span className="flex items-center gap-1.5 text-rose-300">
              <span className="w-2.5 h-2.5 rounded-full bg-rose-500"></span>
              {multiHorizonMode === "stacked" ? "Critical (0-10 cyc)" : "P(fail ≤ 10)"}
            </span>
            <span className="flex items-center gap-1.5 text-amber-300">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-500"></span>
              {multiHorizonMode === "stacked" ? "Overhaul (10-30 cyc)" : "P(fail ≤ 30)"}
            </span>
            <span className="flex items-center gap-1.5 text-indigo-300">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
              {multiHorizonMode === "stacked" ? "Advisory (30-60 cyc)" : "P(fail ≤ 60)"}
            </span>
          </div>
          <span className="font-mono text-slate-500 text-[10px]">
            Model: Multi-Horizon GBDT Classifiers with Monotone Risk Calibration
          </span>
        </div>
      </div>

      {/* 360° Gas-Path Attribution Share Radar */}
      <div className="glass-panel p-6 rounded-3xl space-y-4 border-indigo-500/20">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-purple-500/10 border border-purple-500/20 text-purple-400">
              <Compass className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h3 className="font-bold text-sm text-white">Attribution share</h3>
                <span className="text-[10px] font-mono px-2 py-0.5 rounded-md bg-purple-500/10 text-purple-300 border border-purple-500/20 uppercase font-bold">
                  Station |SHAP| (100% Sum)
                </span>
              </div>
              <p className="text-[11px] text-slate-400">
                Proportional |SHAP| attribution shares across 5 engine stations (Fan/LPC, HPC, Combustor, HPT, LPT/Exhaust). Uniform baseline is 20%.
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-center">
          {/* Radar Spider Visualizer */}
          <div className="lg:col-span-7 h-80 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <RadarChart cx="50%" cy="50%" outerRadius="75%" data={subsystemHealthData}>
                <PolarGrid stroke="#334155" strokeDasharray="3 3" />
                <PolarAngleAxis 
                  dataKey="subsystem" 
                  stroke="#94a3b8" 
                  fontSize={11} 
                  tick={{ fill: '#e2e8f0', fontWeight: 600 }} 
                />
                <PolarRadiusAxis 
                  angle={90} 
                  domain={[0, 100]} 
                  stroke="#475569" 
                  fontSize={10} 
                  tick={{ fill: '#64748b' }} 
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0f172a", borderColor: "#334155", borderRadius: "12px", fontSize: "12px" }}
                  formatter={(val: any, name: any) => [`${val}%`, name]}
                />
                <Radar 
                  name="Station Attribution Share (%)"
                  dataKey="attribution_share" 
                  stroke="#a855f7" 
                  fill="#a855f7" 
                  fillOpacity={0.35} 
                />
                <Radar 
                  name="Uniform Baseline (20%)" 
                  dataKey="baseline" 
                  stroke="#38bdf8" 
                  fill="#38bdf8" 
                  fillOpacity={0.08} 
                  strokeDasharray="4 4" 
                />
                <Legend wrapperStyle={{ fontSize: "11px", paddingTop: "8px" }} />
              </RadarChart>
            </ResponsiveContainer>
          </div>

          {/* Subsystem Health Cards Breakdown */}
          <div className="lg:col-span-5 space-y-2.5">
            <div className="flex items-center justify-between text-xs text-slate-400 pb-1 border-b border-slate-800">
              <span className="font-semibold uppercase tracking-wider text-[10px]">Gas-Path Station</span>
              <span className="font-mono text-[10px]">Attribution Share</span>
            </div>
            {subsystemHealthData.map((item: any, idx: number) => {
              const rank = item.driver_rank || (idx + 1);
              const isTop = rank === 1;
              const isSecond = rank === 2;
              const share = item.attribution_share !== undefined ? item.attribution_share : item.degradation;
              return (
                <div key={idx} className="bg-slate-900/80 hover:bg-slate-850 p-3 rounded-2xl border border-slate-800 transition-all space-y-1.5">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-2">
                      <span className="font-bold text-xs text-white tracking-wide">{item.subsystem}</span>
                      <span className={`text-[9px] font-mono px-2 py-0.5 rounded-full font-bold uppercase tracking-wider ${
                        isTop ? "bg-purple-500/20 text-purple-300 border border-purple-500/40" :
                        isSecond ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/40" :
                        "bg-slate-800 text-slate-400 border border-slate-700"
                      }`}>
                        #{rank} Driver
                      </span>
                    </div>
                    <div className="text-xs font-mono font-bold">
                      <span className={isTop ? "text-purple-300" : isSecond ? "text-indigo-300" : "text-slate-300"}>
                        {share}%
                      </span>
                    </div>
                  </div>

                  {/* Gradient gauge bar */}
                  <div className="h-2 w-full bg-slate-950 rounded-full overflow-hidden flex shadow-inner">
                    <div 
                      className={`h-full transition-all duration-500 ${
                        isTop ? "bg-gradient-to-r from-indigo-500 to-purple-500" :
                        isSecond ? "bg-gradient-to-r from-blue-500 to-indigo-500" :
                        "bg-slate-700"
                      }`}
                      style={{ width: `${Math.min(100, Math.max(0, share))}%` }}
                    />
                  </div>
                </div>
              );
            })}
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

        {/* Explainability / SHAP Sensor Attribution & Multi-Horizon Risk (1 Col) */}
        <div className="glass-panel p-6 rounded-3xl flex flex-col justify-between space-y-6">
          <div>
            <div className="flex items-center justify-between mb-2">
              <div className="flex items-center gap-2">
                <BarChart3 className="w-5 h-5 text-indigo-400" />
                <h3 className="font-bold text-sm text-white">Top Contributing Sensors (SHAP)</h3>
              </div>
              <span className="text-[10px] font-mono text-indigo-300 bg-indigo-500/10 px-2 py-0.5 rounded-md border border-indigo-500/20">
                TreeExplainer
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mb-3">
              Signed SHAP contributions to remaining life. Red accelerates degradation; blue indicates slower wear relative to baseline.
            </p>

            {/* Signed SHAP Bars */}
            <div className="space-y-3">
              {shapBars.map((item: any, idx: number) => {
                const isShortening = item.direction === "shortens_life" || (item.shap_value !== undefined && item.shap_value < 0);
                const absVal = Math.abs(item.shap_value !== undefined ? item.shap_value : (item.contribution || 0.1));
                const widthPct = Math.min(100, Math.max(12, absVal * 30));
                return (
                  <div key={idx} className="space-y-1 bg-slate-900/60 p-2.5 rounded-xl border border-slate-800">
                    <div className="flex items-center justify-between text-xs">
                      <div className="flex items-center gap-1.5">
                        <span className="font-mono text-white font-bold">{item.sensor}</span>
                        <span className="text-slate-400 text-[11px]">({item.name || item.sensor})</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-[10px] px-2 py-0.5 rounded-md font-mono bg-purple-500/15 text-purple-300 border border-purple-500/30">
                          {item.subsystem}
                        </span>
                        <span className={`text-[10px] font-bold uppercase tracking-wider ${
                          isShortening ? "text-rose-400" : "text-sky-400"
                        }`}>
                          {isShortening ? "Shortens Life" : "Prolongs Life"}
                        </span>
                      </div>
                    </div>
                    <div className="flex items-center gap-2 pt-1">
                      <div className="h-2 flex-1 bg-slate-800 rounded-full overflow-hidden">
                        <div 
                          className={`h-full rounded-full ${
                            isShortening 
                              ? "bg-gradient-to-r from-amber-500 to-rose-500" 
                              : "bg-gradient-to-r from-cyan-500 to-blue-500"
                          }`}
                          style={{ width: `${widthPct}%` }}
                        ></div>
                      </div>
                      <span className="text-[10px] font-mono text-slate-300 min-w-[42px] text-right">
                        {item.shap_value !== undefined ? (item.shap_value > 0 ? `+${item.shap_value.toFixed(2)}` : item.shap_value.toFixed(2)) : `${(item.contribution * 10).toFixed(2)}`}
                      </span>
                    </div>
                  </div>
                );
              })}
            </div>

            {/* Disclaimer */}
            <p className="text-[10px] text-slate-500 italic mt-3">
              {engineData.subsystem_disclaimer || "Subsystem mapping is an indicative association, not a diagnosis. For decision support only."}
            </p>
          </div>

          {/* Multi-Horizon Failure Risk */}
          <div className="pt-4 border-t border-slate-800 space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-bold text-white uppercase tracking-wider">Multi-Horizon Risk</span>
              <span className="text-[10px] font-mono text-slate-400">Calibrated P(Fail ≤ H)</span>
            </div>
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: "10 Cycles", prob: pred.fail_prob_10 ?? 0.05, thresh: 0.35 },
                { label: "30 Cycles", prob: pred.fail_prob_30 ?? 0.45, thresh: pred.classifier_threshold_30 ?? 0.50 },
                { label: "60 Cycles", prob: pred.fail_prob_60 ?? 0.85, thresh: 0.50 }
              ].map((h, i) => {
                const pct = Math.round(h.prob * 100);
                const isHigh = h.prob >= h.thresh;
                return (
                  <div key={i} className="bg-slate-900/90 p-2 rounded-xl border border-slate-800 text-center space-y-1">
                    <div className="text-[10px] text-slate-400 font-medium">{h.label}</div>
                    <div className={`text-base font-black font-mono ${isHigh ? "text-rose-400" : "text-emerald-400"}`}>
                      {pct}%
                    </div>
                    <div className="text-[9px] text-slate-500">
                      Thresh: {Math.round(h.thresh * 100)}%
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Linked Work Order Snippet */}
          {workOrder && (
            <div className="pt-3 border-t border-slate-800 space-y-2">
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
