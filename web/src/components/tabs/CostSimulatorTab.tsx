import React, { useState, useEffect } from "react";
import { authHeaders, hasApiKey } from "../../lib/api";
import toast from "react-hot-toast";
import { 
  DollarSign, 
  TrendingUp, 
  ShieldCheck, 
  AlertTriangle, 
  Sliders, 
  Zap, 
  Plane, 
  ArrowRight, 
  CheckCircle2, 
  RefreshCw, 
  PieChart as PieIcon, 
  BarChart3, 
  Info,
  Scale,
  Sparkles,
  FileSpreadsheet
} from "lucide-react";
import { 
  ResponsiveContainer, 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  ReferenceLine, 
  Legend, 
  BarChart, 
  Bar, 
  Cell 
} from "recharts";

interface CostSimulatorTabProps {
  currentSubset: string;
}

export const CostSimulatorTab: React.FC<CostSimulatorTabProps> = ({ currentSubset }) => {
  const [loading, setLoading] = useState<boolean>(true);
  const [curveLoading, setCurveLoading] = useState<boolean>(false);
  
  // Interactive Simulation Parameters
  const [threshold, setThreshold] = useState<number>(30);
  const [useConservative, setUseConservative] = useState<boolean>(true);
  const [costCatastrophic, setCostCatastrophic] = useState<number>(2500000);
  const [costOverhaul, setCostOverhaul] = useState<number>(220000);
  const [costWastedCycle, setCostWastedCycle] = useState<number>(1200);
  const [costInspection, setCostInspection] = useState<number>(12500);

  // Results State
  const [simData, setSimData] = useState<any | null>(null);
  const [curveData, setCurveData] = useState<any | null>(null);
  const [authRequired, setAuthRequired] = useState<boolean>(false);

  interface SimulationOverrides {
    threshold?: number;
    costCatastrophic?: number;
    costOverhaul?: number;
    costWastedCycle?: number;
    costInspection?: number;
  }

  // Fetch simulation. Authenticated clients can POST custom unit economics;
  // unauthenticated viewers still get the server-default simulation through
  // the read-only GET route instead of a page full of em-dashes.
  const fetchSimulation = (overrides: SimulationOverrides = {}) => {
    const effectiveThreshold = overrides.threshold ?? threshold;
    const effectiveCatastrophic = overrides.costCatastrophic ?? costCatastrophic;
    const effectiveOverhaul = overrides.costOverhaul ?? costOverhaul;
    const effectiveWasted = overrides.costWastedCycle ?? costWastedCycle;
    const effectiveInspection = overrides.costInspection ?? costInspection;
    const canSubmitCustomCosts = hasApiKey();

    setLoading(true);
    const payload = {
      subset: currentSubset,
      threshold: effectiveThreshold,
      use_conservative: useConservative,
      cost_catastrophic: effectiveCatastrophic,
      cost_overhaul: effectiveOverhaul,
      cost_wasted_cycle: effectiveWasted,
      cost_inspection: effectiveInspection
    };
    const endpoint = canSubmitCustomCosts
      ? "/api/v1/economics/simulate"
      : `/api/v1/economics/simulate?subset=${encodeURIComponent(currentSubset)}&threshold=${effectiveThreshold}&use_conservative=${useConservative}`;

    fetch(endpoint, canSubmitCustomCosts
      ? {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify(payload)
        }
      : undefined)
      .then(async (response) => {
        if (!response.ok) {
          if (response.status === 401 || response.status === 403) {
            setAuthRequired(true);
          }
          throw new Error(`Simulation request failed (${response.status})`);
        }
        return response.json();
      })
      .then((data) => {
        setSimData(data);
        setAuthRequired(false);
        setLoading(false);
      })
      .catch((err) => {
        console.error("Simulation fetch error:", err);
        setLoading(false);
      });
  };

  // Fetch U-Curve
  const fetchCurve = () => {
    setCurveLoading(true);
    fetch(`/api/v1/economics/curve?subset=${currentSubset}&use_conservative=${useConservative}&min_threshold=5&max_threshold=60`)
      .then((r) => r.json())
      .then((data) => {
        setCurveData(data);
        setCurveLoading(false);
      })
      .catch((err) => {
        console.error("Curve fetch error:", err);
        setCurveLoading(false);
      });
  };

  useEffect(() => {
    fetchSimulation();
    fetchCurve();
  }, [currentSubset, useConservative]);

  const handleApplyPreset = (preset: "commercial" | "cargo" | "military") => {
    if (preset === "commercial") {
      setCostCatastrophic(3000000);
      setCostOverhaul(240000);
      setCostWastedCycle(1400);
      setCostInspection(15000);
      fetchSimulation({
        costCatastrophic: 3000000,
        costOverhaul: 240000,
        costWastedCycle: 1400,
        costInspection: 15000
      });
      toast.success("Applied Commercial Airline Preset ($3.0M AOG, $240k Overhaul)");
    } else if (preset === "cargo") {
      setCostCatastrophic(2000000);
      setCostOverhaul(200000);
      setCostWastedCycle(1000);
      setCostInspection(10000);
      fetchSimulation({
        costCatastrophic: 2000000,
        costOverhaul: 200000,
        costWastedCycle: 1000,
        costInspection: 10000
      });
      toast.success("Applied Cargo Operator Preset ($2.0M AOG, $200k Overhaul)");
    } else {
      setCostCatastrophic(4500000);
      setCostOverhaul(350000);
      setCostWastedCycle(2200);
      setCostInspection(25000);
      fetchSimulation({
        costCatastrophic: 4500000,
        costOverhaul: 350000,
        costWastedCycle: 2200,
        costInspection: 25000
      });
      toast.success("Applied Military / Heavy Preset ($4.5M AOG, $350k Overhaul)");
    }
  };

  const summary = simData?.summary;
  const policies = simData?.policies;
  const breakdownChart = simData?.breakdown_chart || [];
  const ledger = simData?.ledger || [];
  const optimalTau = curveData?.optimal_threshold || 11;
  const curvePoints = curveData?.curve || [];

  return (
    <div className="space-y-6">
      {(!hasApiKey() || authRequired) && (
        <div className="bg-slate-900/70 border border-slate-700 text-slate-300 p-4 rounded-2xl flex items-center gap-3">
          <Info className="w-5 h-5 text-cyan-400 shrink-0" />
          <div className="text-xs space-y-0.5">
            <div className="font-bold text-slate-200">Read-only default economics</div>
            <div>
              Custom AOG, overhaul, inspection, and wasted-life inputs require an API key.
              Click <span className="font-mono text-cyan-300">CONNECT</span> in the header to
              apply your own assumptions.
            </div>
          </div>
        </div>
      )}

      {/* Stale Artifact Warning Banner */}
      {simData?.stale && (
        <div className="bg-amber-500/15 border border-amber-500/30 text-amber-300 p-4 rounded-2xl flex items-center gap-3">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
          <div className="text-xs space-y-0.5">
            <div className="font-bold text-amber-200">Artifact Stale Warning</div>
            <div>
              Simulation bundle hash on disk does not match run manifest. Run <code className="bg-black/30 px-1 py-0.5 rounded font-mono">make evaluate</code> to synchronize.
              {simData?.stale_reason && <span className="ml-1 text-amber-400/80">({simData.stale_reason})</span>}
            </div>
          </div>
        </div>
      )}

      {/* Top Banner: Financial ROI Overview */}
      <div className="glass-panel p-6 rounded-3xl relative overflow-hidden">
        <div className="absolute top-0 right-0 w-96 h-96 bg-emerald-500/10 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-1/3 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-xl bg-emerald-500/20 border border-emerald-500/30 flex items-center justify-center text-emerald-400">
                <DollarSign className="w-5 h-5" />
              </div>
              <h2 className="text-xl font-black text-white tracking-tight">
                Operational Cost &amp; Fleet Maintenance ROI Simulator
              </h2>
              <span className="text-[10px] font-mono px-2.5 py-0.5 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 font-bold">
                {currentSubset} Fleet Analytics
              </span>
            </div>
            <p className="text-xs text-slate-400 max-w-3xl leading-relaxed">
              Mathematical lifecycle economics modeling: compare unscheduled Aircraft on Ground (AOG) catastrophic failure penalties against AETURNUS conservative predictive maintenance dispatch and static overhaul policies.
            </p>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={() => {
                setThreshold(optimalTau);
                fetchSimulation({ threshold: optimalTau });
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-xs font-semibold transition-all shadow-md"
            >
              <Sparkles className="w-4 h-4 text-cyan-300" />
              <span>Snap to Optimal Threshold (τ* = {optimalTau} cyc)</span>
            </button>
            <button
              onClick={() => fetchSimulation()}
              className="p-2 rounded-xl bg-slate-900 border border-slate-700 text-slate-400 hover:text-white transition-colors"
              title="Refresh Simulation"
            >
              <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
            </button>
          </div>
        </div>

        {/* 4 Financial & Break-Even Metric Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-6">
          {/* Lead Metric: Break-Even Cost Ratio */}
          <div className="bg-slate-950/70 p-4 rounded-2xl border border-indigo-500/30 space-y-1 shadow-lg shadow-indigo-950/20">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-semibold uppercase tracking-wider">
              <span>Break-Even Cost Ratio</span>
              <span className="text-[10px] text-cyan-400 font-bold bg-cyan-500/10 px-1.5 py-0.5 rounded font-mono">
                C_cat / C_ovh Parity
              </span>
            </div>
            <div className="text-2xl font-black font-mono text-cyan-300">
              {summary?.breakeven_cost_ratio !== undefined ? `${summary.breakeven_cost_ratio}x` : (loading ? "..." : "—")}
            </div>
            <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between">
              <span>Assumed Ratio:</span>
              <span className="text-emerald-300 font-bold">
                {summary?.current_cost_ratio !== undefined ? `${summary.current_cost_ratio}x (Favorable)` : "—"}
              </span>
            </div>
          </div>

          {/* Net Fleet Savings vs Fixed Interval & vs Reactive */}
          <div className="bg-slate-950/70 p-4 rounded-2xl border border-emerald-500/30 space-y-1 shadow-lg shadow-emerald-950/20">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-semibold uppercase tracking-wider">
              <span>Net Fleet Savings</span>
              <span className="text-[10px] text-emerald-400 font-bold bg-emerald-500/10 px-1.5 py-0.5 rounded">
                vs Fixed Schedule
              </span>
            </div>
            <div className={`text-2xl font-black font-mono ${summary && summary.net_savings_vs_fixed < 0 ? "text-amber-400" : "text-emerald-400"}`}>
              {summary?.net_savings_vs_fixed !== undefined 
                ? `${summary.net_savings_vs_fixed >= 0 ? "+" : ""}$${(summary.net_savings_vs_fixed / 1e6).toFixed(1)}M` 
                : (loading ? "..." : "—")}
            </div>
            <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between">
              <span>vs Reactive Run-to-Failure:</span>
              <span className="text-emerald-300 font-bold">
                {summary?.net_savings_vs_reactive !== undefined 
                  ? `${summary.net_savings_vs_reactive >= 0 ? "+" : ""}$${(summary.net_savings_vs_reactive / 1e6).toFixed(1)}M` 
                  : "—"}
              </span>
            </div>
          </div>

          {/* AETURNUS Total Cost */}
          <div className="bg-slate-950/70 p-4 rounded-2xl border border-slate-700/60 space-y-1">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-semibold uppercase tracking-wider">
              <span>AETURNUS Fleet Cost</span>
              <span className="text-[10px] text-indigo-400 font-bold bg-indigo-500/10 px-1.5 py-0.5 rounded">
                Predictive AI
              </span>
            </div>
            <div className="text-2xl font-black font-mono text-white">
              {summary?.aeturnus_total_cost !== undefined ? `$${(summary.aeturnus_total_cost / 1e6).toFixed(2)}M` : (loading ? "..." : "—")}
            </div>
            <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between">
              <span>Fixed Baseline:</span>
              <span className="text-slate-400">
                {summary?.fixed_total_cost !== undefined ? `$${(summary.fixed_total_cost / 1e6).toFixed(1)}M` : "—"}
              </span>
            </div>
          </div>

          {/* Optimal Threshold Tau */}
          <div className="bg-slate-950/70 p-4 rounded-2xl border border-purple-500/30 space-y-1">
            <div className="flex items-center justify-between text-[11px] text-slate-400 font-semibold uppercase tracking-wider">
              <span>Optimal Threshold τ*</span>
              <span className="text-[10px] text-purple-300 font-bold bg-purple-500/10 px-1.5 py-0.5 rounded font-mono">
                Min Cost Point
              </span>
            </div>
            <div className="text-2xl font-black font-mono text-purple-300 flex items-center gap-1.5">
              <span>{optimalTau}</span>
              <span className="text-xs font-sans text-slate-500 font-normal">cycles</span>
            </div>
            <div className="text-[10px] text-slate-400 font-mono flex items-center justify-between">
              <span>Active Threshold:</span>
              <span className="text-cyan-400 font-bold">{threshold} cyc</span>
            </div>
          </div>
        </div>
      </div>

      {/* Middle Row: Interactive U-Curve Optimizer & Policy Breakdown Stacked Bar Chart */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left Column: Mathematical U-Shaped Cost Curve (7 Cols) */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Scale className="w-5 h-5 text-indigo-400" />
              <h3 className="font-bold text-sm text-white">
                Threshold Optimization Curve (U-Shaped Expected Cost)
              </h3>
            </div>
            <span className="text-[10px] font-mono text-cyan-300 bg-slate-900 px-2 py-0.5 rounded border border-cyan-500/20">
              Optimal τ* = {optimalTau} cyc
            </span>
          </div>
          <p className="text-xs text-slate-400">
            Sweeps maintenance decision threshold $\tau$ from 5 to 60 cycles. Demonstrates the mathematical trade-off: late thresholds trigger catastrophic failure penalties, whereas early thresholds waste unexpired RUL life.
          </p>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={curvePoints} margin={{ top: 10, right: 20, left: 0, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis 
                  dataKey="threshold" 
                  stroke="#64748b" 
                  fontSize={11}
                  label={{ value: "Decision Threshold τ (Cycles)", position: "insideBottom", offset: -10, fill: "#94a3b8", fontSize: 11 }}
                />
                <YAxis 
                  stroke="#64748b" 
                  fontSize={11}
                  unit="$M"
                  domain={['auto', 'auto']}
                />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0b0d19", border: "1px solid rgba(99, 102, 241, 0.3)", borderRadius: "1rem", fontSize: "11px" }}
                  formatter={(value: any, name: any) => [`$${value}M`, name]}
                  labelFormatter={(lbl) => `Threshold: ${lbl} cycles`}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: "11px" }} />
                
                <ReferenceLine 
                  x={optimalTau} 
                  stroke="#22c55e" 
                  strokeDasharray="4 4" 
                  strokeWidth={2}
                  label={{ value: `Optimal τ* (${optimalTau} cyc)`, fill: "#22c55e", fontSize: 10, position: "top" }} 
                />
                <ReferenceLine 
                  x={threshold} 
                  stroke="#38bdf8" 
                  strokeDasharray="2 2" 
                  strokeWidth={2}
                  label={{ value: `Active (${threshold} cyc)`, fill: "#38bdf8", fontSize: 10, position: "insideTopLeft" }} 
                />

                <Line 
                  type="monotone" 
                  dataKey="total_cost_m" 
                  name="Total Fleet Cost ($M)" 
                  stroke="#818cf8" 
                  strokeWidth={3} 
                  dot={false}
                  activeDot={{ r: 6, fill: "#818cf8" }}
                />
                <Line 
                  type="monotone" 
                  dataKey="catastrophic_cost_m" 
                  name="AOG Failure Penalty ($M)" 
                  stroke="#f43f5e" 
                  strokeWidth={1.5} 
                  dot={false} 
                  strokeDasharray="5 5"
                />
                <Line 
                  type="monotone" 
                  dataKey="wasted_life_cost_m" 
                  name="Wasted Life Depreciation ($M)" 
                  stroke="#eab308" 
                  strokeWidth={1.5} 
                  dot={false}
                  strokeDasharray="3 3"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400 bg-slate-900/60 p-3 rounded-2xl border border-slate-800">
            <span>Low Threshold (&lt;15 cyc): High in-flight failure risk</span>
            <span className="text-emerald-400 font-bold font-mono">Sweet Spot: τ* = {optimalTau} cyc</span>
            <span>High Threshold (&gt;35 cyc): Excessive wasted life</span>
          </div>
        </div>

        {/* Right Column: Comparative Policy Breakdown (5 Cols) */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <BarChart3 className="w-5 h-5 text-cyan-400" />
              <h3 className="font-bold text-sm text-white">Policy Cost Breakdown ($M)</h3>
            </div>
            <span className="text-xs text-slate-500 font-mono">3-Policy Comparison</span>
          </div>
          <p className="text-xs text-slate-400">
            Expenditure across scheduled overhauls, emergency AOG recovery, and unexpired life depreciation.
          </p>

          <div className="h-72 w-full pt-2">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={breakdownChart} margin={{ top: 20, right: 10, left: -10, bottom: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis dataKey="policy" stroke="#64748b" fontSize={11} />
                <YAxis stroke="#64748b" fontSize={11} unit="$M" />
                <Tooltip 
                  contentStyle={{ backgroundColor: "#0b0d19", border: "1px solid rgba(99, 102, 241, 0.3)", borderRadius: "1rem", fontSize: "11px" }}
                  formatter={(v: any) => [`$${v}M`, ""]}
                />
                <Legend verticalAlign="top" height={36} wrapperStyle={{ fontSize: "10px" }} />
                <Bar dataKey="Catastrophic AOG ($M)" stackId="a" fill="#f43f5e" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Scheduled Overhauls ($M)" stackId="a" fill="#06b6d4" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Wasted Depreciation ($M)" stackId="a" fill="#eab308" radius={[0, 0, 0, 0]} />
                <Bar dataKey="Line Inspection ($M)" stackId="a" fill="#a855f7" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="grid grid-cols-3 gap-2 text-center text-[10px] font-mono">
            <div className="p-2 rounded-xl bg-slate-900 border border-rose-500/20 text-rose-300">
              <div className="opacity-75">Reactive</div>
              <div className="font-bold text-sm text-white">
                {policies?.reactive?.total_cost_m !== undefined ? `$${policies.reactive.total_cost_m}M` : (loading ? "..." : "—")}
              </div>
            </div>
            <div className="p-2 rounded-xl bg-slate-900 border border-amber-500/20 text-amber-300">
              <div className="opacity-75">Fixed Schedule</div>
              <div className="font-bold text-sm text-white">
                {policies?.fixed_interval?.total_cost_m !== undefined ? `$${policies.fixed_interval.total_cost_m}M` : (loading ? "..." : "—")}
              </div>
            </div>
            <div className="p-2 rounded-xl bg-slate-900 border border-emerald-500/20 text-emerald-300">
              <div className="opacity-75">AETURNUS</div>
              <div className="font-bold text-sm text-white">
                {policies?.aeturnus?.total_cost_m !== undefined ? `$${policies.aeturnus.total_cost_m}M` : (loading ? "..." : "—")}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Bottom Controls: Financial Sliders & Interactive Decision Engine */}
      <div className="glass-panel p-6 rounded-3xl space-y-5">
        {/* Illustrative Unit Economics Disclaimer Banner */}
        <div className="flex items-start gap-2.5 p-3.5 rounded-2xl bg-amber-500/10 border border-amber-500/30 text-amber-200 text-xs">
          <Info className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold">Illustrative Unit Economics Disclaimer:</span> All monetary figures ($ catastrophic AOG penalty, depot overhaul, line inspection, and unexpired life depreciation) are editable illustrative assumptions for comparative scenario modeling. Modify these inputs to reflect your fleet's actual contractual unit economics.
          </div>
        </div>

        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4 border-b border-indigo-500/15 pb-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            <div>
              <h3 className="font-bold text-base text-white">Illustrative Operational Assumptions (Editable)</h3>
              <p className="text-xs text-slate-400">
                Adjust airline operational unit economics and observe instant recalculation across the entire fleet.
              </p>
            </div>
          </div>

          {/* Scenario Presets */}
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-medium mr-1">Scenario Presets (Illustrative):</span>
            <button
              onClick={() => handleApplyPreset("commercial")}
              className="px-2.5 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition-all"
            >
              Commercial Airline
            </button>
            <button
              onClick={() => handleApplyPreset("cargo")}
              className="px-2.5 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition-all"
            >
              Cargo Operator
            </button>
            <button
              onClick={() => handleApplyPreset("military")}
              className="px-2.5 py-1 rounded-lg text-xs bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition-all"
            >
              Military / Heavy
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Threshold Slider */}
          <div className="space-y-2 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center gap-1.5">
                <span className="text-slate-300 font-semibold">Dispatch Threshold (τ)</span>
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-semibold">POLICY</span>
              </div>
              <span className="font-mono text-cyan-300 font-bold text-sm">{threshold} cycles</span>
            </div>
            <input 
              type="range" 
              min="5" 
              max="60" 
              step="1"
              value={threshold}
              onChange={(e) => setThreshold(parseInt(e.target.value))}
              onMouseUp={() => fetchSimulation()}
              onTouchEnd={() => fetchSimulation()}
              className="w-full accent-cyan-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>5 cyc (Aggressive)</span>
              <span>60 cyc (Conservative)</span>
            </div>
          </div>

          {/* Decision Uncertainty Mode */}
          <div className="space-y-2 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <span className="text-slate-300 font-semibold">Uncertainty Risk Boundary</span>
              <span className="font-mono text-emerald-400 font-bold text-xs">
                {useConservative ? "Conservative (p10 Bound)" : "Median Point (p50)"}
              </span>
            </div>
            <div className="grid grid-cols-2 gap-2 pt-1">
              <button
                onClick={() => setUseConservative(true)}
                className={`py-1.5 px-3 rounded-xl text-xs font-semibold transition-all border ${
                  useConservative 
                    ? "bg-emerald-600 text-white border-emerald-500 shadow-md shadow-emerald-600/30" 
                    : "bg-slate-800 text-slate-400 border-slate-700 hover:text-white"
                }`}
              >
                Conservative p10
              </button>
              <button
                onClick={() => setUseConservative(false)}
                className={`py-1.5 px-3 rounded-xl text-xs font-semibold transition-all border ${
                  !useConservative 
                    ? "bg-indigo-600 text-white border-indigo-500 shadow-md shadow-indigo-600/30" 
                    : "bg-slate-800 text-slate-400 border-slate-700 hover:text-white"
                }`}
              >
                Point Model p50
              </button>
            </div>
            <div className="text-[10px] text-slate-500">
              p10 protects against down-side late failure risk in aviation.
            </div>
          </div>

          {/* Catastrophic Failure Penalty Slider */}
          <div className="space-y-2 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center gap-1.5">
                <span className="text-slate-300 font-semibold">Catastrophic AOG Cost ($)</span>
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-semibold">ASSUMPTION</span>
              </div>
              <div className="flex items-center gap-1 font-mono text-rose-400 font-bold text-xs">
                <span>$</span>
                <input
                  type="number"
                  value={costCatastrophic}
                  onChange={(e) => setCostCatastrophic(Math.max(0, parseFloat(e.target.value) || 0))}
                  onBlur={() => fetchSimulation()}
                  onKeyDown={(e) => { if (e.key === "Enter") fetchSimulation(); }}
                  className="w-24 px-1.5 py-0.5 rounded bg-slate-950 border border-slate-700 text-right text-rose-300 font-mono text-xs focus:border-rose-400 focus:outline-none"
                />
              </div>
            </div>
            <input 
              type="range" 
              min="1000000" 
              max="5000000" 
              step="250000"
              value={costCatastrophic}
              onChange={(e) => setCostCatastrophic(parseFloat(e.target.value))}
              onMouseUp={() => fetchSimulation()}
              onTouchEnd={() => fetchSimulation()}
              className="w-full accent-rose-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>$1.0M</span>
              <span>$5.0M</span>
            </div>
          </div>

          {/* Scheduled Depot Overhaul Cost */}
          <div className="space-y-2 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center gap-1.5">
                <span className="text-slate-300 font-semibold">Scheduled Depot Overhaul ($)</span>
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-semibold">ASSUMPTION</span>
              </div>
              <div className="flex items-center gap-1 font-mono text-indigo-300 font-bold text-xs">
                <span>$</span>
                <input
                  type="number"
                  value={costOverhaul}
                  onChange={(e) => setCostOverhaul(Math.max(0, parseFloat(e.target.value) || 0))}
                  onBlur={() => fetchSimulation()}
                  onKeyDown={(e) => { if (e.key === "Enter") fetchSimulation(); }}
                  className="w-24 px-1.5 py-0.5 rounded bg-slate-950 border border-slate-700 text-right text-indigo-300 font-mono text-xs focus:border-indigo-400 focus:outline-none"
                />
              </div>
            </div>
            <input 
              type="range" 
              min="80000" 
              max="500000" 
              step="20000"
              value={costOverhaul}
              onChange={(e) => setCostOverhaul(parseFloat(e.target.value))}
              onMouseUp={() => fetchSimulation()}
              onTouchEnd={() => fetchSimulation()}
              className="w-full accent-indigo-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>$80k</span>
              <span>$500k</span>
            </div>
          </div>

          {/* Wasted Depreciation Cost */}
          <div className="space-y-2 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center gap-1.5">
                <span className="text-slate-300 font-semibold">Wasted Useful Life ($/cyc)</span>
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-semibold">ASSUMPTION</span>
              </div>
              <div className="flex items-center gap-1 font-mono text-amber-300 font-bold text-xs">
                <span>$</span>
                <input
                  type="number"
                  value={costWastedCycle}
                  onChange={(e) => setCostWastedCycle(Math.max(0, parseFloat(e.target.value) || 0))}
                  onBlur={() => fetchSimulation()}
                  onKeyDown={(e) => { if (e.key === "Enter") fetchSimulation(); }}
                  className="w-20 px-1.5 py-0.5 rounded bg-slate-950 border border-slate-700 text-right text-amber-300 font-mono text-xs focus:border-amber-400 focus:outline-none"
                />
              </div>
            </div>
            <input 
              type="range" 
              min="400" 
              max="3000" 
              step="100"
              value={costWastedCycle}
              onChange={(e) => setCostWastedCycle(parseFloat(e.target.value))}
              onMouseUp={() => fetchSimulation()}
              onTouchEnd={() => fetchSimulation()}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>$400/cyc</span>
              <span>$3,000/cyc</span>
            </div>
          </div>

          {/* Line Triage Inspection Cost */}
          <div className="space-y-2 bg-slate-900/60 p-4 rounded-2xl border border-slate-800">
            <div className="flex justify-between items-center text-xs">
              <div className="flex items-center gap-1.5">
                <span className="text-slate-300 font-semibold">Borescopic Line Inspection ($)</span>
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded bg-amber-500/10 text-amber-300 border border-amber-500/20 font-semibold">ASSUMPTION</span>
              </div>
              <div className="flex items-center gap-1 font-mono text-purple-300 font-bold text-xs">
                <span>$</span>
                <input
                  type="number"
                  value={costInspection}
                  onChange={(e) => setCostInspection(Math.max(0, parseFloat(e.target.value) || 0))}
                  onBlur={() => fetchSimulation()}
                  onKeyDown={(e) => { if (e.key === "Enter") fetchSimulation(); }}
                  className="w-20 px-1.5 py-0.5 rounded bg-slate-950 border border-slate-700 text-right text-purple-300 font-mono text-xs focus:border-purple-400 focus:outline-none"
                />
              </div>
            </div>
            <input 
              type="range" 
              min="2000" 
              max="30000" 
              step="1000"
              value={costInspection}
              onChange={(e) => setCostInspection(parseFloat(e.target.value))}
              onMouseUp={() => fetchSimulation()}
              onTouchEnd={() => fetchSimulation()}
              className="w-full accent-purple-500 cursor-pointer"
            />
            <div className="flex justify-between text-[10px] text-slate-500 font-mono">
              <span>$2.0k</span>
              <span>$30.0k</span>
            </div>
          </div>
        </div>
      </div>

      {/* Per-Engine ROI Ledger Table */}
      <div className="glass-panel p-6 rounded-3xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <FileSpreadsheet className="w-5 h-5 text-emerald-400" />
            <h3 className="font-bold text-base text-white">
              Fleet Engine ROI &amp; Dispatch Ledger ({currentSubset})
            </h3>
          </div>
          <span className="text-xs font-mono text-slate-400">
            Top 30 Highest-Risk Turbofans
          </span>
        </div>
        <p className="text-xs text-slate-400">
          Individual engine outcomes under the active AETURNUS threshold policy ($\tau = {threshold}$ cycles):
        </p>

        <div className="overflow-x-auto">
          <table className="w-full text-xs text-left">
            <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
              <tr>
                <th className="p-3">Engine Unit</th>
                <th className="p-3 font-mono">Actual RUL</th>
                <th className="p-3 font-mono">Predicted RUL</th>
                <th className="p-3 font-mono">Conservative p10</th>
                <th className="p-3 font-mono">Pull Safety Margin</th>
                <th className="p-3 font-mono">Incurred Cost</th>
                <th className="p-3 text-center">Outcome</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-800/60 font-mono">
              {ledger.map((row: any) => {
                const isSafe = row.outcome === "SAFE_PREVENTIVE_OVERHAUL";
                return (
                  <tr key={row.unit} className="hover:bg-slate-800/20 transition-colors">
                    <td className="p-3 font-bold text-white font-sans flex items-center gap-2">
                      <Plane className="w-3.5 h-3.5 text-indigo-400" />
                      <span>{currentSubset}-{String(row.unit).padStart(3, "0")}</span>
                    </td>
                    <td className="p-3 text-cyan-300 font-bold">{row.actual_rul} cyc</td>
                    <td className="p-3 text-slate-300">{row.predicted_rul} cyc</td>
                    <td className="p-3 text-indigo-300">{row.conservative_p10} cyc</td>
                    <td className="p-3">
                      <span className={`px-2 py-0.5 rounded text-[11px] font-bold ${
                        row.effective_pull_cycle_margin > 0 
                          ? "bg-emerald-500/10 text-emerald-300 border border-emerald-500/20" 
                          : "bg-rose-500/20 text-rose-300 border border-rose-500/30"
                      }`}>
                        {row.effective_pull_cycle_margin > 0 ? `+${row.effective_pull_cycle_margin}` : row.effective_pull_cycle_margin} cyc
                      </span>
                    </td>
                    <td className="p-3 text-white font-bold">
                      ${row.cost.toLocaleString()}
                    </td>
                    <td className="p-3 text-center">
                      <span className={`inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold border ${
                        isSafe 
                          ? "bg-emerald-500/20 text-emerald-300 border-emerald-500/30" 
                          : "bg-rose-500/20 text-rose-300 border-rose-500/30"
                      }`}>
                        {isSafe ? <CheckCircle2 className="w-3 h-3" /> : <AlertTriangle className="w-3 h-3" />}
                        {isSafe ? "Disaster Avoided" : "Catastrophic Failure"}
                      </span>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
