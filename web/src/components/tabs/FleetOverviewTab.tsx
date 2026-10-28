import React, { useState, useMemo } from "react";
import { 
  AlertTriangle, 
  CheckCircle2, 
  Clock, 
  Activity, 
  FileText, 
  ShieldAlert, 
  TrendingDown, 
  ChevronRight,
  Filter,
  Zap,
  ArrowUpRight,
  PieChart as PieIcon,
  ShieldCheck,
  HeartPulse,
  Info
} from "lucide-react";
import { 
  ResponsiveContainer, 
  AreaChart, 
  Area, 
  XAxis, 
  YAxis, 
  Tooltip, 
  CartesianGrid, 
  ReferenceLine, 
  PieChart, 
  Pie, 
  Cell 
} from "recharts";

interface EngineItem {
  engine_id: string;
  unit: number;
  current_cycle: number;
  conservative_rul: number;
  rul_p50: number;
  health_index: number;
  anomaly_score: number;
  severity: "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "HEALTHY";
  has_open_order: boolean;
  sparkline: number[];
  suspected_subsystem: string;
}

interface FleetOverviewTabProps {
  fleetData: any;
  onSelectEngine: (engineId: string) => void;
  recentAlerts: any[];
}

export const FleetOverviewTab: React.FC<FleetOverviewTabProps> = ({
  fleetData,
  onSelectEngine,
  recentAlerts
}) => {
  const [severityFilter, setSeverityFilter] = useState<string>("ALL");
  const [searchTerm, setSearchTerm] = useState<string>("");

  const kpis = fleetData?.kpis || {};
  const engines: EngineItem[] = fleetData?.engines || [];
  const dist = fleetData?.severity_distribution || {};

  // Survival Curve Data: prefer server calculation or compute dynamically from engines
  const survivalData = useMemo(() => {
    if (fleetData?.survival_curve && fleetData.survival_curve.length > 0) {
      return fleetData.survival_curve;
    }
    if (!engines || engines.length === 0) return [];
    const n = engines.length;
    const p50s = engines.map((e) => e.rul_p50);
    const p10s = engines.map((e) => e.conservative_rul);
    const maxH = Math.min(160, Math.max(30, Math.ceil(Math.max(...p50s) / 10) * 10));
    const points = [];
    for (let t = 0; t <= maxH; t += 5) {
      const surviving = p50s.filter((r) => r > t).length;
      const sP50 = (surviving / n) * 100;
      const sP10 = (p10s.filter((r) => r > t).length / n) * 100;
      points.push({
        cycle: t,
        survival_rate: Number(sP50.toFixed(1)),
        conservative_rate: Number(sP10.toFixed(1)),
        surviving_count: surviving,
        at_risk_count: n - surviving
      });
    }
    return points;
  }, [fleetData?.survival_curve, engines]);

  // Key Survival Milestones
  const medianLife = useMemo(() => {
    const point = survivalData.find((p: any) => p.survival_rate <= 50);
    return point ? point.cycle : "> 120";
  }, [survivalData]);

  const p75Life = useMemo(() => {
    const point = survivalData.find((p: any) => p.survival_rate <= 75);
    return point ? point.cycle : "> 80";
  }, [survivalData]);

  const criticalHorizonSurvival = useMemo(() => {
    const point = survivalData.find((p: any) => p.cycle === 15) || survivalData[3];
    return point ? point.survival_rate : 100;
  }, [survivalData]);

  // Risk Distribution Donut Data
  const donutData = useMemo(() => {
    const colorMap: Record<string, { color: string; label: string; action: string }> = {
      CRITICAL: { color: "#f43f5e", label: "Critical Hazard (≤15 cyc)", action: "Immediate grounding & borescope inspection" },
      HIGH: { color: "#f97316", label: "High Risk (16-30 cyc)", action: "Schedule depot overhaul within 3 days" },
      MEDIUM: { color: "#eab308", label: "Medium Wear (31-60 cyc)", action: "Elevated telemetry frequency & sensor review" },
      LOW: { color: "#06b6d4", label: "Low Risk (61-100 cyc)", action: "Standard flight operations, regular checks" },
      HEALTHY: { color: "#10b981", label: "Healthy (>100 cyc)", action: "Optimal flight readiness, full mission endurance" }
    };

    const keys: Array<"CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "HEALTHY"> = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "HEALTHY"];
    const total = engines.length || 1;

    return keys.map((k) => {
      const count = dist[k] || 0;
      const pct = Number(((count / total) * 100).toFixed(1));
      return {
        name: k,
        label: colorMap[k].label,
        value: count,
        percentage: pct,
        color: colorMap[k].color,
        action: colorMap[k].action
      };
    }).filter((d) => d.value > 0);
  }, [dist, engines]);

  const filteredEngines = engines.filter((e) => {
    const matchesFilter = severityFilter === "ALL" || e.severity === severityFilter;
    const matchesSearch = e.engine_id.toLowerCase().includes(searchTerm.toLowerCase());
    return matchesFilter && matchesSearch;
  });

  if (!fleetData) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="flex items-center gap-3 text-slate-400">
          <Activity className="w-6 h-6 animate-spin text-indigo-400" />
          <span>Loading telemetry fleet state...</span>
        </div>
      </div>
    );
  }

  const getSeverityBadgeClass = (sev: string) => {
    switch (sev) {
      case "CRITICAL": return "badge-critical";
      case "HIGH": return "badge-high";
      case "MEDIUM": return "badge-medium";
      case "LOW": return "badge-low";
      default: return "badge-healthy";
    }
  };

  return (
    <div className="space-y-6">
      {/* KPI Cards Strip */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {/* Monitored */}
        <div className="glass-panel p-4 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Monitored</span>
            <Activity className="w-4 h-4 text-indigo-400" />
          </div>
          <div className="text-2xl font-bold text-white font-mono">{kpis.engines_monitored || 0}</div>
          <div className="text-[11px] text-slate-400 mt-1">Turbofan units</div>
          <div className="absolute -bottom-6 -right-6 w-16 h-16 bg-indigo-500/10 rounded-full blur-xl group-hover:bg-indigo-500/20 transition-all"></div>
        </div>

        {/* Critical */}
        <div className="glass-panel p-4 rounded-2xl relative overflow-hidden group border-rose-500/20">
          <div className="flex items-center justify-between text-rose-300 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Critical</span>
            <AlertTriangle className="w-4 h-4 text-rose-400 animate-pulse" />
          </div>
          <div className="text-2xl font-bold text-rose-400 font-mono">{kpis.critical_count || 0}</div>
          <div className="text-[11px] text-rose-300/80 mt-1">RUL ≤ 15 cycles</div>
          <div className="absolute -bottom-6 -right-6 w-16 h-16 bg-rose-500/10 rounded-full blur-xl group-hover:bg-rose-500/20 transition-all"></div>
        </div>

        {/* Open Work Orders */}
        <div className="glass-panel p-4 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Work Orders</span>
            <FileText className="w-4 h-4 text-purple-400" />
          </div>
          <div className="text-2xl font-bold text-purple-300 font-mono">{kpis.open_work_orders || 0}</div>
          <div className="text-[11px] text-slate-400 mt-1">Auto-dispatched</div>
          <div className="absolute -bottom-6 -right-6 w-16 h-16 bg-purple-500/10 rounded-full blur-xl group-hover:bg-purple-500/20 transition-all"></div>
        </div>

        {/* Exception Queue */}
        <div className={`glass-panel p-4 rounded-2xl relative overflow-hidden group ${kpis.exception_queue_size > 0 ? "border-amber-500/30 bg-amber-500/5" : ""}`}>
          <div className="flex items-center justify-between text-amber-300 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Human Review</span>
            <ShieldAlert className="w-4 h-4 text-amber-400" />
          </div>
          <div className="text-2xl font-bold text-amber-400 font-mono">{kpis.exception_queue_size || 0}</div>
          <div className="text-[11px] text-amber-300/80 mt-1">Exceptions pending</div>
          <div className="absolute -bottom-6 -right-6 w-16 h-16 bg-amber-500/10 rounded-full blur-xl group-hover:bg-amber-500/20 transition-all"></div>
        </div>

        {/* Mean RUL */}
        <div className="glass-panel p-4 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Fleet Mean RUL</span>
            <Clock className="w-4 h-4 text-cyan-400" />
          </div>
          <div className="text-2xl font-bold text-cyan-300 font-mono">{kpis.mean_conservative_rul || 0}</div>
          <div className="text-[11px] text-slate-400 mt-1">Cycles remaining</div>
          <div className="absolute -bottom-6 -right-6 w-16 h-16 bg-cyan-500/10 rounded-full blur-xl group-hover:bg-cyan-500/20 transition-all"></div>
        </div>

        {/* Model Version */}
        <div className="glass-panel p-4 rounded-2xl relative overflow-hidden group">
          <div className="flex items-center justify-between text-slate-400 mb-2">
            <span className="text-xs font-semibold uppercase tracking-wider">Active Model</span>
            <Zap className="w-4 h-4 text-emerald-400" />
          </div>
          <div className="text-sm font-bold text-emerald-300 font-mono truncate">{kpis.live_model_version || "champion"}</div>
          <div className="text-[11px] text-emerald-400/80 mt-1 flex items-center gap-1">
            <CheckCircle2 className="w-3 h-3 text-emerald-400" />
            <span>Serving verified</span>
          </div>
          <div className="absolute -bottom-6 -right-6 w-16 h-16 bg-emerald-500/10 rounded-full blur-xl group-hover:bg-emerald-500/20 transition-all"></div>
        </div>
      </div>

      {/* Fleet Reliability & Risk Intelligence: Survival Curve & Risk Donut */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Fleet Survival Curve S(t) (7 cols) */}
        <div className="lg:col-span-7 glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div>
              <div className="flex items-center gap-2">
                <TrendingDown className="w-5 h-5 text-cyan-400" />
                <h3 className="font-bold text-base text-white">Fleet Survival Function S(t)</h3>
                <span className="text-[10px] font-mono text-cyan-300 bg-slate-900 px-2 py-0.5 rounded-md border border-cyan-500/20">
                  Kaplan-Meier Reliability
                </span>
              </div>
              <p className="text-xs text-slate-400 mt-1">
                Forward survival probability across additional flight cycles. Shaded band represents conservative (p10) safety margin.
              </p>
            </div>
            {/* Survival KPI Chips */}
            <div className="flex items-center gap-2 self-start sm:self-auto flex-shrink-0">
              <div className="bg-slate-900/80 px-3 py-1.5 rounded-xl border border-slate-800 text-right">
                <div className="text-[9px] text-slate-400 uppercase font-mono">Median Life (t₅₀)</div>
                <div className="text-sm font-bold text-cyan-300 font-mono">{medianLife} cyc</div>
              </div>
              <div className="bg-slate-900/80 px-3 py-1.5 rounded-xl border border-slate-800 text-right">
                <div className="text-[9px] text-slate-400 uppercase font-mono">15-Cyc Survival</div>
                <div className="text-sm font-bold text-emerald-400 font-mono">{criticalHorizonSurvival}%</div>
              </div>
            </div>
          </div>

          {/* Area Chart Container */}
          <div className="h-64 w-full pt-1">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={survivalData} margin={{ top: 15, right: 15, bottom: 20, left: -10 }}>
                <defs>
                  <linearGradient id="fleetSurvivalGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06b6d4" stopOpacity={0.35} />
                    <stop offset="95%" stopColor="#06b6d4" stopOpacity={0.0} />
                  </linearGradient>
                  <linearGradient id="fleetConservativeGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#818cf8" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#818cf8" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                <XAxis 
                  dataKey="cycle" 
                  stroke="#64748b" 
                  fontSize={10} 
                  tickLine={false}
                  unit=" cyc"
                  label={{ value: "Operating Flight Cycles Horizon (t)", position: "insideBottom", offset: -12, fill: "#94a3b8", fontSize: 10 }}
                />
                <YAxis 
                  domain={[0, 100]} 
                  stroke="#64748b" 
                  fontSize={10} 
                  tickLine={false} 
                  unit="%" 
                />
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-slate-900/95 backdrop-blur-md border border-slate-700/80 p-3 rounded-2xl shadow-xl text-xs space-y-1.5 min-w-[210px]">
                          <div className="font-bold text-white border-b border-slate-800 pb-1 flex items-center justify-between">
                            <span>Horizon: +{d.cycle} Cycles</span>
                            <span className="text-[10px] font-mono text-cyan-300">{d.surviving_count} Active</span>
                          </div>
                          <div className="flex justify-between text-slate-300">
                            <span className="text-cyan-400">Expected Survival S(t):</span>
                            <span className="font-bold font-mono">{d.survival_rate}%</span>
                          </div>
                          <div className="flex justify-between text-slate-400">
                            <span className="text-indigo-400">Conservative Margin:</span>
                            <span className="font-mono">{d.conservative_rate}%</span>
                          </div>
                          <div className="flex justify-between text-slate-500 pt-1 border-t border-slate-800/80 text-[10px]">
                            <span>At-Risk By Horizon:</span>
                            <span className="text-rose-400 font-mono font-bold">{d.at_risk_count} engines</span>
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <ReferenceLine 
                  y={50} 
                  stroke="#818cf8" 
                  strokeDasharray="3 3" 
                  strokeWidth={1.5}
                  label={{ value: "50% Median Life", fill: "#818cf8", fontSize: 9, position: "insideTopLeft", offset: 6 }} 
                />
                <ReferenceLine 
                  x={15} 
                  stroke="#f43f5e" 
                  strokeDasharray="3 3" 
                  strokeWidth={1.5}
                  label={{ value: "15-Cyc Critical", fill: "#fb7185", fontSize: 9, position: "insideTopRight", offset: 6 }} 
                />
                <Area 
                  type="monotone" 
                  dataKey="survival_rate" 
                  stroke="#06b6d4" 
                  strokeWidth={2.5} 
                  fillOpacity={1} 
                  fill="url(#fleetSurvivalGrad)" 
                  name="Expected Survival (p50)"
                />
                <Area 
                  type="monotone" 
                  dataKey="conservative_rate" 
                  stroke="#818cf8" 
                  strokeWidth={1.5} 
                  strokeDasharray="4 4"
                  fillOpacity={1} 
                  fill="url(#fleetConservativeGrad)" 
                  name="Conservative Margin (p10)"
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-500 pt-2 border-t border-slate-800/80">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-cyan-400"></span> Expected S(t)
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400 ml-2"></span> Conservative Margin (p10)
            </span>
            <span className="font-mono text-slate-400">
              75% Survival Milestone: <strong className="text-cyan-300">{p75Life} cyc</strong>
            </span>
          </div>
        </div>

        {/* Right: Risk Distribution Donut (5 cols) */}
        <div className="lg:col-span-5 glass-panel p-6 rounded-3xl flex flex-col justify-between space-y-4">
          <div>
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <PieIcon className="w-5 h-5 text-indigo-400" />
                <h3 className="font-bold text-base text-white">Fleet Risk Distribution</h3>
              </div>
              <span className="text-[10px] font-mono text-indigo-300 bg-slate-900 px-2 py-0.5 rounded-md border border-indigo-500/20">
                Triage Donut
              </span>
            </div>
            <p className="text-xs text-slate-400 mt-1">
              Active engines partitioned by operational health tier. Click any slice or badge to filter the fleet list below.
            </p>
          </div>

          {/* Donut Chart with Centered Metric */}
          <div className="h-52 w-full relative flex items-center justify-center">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={donutData}
                  cx="50%"
                  cy="50%"
                  innerRadius={55}
                  outerRadius={80}
                  paddingAngle={4}
                  dataKey="value"
                  onClick={(entry) => {
                    if (entry && entry.name) {
                      setSeverityFilter(entry.name === severityFilter ? "ALL" : entry.name);
                    }
                  }}
                  className="cursor-pointer"
                >
                  {donutData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={entry.color} 
                      stroke={severityFilter === entry.name ? "#ffffff" : "#0f172a"}
                      strokeWidth={severityFilter === entry.name ? 2 : 1.5}
                      className="transition-all duration-300 hover:opacity-80"
                    />
                  ))}
                </Pie>
                <Tooltip
                  content={({ active, payload }) => {
                    if (active && payload && payload.length) {
                      const d = payload[0].payload;
                      return (
                        <div className="bg-slate-900/95 backdrop-blur-md border border-slate-700/80 p-3 rounded-2xl shadow-xl text-xs space-y-1.5 min-w-[200px]">
                          <div className="flex items-center justify-between border-b border-slate-800 pb-1">
                            <span className="font-bold text-white flex items-center gap-1.5">
                              <span className="w-2.5 h-2.5 rounded-full" style={{ backgroundColor: d.color }}></span>
                              {d.name}
                            </span>
                            <span className="font-mono text-indigo-300 font-bold">{d.percentage}%</span>
                          </div>
                          <div className="flex justify-between text-slate-300 font-mono text-[11px]">
                            <span>Units:</span>
                            <span className="font-bold text-white">{d.value} engines</span>
                          </div>
                          <div className="text-[10px] text-slate-400 pt-1 border-t border-slate-800/80 italic">
                            {d.action}
                          </div>
                        </div>
                      );
                    }
                    return null;
                  }}
                />
              </PieChart>
            </ResponsiveContainer>

            {/* Centered Donut Badge */}
            <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-xl font-bold font-mono text-white tracking-tight">
                {engines.length}
              </span>
              <span className="text-[9px] font-mono text-slate-400 uppercase tracking-wider">
                Total Fleet
              </span>
            </div>
          </div>

          {/* Interactive Tier Badges Legend */}
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 pt-1 border-t border-slate-800/80">
            {donutData.map((tier) => {
              const isActive = severityFilter === tier.name;
              return (
                <button
                  key={tier.name}
                  onClick={() => setSeverityFilter(isActive ? "ALL" : tier.name)}
                  className={`p-2 rounded-xl text-left border transition-all text-xs flex flex-col justify-between ${
                    isActive
                      ? "bg-slate-800/90 border-indigo-500 shadow-md ring-1 ring-indigo-500/50"
                      : "bg-slate-900/40 border-slate-800/80 hover:bg-slate-800/50"
                  }`}
                >
                  <div className="flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full" style={{ backgroundColor: tier.color }} />
                    <span className="font-semibold text-slate-200 text-[11px] truncate">{tier.name}</span>
                  </div>
                  <div className="flex items-baseline justify-between mt-1 font-mono text-[11px]">
                    <span className="text-slate-400">{tier.value} units</span>
                    <span className="font-bold" style={{ color: tier.color }}>{tier.percentage}%</span>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      </div>

      {/* Severity Breakdown Bar & Search Filter */}
      <div className="glass-panel p-5 rounded-2xl flex flex-col md:flex-row md:items-center justify-between gap-4">
        {/* Severity Tabs Filter */}
        <div className="flex items-center gap-2 overflow-x-auto pb-1 md:pb-0">
          <span className="text-xs text-slate-400 font-semibold mr-1 flex items-center gap-1">
            <Filter className="w-3.5 h-3.5" /> Filter:
          </span>
          {[
            { id: "ALL", label: "All Fleet", count: engines.length },
            { id: "CRITICAL", label: "Critical", count: dist.CRITICAL || 0, badge: "badge-critical" },
            { id: "HIGH", label: "High", count: dist.HIGH || 0, badge: "badge-high" },
            { id: "MEDIUM", label: "Medium", count: dist.MEDIUM || 0, badge: "badge-medium" },
            { id: "LOW", label: "Low", count: dist.LOW || 0, badge: "badge-low" },
            { id: "HEALTHY", label: "Healthy", count: dist.HEALTHY || 0, badge: "badge-healthy" },
          ].map((btn) => (
            <button
              key={btn.id}
              onClick={() => setSeverityFilter(btn.id)}
              className={`px-3 py-1.5 rounded-xl text-xs font-medium transition-all flex items-center gap-1.5 whitespace-nowrap ${
                severityFilter === btn.id
                  ? "bg-indigo-600 text-white font-semibold shadow-md shadow-indigo-600/30"
                  : "bg-slate-900/60 text-slate-400 hover:text-slate-200 border border-slate-800"
              }`}
            >
              <span>{btn.label}</span>
              <span className={`text-[10px] px-1.5 py-0.2 rounded-full ${btn.badge ? btn.badge : "bg-slate-800 text-slate-300"}`}>
                {btn.count}
              </span>
            </button>
          ))}
        </div>

        {/* Search */}
        <div className="w-full md:w-64">
          <input
            type="text"
            placeholder="Search engine ID (e.g. FD001-012)..."
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            className="w-full px-3.5 py-1.5 text-xs bg-slate-900/90 text-white rounded-xl border border-indigo-500/20 focus:outline-none focus:border-indigo-500 placeholder-slate-500"
          />
        </div>
      </div>

      {/* Main Grid: Engines Cards + Replay Alert Feed */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        {/* Engine Cards (3 Columns) */}
        <div className="lg:col-span-3 grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {filteredEngines.map((engine) => (
            <div
              key={engine.engine_id}
              onClick={() => onSelectEngine(engine.engine_id)}
              className="glass-panel p-5 rounded-2xl cursor-pointer transition-all duration-300 hover:scale-[1.01] hover:border-indigo-500/40 group relative flex flex-col justify-between"
            >
              {/* Header */}
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <span className="font-mono font-bold text-base text-white group-hover:text-indigo-300 transition-colors">
                    {engine.engine_id}
                  </span>
                  <span className="text-[10px] text-slate-400 font-mono">
                    Cycle {engine.current_cycle}
                  </span>
                </div>
                <span className={`text-[10px] px-2.5 py-1 rounded-full font-bold uppercase tracking-wider ${getSeverityBadgeClass(engine.severity)}`}>
                  {engine.severity}
                </span>
              </div>

              {/* RUL & Health Index Progress */}
              <div className="space-y-3 mb-4">
                <div className="flex items-baseline justify-between">
                  <span className="text-xs text-slate-400">Conservative RUL (p10):</span>
                  <div className="flex items-baseline gap-1">
                    <span className="font-mono text-xl font-extrabold text-white">
                      {engine.conservative_rul}
                    </span>
                    <span className="text-xs text-slate-400 font-mono">cycles</span>
                  </div>
                </div>

                {/* Health Index Bar */}
                <div>
                  <div className="flex items-center justify-between text-[11px] text-slate-400 mb-1">
                    <span>Health Index:</span>
                    <span className="font-mono font-semibold text-slate-300">{(engine.health_index * 100).toFixed(0)}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-slate-800 rounded-full overflow-hidden">
                    <div 
                      className={`h-full rounded-full transition-all duration-500 ${
                        engine.health_index > 0.6 ? "bg-emerald-400" : engine.health_index > 0.3 ? "bg-amber-400" : "bg-rose-500"
                      }`}
                      style={{ width: `${Math.max(5, engine.health_index * 100)}%` }}
                    ></div>
                  </div>
                </div>

                {/* Sparkline */}
                {engine.sparkline && engine.sparkline.length > 0 && (
                  <div className="h-10 w-full mt-2">
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={engine.sparkline.map((val, idx) => ({ idx, val }))}>
                        <defs>
                          <linearGradient id={`spark-${engine.engine_id}`} x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={engine.severity === "CRITICAL" ? "#f43f5e" : "#6366f1"} stopOpacity={0.4}/>
                            <stop offset="95%" stopColor={engine.severity === "CRITICAL" ? "#f43f5e" : "#6366f1"} stopOpacity={0.0}/>
                          </linearGradient>
                        </defs>
                        <Area 
                          type="monotone" 
                          dataKey="val" 
                          stroke={engine.severity === "CRITICAL" ? "#f43f5e" : "#6366f1"} 
                          fillOpacity={1} 
                          fill={`url(#spark-${engine.engine_id})`} 
                          strokeWidth={2}
                          isAnimationActive={false}
                        />
                      </AreaChart>
                    </ResponsiveContainer>
                  </div>
                )}
              </div>

              {/* Card Footer */}
              <div className="pt-3 border-t border-slate-800/80 flex items-center justify-between text-xs text-slate-400">
                <span className="text-[11px] text-indigo-400/90 font-medium">
                  Subsystem: {engine.suspected_subsystem}
                </span>
                <div className="flex items-center gap-1 group-hover:text-indigo-300 transition-colors">
                  <span>Deep-dive</span>
                  <ArrowUpRight className="w-3.5 h-3.5" />
                </div>
              </div>
            </div>
          ))}
        </div>

        {/* Right Replay Alert Feed (1 Column) */}
        <div className="glass-panel p-5 rounded-2xl flex flex-col h-full">
          <div className="flex items-center justify-between mb-4 pb-3 border-b border-indigo-500/10">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-cyan-400" />
              <h3 className="font-semibold text-sm text-white">Replay Decision Feed</h3>
            </div>
            <span className="text-[10px] bg-slate-800 px-2 py-0.5 rounded-full text-slate-400 font-mono">
              WebSocket
            </span>
          </div>

          <div className="space-y-3 flex-1 overflow-y-auto max-h-[600px] pr-1">
            {recentAlerts.length === 0 ? (
              <div className="text-center py-12 text-slate-500 text-xs">
                No active events yet. Press Play in header to stream telemetry.
              </div>
            ) : (
              recentAlerts.slice(0, 15).map((alert, idx) => (
                <div 
                  key={idx}
                  className="p-3 rounded-xl bg-slate-900/70 border border-slate-800 hover:border-indigo-500/30 transition-all text-xs"
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono font-bold text-white text-[11px]">
                      {alert.engine_id || `WO #${alert.work_order_id?.slice(-6)}`}
                    </span>
                    <span className={`text-[9px] px-2 py-0.5 rounded-full font-bold ${getSeverityBadgeClass(alert.severity || "MEDIUM")}`}>
                      {alert.severity || "EVENT"}
                    </span>
                  </div>
                  <p className="text-[11px] text-slate-300 line-clamp-2">
                    {alert.recommended_action || alert.event || "Work order evaluated"}
                  </p>
                  <div className="mt-2 text-[10px] text-slate-500 font-mono flex items-center justify-between">
                    <span>Cycle {alert.created_at_cycle || alert.cycle || "--"}</span>
                    <span>{alert.status || "AUTO_DISPATCHED"}</span>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};
