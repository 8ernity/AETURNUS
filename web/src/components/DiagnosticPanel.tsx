import React, { useState } from "react";
import { 
  Activity, 
  AlertTriangle, 
  ChevronDown, 
  ChevronUp, 
  ShieldCheck, 
  Cpu, 
  Radio, 
  Clock,
  Layers,
  Wrench
} from "lucide-react";

interface DiagnosticPanelProps {
  currentSubset: string;
  currentCycle: number;
  isPlaying: boolean;
  exceptionCount: number;
  wsConnected: boolean;
  fleetData?: any;
  recentAlerts?: any[];
}

export const DiagnosticPanel: React.FC<DiagnosticPanelProps> = ({
  currentSubset,
  currentCycle,
  isPlaying,
  exceptionCount,
  wsConnected,
  fleetData,
  recentAlerts = []
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  // Derived metrics directly from /fleet endpoint payload
  const engines = fleetData?.engines || [];
  const totalEngines = fleetData?.total_engines || engines.length || 0;
  const criticalCount = fleetData?.critical_count || engines.filter((e: any) => e.severity === "CRITICAL").length || 0;
  const highCount = engines.filter((e: any) => e.severity === "HIGH").length || 0;
  const mediumCount = engines.filter((e: any) => e.severity === "MEDIUM").length || 0;
  const healthyCount = engines.filter((e: any) => e.severity === "HEALTHY" || e.severity === "LOW").length || 0;
  const openOrdersCount = fleetData?.open_work_orders ?? engines.filter((e: any) => e.has_open_order).length ?? 0;
  const avgRul = fleetData?.avg_rul ?? (engines.length > 0 
    ? (engines.reduce((acc: number, e: any) => acc + (e.conservative_rul || 0), 0) / engines.length).toFixed(1) 
    : "--");

  // Subsystems aggregated from engine predictions
  const subsystemCounts: Record<string, number> = {};
  engines.forEach((e: any) => {
    const sub = e.suspected_subsystem || "HPC";
    subsystemCounts[sub] = (subsystemCounts[sub] || 0) + 1;
  });
  const topSubsystems = Object.entries(subsystemCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 3);

  const statusColor = criticalCount > 0 
    ? "bg-rose-500" 
    : highCount > 0 
    ? "bg-amber-500" 
    : isPlaying 
    ? "bg-cyan-500" 
    : "bg-emerald-500";

  const statusLabel = criticalCount > 0 
    ? `${criticalCount} Critical Engines` 
    : highCount > 0 
    ? `${highCount} High Priority` 
    : isPlaying 
    ? "Telemetry Streaming Active" 
    : "Fleet Standby / Nominal";

  return (
    <div className="fixed bottom-5 right-5 z-40 flex flex-col items-end">
      {/* Expanded Diagnostics Card */}
      {isOpen && (
        <div className="mb-3 w-80 sm:w-96 rounded-2xl bg-slate-900/95 border border-slate-700/80 shadow-2xl backdrop-blur-xl p-5 space-y-4 text-xs animate-in fade-in slide-in-from-bottom-4 duration-200">
          {/* Header */}
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-cyan-400" />
              <span className="font-bold text-sm text-white tracking-wide">Fleet Diagnostic Panel</span>
            </div>
            <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-800 text-slate-300 border border-slate-700">
              {currentSubset} · Cycle {currentCycle}
            </span>
          </div>

          {/* Operational Status Banner */}
          <div className="flex items-center gap-2.5 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
            <span className={`w-2.5 h-2.5 rounded-full ${statusColor} animate-pulse`} />
            <div className="flex-1">
              <div className="font-semibold text-slate-200">{statusLabel}</div>
              <div className="text-[11px] text-slate-400">
                {wsConnected ? "WebSocket live telemetry synchronized" : "HTTP polling fallback"}
              </div>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="grid grid-cols-2 gap-2.5">
            <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
              <div className="text-slate-400 text-[10px] flex items-center gap-1">
                <Radio className="w-3 h-3 text-cyan-400" /> Total Engines
              </div>
              <div className="text-base font-bold font-mono text-white mt-0.5">{totalEngines}</div>
              <div className="text-[10px] text-emerald-400 mt-0.5">{healthyCount} nominal</div>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
              <div className="text-slate-400 text-[10px] flex items-center gap-1">
                <Clock className="w-3 h-3 text-indigo-400" /> Fleet Avg RUL
              </div>
              <div className="text-base font-bold font-mono text-cyan-300 mt-0.5">{avgRul} cyc</div>
              <div className="text-[10px] text-slate-400 mt-0.5">Conservative P10-derived</div>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
              <div className="text-slate-400 text-[10px] flex items-center gap-1">
                <AlertTriangle className="w-3 h-3 text-rose-400" /> Critical / High
              </div>
              <div className="text-base font-bold font-mono text-rose-300 mt-0.5">
                {criticalCount} <span className="text-slate-500 font-normal text-xs">/</span> {highCount}
              </div>
              <div className="text-[10px] text-slate-400 mt-0.5">{mediumCount} medium advisory</div>
            </div>

            <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
              <div className="text-slate-400 text-[10px] flex items-center gap-1">
                <Wrench className="w-3 h-3 text-amber-400" /> Open Work Orders
              </div>
              <div className="text-base font-bold font-mono text-amber-300 mt-0.5">{openOrdersCount}</div>
              <div className="text-[10px] text-slate-400 mt-0.5">
                {exceptionCount > 0 ? `${exceptionCount} exceptions pending` : "No pending exceptions"}
              </div>
            </div>
          </div>

          {/* Subsystem Telemetry Breakdown */}
          {topSubsystems.length > 0 && (
            <div className="space-y-1.5 pt-1">
              <div className="flex items-center gap-1.5 text-[11px] font-semibold text-slate-300">
                <Layers className="w-3 h-3 text-purple-400" /> Suspected Subsystem Distribution
              </div>
              <div className="flex gap-2">
                {topSubsystems.map(([name, count]) => (
                  <div key={name} className="flex-1 p-2 rounded-lg bg-slate-950/40 border border-slate-800 text-center">
                    <div className="font-mono font-bold text-slate-200">{name}</div>
                    <div className="text-[10px] text-slate-400">{count} units</div>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Recent Alert Snippet */}
          {recentAlerts.length > 0 && (
            <div className="p-2 rounded-xl bg-rose-950/30 border border-rose-800/40 text-[11px] text-rose-300 flex items-start gap-2">
              <AlertTriangle className="w-3.5 h-3.5 text-rose-400 shrink-0 mt-0.5" />
              <div className="truncate">
                <span className="font-semibold">{recentAlerts[0]?.engine_id || "Engine Alert"}:</span>{" "}
                {recentAlerts[0]?.severity || "CRITICAL"} severity on cycle {recentAlerts[0]?.cycle || currentCycle}
              </div>
            </div>
          )}
        </div>
      )}

      {/* Floating Toggle Pill */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2.5 px-4 py-2.5 rounded-full bg-slate-900 border border-slate-700 hover:border-slate-600 text-white shadow-xl hover:shadow-cyan-500/10 transition-all cursor-pointer backdrop-blur-md"
      >
        <span className={`w-2.5 h-2.5 rounded-full ${statusColor}`} />
        <span className="text-xs font-semibold tracking-wide">Diagnostic Panel</span>
        {criticalCount > 0 && (
          <span className="text-[10px] font-bold px-1.5 py-0.2 rounded-full bg-rose-500/30 text-rose-300 border border-rose-500/40">
            {criticalCount}
          </span>
        )}
        {isOpen ? <ChevronDown className="w-4 h-4 text-slate-400" /> : <ChevronUp className="w-4 h-4 text-slate-400" />}
      </button>
    </div>
  );
};
