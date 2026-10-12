import React, { useState } from "react";

interface DiagnosticPanelProps {
  currentSubset: string;
  currentCycle: number;
  isPlaying: boolean;
  exceptionCount: number;
  wsConnected: boolean;
  fleetData?: any;
  recentAlerts?: any[];
  onSelectEngine?: (engineId: string) => void;
}

export const DiagnosticPanel: React.FC<DiagnosticPanelProps> = ({
  currentSubset,
  currentCycle,
  isPlaying,
  exceptionCount,
  wsConnected,
  fleetData,
  recentAlerts = [],
  onSelectEngine
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);

  // Derived metrics from /fleet data
  const engines = fleetData?.engines || [];
  const totalEngines = fleetData?.total_engines || engines.length || 0;
  const criticalEngines = engines.filter((e: any) => e.severity === "CRITICAL");
  const highEngines = engines.filter((e: any) => e.severity === "HIGH");
  const mediumEngines = engines.filter((e: any) => e.severity === "MEDIUM");
  const healthyEngines = engines.filter((e: any) => e.severity === "HEALTHY" || e.severity === "LOW");

  const criticalCount = fleetData?.critical_count ?? criticalEngines.length;
  const highCount = highEngines.length;
  const mediumCount = mediumEngines.length;
  const healthyCount = healthyEngines.length;
  const openOrdersCount = fleetData?.open_work_orders ?? engines.filter((e: any) => e.has_open_order).length ?? 0;
  const avgRul = fleetData?.avg_rul ?? (engines.length > 0
    ? (engines.reduce((acc: number, e: any) => acc + (e.conservative_rul || 0), 0) / engines.length).toFixed(1)
    : "—");

  const flaggedEngines = [...criticalEngines, ...highEngines];

  return (
    <aside aria-label="Fleet telemetry status panel" className="fixed bottom-4 right-4 z-40 flex flex-col items-end">
      {isOpen && (
        <section aria-labelledby="status-panel-title" className="mb-2 w-96 rounded-lg bg-slate-900 border border-slate-700 shadow-xl text-slate-200 text-xs overflow-hidden flex flex-col">
          {/* Header */}
          <div className="flex items-center justify-between px-3.5 py-2.5 bg-slate-800 border-b border-slate-700">
            <div className="flex items-center gap-2">
              <span
                className={`w-2 h-2 rounded-full ${
                  criticalCount > 0 ? "bg-red-500" : highCount > 0 ? "bg-amber-500" : "bg-emerald-500"
                }`}
              />
              <span id="status-panel-title" className="font-semibold text-slate-100">Fleet Status — {currentSubset}</span>
            </div>
            <div className="flex items-center gap-2 text-[11px] text-slate-400">
              <span>Cycle {currentCycle}</span>
              <span className={`px-1.5 py-0.5 rounded text-[10px] ${wsConnected ? "bg-emerald-950 text-emerald-400 border border-emerald-800" : "bg-slate-800 text-slate-400"}`}>
                {wsConnected ? "LIVE" : "DISCONNECTED"}
              </span>
            </div>
          </div>

          {/* Metrics Grid */}
          <div className="p-3 space-y-3">
            <div className="grid grid-cols-2 gap-2 text-[11px]">
              <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
                <div className="text-slate-400">Total Units</div>
                <div className="text-sm font-semibold text-white font-mono mt-0.5">{totalEngines}</div>
                <div className="text-[10px] text-emerald-400">{healthyCount} nominal</div>
              </div>

              <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
                <div className="text-slate-400">Fleet Avg RUL</div>
                <div className="text-sm font-semibold text-cyan-300 font-mono mt-0.5">{avgRul} cyc</div>
                <div className="text-[10px] text-slate-400">Conservative p10</div>
              </div>

              <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
                <div className="text-slate-400">Critical / High</div>
                <div className="text-sm font-semibold text-red-300 font-mono mt-0.5">
                  {criticalCount} <span className="text-slate-500 text-xs">/</span> {highCount}
                </div>
                <div className="text-[10px] text-amber-400">{mediumCount} medium</div>
              </div>

              <div className="p-2 rounded bg-slate-800/80 border border-slate-700/60">
                <div className="text-slate-400">Open Work Orders</div>
                <div className="text-sm font-semibold text-amber-300 font-mono mt-0.5">{openOrdersCount}</div>
                <div className="text-[10px] text-slate-400">{exceptionCount} exceptions</div>
              </div>
            </div>

            {/* Flagged Units List */}
            <div>
              <div className="text-[11px] font-semibold text-slate-300 mb-1.5 flex items-center justify-between">
                <span>Flagged Assets ({flaggedEngines.length})</span>
                <span className="text-[10px] text-slate-500">Sorted by priority</span>
              </div>

              {flaggedEngines.length === 0 ? (
                <div className="p-2 rounded bg-slate-800/40 text-slate-400 text-[11px] text-center border border-slate-800">
                  All units within nominal operational limits.
                </div>
              ) : (
                <div className="space-y-1 max-h-36 overflow-y-auto pr-1">
                  {flaggedEngines.slice(0, 6).map((eng: any) => (
                    <div
                      key={eng.engine_id}
                      onClick={() => onSelectEngine?.(eng.engine_id)}
                      className="flex items-center justify-between p-1.5 rounded bg-slate-800/60 hover:bg-slate-750 border border-slate-700/50 cursor-pointer transition-colors"
                    >
                      <div className="flex items-center gap-1.5">
                        <span
                          className={`w-1.5 h-1.5 rounded-full ${
                            eng.severity === "CRITICAL" ? "bg-red-500" : "bg-orange-500"
                          }`}
                        />
                        <span className="font-mono text-slate-200">{eng.engine_id}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-slate-400 font-mono text-[10px]">
                          RUL: {eng.conservative_rul ? eng.conservative_rul.toFixed(0) : "—"}
                        </span>
                        <span
                          className={`px-1.5 py-0.2 rounded text-[9px] font-bold ${
                            eng.severity === "CRITICAL"
                              ? "bg-red-950 text-red-300 border border-red-800"
                              : "bg-orange-950 text-orange-300 border border-orange-800"
                          }`}
                        >
                          {eng.severity}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </section>
      )}

      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 px-3 py-2 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-600 text-slate-200 text-xs font-medium shadow-md transition-all cursor-pointer"
        title={isOpen ? "Collapse Status Panel" : "Expand Fleet Status Panel"}
      >
        <span
          className={`w-2 h-2 rounded-full ${
            criticalCount > 0 ? "bg-red-500" : highCount > 0 ? "bg-amber-500" : "bg-emerald-500"
          }`}
        />
        <span>Fleet Status ({currentSubset})</span>
        {criticalCount > 0 && (
          <span className="px-1.5 py-0.2 rounded text-[10px] font-bold bg-red-600 text-white">
            {criticalCount}
          </span>
        )}
        <span className="text-slate-400 text-[10px]">{isOpen ? "▲" : "▼"}</span>
      </button>
    </aside>
  );
};
