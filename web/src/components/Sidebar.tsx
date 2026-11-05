import React from "react";
import { 
  Activity, 
  Cpu, 
  ClipboardList, 
  Layers, 
  GitBranch, 
  Database, 
  HelpCircle, 
  ChevronLeft, 
  ChevronRight,
  DollarSign,
  ShieldCheck,
  Flame
} from "lucide-react";

import aeturnusName from "../assets/aeturnus-name.png";

interface SidebarProps {
  activeTab: string;
  setActiveTab: (tab: string) => void;
  collapsed: boolean;
  setCollapsed: (collapsed: boolean) => void;
  exceptionCount: number;
}

export const Sidebar: React.FC<SidebarProps> = ({
  activeTab,
  setActiveTab,
  collapsed,
  setCollapsed,
  exceptionCount
}) => {
  const navItems = [
    { id: "fleet", label: "Fleet Overview", icon: Activity, badge: "Replay" },
    { id: "engine", label: "Engine Detail", icon: Cpu, badge: "Telemetry" },
    { id: "work_orders", label: "Work Orders & Triage", icon: ClipboardList, badge: exceptionCount > 0 ? `${exceptionCount} Review` : "Agent", badgeAlert: exceptionCount > 0 },
    { id: "cost_simulator", label: "Cost & ROI Simulator", icon: DollarSign, badge: "Finance" },
    { id: "model_lab", label: "Model Lab", icon: Layers, badge: "Benchmark" },
    { id: "mlops", label: "MLOps & Monitoring", icon: GitBranch, badge: "Drift" },
    { id: "pipeline", label: "Data Pipeline", icon: Database, badge: "ETL" },
    { id: "how_it_works", label: "How It Works", icon: HelpCircle, badge: "Docs" },
  ];

  return (
    <aside
      className={`fixed top-0 left-0 h-screen z-40 sidebar-mesh border-r border-indigo-500/15 flex flex-col transition-all duration-300 ${
        collapsed ? "w-20 overflow-visible" : "w-72"
      }`}
    >
      {/* Brand Header */}
      <div className="h-20 flex items-center px-4 border-b border-indigo-500/10 justify-between">
        {!collapsed ? (
          <div className="flex items-center gap-3 overflow-hidden">
            <img 
              src="/aeturnus-mark.png" 
              alt="AETURNUS Emblem" 
              className="h-10 w-auto max-w-[48px] object-contain drop-shadow-md flex-shrink-0" 
            />
            <img 
              src={aeturnusName} 
              alt="AETURNUS" 
              className="h-6.5 max-h-[26px] w-auto max-w-[160px] object-contain drop-shadow-sm flex-shrink-0" 
            />
          </div>
        ) : (
          <div className="mx-auto relative group">
            <img 
              src="/aeturnus-mark.png" 
              alt="AETURNUS" 
              className="h-10 w-auto max-w-[42px] object-contain mx-auto drop-shadow-md cursor-pointer" 
            />
            <div
              role="tooltip"
              className="absolute left-full ml-4 top-1/2 -translate-y-1/2 z-50 pointer-events-none opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 -translate-x-2 group-hover:translate-x-0 transition-all duration-200 ease-out px-3 py-1.5 rounded-xl bg-[#0c0f20]/95 backdrop-blur-xl border border-indigo-500/30 shadow-2xl shadow-black/80 whitespace-nowrap text-xs font-bold text-white flex items-center gap-2"
            >
              <div className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-2.5 h-2.5 rotate-45 bg-[#0c0f20] border-l border-b border-indigo-500/30" />
              <span>AETURNUS</span>
              <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/15 border border-cyan-500/30 px-1.5 py-0.5 rounded-full">
                Cockpit
              </span>
            </div>
          </div>
        )}

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors relative group"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
          aria-label={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          {collapsed && (
            <div
              role="tooltip"
              className="absolute left-full ml-3 top-1/2 -translate-y-1/2 z-50 pointer-events-none opacity-0 group-hover:opacity-100 -translate-x-2 group-hover:translate-x-0 transition-all duration-200 ease-out px-2.5 py-1.5 rounded-xl bg-[#0c0f20]/95 backdrop-blur-xl border border-indigo-500/30 shadow-2xl shadow-black/80 whitespace-nowrap text-[11px] font-semibold text-slate-200 flex items-center gap-1.5"
            >
              <div className="absolute -left-1 top-1/2 -translate-y-1/2 w-2 h-2 rotate-45 bg-[#0c0f20] border-l border-b border-indigo-500/30" />
              <span>Expand sidebar</span>
            </div>
          )}
        </button>
      </div>

      {/* Navigation Items */}
      <div className={`flex-1 py-5 px-3 space-y-1.5 ${collapsed ? "overflow-visible" : "overflow-y-auto"}`}>
        <div className={`px-3 mb-2 text-[11px] font-semibold uppercase tracking-wider text-slate-500 ${collapsed ? "hidden" : "block"}`}>
          System Modules
        </div>
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = activeTab === item.id;
          return (
            <button
              key={item.id}
              onClick={() => setActiveTab(item.id)}
              aria-label={item.label}
              className={`w-full flex items-center rounded-xl text-sm font-medium transition-all group relative ${
                collapsed ? "justify-center px-0 py-3" : "gap-3.5 px-3.5 py-3"
              } ${
                isActive
                  ? "bg-gradient-to-r from-indigo-600/30 to-purple-600/20 text-indigo-300 border border-indigo-500/30 shadow-lg shadow-indigo-500/10"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
              }`}
            >
              <Icon className={`w-5 h-5 transition-transform group-hover:scale-110 flex-shrink-0 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
              
              {!collapsed && (
                <div className="flex-1 flex items-center justify-between min-w-0">
                  <span className="truncate">{item.label}</span>
                  {item.badge && (
                    <span className={`text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                      item.badgeAlert
                        ? "bg-rose-500/20 text-rose-300 border border-rose-500/30 animate-pulse"
                        : isActive 
                        ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30" 
                        : "bg-slate-800 text-slate-400 border border-slate-700/50"
                    }`}>
                      {item.badge}
                    </span>
                  )}
                </div>
              )}

              {/* Slide Name Tooltip / Flyout when slider is closed */}
              {collapsed && (
                <div
                  role="tooltip"
                  className="absolute left-full ml-3.5 top-1/2 -translate-y-1/2 z-50 pointer-events-none opacity-0 group-hover:opacity-100 group-focus-visible:opacity-100 -translate-x-2 group-hover:translate-x-0 group-focus-visible:translate-x-0 transition-all duration-200 ease-out flex items-center gap-2.5 px-3 py-2 rounded-xl bg-[#0c0f20]/95 backdrop-blur-xl border border-indigo-500/30 shadow-2xl shadow-black/80 whitespace-nowrap"
                >
                  {/* Arrow pointing left toward the icon box */}
                  <div className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 rotate-45 bg-[#0c0f20] border-l border-b border-indigo-500/30" />
                  
                  <div className="relative z-10 flex items-center gap-2">
                    {isActive && (
                      <span className="w-2 h-2 rounded-full bg-cyan-400 shadow-sm shadow-cyan-400 animate-pulse" />
                    )}
                    <span className="text-xs font-bold text-white tracking-wide">
                      {item.label}
                    </span>
                  </div>

                  {item.badge && (
                    <span
                      className={`relative z-10 text-[10px] px-2 py-0.5 rounded-full font-semibold ${
                        item.badgeAlert
                          ? "bg-rose-500/20 text-rose-300 border border-rose-500/30 animate-pulse"
                          : isActive
                          ? "bg-indigo-500/20 text-indigo-300 border border-indigo-500/30"
                          : "bg-slate-800 text-slate-300 border border-slate-700/60"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </div>
              )}
            </button>
          );
        })}
      </div>

      {/* Footer System Status */}
      {!collapsed && (
        <div className="p-4 m-3 rounded-xl bg-slate-900/60 border border-indigo-500/15">
          <div className="flex items-center gap-2 mb-1.5">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span className="text-xs font-semibold text-slate-300">Rule-Based Triage (No-LLM)</span>
          </div>
          <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
            <ShieldCheck className="w-3.5 h-3.5 text-indigo-400" />
            <span>Deterministic Policy · v2026-10-07</span>
          </div>
        </div>
      )}
    </aside>
  );
};
