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
        collapsed ? "w-20" : "w-72"
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
          <img 
            src="/aeturnus-mark.png" 
            alt="AETURNUS" 
            className="h-10 w-auto max-w-[42px] object-contain mx-auto drop-shadow-md" 
          />
        )}

        <button
          onClick={() => setCollapsed(!collapsed)}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 transition-colors"
          title={collapsed ? "Expand sidebar" : "Collapse sidebar"}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Navigation Items */}
      <div className="flex-1 py-5 px-3 space-y-1.5 overflow-y-auto">
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
              className={`w-full flex items-center gap-3.5 px-3.5 py-3 rounded-xl text-sm font-medium transition-all group relative ${
                isActive
                  ? "bg-gradient-to-r from-indigo-600/30 to-purple-600/20 text-indigo-300 border border-indigo-500/30 shadow-lg shadow-indigo-500/10"
                  : "text-slate-400 hover:text-slate-200 hover:bg-slate-800/40"
              }`}
            >
              <Icon className={`w-5 h-5 transition-transform group-hover:scale-110 ${isActive ? "text-indigo-400" : "text-slate-400"}`} />
              {!collapsed && (
                <div className="flex-1 flex items-center justify-between">
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
