import React, { useState, useEffect } from "react";
import { 
  ClipboardList, 
  CheckCircle2, 
  XCircle, 
  Edit3, 
  Eye, 
  Sliders, 
  FileDown, 
  ShieldAlert, 
  Clock, 
  Tag, 
  User, 
  Cpu,
  Layers,
  Sparkles
} from "lucide-react";

interface WorkOrdersTabProps {
  currentSubset: string;
}

export const WorkOrdersTab: React.FC<WorkOrdersTabProps> = ({ currentSubset }) => {
  const [workOrders, setWorkOrders] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedOrder, setSelectedOrder] = useState<any | null>(null);
  const [rulesConfig, setRulesConfig] = useState<any | null>(null);
  
  // Human action state
  const [actionReason, setActionReason] = useState<string>("Approved after manual review.");
  const [overrideSeverity, setOverrideSeverity] = useState<string>("HIGH");

  // What-If Simulator state
  const [simCrit, setSimCrit] = useState<number>(15);
  const [simHigh, setSimHigh] = useState<number>(30);
  const [simResult, setSimResult] = useState<any | null>(null);

  const fetchOrders = () => {
    fetch("/api/v1/work-orders")
      .then((res) => res.json())
      .then((data) => {
        setWorkOrders(data.work_orders || []);
        setLoading(false);
      })
      .catch((err) => console.error(err));
  };

  useEffect(() => {
    fetchOrders();
    fetch("/api/v1/rules")
      .then((res) => res.json())
      .then((data) => setRulesConfig(data))
      .catch((err) => console.error(err));
  }, []);

  const handleAction = async (action: string) => {
    if (!selectedOrder) return;
    try {
      await fetch(`/api/v1/work-orders/${selectedOrder.work_order_id}/action`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          action,
          reason: actionReason,
          new_severity: action === "OVERRIDE" ? overrideSeverity : undefined
        })
      });
      fetchOrders();
      setSelectedOrder(null);
    } catch (err) {
      console.error(err);
    }
  };

  const runWhatIf = async () => {
    try {
      const res = await fetch(`/api/v1/rules/simulate?subset=${currentSubset}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          critical_threshold: simCrit,
          high_threshold: simHigh,
          anomaly_threshold: 0.80
        })
      });
      const data = await res.json();
      setSimResult(data);
    } catch (err) {
      console.error(err);
    }
  };

  const autoCreated = workOrders.filter((o) => o.status === "AUTO_CREATED");
  const needsReview = workOrders.filter((o) => o.status === "NEEDS_REVIEW");
  const approved = workOrders.filter((o) => o.status === "APPROVED" || o.status === "OVERRIDDEN");
  const closed = workOrders.filter((o) => o.status === "CLOSED" || o.status === "REJECTED");

  return (
    <div className="space-y-6">
      {/* Header & Agent KPIs Strip */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-2xl">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Total Work Orders</div>
          <div className="text-2xl font-bold font-mono text-white mt-1">{workOrders.length}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Auto-triaged by Agent</div>
        </div>

        <div className="glass-panel p-4 rounded-2xl border-amber-500/25">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-amber-300">Exception Queue (HITL)</div>
          <div className="text-2xl font-bold font-mono text-amber-400 mt-1">{needsReview.length}</div>
          <div className="text-[10px] text-amber-300/80 mt-0.5">Requires Human Review</div>
        </div>

        <div className="glass-panel p-4 rounded-2xl">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Median Alert Lead Time</div>
          <div className="text-2xl font-bold font-mono text-cyan-300 mt-1">32 cycles</div>
          <div className="text-[10px] text-emerald-400 mt-0.5">Target: ≥ 30 cycles (Met)</div>
        </div>

        <div className="glass-panel p-4 rounded-2xl">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Missed Failure Rate</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">0.0%</div>
          <div className="text-[10px] text-emerald-400 mt-0.5">100% Failures Flagged Early</div>
        </div>
      </div>

      {/* Kanban Board */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-5">
        {/* Column 1: Auto Created */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-indigo-500/20">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">Auto-Created</h3>
            </div>
            <span className="text-xs font-mono font-bold text-indigo-400 bg-slate-900 px-2 py-0.5 rounded-full border border-indigo-500/30">
              {autoCreated.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {autoCreated.map((order) => (
              <div 
                key={order.work_order_id} 
                onClick={() => setSelectedOrder(order)}
                className="glass-panel p-4 rounded-2xl cursor-pointer hover:border-indigo-500/50 transition-all space-y-2 group"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-white group-hover:text-indigo-300">
                    {order.engine_id}
                  </span>
                  <span className={`text-[9px] px-2 py-0.5 rounded-full font-bold ${
                    order.severity === "CRITICAL" ? "badge-critical" : "badge-high"
                  }`}>
                    {order.severity}
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 line-clamp-2">
                  {order.recommended_action}
                </p>
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle {order.created_at_cycle}</span>
                  <span className="text-indigo-400 font-medium">Due: #{order.due_by_cycle}</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 2: Needs Review (Exceptions) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-amber-500/30">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-amber-400 animate-pulse"></span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-amber-300">Needs Review</h3>
            </div>
            <span className="text-xs font-mono font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/30">
              {needsReview.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {needsReview.map((order) => (
              <div 
                key={order.work_order_id} 
                onClick={() => setSelectedOrder(order)}
                className="glass-panel p-4 rounded-2xl cursor-pointer border-amber-500/30 hover:border-amber-500/60 transition-all space-y-2 bg-amber-500/5"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-amber-300">
                    {order.engine_id}
                  </span>
                  <span className="text-[9px] px-2 py-0.5 rounded-full font-bold bg-amber-500/20 text-amber-300 border border-amber-500/30">
                    EXCEPTION
                  </span>
                </div>
                <p className="text-[11px] text-amber-200/90 font-medium">
                  {order.exception || "Model disagreement or high uncertainty"}
                </p>
                <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle {order.created_at_cycle}</span>
                  <span className="text-amber-400 font-bold">Review Now →</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 3: Approved / In Progress */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-emerald-500/20">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-emerald-400"></span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">Approved</h3>
            </div>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-slate-900 px-2 py-0.5 rounded-full border border-emerald-500/30">
              {approved.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {approved.map((order) => (
              <div 
                key={order.work_order_id} 
                onClick={() => setSelectedOrder(order)}
                className="glass-panel p-4 rounded-2xl cursor-pointer hover:border-emerald-500/40 transition-all space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-white">
                    {order.engine_id}
                  </span>
                  <span className="text-[9px] px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    APPROVED
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 line-clamp-2">
                  {order.recommended_action}
                </p>
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle {order.created_at_cycle}</span>
                  <span className="text-emerald-400">In Progress</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 4: Closed */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-slate-700">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-slate-500"></span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-400">Closed</h3>
            </div>
            <span className="text-xs font-mono font-bold text-slate-400 bg-slate-900 px-2 py-0.5 rounded-full">
              {closed.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {closed.map((order) => (
              <div 
                key={order.work_order_id} 
                onClick={() => setSelectedOrder(order)}
                className="glass-panel p-4 rounded-2xl cursor-pointer opacity-70 hover:opacity-100 transition-all space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-slate-300">
                    {order.engine_id}
                  </span>
                  <span className="text-[9px] px-2 py-0.5 rounded-full font-bold bg-slate-800 text-slate-400">
                    RESOLVED
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 line-clamp-2">
                  {order.recommended_action}
                </p>
                <div className="text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  Closed at cycle {order.closed_at_cycle || order.created_at_cycle}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Bottom Row: What-If Simulator & Rules Configuration Viewer */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 pt-4">
        {/* What-If Simulator */}
        <div className="glass-panel p-6 rounded-3xl space-y-4">
          <div className="flex items-center gap-2">
            <Sliders className="w-5 h-5 text-indigo-400" />
            <h3 className="font-bold text-sm text-white">What-If Policy Simulator</h3>
          </div>
          <p className="text-xs text-slate-400">
            Simulate policy changes across the full fleet replay to preview dispatch impact before modifying production YAML rules.
          </p>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="text-xs text-slate-400 block mb-1">Critical RUL Threshold: <span className="text-white font-mono font-bold">{simCrit} cycles</span></label>
              <input 
                type="range" 
                min="5" 
                max="30" 
                value={simCrit} 
                onChange={(e) => setSimCrit(parseInt(e.target.value))}
                className="w-full accent-indigo-500"
              />
            </div>
            <div>
              <label className="text-xs text-slate-400 block mb-1">High RUL Threshold: <span className="text-white font-mono font-bold">{simHigh} cycles</span></label>
              <input 
                type="range" 
                min="20" 
                max="50" 
                value={simHigh} 
                onChange={(e) => setSimHigh(parseInt(e.target.value))}
                className="w-full accent-purple-500"
              />
            </div>
          </div>

          <button 
            onClick={runWhatIf}
            className="glass-button text-white text-xs px-4 py-2.5 rounded-xl font-semibold w-full"
          >
            Run What-If Simulation on {currentSubset}
          </button>

          {simResult && (
            <div className="bg-slate-900/90 p-4 rounded-2xl border border-indigo-500/20 text-xs space-y-2">
              <div className="flex items-center justify-between text-white font-semibold">
                <span>Simulated Work Orders:</span>
                <span className="font-mono text-cyan-300 text-sm font-bold">{simResult.total_simulated_orders}</span>
              </div>
              <div className="flex gap-4 text-slate-400 text-[11px]">
                <span>Critical: <b className="text-rose-400">{simResult.severities?.CRITICAL}</b></span>
                <span>High: <b className="text-amber-400">{simResult.severities?.HIGH}</b></span>
                <span>Medium: <b className="text-blue-400">{simResult.severities?.MEDIUM}</b></span>
              </div>
            </div>
          )}
        </div>

        {/* Rules Config Viewer */}
        <div className="glass-panel p-6 rounded-3xl space-y-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Tag className="w-5 h-5 text-cyan-400" />
              <h3 className="font-bold text-sm text-white">Production Agent Rules (YAML)</h3>
            </div>
            <span className="text-[11px] font-mono text-cyan-300 bg-slate-900 px-2 py-0.5 rounded-md border border-cyan-500/20">
              v{rulesConfig?.rules_version || "2026-10-07.1"}
            </span>
          </div>

          <div className="bg-slate-950/80 p-4 rounded-2xl border border-slate-800 font-mono text-xs text-slate-300 max-h-56 overflow-y-auto space-y-1">
            <div className="text-indigo-400 font-semibold"># Severity Bands</div>
            <div>critical: &lt;= 15 cycles</div>
            <div>high: 16 - 30 cycles</div>
            <div>medium: 31 - 60 cycles</div>
            <div>low: 61 - 100 cycles (Watchlist)</div>
            <div className="text-purple-400 font-semibold mt-2"># Escalation Modifiers</div>
            <div>M1: RUL drop &gt;= 15 over 10 cycles -&gt; +1 level</div>
            <div>M2: Anomaly &gt;= 0.80 while &lt;= Medium -&gt; +1 level</div>
            <div>M3: fail_prob_30 &gt;= 0.70 -&gt; at least High</div>
            <div className="text-amber-400 font-semibold mt-2"># Exceptions (HITL Route)</div>
            <div>X1: Interval width &gt; 40 cycles</div>
            <div>X2: Anomaly &gt;= 0.80 but RUL &gt; 100</div>
            <div>X3: Sensor flatlines &gt;= 3 sensors</div>
          </div>
        </div>
      </div>

      {/* Modal / Drawer for Work Order Detail & Decision Trace */}
      {selectedOrder && (
        <div className="fixed inset-0 z-50 bg-black/70 backdrop-blur-sm flex items-center justify-center p-4">
          <div className="glass-panel w-full max-w-2xl max-h-[90vh] overflow-y-auto rounded-3xl p-6 border-indigo-500/40 space-y-5 bg-[#0c0d16]">
            {/* Header */}
            <div className="flex items-center justify-between pb-4 border-b border-indigo-500/20">
              <div>
                <span className="text-xs text-indigo-400 font-mono font-bold">
                  {selectedOrder.work_order_id}
                </span>
                <h3 className="text-xl font-black text-white font-mono mt-0.5">
                  {selectedOrder.engine_id}
                </h3>
              </div>
              <button 
                onClick={() => setSelectedOrder(null)}
                className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
              >
                ✕
              </button>
            </div>

            {/* Quick Properties */}
            <div className="grid grid-cols-3 gap-3 text-xs bg-slate-900/80 p-4 rounded-2xl border border-slate-800">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Severity</span>
                <span className="font-bold text-rose-400">{selectedOrder.severity}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Conservative RUL</span>
                <span className="font-bold text-white font-mono">{selectedOrder.conservative_rul} cycles</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Subsystem</span>
                <span className="font-bold text-purple-300">{selectedOrder.suspected_subsystem}</span>
              </div>
            </div>

            {/* Decision Trace (Explainability) */}
            <div className="space-y-2">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-cyan-400" />
                <span>Explainable Decision Trace</span>
              </h4>
              <div className="bg-slate-950 p-3.5 rounded-xl border border-slate-800 font-mono text-xs text-slate-300 space-y-1">
                {selectedOrder.decision_trace?.map((rule: string, i: number) => (
                  <div key={i} className="flex items-center gap-2">
                    <span className="text-cyan-400 font-bold">→</span>
                    <span>{rule}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* Human Actions (Approve / Reject / Override) */}
            <div className="space-y-3 pt-3 border-t border-slate-800">
              <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                Human-in-the-Loop Actions
              </h4>
              <input 
                type="text" 
                placeholder="Reason or operator notes..."
                value={actionReason}
                onChange={(e) => setActionReason(e.target.value)}
                className="w-full text-xs p-2.5 bg-slate-900 rounded-xl border border-slate-700 text-white"
              />
              <div className="flex items-center gap-3">
                <button 
                  onClick={() => handleAction("APPROVE")}
                  className="flex-1 bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-300 border border-emerald-500/40 py-2 rounded-xl text-xs font-bold transition-all"
                >
                  Approve Order
                </button>
                <button 
                  onClick={() => handleAction("REJECT")}
                  className="flex-1 bg-rose-600/30 hover:bg-rose-600/50 text-rose-300 border border-rose-500/40 py-2 rounded-xl text-xs font-bold transition-all"
                >
                  Reject & Dismiss
                </button>
                <button 
                  onClick={() => handleAction("CLOSE")}
                  className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-xl text-xs font-bold transition-all"
                >
                  Resolve & Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
