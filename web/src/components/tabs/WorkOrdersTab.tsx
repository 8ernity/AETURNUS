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
  Sparkles,
  Plane,
  Wrench,
  Truck,
  PackageCheck,
  FileCheck,
  ChevronRight,
  AlertTriangle,
  ArrowRight
} from "lucide-react";

interface WorkOrdersTabProps {
  currentSubset: string;
}

const LIFECYCLE_STAGES = [
  "PROPOSED",
  "DISPATCHED",
  "PARTS_ORDERED",
  "IN_PROGRESS",
  "POST_MAINTENANCE_RUNUP",
  "AIRWORTHY_CLOSED"
];

export const WorkOrdersTab: React.FC<WorkOrdersTabProps> = ({ currentSubset }) => {
  const [workOrders, setWorkOrders] = useState<any[]>([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [selectedOrder, setSelectedOrder] = useState<any | null>(null);
  const [rulesConfig, setRulesConfig] = useState<any | null>(null);
  const [actionLoading, setActionLoading] = useState<boolean>(false);
  
  // Human action state
  const [actionReason, setActionReason] = useState<string>("Approved and verified by maintenance coordinator.");
  const [overrideSeverity, setOverrideSeverity] = useState<string>("HIGH");

  // What-If Simulator state
  const [simCrit, setSimCrit] = useState<number>(15);
  const [simHigh, setSimHigh] = useState<number>(30);
  const [simResult, setSimResult] = useState<any | null>(null);

  const fetchOrders = () => {
    fetch("/api/v1/work-orders")
      .then((res) => res.json())
      .then((data) => {
        const orders = data.work_orders || [];
        setWorkOrders(orders);
        setLoading(false);
        // If an order is currently selected, refresh its data
        if (selectedOrder) {
          const updated = orders.find((o: any) => o.work_order_id === selectedOrder.work_order_id);
          if (updated) setSelectedOrder(updated);
        }
      })
      .catch((err) => {
        console.error(err);
        setLoading(false);
      });
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
    setActionLoading(true);
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
    } catch (err) {
      console.error(err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleAdvance = async () => {
    if (!selectedOrder) return;
    setActionLoading(true);
    try {
      const res = await fetch(`/api/v1/work-orders/${selectedOrder.work_order_id}/advance`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          note: actionReason
        })
      });
      const data = await res.json();
      if (data.work_order) {
        setSelectedOrder(data.work_order);
      }
      fetchOrders();
    } catch (err) {
      console.error("Error advancing lifecycle:", err);
    } finally {
      setActionLoading(false);
    }
  };

  const handleDerate = async () => {
    if (!selectedOrder) return;
    setActionLoading(true);
    try {
      const res = await fetch(`/api/v1/work-orders/${selectedOrder.work_order_id}/derate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          thrust_derate_pct: 5.5,
          maintenance_hub: "JFK Intl (Base Hangar 4)"
        })
      });
      const data = await res.json();
      if (data.work_order) {
        setSelectedOrder(data.work_order);
      }
      fetchOrders();
    } catch (err) {
      console.error("Error applying derate:", err);
    } finally {
      setActionLoading(false);
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

  // Group into industrial lifecycle pipeline categories
  const proposedOrders = workOrders.filter(
    (o) => o.status === "PROPOSED" || o.status === "AUTO_CREATED"
  );
  const exceptionOrders = workOrders.filter((o) => o.status === "NEEDS_REVIEW");
  const logisticsOrders = workOrders.filter(
    (o) => o.status === "DISPATCHED" || o.status === "PARTS_ORDERED"
  );
  const maintenanceOrders = workOrders.filter(
    (o) => o.status === "IN_PROGRESS" || o.status === "POST_MAINTENANCE_RUNUP" || o.status === "APPROVED" || o.status === "OVERRIDDEN"
  );
  const closedOrders = workOrders.filter(
    (o) => o.status === "AIRWORTHY_CLOSED" || o.status === "CLOSED" || o.status === "REJECTED"
  );

  const getStageIndex = (status: string) => {
    const idx = LIFECYCLE_STAGES.indexOf(status);
    if (idx !== -1) return idx;
    if (status === "AUTO_CREATED") return 0;
    if (status === "APPROVED" || status === "OVERRIDDEN") return 3; // IN_PROGRESS
    if (status === "CLOSED") return 5; // AIRWORTHY_CLOSED
    return -1;
  };

  return (
    <div className="space-y-6">
      {/* Header & Aerospace PdM KPIs Strip */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="glass-panel p-4 rounded-2xl">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Total Work Orders</div>
          <div className="text-2xl font-bold font-mono text-white mt-1">{workOrders.length}</div>
          <div className="text-[10px] text-slate-500 mt-0.5">Fleet-wide Requisition Pipeline</div>
        </div>

        <div className="glass-panel p-4 rounded-2xl border-amber-500/25">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-amber-300">Exception Queue (HITL)</div>
          <div className="text-2xl font-bold font-mono text-amber-400 mt-1">{exceptionOrders.length}</div>
          <div className="text-[10px] text-amber-300/80 mt-0.5">Safety & Disagreement Triage</div>
        </div>

        <div className="glass-panel p-4 rounded-2xl border-indigo-500/25">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-indigo-300">Logistics & Hangar</div>
          <div className="text-2xl font-bold font-mono text-indigo-400 mt-1">{logisticsOrders.length + maintenanceOrders.length}</div>
          <div className="text-[10px] text-indigo-300/80 mt-0.5">Spares Dispatched / In-Work</div>
        </div>

        <div className="glass-panel p-4 rounded-2xl border-emerald-500/25">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-emerald-300">Airworthy Releases</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">{closedOrders.filter(o => o.status === "AIRWORTHY_CLOSED" || o.status === "CLOSED").length}</div>
          <div className="text-[10px] text-emerald-400 mt-0.5">Form 8130-3 Certified</div>
        </div>
      </div>

      {/* Industrial Lifecycle Pipeline Kanban Board */}
      <div className="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4">
        {/* Column 1: Proposed / Initial */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-indigo-500/20">
            <div className="flex items-center gap-2">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-400"></span>
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">1. Proposed</h3>
            </div>
            <span className="text-xs font-mono font-bold text-indigo-400 bg-slate-900 px-2 py-0.5 rounded-full border border-indigo-500/30">
              {proposedOrders.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {proposedOrders.map((order) => (
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
                {order.derate_plan && (
                  <div className="text-[9px] bg-indigo-950/60 text-indigo-300 px-2 py-0.5 rounded border border-indigo-800/40 font-mono">
                    ✈️ De-rate: +{order.derate_plan.life_extension_cycles} cyc
                  </div>
                )}
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle #{order.created_at_cycle}</span>
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
              <h3 className="text-xs font-bold uppercase tracking-wider text-amber-300">Exceptions</h3>
            </div>
            <span className="text-xs font-mono font-bold text-amber-400 bg-amber-500/10 px-2 py-0.5 rounded-full border border-amber-500/30">
              {exceptionOrders.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {exceptionOrders.map((order) => (
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
                    HITL
                  </span>
                </div>
                <p className="text-[11px] text-amber-200/90 font-medium">
                  {order.exception || "Model disagreement or high uncertainty"}
                </p>
                <div className="flex items-center justify-between text-[10px] text-slate-400 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle #{order.created_at_cycle}</span>
                  <span className="text-amber-400 font-bold">Review Now →</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 3: Logistics & Spares (Dispatched & Parts Ordered) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-cyan-500/20">
            <div className="flex items-center gap-2">
              <Truck className="w-3.5 h-3.5 text-cyan-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">2. Logistics</h3>
            </div>
            <span className="text-xs font-mono font-bold text-cyan-400 bg-slate-900 px-2 py-0.5 rounded-full border border-cyan-500/30">
              {logisticsOrders.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {logisticsOrders.map((order) => (
              <div 
                key={order.work_order_id} 
                onClick={() => setSelectedOrder(order)}
                className="glass-panel p-4 rounded-2xl cursor-pointer hover:border-cyan-500/40 transition-all space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-white">
                    {order.engine_id}
                  </span>
                  <span className="text-[9px] px-2 py-0.5 rounded-full font-bold bg-cyan-500/20 text-cyan-300 border border-cyan-500/30">
                    {order.status}
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 line-clamp-2">
                  {order.recommended_action}
                </p>
                {order.parts_bom && (
                  <div className="text-[9px] bg-slate-900 text-cyan-300 px-2 py-0.5 rounded border border-slate-700 font-mono">
                    📦 {order.parts_bom.total_items} Parts · Est ${order.parts_bom.total_cost_usd?.toLocaleString()}
                  </div>
                )}
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle #{order.created_at_cycle}</span>
                  <span className="text-cyan-400 font-medium">BOM Dispatched</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 4: Hangar Maintenance (In Progress & Runup) */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-purple-500/20">
            <div className="flex items-center gap-2">
              <Wrench className="w-3.5 h-3.5 text-purple-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">3. Hangar Work</h3>
            </div>
            <span className="text-xs font-mono font-bold text-purple-400 bg-slate-900 px-2 py-0.5 rounded-full border border-purple-500/30">
              {maintenanceOrders.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {maintenanceOrders.map((order) => (
              <div 
                key={order.work_order_id} 
                onClick={() => setSelectedOrder(order)}
                className="glass-panel p-4 rounded-2xl cursor-pointer hover:border-purple-500/40 transition-all space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-white">
                    {order.engine_id}
                  </span>
                  <span className="text-[9px] px-2 py-0.5 rounded-full font-bold bg-purple-500/20 text-purple-300 border border-purple-500/30">
                    {order.status}
                  </span>
                </div>
                <p className="text-[11px] text-slate-300 line-clamp-2">
                  {order.recommended_action}
                </p>
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle #{order.created_at_cycle}</span>
                  <span className="text-purple-400 font-medium">Tech Assigned</span>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Column 5: Airworthy Release / Closed */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-emerald-500/30">
            <div className="flex items-center gap-2">
              <FileCheck className="w-3.5 h-3.5 text-emerald-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">4. Airworthy</h3>
            </div>
            <span className="text-xs font-mono font-bold text-emerald-400 bg-slate-900 px-2 py-0.5 rounded-full border border-emerald-500/30">
              {closedOrders.length}
            </span>
          </div>

          <div className="space-y-3 max-h-[550px] overflow-y-auto pr-1">
            {closedOrders.map((order) => (
              <div 
                key={order.work_order_id} 
                onClick={() => setSelectedOrder(order)}
                className="glass-panel p-4 rounded-2xl cursor-pointer hover:border-emerald-500/40 transition-all space-y-2"
              >
                <div className="flex items-center justify-between">
                  <span className="font-mono text-xs font-bold text-slate-200">
                    {order.engine_id}
                  </span>
                  <span className="text-[9px] px-2 py-0.5 rounded-full font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">
                    {order.status === "AIRWORTHY_CLOSED" ? "CERTIFIED" : order.status}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 line-clamp-2">
                  {order.recommended_action}
                </p>
                <div className="text-[10px] text-emerald-400 font-mono pt-1 border-t border-slate-800 flex items-center justify-between">
                  <span>FAA 8130-3</span>
                  <span>Closed #{order.closed_at_cycle || order.created_at_cycle}</span>
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
            <div>critical: &lt;= 15 cycles (Triggers Throttle De-rate &amp; Parts BOM)</div>
            <div>high: 16 - 30 cycles</div>
            <div>medium: 31 - 60 cycles</div>
            <div>low: 61 - 100 cycles (Watchlist)</div>
            <div className="text-purple-400 font-semibold mt-2"># Multi-Stage Industrial Lifecycle</div>
            <div>PROPOSED → DISPATCHED → PARTS_ORDERED → IN_PROGRESS → POST_MAINTENANCE_RUNUP → AIRWORTHY_CLOSED</div>
            <div className="text-amber-400 font-semibold mt-2"># Exceptions (HITL Route)</div>
            <div>X1: Interval width &gt; 40 cycles</div>
            <div>X2: Anomaly &gt;= 0.80 but RUL &gt; 100</div>
            <div>X3: Sensor flatlines &gt;= 3 sensors</div>
          </div>
        </div>
      </div>

      {/* Modal / Drawer for Work Order Detail & Aerospace Lifecycle Stepper */}
      {selectedOrder && (
        <div className="fixed inset-0 z-50 bg-black/75 backdrop-blur-md flex items-center justify-center p-4">
          <div className="glass-panel w-full max-w-4xl max-h-[92vh] overflow-y-auto rounded-3xl p-6 border-indigo-500/40 space-y-6 bg-[#0c0d16]">
            {/* Header */}
            <div className="flex items-center justify-between pb-4 border-b border-indigo-500/20">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-xl bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center">
                  <Plane className="w-5 h-5 text-indigo-400" />
                </div>
                <div>
                  <span className="text-xs text-indigo-400 font-mono font-bold">
                    {selectedOrder.work_order_id}
                  </span>
                  <h3 className="text-xl font-black text-white font-mono mt-0.5">
                    {selectedOrder.engine_id}
                  </h3>
                </div>
              </div>
              <div className="flex items-center gap-3">
                {/* One-click PDF download */}
                <a
                  href={`/api/v1/work-orders/${selectedOrder.work_order_id}/airworthiness-report.pdf`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-xs font-semibold transition-all shadow-md"
                >
                  <FileDown className="w-4 h-4 text-indigo-300" />
                  <span>Certified PDF Report</span>
                </a>
                <button 
                  onClick={() => setSelectedOrder(null)}
                  className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800"
                >
                  ✕
                </button>
              </div>
            </div>

            {/* Industrial Multi-Stage Stepper */}
            <div className="space-y-2 bg-slate-950/80 p-4 rounded-2xl border border-slate-800">
              <div className="flex items-center justify-between text-xs text-slate-400 mb-2">
                <span className="font-semibold uppercase tracking-wider text-slate-300">Work Order Industrial Lifecycle</span>
                <span className="font-mono text-cyan-300 font-bold">Current Stage: {selectedOrder.status}</span>
              </div>
              <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
                {LIFECYCLE_STAGES.map((stage, idx) => {
                  const currentIdx = getStageIndex(selectedOrder.status);
                  const isCompleted = currentIdx > idx;
                  const isCurrent = currentIdx === idx;
                  return (
                    <div 
                      key={stage}
                      className={`p-2 rounded-xl text-center border text-[11px] font-mono transition-all ${
                        isCurrent 
                          ? "bg-indigo-600/30 border-indigo-500 text-white font-bold ring-1 ring-indigo-500" 
                          : isCompleted 
                            ? "bg-emerald-950/40 border-emerald-500/40 text-emerald-300" 
                            : "bg-slate-900/50 border-slate-800 text-slate-500"
                      }`}
                    >
                      <div className="text-[9px] uppercase opacity-75">Step {idx + 1}</div>
                      <div className="truncate mt-0.5">{stage.replace(/_/g, " ")}</div>
                    </div>
                  );
                })}
              </div>
            </div>

            {/* Quick Properties */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-xs bg-slate-900/80 p-4 rounded-2xl border border-slate-800">
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Severity</span>
                <span className={`font-bold ${selectedOrder.severity === "CRITICAL" ? "text-rose-400" : "text-amber-400"}`}>
                  {selectedOrder.severity}
                </span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Conservative RUL</span>
                <span className="font-bold text-white font-mono">{selectedOrder.conservative_rul} cycles</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Suspected Subsystem</span>
                <span className="font-bold text-purple-300">{selectedOrder.suspected_subsystem || "HPC"}</span>
              </div>
              <div>
                <span className="text-slate-400 block text-[10px] uppercase">Target Hub</span>
                <span className="font-bold text-cyan-300">{selectedOrder.derate_plan?.maintenance_hub || "JFK Hangar 4"}</span>
              </div>
            </div>

            {/* Self-Healing Flight Throttle De-Rating Recommendation */}
            {selectedOrder.derate_plan ? (
              <div className="bg-gradient-to-r from-indigo-950/60 to-purple-950/60 p-5 rounded-2xl border border-indigo-500/30 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Plane className="w-5 h-5 text-indigo-400" />
                    <h4 className="text-sm font-bold text-white uppercase tracking-wider">
                      Self-Healing Mission Re-planning: FADEC Engine De-rate
                    </h4>
                  </div>
                  <span className="text-xs px-2.5 py-0.5 rounded-full font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40">
                    {selectedOrder.derate_plan.fadec_status}
                  </span>
                </div>
                <p className="text-xs text-slate-300">
                  {selectedOrder.derate_plan.operational_recommendation}
                </p>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-2 text-xs font-mono">
                  <div className="bg-slate-900/80 p-2.5 rounded-xl border border-indigo-500/20">
                    <span className="text-slate-400 text-[10px] block">Thrust De-rate</span>
                    <span className="text-amber-300 font-bold text-sm">-{selectedOrder.derate_plan.thrust_derate_pct}% Max Takeoff</span>
                  </div>
                  <div className="bg-slate-900/80 p-2.5 rounded-xl border border-indigo-500/20">
                    <span className="text-slate-400 text-[10px] block">RUL Extended By</span>
                    <span className="text-emerald-400 font-bold text-sm">+{selectedOrder.derate_plan.life_extension_cycles} cycles</span>
                  </div>
                  <div className="bg-slate-900/80 p-2.5 rounded-xl border border-indigo-500/20">
                    <span className="text-slate-400 text-[10px] block">Effective RUL</span>
                    <span className="text-cyan-300 font-bold text-sm">{selectedOrder.derate_plan.effective_rul_cycles} cycles</span>
                  </div>
                  <div className="bg-slate-900/80 p-2.5 rounded-xl border border-indigo-500/20">
                    <span className="text-slate-400 text-[10px] block">Routing Destination</span>
                    <span className="text-white font-bold text-sm">{selectedOrder.derate_plan.maintenance_hub}</span>
                  </div>
                </div>
              </div>
            ) : (
              <div className="p-4 rounded-2xl border border-slate-800 bg-slate-900/40 flex items-center justify-between">
                <div>
                  <h4 className="text-xs font-bold text-slate-200 uppercase">Self-Healing Throttle De-Rating Available</h4>
                  <p className="text-[11px] text-slate-400 mt-0.5">
                    Calculate Arrhenius creep life extension (+15–20 cycles) and dispatch FADEC advisory.
                  </p>
                </div>
                <button
                  onClick={handleDerate}
                  disabled={actionLoading}
                  className="px-3.5 py-2 bg-indigo-600 hover:bg-indigo-500 text-white rounded-xl text-xs font-bold transition-all shadow-md"
                >
                  Apply FADEC De-rate
                </button>
              </div>
            )}

            {/* Parts Bill of Materials (BOM) & Inventory Dispatch */}
            {selectedOrder.parts_bom && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300 flex items-center gap-1.5">
                    <PackageCheck className="w-4 h-4 text-cyan-400" />
                    <span>Parts Requisition &amp; Spares Inventory Dispatch (BOM)</span>
                  </h4>
                  <span className="text-xs text-slate-400 font-mono">
                    Total: <b className="text-cyan-300">${selectedOrder.parts_bom.total_cost_usd?.toLocaleString()}</b> · Lead: <b className="text-slate-200">{selectedOrder.parts_bom.max_lead_time_days} days</b>
                  </span>
                </div>
                
                <div className="overflow-x-auto rounded-xl border border-slate-800 bg-slate-950/60">
                  <table className="w-full text-left text-xs font-mono">
                    <thead className="bg-slate-900/80 text-slate-400 text-[10px] uppercase border-b border-slate-800">
                      <tr>
                        <th className="p-2.5">Component / LRU</th>
                        <th className="p-2.5">Part Number</th>
                        <th className="p-2.5">Bin Loc</th>
                        <th className="p-2.5">Qty</th>
                        <th className="p-2.5">Unit Cost</th>
                        <th className="p-2.5">Lead Time</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-slate-800/60 text-slate-300">
                      {selectedOrder.parts_bom.line_items?.map((item: any, i: number) => (
                        <tr key={i} className="hover:bg-slate-900/40">
                          <td className="p-2.5 font-semibold text-white">{item.component_name}</td>
                          <td className="p-2.5 text-cyan-300">{item.part_number}</td>
                          <td className="p-2.5 text-slate-400">{item.bin_location}</td>
                          <td className="p-2.5">{item.quantity}</td>
                          <td className="p-2.5">${item.unit_cost_usd?.toLocaleString()}</td>
                          <td className="p-2.5 text-amber-300">{item.lead_time_days} d</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}

            {/* Explainable Decision Trace */}
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

            {/* Human Actions & Multi-Stage Transition Controls */}
            <div className="space-y-3 pt-3 border-t border-slate-800">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-bold uppercase tracking-wider text-slate-300">
                  Industrial Lifecycle Controls &amp; Sign-off
                </h4>
                {selectedOrder.status !== "AIRWORTHY_CLOSED" && (
                  <button
                    onClick={handleAdvance}
                    disabled={actionLoading}
                    className="flex items-center gap-2 bg-gradient-to-r from-indigo-600 to-cyan-600 hover:from-indigo-500 hover:to-cyan-500 text-white px-4 py-2 rounded-xl text-xs font-bold shadow-lg shadow-indigo-600/30 transition-all"
                  >
                    <span>Advance to Next Lifecycle Stage</span>
                    <ArrowRight className="w-4 h-4" />
                  </button>
                )}
              </div>

              <input 
                type="text" 
                placeholder="Operator signoff comments or FAA maintenance log notes..."
                value={actionReason}
                onChange={(e) => setActionReason(e.target.value)}
                className="w-full text-xs p-2.5 bg-slate-900 rounded-xl border border-slate-700 text-white"
              />

              <div className="flex items-center gap-3">
                <button 
                  onClick={() => handleAction("APPROVE")}
                  disabled={actionLoading}
                  className="flex-1 bg-emerald-600/30 hover:bg-emerald-600/50 text-emerald-300 border border-emerald-500/40 py-2 rounded-xl text-xs font-bold transition-all"
                >
                  Authorize Order
                </button>
                <button 
                  onClick={() => handleAction("REJECT")}
                  disabled={actionLoading}
                  className="flex-1 bg-rose-600/30 hover:bg-rose-600/50 text-rose-300 border border-rose-500/40 py-2 rounded-xl text-xs font-bold transition-all"
                >
                  Reject &amp; Dismiss
                </button>
                <button 
                  onClick={() => handleAction("CLOSE")}
                  disabled={actionLoading}
                  className="flex-1 bg-slate-800 hover:bg-slate-700 text-slate-300 py-2 rounded-xl text-xs font-bold transition-all"
                >
                  Certify &amp; Close Out
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
