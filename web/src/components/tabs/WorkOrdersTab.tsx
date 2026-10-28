import React, { useState, useEffect } from "react";
import toast from "react-hot-toast";
import { authHeaders } from "../../lib/api";
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
  ArrowRight,
  Loader2
} from "lucide-react";

interface WorkOrdersTabProps {
  currentSubset: string;
}

const LIFECYCLE_STAGES = [
  "PROPOSED",
  "APPROVED",
  "DISPATCHED",
  "PARTS_ORDERED",
  "IN_PROGRESS",
  "CLOSED"
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
  const [isSimulating, setIsSimulating] = useState<boolean>(false);
  const [isDownloadingPdf, setIsDownloadingPdf] = useState<boolean>(false);

  const getResponseError = async (res: Response): Promise<string> => {
    try {
      const body = await res.json();
      if (body?.detail) return String(body.detail);
    } catch {
      // Response was not JSON; fall back to the HTTP status below.
    }
    return `HTTP ${res.status}: ${res.statusText || "Request failed"}`;
  };

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

  // Maintenance Queue state
  const [maintenanceQueue, setMaintenanceQueue] = useState<any[]>([]);
  const [queueConfig, setQueueConfig] = useState<any | null>(null);
  const [dailyCapacity, setDailyCapacity] = useState<number>(5);
  const [showQueue, setShowQueue] = useState<boolean>(true);

  const fetchQueue = () => {
    fetch(`/api/v1/maintenance-queue?subset=${currentSubset}&capacity=${dailyCapacity}`)
      .then((res) => res.json())
      .then((data) => {
        setMaintenanceQueue(data.queue || []);
        if (data.config) setQueueConfig(data.config);
      })
      .catch((err) => console.error(err));
  };

  useEffect(() => {
    fetchOrders();
    fetchQueue();
    fetch("/api/v1/rules")
      .then((res) => res.json())
      .then((data) => setRulesConfig(data))
      .catch((err) => console.error(err));
  }, [currentSubset, dailyCapacity]);

  const handleAction = async (action: string) => {
    if (!selectedOrder) return;
    setActionLoading(true);
    try {
      const res = await fetch(`/api/v1/work-orders/${selectedOrder.work_order_id}/action`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          action,
          reason: actionReason,
          new_severity: action === "OVERRIDE" ? overrideSeverity : undefined
        })
      });
      if (!res.ok) throw new Error(await getResponseError(res));
      const data = await res.json();
      if (data.work_order) {
        setSelectedOrder(
          ["CLOSED", "REJECTED"].includes(data.work_order.status) ? null : data.work_order
        );
      }
      fetchOrders();
      toast.success(`Work order action applied: ${action}`);
    } catch (err) {
      console.error("Work order action failed:", err);
      const message = err instanceof Error ? err.message : "Unknown error";
      toast.error(`Work order action failed: ${message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const handleAdvance = async () => {
    if (!selectedOrder) return;
    setActionLoading(true);
    try {
      const note = actionReason.trim() || "Stage progression";
      const res = await fetch(
        `/api/v1/work-orders/${selectedOrder.work_order_id}/advance`,
        {
          method: "POST",
          headers: authHeaders({ "Content-Type": "application/json" }),
          body: JSON.stringify({ note })
        }
      );
      if (!res.ok) throw new Error(await getResponseError(res));
      const data = await res.json();
      if (data.work_order) {
        setSelectedOrder(data.work_order.status === "CLOSED" ? null : data.work_order);
        toast.success(`Work order advanced to ${data.work_order.status}`);
      }
      fetchOrders();
    } catch (err) {
      console.error("Error advancing lifecycle:", err);
      const message = err instanceof Error ? err.message : "Unknown error";
      toast.error(`Unable to advance work order: ${message}`);
    } finally {
      setActionLoading(false);
    }
  };

  const runWhatIf = async () => {
    if (isSimulating) return;
    setIsSimulating(true);
    const toastId = toast.loading(`Running What-If Policy Simulation on ${currentSubset}...`);
    try {
      const res = await fetch(`/api/v1/rules/simulate?subset=${currentSubset}`, {
        method: "POST",
        headers: authHeaders({ "Content-Type": "application/json" }),
        body: JSON.stringify({
          critical_threshold: simCrit,
          high_threshold: simHigh,
          anomaly_threshold: 0.80
        })
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}: Simulation error`);
      const data = await res.json();
      setSimResult(data);
      toast.success(
        `What-If Simulation Complete: ${data.total_simulated_orders} work orders evaluated across ${currentSubset}`,
        { id: toastId }
      );
    } catch (err: any) {
      console.error("Simulation failed:", err);
      toast.error(`Simulation failed: ${err.message || "Error running policy simulation"}`, { id: toastId });
    } finally {
      setIsSimulating(false);
    }
  };

  const handleDownloadWorkOrderPdf = async (wo: any) => {
    if (isDownloadingPdf || !wo) return;
    setIsDownloadingPdf(true);
    const toastId = toast.loading(`Generating Maintenance Report PDF for ${wo.work_order_id}...`);
    try {
      const res = await fetch(`/api/v1/work-orders/${wo.work_order_id}/maintenance-report.pdf`);
      if (!res.ok) throw new Error(`HTTP ${res.status}: Failed to generate report`);
      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `maintenance_report_${wo.work_order_id}.pdf`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
      toast.success(`Maintenance Report PDF downloaded for ${wo.work_order_id}`, { id: toastId });
    } catch (err: any) {
      console.error("PDF download error:", err);
      toast.error(`Download failed: ${err.message || "Error generating report"}`, { id: toastId });
    } finally {
      setIsDownloadingPdf(false);
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
    (o) => o.status === "IN_PROGRESS" || o.status === "APPROVED" || o.status === "OVERRIDDEN"
  );
  const closedOrders = workOrders.filter(
    (o) => o.status === "CLOSED" || o.status === "REJECTED"
  );

  const getStageIndex = (status: string) => {
    const idx = LIFECYCLE_STAGES.indexOf(status);
    if (idx !== -1) return idx;
    if (status === "AUTO_CREATED" || status === "NEEDS_REVIEW") return 0;
    if (status === "OVERRIDDEN") return 4;
    return -1;
  };

  const queueWeights = queueConfig?.weights;
  const queueConfigSource = queueConfig?.source || "configs/queue.yaml";
  const queueWeightSummary = queueWeights
    ? `w_rul=${Number(queueWeights.w_rul).toFixed(2)}, w_fail=${Number(queueWeights.w_fail).toFixed(2)}, w_anom=${Number(queueWeights.w_anom).toFixed(2)}, w_trend=${Number(queueWeights.w_trend).toFixed(2)}`
    : "loading configured weights from the queue policy…";

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
          <div className="text-[11px] font-semibold uppercase tracking-wider text-emerald-300">Closed &amp; Resolved</div>
          <div className="text-2xl font-bold font-mono text-emerald-400 mt-1">{closedOrders.length}</div>
          <div className="text-[10px] text-emerald-400 mt-0.5">Maintenance Signoff Completed</div>
        </div>
      </div>

      {/* Ranked Maintenance Priority Queue (Capacity-Constrained) */}
      <div className="glass-panel p-6 rounded-3xl space-y-4 border-indigo-500/30">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Wrench className="w-5 h-5 text-indigo-400" />
            <div>
              <h3 className="font-bold text-sm text-white">Ranked Maintenance Priority Queue</h3>
              <p className="text-[11px] text-slate-400">
                Composite urgency score $S_i$ ranked descending. Top {dailyCapacity} engines flagged as actionable today.
              </p>
            </div>
          </div>

          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 bg-slate-900/90 px-3 py-1.5 rounded-xl border border-slate-800 text-xs">
              <span className="text-slate-400">Daily Service Capacity:</span>
              <select
                value={dailyCapacity}
                onChange={(e) => setDailyCapacity(Number(e.target.value))}
                className="bg-slate-800 text-indigo-300 font-bold px-2 py-0.5 rounded border border-indigo-500/30 font-mono text-xs focus:outline-none"
              >
                <option value={3}>3 engines/shift</option>
                <option value={5}>5 engines/shift</option>
                <option value={8}>8 engines/shift</option>
                <option value={10}>10 engines/shift</option>
              </select>
            </div>
            <button
              onClick={() => setShowQueue(!showQueue)}
              className="text-xs px-3 py-1.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition-colors"
            >
              {showQueue ? "Collapse" : "Expand"}
            </button>
          </div>
        </div>

        {showQueue && maintenanceQueue.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full text-xs text-left">
              <thead className="bg-slate-900/80 text-slate-400 uppercase tracking-wider font-semibold border-b border-slate-800">
                <tr>
                  <th className="p-3">Rank</th>
                  <th className="p-3">Engine ID</th>
                  <th className="p-3">Subsystem</th>
                  <th className="p-3 font-mono">Priority Score</th>
                  <th className="p-3 font-mono">RUL (p10)</th>
                  <th className="p-3 font-mono">P(Fail ≤ 30)</th>
                  <th className="p-3 font-mono">Anomaly Score</th>
                  <th className="p-3 text-center">Shift Status</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800/60 font-mono">
                {maintenanceQueue.map((item: any) => (
                  <tr 
                    key={item.rank}
                    className={`transition-colors ${
                      item.actionable_today ? "bg-emerald-950/15 hover:bg-emerald-950/25" : "hover:bg-slate-800/20 text-slate-400"
                    }`}
                  >
                    <td className="p-3 font-bold text-white">#{item.rank}</td>
                    <td className="p-3 font-bold text-cyan-300 font-mono">{item.engine_id}</td>
                    <td className="p-3 font-sans text-slate-300">
                      <span className="px-2 py-0.5 rounded bg-purple-500/15 text-purple-300 border border-purple-500/30 text-[10px]">
                        {item.suspected_subsystem}
                      </span>
                    </td>
                    <td className="p-3 font-bold text-white">
                      <div className="flex items-center gap-2">
                        <span>{item.priority_score.toFixed(3)}</span>
                        <div className="w-16 h-1.5 bg-slate-800 rounded-full overflow-hidden hidden sm:block">
                          <div 
                            className="h-full bg-indigo-500 rounded-full" 
                            style={{ width: `${Math.min(100, item.priority_score * 100)}%` }}
                          />
                        </div>
                      </div>
                    </td>
                    <td className="p-3 text-amber-300">{item.conservative_rul.toFixed(1)} cyc</td>
                    <td className="p-3 text-rose-300">{(item.fail_prob_30 * 100).toFixed(0)}%</td>
                    <td className="p-3 text-slate-300">{item.anomaly_score.toFixed(2)}</td>
                    <td className="p-3 text-center">
                      {item.actionable_today ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-bold bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 uppercase tracking-wider">
                          <CheckCircle2 className="w-3 h-3" /> Actionable Today
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold bg-slate-800 text-slate-400 border border-slate-700 uppercase tracking-wider">
                          Backlog
                        </span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}

        <div className="text-[10px] text-slate-500 italic">
          * Configured starting weights ({queueWeightSummary}) from <span className="font-mono">{queueConfigSource}</span> for decision support, not empirical physical claims.
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
                <div className="flex items-center justify-between text-[10px] text-slate-500 font-mono pt-1 border-t border-slate-800">
                  <span>Cycle #{order.created_at_cycle}</span>
                  <span className="text-cyan-400 font-medium">Logistics Dispatched</span>
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

        {/* Column 5: Closed & Resolved */}
        <div className="space-y-3">
          <div className="flex items-center justify-between pb-2 border-b border-emerald-500/30">
            <div className="flex items-center gap-2">
              <FileCheck className="w-3.5 h-3.5 text-emerald-400" />
              <h3 className="text-xs font-bold uppercase tracking-wider text-slate-300">4. Closed</h3>
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
                    {order.status}
                  </span>
                </div>
                <p className="text-[11px] text-slate-400 line-clamp-2">
                  {order.recommended_action}
                </p>
                <div className="text-[10px] text-emerald-400 font-mono pt-1 border-t border-slate-800 flex items-center justify-between">
                  <span>Resolved</span>
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
              <label className="text-xs text-slate-400 block mb-1 flex items-center justify-between">
                <span>Critical RUL Threshold: <span className="text-white font-mono font-bold">{simCrit} cycles</span></span>
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-semibold">POLICY</span>
              </label>
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
              <label className="text-xs text-slate-400 block mb-1 flex items-center justify-between">
                <span>High RUL Threshold: <span className="text-white font-mono font-bold">{simHigh} cycles</span></span>
                <span className="text-[9px] uppercase font-mono px-1.5 py-0.2 rounded bg-cyan-500/10 text-cyan-300 border border-cyan-500/20 font-semibold">POLICY</span>
              </label>
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
            disabled={isSimulating}
            className="glass-button text-white text-xs px-4 py-2.5 rounded-xl font-semibold w-full flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
          >
            {isSimulating ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin text-cyan-300" />
                <span>Simulating Fleet Policies on {currentSubset}...</span>
              </>
            ) : (
              <span>Run What-If Simulation on {currentSubset}</span>
            )}
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
            <div>critical: &lt;= 15 cycles (Emergency Maintenance Dispatch)</div>
            <div>high: 16 - 30 cycles</div>
            <div>medium: 31 - 60 cycles</div>
            <div>low: 61 - 100 cycles (Watchlist)</div>
            <div className="text-purple-400 font-semibold mt-2"># Multi-Stage Industrial Lifecycle</div>
            <div>PROPOSED → APPROVED → DISPATCHED → PARTS_ORDERED → IN_PROGRESS → CLOSED</div>
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
                <button
                  onClick={() => handleDownloadWorkOrderPdf(selectedOrder)}
                  disabled={isDownloadingPdf}
                  className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-xs font-semibold transition-all shadow-md disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isDownloadingPdf ? (
                    <Loader2 className="w-4 h-4 animate-spin text-indigo-300" />
                  ) : (
                    <FileDown className="w-4 h-4 text-indigo-300" />
                  )}
                  <span>Maintenance Report PDF</span>
                </button>
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
                <span className="text-slate-400 block text-[10px] uppercase">Due By</span>
                <span className="font-bold text-cyan-300">Cycle #{selectedOrder.due_by_cycle || "—"}</span>
              </div>
            </div>

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
                {selectedOrder.status !== "CLOSED" && (
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
                placeholder="Operator signoff comments or maintenance log notes..."
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
                  Close Out Order
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
