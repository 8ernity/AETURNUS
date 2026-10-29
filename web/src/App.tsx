import React, { useState, useEffect, useRef } from "react";
import { Toaster } from "react-hot-toast";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { FleetOverviewTab } from "./components/tabs/FleetOverviewTab";
import { EngineDetailTab } from "./components/tabs/EngineDetailTab";
import { WorkOrdersTab } from "./components/tabs/WorkOrdersTab";
import { ModelLabTab } from "./components/tabs/ModelLabTab";
import { CostSimulatorTab } from "./components/tabs/CostSimulatorTab";
import { MLOpsTab } from "./components/tabs/MLOpsTab";
import { DataPipelineTab } from "./components/tabs/DataPipelineTab";
import { HowItWorksTab } from "./components/tabs/HowItWorksTab";
import { DiagnosticPanel } from "./components/DiagnosticPanel";
import { ErrorBoundary } from "./components/ErrorBoundary";
import { ApiKeyDialog } from "./components/ApiKeyDialog";
import { authHeaders, hasApiKey, websocketUrl } from "./lib/api";

export function App() {
  const [activeTab, setActiveTab] = useState<string>("fleet");
  const [collapsed, setCollapsed] = useState<boolean>(false);
  const [currentSubset, setCurrentSubset] = useState<string>("FD001");
  const [selectedEngineId, setSelectedEngineId] = useState<string>("FD001-001");

  // Replay & Telemetry state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [speed, setSpeed] = useState<number>(1);
  const [currentCycle, setCurrentCycle] = useState<number>(30);
  const [wsConnected, setWsConnected] = useState<boolean>(false);
  const [wsAuthFailed, setWsAuthFailed] = useState<boolean>(false);
  const [apiKeyOpen, setApiKeyOpen] = useState<boolean>(false);
  const [hasKey, setHasKey] = useState<boolean>(hasApiKey());

  // Data cache
  const [fleetData, setFleetData] = useState<any>(null);
  const [recentAlerts, setRecentAlerts] = useState<any[]>([]);

  const wsRef = useRef<WebSocket | null>(null);

  // Fetch Fleet summary
  const fetchFleet = () => {
    fetch(`/api/v1/fleet?subset=${currentSubset}`)
      .then((res) => res.json())
      .then((data) => {
        setFleetData(data);
        if (data.engines && data.engines.length > 0 && !selectedEngineId) {
          setSelectedEngineId(data.engines[0].engine_id);
        }
      })
      .catch((err) => console.log("Waiting for backend API..."));
  };

  useEffect(() => {
    fetchFleet();
    const interval = setInterval(fetchFleet, 5000);
    return () => clearInterval(interval);
  }, [currentSubset]);

  // WebSocket Connection
  useEffect(() => {
    const wsUrl = websocketUrl("/ws");

    const connectWs = () => {
      const socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        setWsConnected(true);
        setWsAuthFailed(false);
      };

      socket.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.type === "tick") {
            setCurrentCycle(msg.data.cycle || 30);
            if (msg.data.order) {
              setRecentAlerts((prev) => [msg.data.order, ...prev.slice(0, 30)]);
            }
          } else if (msg.type === "work_order_created") {
            setRecentAlerts((prev) => [msg.data, ...prev.slice(0, 30)]);
          }
        } catch (e) {
          console.error(e);
        }
      };

      socket.onclose = (event) => {
        setWsConnected(false);
        if (event.code === 1008) {
          // Authentication required/rejected. Retrying the same credential
          // every three seconds just fills the console; let the operator fix
          // the key in the header dialog instead.
          setWsAuthFailed(true);
          setHasKey(hasApiKey());
          if (!hasApiKey()) setApiKeyOpen(true);
          return;
        }
        setTimeout(connectWs, 3000);
      };

      socket.onerror = () => {
        socket.close();
      };
    };

    connectWs();
    return () => {
      if (wsRef.current) wsRef.current.close();
    };
  }, []);

  const handleAuthResponse = (res: Response) => {
    if (!res.ok && (res.status === 401 || res.status === 403)) {
      setWsAuthFailed(true);
      setHasKey(hasApiKey());
      setApiKeyOpen(true);
    }
  };

  // Play / Pause / Speed Replay Handlers
  const handleTogglePlay = async () => {
    const nextState = !isPlaying;
    setIsPlaying(nextState);
    const res = nextState
      ? await fetch(`/api/v1/replay/start?speed=${speed}`, {
          method: "POST",
          headers: authHeaders()
        })
      : await fetch(`/api/v1/replay/pause`, {
          method: "POST",
          headers: authHeaders()
        });
    if (!res.ok) {
      setIsPlaying(!nextState);
      handleAuthResponse(res);
      console.error("Replay control rejected:", res.status);
    }
  };

  const handleChangeSpeed = async (newSpeed: number) => {
    setSpeed(newSpeed);
    if (isPlaying) {
      const res = await fetch(`/api/v1/replay/start?speed=${newSpeed}`, {
        method: "POST",
        headers: authHeaders()
      });
      handleAuthResponse(res);
    }
  };

  const handleReset = async () => {
    setIsPlaying(false);
    const res = await fetch(`/api/v1/replay/reset`, { method: "POST", headers: authHeaders() });
    handleAuthResponse(res);
    setCurrentCycle(20);
    fetchFleet();
  };

  const handleLifecycleDemo = async () => {
    // 1-Click Lifecycle Demo Story
    setIsPlaying(true);
    setSpeed(5);
    setActiveTab("engine");
    setSelectedEngineId(`${currentSubset}-001`);
    const res = await fetch(`/api/v1/replay/start?speed=5&mode=lifecycle_demo&unit=1`, {
      method: "POST",
      headers: authHeaders()
    });
    handleAuthResponse(res);
  };

  const handleSelectEngine = (engineId: string) => {
    setSelectedEngineId(engineId);
    setActiveTab("engine");
  };

  const exceptionCount = fleetData?.kpis?.exception_queue_size || 0;

  return (
    <div className="min-h-screen bg-[#090a10] text-[#f8fafc] flex">
      {/* Collapsible Sidebar */}
      <Sidebar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        collapsed={collapsed}
        setCollapsed={setCollapsed}
        exceptionCount={exceptionCount}
      />

      {/* Main Content Area */}
      <div 
        className={`flex-1 min-w-0 transition-all duration-300 main-mesh min-h-screen flex flex-col ${
          collapsed ? "ml-20" : "ml-72"
        }`}
      >
        {/* Sticky Header with Replay Controls */}
        <Header
          currentSubset={currentSubset}
          onSelectSubset={(sub) => {
            setCurrentSubset(sub);
            setSelectedEngineId(`${sub}-001`);
          }}
          isPlaying={isPlaying}
          onTogglePlay={handleTogglePlay}
          speed={speed}
          onChangeSpeed={handleChangeSpeed}
          onReset={handleReset}
          onLifecycleDemo={handleLifecycleDemo}
          currentCycle={currentCycle}
          wsConnected={wsConnected}
          wsAuthFailed={wsAuthFailed}
          hasApiKey={hasKey}
          onOpenApiKey={() => setApiKeyOpen(true)}
        />

        {/* Global data-provenance banner: every live-looking surface is replay data. */}
        <div className="border-b border-amber-500/20 bg-amber-500/10 px-8 py-2 text-[11px] font-medium text-amber-200 flex items-center gap-2">
          <span className="uppercase tracking-wider font-bold text-amber-300">Simulation</span>
          <span>
            Replay of NASA C-MAPSS data (simulated). Telemetry, decisions, and maintenance
            actions are demonstrations, not live aircraft operations.
          </span>
        </div>

        {/* Dynamic Tab Body */}
        <main className="flex-1 min-w-0 p-8 overflow-y-auto">
          {activeTab === "fleet" && (
            <ErrorBoundary fallbackTitle="Fleet Overview">
              <FleetOverviewTab
                fleetData={fleetData}
                onSelectEngine={handleSelectEngine}
                recentAlerts={recentAlerts}
              />
            </ErrorBoundary>
          )}

          {activeTab === "engine" && (
            <ErrorBoundary fallbackTitle="Engine Telemetry View">
              <EngineDetailTab
                selectedEngineId={selectedEngineId}
                currentSubset={currentSubset}
                onBackToFleet={() => setActiveTab("fleet")}
              />
            </ErrorBoundary>
          )}

          {activeTab === "work_orders" && (
            <WorkOrdersTab currentSubset={currentSubset} />
          )}

          {activeTab === "cost_simulator" && (
            <CostSimulatorTab currentSubset={currentSubset} />
          )}

          {activeTab === "model_lab" && (
            <ModelLabTab />
          )}

          {activeTab === "mlops" && (
            <MLOpsTab currentSubset={currentSubset} />
          )}

          {activeTab === "pipeline" && (
            <DataPipelineTab currentSubset={currentSubset} />
          )}

          {activeTab === "how_it_works" && (
            <HowItWorksTab />
          )}
        </main>
      </div>

      {/* Fleet Diagnostic Panel & Grok Bot Copilot */}
      <DiagnosticPanel
        currentSubset={currentSubset}
        currentCycle={currentCycle}
        isPlaying={isPlaying}
        exceptionCount={fleetData?.exception_count || 0}
        wsConnected={wsConnected}
        fleetData={fleetData}
        recentAlerts={recentAlerts}
        onSelectEngine={handleSelectEngine}
      />

      <ApiKeyDialog open={apiKeyOpen} onClose={() => setApiKeyOpen(false)} />

      <Toaster 
        position="top-right" 
        toastOptions={{
          style: {
            background: "#0f172a",
            color: "#f8fafc",
            border: "1px solid rgba(99, 102, 241, 0.3)",
            fontSize: "12px",
            borderRadius: "1rem"
          }
        }}
      />
    </div>
  );
}

export default App;
