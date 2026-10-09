import React, { useState, useEffect, useRef } from "react";
import { Sidebar } from "./components/Sidebar";
import { Header } from "./components/Header";
import { FleetOverviewTab } from "./components/tabs/FleetOverviewTab";
import { EngineDetailTab } from "./components/tabs/EngineDetailTab";
import { WorkOrdersTab } from "./components/tabs/WorkOrdersTab";
import { ModelLabTab } from "./components/tabs/ModelLabTab";
import { MLOpsTab } from "./components/tabs/MLOpsTab";
import { DataPipelineTab } from "./components/tabs/DataPipelineTab";
import { HowItWorksTab } from "./components/tabs/HowItWorksTab";

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
    const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
    const host = window.location.host;
    const wsUrl = `${protocol}//${host}/ws`;

    const connectWs = () => {
      const socket = new WebSocket(wsUrl);
      wsRef.current = socket;

      socket.onopen = () => {
        setWsConnected(true);
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

      socket.onclose = () => {
        setWsConnected(false);
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

  // Play / Pause / Speed Replay Handlers
  const handleTogglePlay = async () => {
    const nextState = !isPlaying;
    setIsPlaying(nextState);
    if (nextState) {
      await fetch(`/api/v1/replay/start?speed=${speed}`, { method: "POST" });
    } else {
      await fetch(`/api/v1/replay/pause`, { method: "POST" });
    }
  };

  const handleChangeSpeed = async (newSpeed: number) => {
    setSpeed(newSpeed);
    if (isPlaying) {
      await fetch(`/api/v1/replay/start?speed=${newSpeed}`, { method: "POST" });
    }
  };

  const handleReset = async () => {
    setIsPlaying(false);
    await fetch(`/api/v1/replay/reset`, { method: "POST" });
    setCurrentCycle(20);
    fetchFleet();
  };

  const handleLifecycleDemo = async () => {
    // 1-Click Lifecycle Demo Story
    setIsPlaying(true);
    setSpeed(5);
    setActiveTab("engine");
    setSelectedEngineId(`${currentSubset}-001`);
    await fetch(`/api/v1/replay/start?speed=5&mode=lifecycle_demo&unit=1`, { method: "POST" });
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
        className={`flex-1 transition-all duration-300 main-mesh min-h-screen flex flex-col ${
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
        />

        {/* Dynamic Tab Body */}
        <main className="flex-1 p-8 overflow-y-auto">
          {activeTab === "fleet" && (
            <FleetOverviewTab
              fleetData={fleetData}
              onSelectEngine={handleSelectEngine}
              recentAlerts={recentAlerts}
            />
          )}

          {activeTab === "engine" && (
            <EngineDetailTab
              selectedEngineId={selectedEngineId}
              currentSubset={currentSubset}
              onBackToFleet={() => setActiveTab("fleet")}
            />
          )}

          {activeTab === "work_orders" && (
            <WorkOrdersTab currentSubset={currentSubset} />
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
    </div>
  );
}

export default App;
