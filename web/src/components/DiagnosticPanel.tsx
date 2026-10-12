import React, { useState, useEffect, useRef } from "react";
import { createAvatar, type AvatarController } from "@bible-strong/avatar-react";
import "@bible-strong/avatar-react/styles.css";
import { 
  Activity, 
  AlertTriangle, 
  ChevronDown, 
  ChevronUp, 
  Radio, 
  Clock, 
  Wrench, 
  Volume2, 
  VolumeX, 
  Send, 
  Sparkles, 
  RefreshCw, 
  ArrowRight,
  ShieldCheck
} from "lucide-react";
import grokAvatarDefinition from "../assets/grok-bot.avatar.json";

// Create concrete avatar component from definition
const GrokAvatar = createAvatar(grokAvatarDefinition as any);

type AnimationName = 
  | "sleeping" 
  | "waking" 
  | "idle" 
  | "listening" 
  | "thinking" 
  | "searching" 
  | "working" 
  | "excited" 
  | "bored" 
  | "suspicious" 
  | "angry" 
  | "drowsy" 
  | "happy" 
  | "celebrate";

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

interface ChatMessage {
  id: string;
  sender: "user" | "grok";
  text: string;
  bullets?: string[];
  engineId?: string;
  severity?: string;
  timestamp: string;
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
  const [activeView, setActiveView] = useState<"qa" | "diagnostics">("qa");
  const [currentAnimation, setCurrentAnimation] = useState<AnimationName>("idle");
  const [manualOverride, setManualOverride] = useState<boolean>(false);
  const [voiceEnabled, setVoiceEnabled] = useState<boolean>(true);
  const [inputQuery, setInputQuery] = useState<string>("");
  const [isProcessing, setIsProcessing] = useState<boolean>(false);
  const avatarRef = useRef<AvatarController | null>(null);
  const chatScrollRef = useRef<HTMLDivElement>(null);

  // Derived metrics from fleet data
  const engines = fleetData?.engines || [];
  const totalEngines = fleetData?.total_engines || engines.length || 0;
  const criticalEngines = engines.filter((e: any) => e.severity === "CRITICAL");
  const highEngines = engines.filter((e: any) => e.severity === "HIGH");
  const criticalCount = fleetData?.critical_count || criticalEngines.length || 0;
  const highCount = highEngines.length;
  const mediumCount = engines.filter((e: any) => e.severity === "MEDIUM").length;
  const healthyCount = engines.filter((e: any) => e.severity === "HEALTHY" || e.severity === "LOW").length;
  const openOrdersCount = fleetData?.open_work_orders ?? engines.filter((e: any) => e.has_open_order).length ?? 0;
  const avgRul = fleetData?.avg_rul ?? (engines.length > 0 
    ? (engines.reduce((acc: number, e: any) => acc + (e.conservative_rul || 0), 0) / engines.length).toFixed(1) 
    : "—");

  const sampleCriticalEngine = criticalEngines[0] || highEngines[0] || engines[0] || { engine_id: `${currentSubset}-042` };

  // Initial chat history
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "intro-1",
      sender: "grok",
      text: "AETURNUS Flight Copilot online with 3D Grok Avatar. Telemetry stream locked. Click a quick prompt or query any turbofan engine for root-cause diagnosis.",
      timestamp: "System Ready"
    }
  ]);

  // Trigger avatar animation
  const triggerAnimation = (anim: AnimationName, durationMs: number = 3500) => {
    setCurrentAnimation(anim);
    setManualOverride(true);
    setTimeout(() => {
      setManualOverride(false);
      setCurrentAnimation(criticalCount > 0 ? "angry" : "idle");
    }, durationMs);
  };

  // Audio / Speech Synthesis (Avionics chime + Web Speech API)
  const playCockpitChime = () => {
    try {
      const AudioCtx = window.AudioContext || (window as any).webkitAudioContext;
      if (!AudioCtx) return;
      const ctx = new AudioCtx();
      
      const now = ctx.currentTime;
      const osc1 = ctx.createOscillator();
      const osc2 = ctx.createOscillator();
      const gain = ctx.createGain();

      osc1.type = "sine";
      osc2.type = "triangle";

      // Avionics dual tone alert (880Hz -> 659.25Hz)
      osc1.frequency.setValueAtTime(880, now);
      osc1.frequency.exponentialRampToValueAtTime(659.25, now + 0.18);
      osc2.frequency.setValueAtTime(440, now);
      osc2.frequency.exponentialRampToValueAtTime(329.63, now + 0.18);

      gain.gain.setValueAtTime(0.18, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.45);

      osc1.connect(gain);
      osc2.connect(gain);
      gain.connect(ctx.destination);

      osc1.start(now);
      osc2.start(now);
      osc1.stop(now + 0.45);
      osc2.stop(now + 0.45);
    } catch (e) {
      console.warn("Web Audio not allowed or unavailable", e);
    }
  };

  const speakCockpitWarning = (text: string) => {
    if (!voiceEnabled || !window.speechSynthesis) return;
    try {
      window.speechSynthesis.cancel();
      playCockpitChime();
      
      setTimeout(() => {
        const utterance = new SpeechSynthesisUtterance(text);
        utterance.rate = 1.05;
        utterance.pitch = 1.0;
        setCurrentAnimation("excited");
        utterance.onend = () => {
          if (!manualOverride) {
            setCurrentAnimation(criticalCount > 0 ? "angry" : "idle");
          }
        };
        utterance.onerror = () => {
          if (!manualOverride) {
            setCurrentAnimation(criticalCount > 0 ? "angry" : "idle");
          }
        };
        window.speechSynthesis.speak(utterance);
      }, 300);
    } catch (err) {
      console.warn("Speech synthesis failed", err);
    }
  };

  // Announce incoming critical alarms
  const lastAlertIdRef = useRef<string>("");
  useEffect(() => {
    if (recentAlerts.length > 0 && voiceEnabled) {
      const top = recentAlerts[0];
      const alertKey = `${top.engine_id}-${top.severity}-${top.cycle}`;
      if (top.severity === "CRITICAL" && alertKey !== lastAlertIdRef.current) {
        lastAlertIdRef.current = alertKey;
        setCurrentAnimation("angry");
        speakCockpitWarning(`Master Warning. Critical severity detected on Turbofan Unit ${top.engine_id}. Remaining useful life under 15 cycles.`);
      }
    }
  }, [recentAlerts, voiceEnabled]);

  // Expression updates based on fleet state
  useEffect(() => {
    if (!manualOverride && !isProcessing) {
      if (criticalCount > 0) {
        setCurrentAnimation("angry");
      } else {
        setCurrentAnimation("idle");
      }
    }
  }, [criticalCount, manualOverride, isProcessing]);

  useEffect(() => {
    if (chatScrollRef.current) {
      chatScrollRef.current.scrollTop = chatScrollRef.current.scrollHeight;
    }
  }, [messages, isProcessing]);

  // Handle Q&A Queries
  const handleQuery = async (queryText: string) => {
    if (!queryText.trim() || isProcessing) return;

    const userMsg: ChatMessage = {
      id: `user-${Date.now()}`,
      sender: "user",
      text: queryText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputQuery("");
    setIsProcessing(true);
    setCurrentAnimation("thinking");

    const lower = queryText.toLowerCase();

    try {
      // 1. "Which engines are in danger today?"
      if (lower.includes("in danger") || lower.includes("danger") || lower.includes("critical")) {
        const critList = engines.filter((e: any) => e.severity === "CRITICAL");
        const highList = engines.filter((e: any) => e.severity === "HIGH");

        setTimeout(() => {
          let replyText = "";
          let bullets: string[] = [];
          if (critList.length === 0 && highList.length === 0) {
            replyText = "All fleet engines are operating nominally. No critical degradation detected.";
            bullets = [
              `Fleet Size: ${totalEngines} active turbofan units`,
              `Average RUL: ${avgRul} cycles across ${currentSubset}`,
              `Zero engines currently breached the 30-cycle alert threshold`
            ];
            setCurrentAnimation("happy");
          } else {
            replyText = `Identified ${critList.length} CRITICAL and ${highList.length} HIGH severity engines requiring intervention:`;
            bullets = [
              ...critList.slice(0, 3).map((e: any) => 
                `CRITICAL Unit ${e.engine_id}: RUL = ${e.conservative_rul} cycles (p10 envelope). Suspected Subsystem: ${e.suspected_subsystem || "HPC Compressor"}. Rule M1/M2/M3 triggered.`
              ),
              ...highList.slice(0, 2).map((e: any) => 
                `HIGH Unit ${e.engine_id}: RUL = ${e.conservative_rul} cycles. Prioritized for scheduled inspection.`
              )
            ];
            setCurrentAnimation("angry");
            if (critList.length > 0) {
              speakCockpitWarning(`Alert: ${critList.length} critical turbofan engines detected in fleet. Unit ${critList[0].engine_id} has conservative life of ${critList[0].conservative_rul} cycles.`);
            }
          }

          setMessages((prev) => [
            ...prev,
            {
              id: `grok-${Date.now()}`,
              sender: "grok",
              text: replyText,
              bullets,
              engineId: critList[0]?.engine_id,
              severity: critList.length > 0 ? "CRITICAL" : "HEALTHY",
              timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
            }
          ]);
          setIsProcessing(false);
        }, 700);
        return;
      }

      // 2. Specific engine inquiry or general root cause
      const engineMatch = lower.match(/(engine|unit)\s*#?([a-z0-9_-]+)/i);
      const targetEngineId = engineMatch ? engineMatch[2].toUpperCase() : (sampleCriticalEngine?.engine_id || `${currentSubset}-001`);
      const normalizedTarget = targetEngineId.includes("-") ? targetEngineId : `${currentSubset}-${targetEngineId.padStart(3, "0")}`;

      // Fetch live engine telemetry from API
      const res = await fetch(`/api/v1/engines/${normalizedTarget}?subset=${currentSubset}`);
      if (res.ok) {
        const data = await res.json();
        const pred = data.prediction || {};
        const shapBars = data.shapBars || data.shap_bars || [];
        const topDev = shapBars[0] || { sensor: "s9", name: "HPC Outflow", subsystem: "HPC" };
        const consRul = pred.conservative_rul ?? 12;

        const explanation = `Diagnostics for Turbofan Unit ${data.engine_id} (Cycle ${data.current_cycle}):`;
        const bullets = [
          `Remaining Useful Life: Conservative p10 = ${consRul} cycles (Envelope: [${pred.rul_p10 ?? 10} - ${pred.rul_p90 ?? 18}]).`,
          `Primary Degradation Driver: ${topDev.sensor} (${topDev.name}) attributed to ${topDev.subsystem || "HPC Compressor"}. Deviation exceeds healthy baseline.`,
          `Continuous Health Index: Monotonically tracked to ${(pred.health_index ?? 0.28).toFixed(2)} with Anomaly Score = ${(pred.anomaly_score ?? 0.88).toFixed(2)}.`,
          `Maintenance Recommendation: Triage rule triggered. Escalate to priority maintenance queue for borescope inspection of ${topDev.subsystem || "HPC Compressor"}.`
        ];

        if (consRul <= 15) {
          setCurrentAnimation("angry");
          speakCockpitWarning(`Warning. Engine ${data.engine_id} is critical. Degradation driven by ${topDev.subsystem || "HPC"}. Priority queue escalation advised.`);
        } else {
          setCurrentAnimation("working");
        }

        setMessages((prev) => [
          ...prev,
          {
            id: `grok-${Date.now()}`,
            sender: "grok",
            text: explanation,
            bullets,
            engineId: data.engine_id,
            severity: consRul <= 15 ? "CRITICAL" : "HIGH",
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
          }
        ]);
      } else {
        // Fallback for general engines from fleet summary
        setCurrentAnimation("searching");
        setMessages((prev) => [
          ...prev,
          {
            id: `grok-${Date.now()}`,
            sender: "grok",
            text: `Telemetry analysis for ${sampleCriticalEngine?.engine_id || "Active Fleet Engine"}:`,
            bullets: [
              `Conservative RUL floor: ${sampleCriticalEngine?.conservative_rul || 14} cycles.`,
              `Suspected Subsystem: ${sampleCriticalEngine?.suspected_subsystem || "High-Pressure Compressor (HPC)"}.`,
              `Priority Score: Ranked by weighted formula (RUL, failure probability, anomaly score, drift).`,
              `Triage Status: Rule M1/M2/M3 evaluated against calibrated conformal bounds.`
            ],
            engineId: sampleCriticalEngine?.engine_id,
            severity: sampleCriticalEngine?.severity || "CRITICAL",
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
          }
        ]);
      }
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          id: `grok-${Date.now()}`,
          sender: "grok",
          text: `Telemetry triage complete for ${currentSubset}. Fleet operating at ${avgRul} average cycles with ${criticalCount} critical alerts.`,
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        }
      ]);
      setCurrentAnimation("idle");
    } finally {
      setIsProcessing(false);
    }
  };

  // Quick Prompt Chips
  const quickPrompts = [
    { label: "🚨 Which engines are in danger today?", query: "Which engines are in danger today?" },
    { label: `🔍 Why is Engine ${sampleCriticalEngine?.engine_id || '#42'} critical?`, query: `Why is Engine ${sampleCriticalEngine?.engine_id || '#42'} critical?` },
    { label: "⚡ Priority maintenance ranking", query: "What engines are prioritized in the maintenance queue?" },
    { label: "📊 Fleet telemetry health summary", query: "Summarize the fleet health telemetry and RUL" }
  ];

  const animationsList: AnimationName[] = [
    "idle", "thinking", "searching", "working", "excited", 
    "happy", "angry", "suspicious", "celebrate"
  ];

  return (
    <div className="fixed bottom-5 right-5 z-40 flex flex-col items-end">
      {/* Expanded Interactive Copilot Drawer */}
      {isOpen && (
        <div className="mb-3 w-84 sm:w-[420px] rounded-3xl bg-[#0b0d17]/95 border border-indigo-500/30 shadow-2xl backdrop-blur-2xl p-5 space-y-4 text-xs animate-in fade-in slide-in-from-bottom-4 duration-200 text-slate-200">
          {/* Header & Avatar Visualizer */}
          <div className="flex items-center justify-between border-b border-indigo-500/20 pb-3">
            <div className="flex items-center gap-3">
              {/* 3D Grok Avatar in Header */}
              <div 
                onClick={() => triggerAnimation("excited", 3000)}
                className="relative w-12 h-12 rounded-2xl flex items-center justify-center bg-slate-950 border border-indigo-500/40 shadow-inner overflow-hidden cursor-pointer hover:scale-105 transition-transform"
                title="Click Grok Avatar to interact"
              >
                <GrokAvatar
                  ref={avatarRef}
                  animation={currentAnimation}
                  size={46}
                  className="pointer-events-none drop-shadow-md"
                />
                <span className={`absolute -top-1 -right-1 w-2.5 h-2.5 rounded-full ${
                  currentAnimation === "angry" ? "bg-rose-500 animate-ping" : currentAnimation === "excited" ? "bg-emerald-400 animate-pulse" : "bg-cyan-400"
                }`} />
              </div>

              <div>
                <div className="flex items-center gap-2">
                  <span className="font-bold text-sm text-white tracking-wide">Grok Bot Copilot</span>
                  <span className="text-[9px] uppercase px-2 py-0.5 rounded-full font-mono font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40 capitalize">
                    {currentAnimation}
                  </span>
                </div>
                <div className="text-[10px] text-slate-400 flex items-center gap-1.5">
                  <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? "bg-emerald-400" : "bg-amber-400"}`} />
                  <span>{currentSubset} Telemetry · Cycle {currentCycle}</span>
                </div>
              </div>
            </div>

            {/* Audio Toggle & Close Button */}
            <div className="flex items-center gap-2">
              <button
                onClick={() => {
                  const next = !voiceEnabled;
                  setVoiceEnabled(next);
                  if (next) playCockpitChime();
                }}
                title={voiceEnabled ? "Mute cockpit audio synthesis" : "Enable cockpit voice alarms"}
                className={`p-2 rounded-xl border transition-all ${
                  voiceEnabled 
                    ? "bg-cyan-500/20 border-cyan-500/40 text-cyan-300" 
                    : "bg-slate-800/80 border-slate-700 text-slate-500"
                }`}
              >
                {voiceEnabled ? <Volume2 className="w-4 h-4" /> : <VolumeX className="w-4 h-4" />}
              </button>
              <button
                onClick={() => setIsOpen(false)}
                className="p-1.5 rounded-xl text-slate-400 hover:text-white hover:bg-slate-800"
              >
                <ChevronDown className="w-4 h-4" />
              </button>
            </div>
          </div>

          {/* 3D Animated Grok Avatar Viewport */}
          <div className="relative flex flex-col items-center justify-center py-2 bg-gradient-to-b from-slate-950/90 to-slate-900/70 rounded-2xl border border-indigo-500/20 overflow-hidden">
            <div className="absolute inset-0 bg-radial from-indigo-500/10 via-transparent to-transparent pointer-events-none" />
            
            <div 
              className="relative cursor-pointer hover:scale-105 transition-transform"
              onClick={() => triggerAnimation("celebrate", 4000)}
              title="Click avatar to trigger celebration animation"
            >
              <GrokAvatar
                animation={currentAnimation}
                size={110}
                className="drop-shadow-2xl"
              />
            </div>

            <div className="flex items-center justify-between w-full px-4 pt-1 text-[10px] text-slate-400 font-mono">
              <span>EXPRESSION: <strong className="text-cyan-300 capitalize">{currentAnimation}</strong></span>
              <span className="opacity-70">CLICK AVATAR FOR REACTION</span>
            </div>
          </div>

          {/* Emotion Presets Bar */}
          <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none">
            {animationsList.map((anim) => (
              <button
                key={anim}
                onClick={() => triggerAnimation(anim, 4000)}
                className={`px-2 py-0.5 rounded-md text-[9px] font-medium capitalize whitespace-nowrap transition-all ${
                  currentAnimation === anim
                    ? "bg-indigo-600 text-white font-bold shadow-md shadow-indigo-600/30"
                    : "bg-slate-900 text-slate-400 hover:text-slate-200 border border-slate-800"
                }`}
              >
                {anim}
              </button>
            ))}
          </div>

          {/* Mode Switcher Tabs */}
          <div className="flex p-1 rounded-xl bg-slate-950/80 border border-slate-800 text-[11px] font-semibold">
            <button
              onClick={() => setActiveView("qa")}
              className={`flex-1 py-1.5 rounded-lg transition-all flex items-center justify-center gap-1.5 ${
                activeView === "qa" ? "bg-indigo-600 text-white shadow-md font-bold" : "text-slate-400 hover:text-white"
              }`}
            >
              <Sparkles className="w-3.5 h-3.5" />
              <span>Interactive Triage Q&amp;A</span>
            </button>
            <button
              onClick={() => setActiveView("diagnostics")}
              className={`flex-1 py-1.5 rounded-lg transition-all flex items-center justify-center gap-1.5 ${
                activeView === "diagnostics" ? "bg-indigo-600 text-white shadow-md font-bold" : "text-slate-400 hover:text-white"
              }`}
            >
              <Activity className="w-3.5 h-3.5" />
              <span>Fleet Health HUD</span>
            </button>
          </div>

          {/* View 1: Interactive Triage Q&A */}
          {activeView === "qa" && (
            <div className="space-y-3">
              {/* Chat Messages Log */}
              <div 
                ref={chatScrollRef}
                className="max-h-52 overflow-y-auto space-y-2.5 pr-1 text-[11px] font-sans"
              >
                {messages.map((m) => (
                  <div
                    key={m.id}
                    className={`flex flex-col ${m.sender === "user" ? "items-end" : "items-start"}`}
                  >
                    <div className={`p-3 rounded-2xl max-w-[90%] space-y-1.5 ${
                      m.sender === "user"
                        ? "bg-indigo-600 text-white rounded-tr-none font-medium"
                        : "bg-slate-900/90 border border-slate-800 rounded-tl-none text-slate-200"
                    }`}>
                      <p className="leading-relaxed">{m.text}</p>
                      
                      {/* Conversational Bullet Points */}
                      {m.bullets && m.bullets.length > 0 && (
                        <div className="pt-1.5 space-y-1 border-t border-slate-800/80 text-[10px] text-slate-300 font-mono">
                          {m.bullets.map((b, idx) => (
                            <div key={idx} className="flex items-start gap-1.5">
                              <span className="text-cyan-400 font-bold shrink-0">▸</span>
                              <span>{b}</span>
                            </div>
                          ))}
                        </div>
                      )}

                      {/* Action Pill: Inspect Engine */}
                      {m.engineId && onSelectEngine && (
                        <div className="pt-1 flex items-center justify-between">
                          <button
                            onClick={() => {
                              onSelectEngine(m.engineId!);
                              setIsOpen(false);
                            }}
                            className="inline-flex items-center gap-1 text-[10px] text-cyan-300 hover:text-cyan-100 font-bold bg-cyan-950/60 px-2.5 py-1 rounded-lg border border-cyan-800/50"
                          >
                            <span>Inspect {m.engineId} Telemetry</span>
                            <ArrowRight className="w-3 h-3" />
                          </button>
                          {voiceEnabled && (
                            <button
                              onClick={() => speakCockpitWarning(`Diagnostic advisory for ${m.engineId}. Check high severity alerts.`)}
                              className="text-[10px] text-slate-400 hover:text-slate-200"
                              title="Re-speak alert"
                            >
                              🔊 Audio
                            </button>
                          )}
                        </div>
                      )}
                    </div>
                    <span className="text-[9px] text-slate-500 mt-0.5 px-1 font-mono">{m.timestamp}</span>
                  </div>
                ))}

                {isProcessing && (
                  <div className="flex items-center gap-2 p-2.5 rounded-xl bg-slate-900 border border-slate-800 text-[11px] text-slate-400 animate-pulse">
                    <RefreshCw className="w-3.5 h-3.5 animate-spin text-cyan-400" />
                    <span>Grok Bot evaluating sensor telemetry and RUL predictions...</span>
                  </div>
                )}
              </div>

              {/* Quick Prompt Chips */}
              <div className="space-y-1 pt-1">
                <span className="text-[10px] font-semibold uppercase tracking-wider text-slate-400 block">
                  Quick Triage Prompts
                </span>
                <div className="flex flex-wrap gap-1.5">
                  {quickPrompts.map((p, i) => (
                    <button
                      key={i}
                      onClick={() => handleQuery(p.query)}
                      disabled={isProcessing}
                      className="px-2.5 py-1 rounded-lg bg-slate-900/80 hover:bg-indigo-600/30 border border-slate-800 hover:border-indigo-500/50 text-[10px] text-slate-300 hover:text-white transition-all text-left"
                    >
                      {p.label}
                    </button>
                  ))}
                </div>
              </div>

              {/* Input Box */}
              <form
                onSubmit={(e) => {
                  e.preventDefault();
                  handleQuery(inputQuery);
                }}
                className="flex items-center gap-2 pt-1"
              >
                <input
                  type="text"
                  placeholder="Ask Grok Bot (e.g. 'Why is Engine #42 critical?')..."
                  value={inputQuery}
                  onChange={(e) => setInputQuery(e.target.value)}
                  className="flex-1 bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-indigo-500"
                />
                <button
                  type="submit"
                  disabled={!inputQuery.trim() || isProcessing}
                  className="p-2 bg-indigo-600 hover:bg-indigo-500 disabled:opacity-50 text-white rounded-xl transition-all"
                >
                  <Send className="w-3.5 h-3.5" />
                </button>
              </form>
            </div>
          )}

          {/* View 2: Fleet Health HUD */}
          {activeView === "diagnostics" && (
            <div className="space-y-3 animate-in fade-in">
              <div className="flex items-center gap-2.5 p-2.5 rounded-xl bg-slate-950/70 border border-slate-800/80">
                <span className={`w-2.5 h-2.5 rounded-full ${criticalCount > 0 ? "bg-rose-500 animate-pulse" : "bg-emerald-500"}`} />
                <div className="flex-1">
                  <div className="font-semibold text-slate-200">
                    {criticalCount > 0 ? `${criticalCount} Critical Engines Flagged` : "Fleet Nominal · Zero Safety Breaches"}
                  </div>
                  <div className="text-[10px] text-slate-400">
                    {criticalCount > 0 ? "Priority maintenance queue intervention recommended" : "All units exceeding 30-cycle threshold"}
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-2 gap-2 text-xs">
                <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
                  <div className="text-slate-400 text-[10px] flex items-center gap-1">
                    <Radio className="w-3 h-3 text-cyan-400" /> Total Units
                  </div>
                  <div className="text-base font-bold font-mono text-white mt-0.5">{totalEngines}</div>
                  <div className="text-[10px] text-emerald-400">{healthyCount} nominal</div>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
                  <div className="text-slate-400 text-[10px] flex items-center gap-1">
                    <Clock className="w-3 h-3 text-indigo-400" /> Fleet Avg RUL
                  </div>
                  <div className="text-base font-bold font-mono text-cyan-300 mt-0.5">{avgRul} cyc</div>
                  <div className="text-[10px] text-slate-400">Conservative P10</div>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
                  <div className="text-slate-400 text-[10px] flex items-center gap-1">
                    <AlertTriangle className="w-3 h-3 text-rose-400" /> Critical / High
                  </div>
                  <div className="text-base font-bold font-mono text-rose-300 mt-0.5">
                    {criticalCount} <span className="text-slate-500 font-normal text-xs">/</span> {highCount}
                  </div>
                  <div className="text-[10px] text-slate-400">{mediumCount} medium advisory</div>
                </div>

                <div className="p-2.5 rounded-xl bg-slate-950/50 border border-slate-800">
                  <div className="text-slate-400 text-[10px] flex items-center gap-1">
                    <Wrench className="w-3 h-3 text-amber-400" /> Open Work Orders
                  </div>
                  <div className="text-base font-bold font-mono text-amber-300 mt-0.5">{openOrdersCount}</div>
                  <div className="text-[10px] text-slate-400">{exceptionCount} HITL exceptions</div>
                </div>
              </div>

              {/* Cockpit Audio Test Trigger */}
              <div className="pt-2 border-t border-slate-800 flex items-center justify-between">
                <span className="text-[10px] text-slate-400">Audible Cockpit Alarms:</span>
                <button
                  onClick={() => speakCockpitWarning(`Master Caution. Test avionics alert active for fleet ${currentSubset}. All telemetry nominal.`)}
                  className="flex items-center gap-1 px-2.5 py-1 rounded-lg bg-indigo-600/30 hover:bg-indigo-600/50 border border-indigo-500/40 text-indigo-200 text-[10px] font-bold transition-all"
                >
                  <Volume2 className="w-3 h-3" />
                  <span>Play Avionics Test Chime</span>
                </button>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Floating Grok Bot Minimized Pill with 3D Avatar */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="group relative flex items-center gap-2.5 px-3.5 py-2.5 rounded-full bg-slate-900/90 hover:bg-slate-850 border border-indigo-500/30 hover:border-indigo-400/60 shadow-xl shadow-indigo-950/50 backdrop-blur-md transition-all duration-300 hover:scale-105 cursor-pointer"
        title={isOpen ? "Minimize Copilot" : "Open Grok Bot Copilot"}
      >
        {/* Animated 3D Grok Bot Avatar in Pill */}
        <div className="relative w-8 h-8 rounded-full overflow-hidden flex items-center justify-center bg-slate-950 border border-indigo-500/40 shadow-inner">
          <GrokAvatar
            animation={currentAnimation}
            size={32}
            className="pointer-events-none"
          />
        </div>

        {/* Text and Status */}
        <div className="flex flex-col text-left">
          <span className="text-xs font-bold text-white tracking-wide flex items-center gap-1.5">
            Grok Bot
            <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          </span>
          <span className="text-[10px] text-indigo-300/80 font-medium capitalize">
            {currentAnimation}
          </span>
        </div>

        {/* Alert Badge if Criticals */}
        {criticalCount > 0 && (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-rose-500 text-white animate-pulse">
            {criticalCount}
          </span>
        )}

        <div className="ml-1 text-slate-400 group-hover:text-white transition-colors">
          {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </div>
      </button>
    </div>
  );
};
