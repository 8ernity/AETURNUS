import React, { useState, useEffect, useRef } from "react";
import { createAvatar, type AvatarController } from "@bible-strong/avatar-react";
import "@bible-strong/avatar-react/styles.css";
import { 
  Bot, 
  Sparkles, 
  ChevronDown, 
  ChevronUp, 
  Radio, 
  AlertTriangle, 
  Search, 
  Cpu, 
  TrendingUp, 
  Play, 
  Pause,
  PartyPopper,
  Flame,
  Activity
} from "lucide-react";
import grokAvatarDefinition from "../assets/grok-bot.avatar.json";

// Create concrete avatar component from definition
const GrokAvatar = createAvatar(grokAvatarDefinition);

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
  | "curious" 
  | "confused" 
  | "surprised" 
  | "proud" 
  | "shy" 
  | "sad" 
  | "laughing" 
  | "scared" 
  | "playful" 
  | "celebrate";

interface GrokBotAssistantProps {
  currentSubset: string;
  currentCycle: number;
  isPlaying: boolean;
  exceptionCount: number;
  wsConnected: boolean;
  fleetData?: any;
  recentAlerts?: any[];
}

export const GrokBotAssistant: React.FC<GrokBotAssistantProps> = ({
  currentSubset,
  currentCycle,
  isPlaying,
  exceptionCount,
  wsConnected,
  fleetData,
  recentAlerts = []
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(false);
  const [currentAnimation, setCurrentAnimation] = useState<AnimationName>("idle");
  const [manualOverride, setManualOverride] = useState<boolean>(false);
  const [chatMessage, setChatMessage] = useState<string>(
    "AETURNUS Diagnostic Copilot online. Telemetry stream synchronized."
  );

  const avatarRef = useRef<AvatarController>(null);

  // Synchronize animation with Cockpit Telemetry events (unless user manually triggered)
  useEffect(() => {
    if (manualOverride) return;

    if (exceptionCount > 0 || (recentAlerts.length > 0 && recentAlerts[0]?.severity === "CRITICAL")) {
      setCurrentAnimation("angry");
      setChatMessage(
        `Critical Alert! Engine exception detected on ${currentSubset}. Triage agent assigned immediate review.`
      );
    } else if (isPlaying) {
      setCurrentAnimation("working");
      setChatMessage(
        `Streaming cycle ${currentCycle} across ${fleetData?.total_engines || 100} engines in ${currentSubset}. Causal pipeline active.`
      );
    } else if (wsConnected) {
      setCurrentAnimation("listening");
      setChatMessage(
        `AETURNUS standby on cycle ${currentCycle}. Replay paused. Fleet health nominal.`
      );
    } else {
      setCurrentAnimation("idle");
    }
  }, [isPlaying, currentCycle, exceptionCount, wsConnected, currentSubset, manualOverride, recentAlerts]);

  // Handle manual emotion change
  const triggerAnimation = (anim: AnimationName, message?: string) => {
    setManualOverride(true);
    setCurrentAnimation(anim);
    if (message) {
      setChatMessage(message);
    }
    // Release override after 10 seconds back to reactive telemetry
    setTimeout(() => {
      setManualOverride(false);
    }, 10000);
  };

  // Quick Action Prompts
  const handleScanCriticals = () => {
    const criticals = fleetData?.severity_summary?.CRITICAL || 0;
    triggerAnimation(
      criticals > 0 ? "suspicious" : "happy",
      criticals > 0 
        ? `Scan complete: Found ${criticals} critical engines with RUL ≤ 15 cycles! Inspect work orders tab.`
        : `Scan complete: Zero critical failures in current ${currentSubset} window. Quantiles bounded.`
    );
  };

  const handleDriftCheck = () => {
    triggerAnimation(
      "thinking",
      `Analyzing Population Stability Index (PSI) & Kolmogorov-Smirnov drift statistics on ${currentSubset} sensors...`
    );
  };

  const handleCelebrate = () => {
    triggerAnimation(
      "celebrate",
      `100% Deterministic zero-LLM triage operating at < 5ms latency! AETURNUS uptime preserved.`
    );
  };

  const animationsList: AnimationName[] = [
    "idle", "thinking", "searching", "working", "excited", 
    "happy", "angry", "suspicious", "celebrate", "sleeping"
  ];

  return (
    <div className="fixed bottom-6 right-6 z-50 flex flex-col items-end">
      {/* Expanded Diagnostic Assistant Card */}
      {isOpen && (
        <div className="mb-3 w-80 sm:w-96 rounded-2xl bg-slate-900/95 border border-indigo-500/30 shadow-2xl shadow-indigo-950/60 backdrop-blur-xl p-4 transition-all duration-300 animate-in fade-in slide-in-from-bottom-5">
          {/* Card Header */}
          <div className="flex items-center justify-between pb-3 border-b border-slate-800">
            <div className="flex items-center gap-2.5">
              <div className="w-8 h-8 rounded-lg bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center">
                <Bot className="w-4 h-4 text-indigo-400" />
              </div>
              <div>
                <h4 className="text-sm font-bold text-white tracking-wide flex items-center gap-1.5">
                  Grok Bot
                  <span className="text-[10px] uppercase font-semibold px-1.5 py-0.5 rounded bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">
                    Copilot
                  </span>
                </h4>
                <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
                  <span className={`w-1.5 h-1.5 rounded-full ${wsConnected ? "bg-emerald-400 animate-pulse" : "bg-amber-400"}`} />
                  <span>State: <strong className="text-indigo-300 capitalize">{currentAnimation}</strong></span>
                </div>
              </div>
            </div>

            <button
              onClick={() => setIsOpen(false)}
              className="p-1 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
              title="Minimize Copilot"
            >
              <ChevronDown className="w-5 h-5" />
            </button>
          </div>

          {/* Avatar Display Port */}
          <div className="relative my-3 flex flex-col items-center justify-center py-2 bg-gradient-to-b from-slate-950/80 to-slate-900/60 rounded-xl border border-indigo-500/15 overflow-hidden">
            {/* Ambient Background Glow */}
            <div className="absolute inset-0 bg-radial from-indigo-500/10 via-transparent to-transparent pointer-events-none" />

            <div 
              className="relative cursor-pointer hover:scale-105 transition-transform"
              onClick={() => triggerAnimation("excited", "Hey! Diagnostic copilot ready for fleet command.")}
              title="Click to interact with Grok Bot"
            >
              {/* Render Grok Bot Avatar */}
              <GrokAvatar
                ref={avatarRef}
                animation={currentAnimation}
                size={120}
                className="drop-shadow-xl"
              />
            </div>

            <span className="text-[10px] text-slate-500 font-mono tracking-wider mt-1">
              CLICK AVATAR TO INTERACT
            </span>
          </div>

          {/* Live Diagnostic Speech Bubble */}
          <div className="rounded-xl bg-slate-950/80 border border-indigo-500/20 p-3 mb-3 text-xs leading-relaxed text-slate-200">
            <div className="flex items-start gap-2">
              <Sparkles className="w-4 h-4 text-cyan-400 flex-shrink-0 mt-0.5 animate-pulse" />
              <p className="flex-1 font-medium text-slate-300">
                {chatMessage}
              </p>
            </div>
          </div>

          {/* Quick Action Buttons */}
          <div className="grid grid-cols-3 gap-1.5 mb-3">
            <button
              onClick={handleScanCriticals}
              className="px-2 py-1.5 rounded-lg bg-slate-800/80 hover:bg-indigo-600/40 border border-slate-700/60 hover:border-indigo-400/50 text-[11px] font-medium text-slate-200 transition-colors flex items-center justify-center gap-1"
            >
              <Search className="w-3 h-3 text-amber-400" />
              Scan Fleet
            </button>
            <button
              onClick={handleDriftCheck}
              className="px-2 py-1.5 rounded-lg bg-slate-800/80 hover:bg-indigo-600/40 border border-slate-700/60 hover:border-indigo-400/50 text-[11px] font-medium text-slate-200 transition-colors flex items-center justify-center gap-1"
            >
              <Activity className="w-3 h-3 text-cyan-400" />
              Drift Check
            </button>
            <button
              onClick={handleCelebrate}
              className="px-2 py-1.5 rounded-lg bg-slate-800/80 hover:bg-indigo-600/40 border border-slate-700/60 hover:border-indigo-400/50 text-[11px] font-medium text-slate-200 transition-colors flex items-center justify-center gap-1"
            >
              <PartyPopper className="w-3 h-3 text-purple-400" />
              Uptime
            </button>
          </div>

          {/* Quick Animation Preset Bar */}
          <div>
            <div className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center justify-between">
              <span>Emotion Presets</span>
              {manualOverride && (
                <span className="text-[9px] text-amber-400 font-normal">Manual Mode</span>
              )}
            </div>
            <div className="flex items-center gap-1 overflow-x-auto pb-1 scrollbar-none">
              {animationsList.map((anim) => (
                <button
                  key={anim}
                  onClick={() => triggerAnimation(anim)}
                  className={`px-2 py-1 rounded text-[10px] font-medium capitalize whitespace-nowrap transition-all ${
                    currentAnimation === anim
                      ? "bg-indigo-600 text-white shadow-md shadow-indigo-600/30"
                      : "bg-slate-800/60 text-slate-400 hover:text-slate-200 hover:bg-slate-750"
                  }`}
                >
                  {anim}
                </button>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Floating Minimized Pill / Orb */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="group relative flex items-center gap-2.5 px-3.5 py-2.5 rounded-full bg-slate-900/90 hover:bg-slate-850 border border-indigo-500/30 hover:border-indigo-400/60 shadow-xl shadow-indigo-950/50 backdrop-blur-md transition-all duration-300 hover:scale-105"
        title={isOpen ? "Minimize Copilot" : "Open Grok Bot Assistant"}
      >
        {/* Animated Grok Bot Avatar in Pill */}
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

        {/* Alert Badge if Exceptions */}
        {exceptionCount > 0 && (
          <span className="px-1.5 py-0.5 rounded-full text-[10px] font-bold bg-rose-500 text-white animate-pulse">
            {exceptionCount}
          </span>
        )}

        <div className="ml-1 text-slate-400 group-hover:text-white transition-colors">
          {isOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronUp className="w-4 h-4" />}
        </div>
      </button>
    </div>
  );
};
