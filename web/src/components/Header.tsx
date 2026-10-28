import React from "react";
import { Play, Pause, RotateCcw, FastForward, Radio, Sparkles } from "lucide-react";

interface HeaderProps {
  currentSubset: string;
  onSelectSubset: (sub: string) => void;
  isPlaying: boolean;
  onTogglePlay: () => void;
  speed: number;
  onChangeSpeed: (s: number) => void;
  onReset: () => void;
  onLifecycleDemo: () => void;
  currentCycle: number;
  wsConnected: boolean;
}

export const Header: React.FC<HeaderProps> = ({
  currentSubset,
  onSelectSubset,
  isPlaying,
  onTogglePlay,
  speed,
  onChangeSpeed,
  onReset,
  onLifecycleDemo,
  currentCycle,
  wsConnected
}) => {
  const subsets = [
    { id: "FD001", label: "FD001 · 1 Regime / HPC" },
    { id: "FD002", label: "FD002 · 6 Regimes / HPC" },
    { id: "FD003", label: "FD003 · 1 Regime / 2 Faults" },
    { id: "FD004", label: "FD004 · 6 Regimes / 2 Faults" },
  ];

  return (
    <header className="h-20 border-b border-indigo-500/10 px-8 flex items-center justify-between sticky top-0 z-30 bg-[#090a10]/80 backdrop-blur-md">
      {/* Subset Switcher */}
      <div className="flex items-center gap-3">
        <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">Benchmark:</span>
        <div className="flex items-center bg-slate-900/80 p-1 rounded-xl border border-indigo-500/20">
          {subsets.map((s) => (
            <button
              key={s.id}
              onClick={() => onSelectSubset(s.id)}
              className={`px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${
                currentSubset === s.id
                  ? "bg-gradient-to-r from-indigo-600 to-purple-600 text-white shadow-md shadow-indigo-500/30 font-semibold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {s.id}
            </button>
          ))}
        </div>
      </div>

      {/* Center Replay Controls */}
      <div className="flex items-center gap-3 bg-slate-900/90 px-4 py-2 rounded-2xl border border-indigo-500/25 shadow-lg shadow-indigo-950/40">
        <div className="flex items-center gap-2 pr-3 border-r border-slate-700/60">
          <span className="text-xs text-slate-400 font-medium">Replay Cycle:</span>
          <span className="font-mono font-bold text-sm text-cyan-300 bg-slate-800/80 px-2.5 py-0.5 rounded-md border border-cyan-500/30">
            {currentCycle}
          </span>
        </div>

        {/* Play/Pause */}
        <button
          onClick={onTogglePlay}
          className={`p-2 rounded-xl transition-all ${
            isPlaying
              ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-md shadow-amber-500/20"
              : "glass-button text-white"
          }`}
          title={isPlaying ? "Pause replay" : "Play telemetry stream"}
        >
          {isPlaying ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4 fill-white" />}
        </button>

        {/* Speed Buttons */}
        <div className="flex items-center bg-slate-800/60 p-0.5 rounded-lg border border-slate-700">
          {[1, 5, 20].map((s) => (
            <button
              key={s}
              onClick={() => onChangeSpeed(s)}
              className={`px-2 py-1 text-xs rounded-md font-mono transition-all ${
                speed === s
                  ? "bg-indigo-600 text-white font-bold"
                  : "text-slate-400 hover:text-slate-200"
              }`}
            >
              {s}x
            </button>
          ))}
        </div>

        {/* Reset */}
        <button
          onClick={onReset}
          className="p-1.5 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition-colors"
          title="Reset to initial cycle"
        >
          <RotateCcw className="w-4 h-4" />
        </button>

        {/* Lifecycle Demo Button */}
        <button
          onClick={onLifecycleDemo}
          className="flex items-center gap-1.5 px-3 py-1.5 bg-gradient-to-r from-purple-600/30 to-indigo-600/30 hover:from-purple-600/50 hover:to-indigo-600/50 border border-purple-500/40 rounded-xl text-xs font-semibold text-purple-200 transition-all shadow-sm"
          title="Replay single engine degradation lifecycle to show auto work-order trigger"
        >
          <Sparkles className="w-3.5 h-3.5 text-purple-400" />
          <span>Lifecycle Story</span>
        </button>
      </div>

      {/* Right WebSocket & Telemetry Status */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2 px-3 py-1.5 rounded-xl bg-slate-900/60 border border-indigo-500/20">
          <span className={`w-2.5 h-2.5 rounded-full ${wsConnected ? "bg-emerald-400 animate-pulse" : "bg-rose-500"}`}></span>
          <span className="text-xs font-medium text-slate-300">
            {wsConnected ? "REPLAY CONNECTED" : "DISCONNECTED"}
          </span>
        </div>
      </div>
    </header>
  );
};
