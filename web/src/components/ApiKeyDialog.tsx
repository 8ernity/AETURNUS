import React, { useEffect, useState } from "react";
import { Eye, EyeOff, KeyRound, ShieldCheck, Trash2, X } from "lucide-react";
import { clearApiKey, getApiKey, saveApiKey } from "../lib/api";

interface ApiKeyDialogProps {
  open: boolean;
  onClose: () => void;
}

export const ApiKeyDialog: React.FC<ApiKeyDialogProps> = ({ open, onClose }) => {
  const [value, setValue] = useState<string>("");
  const [reveal, setReveal] = useState<boolean>(false);

  useEffect(() => {
    if (open) {
      setValue(getApiKey());
      setReveal(false);
    }
  }, [open]);

  if (!open) return null;

  const handleSave = () => {
    saveApiKey(value);
    window.location.reload();
  };

  const handleClear = () => {
    clearApiKey();
    setValue("");
    window.location.reload();
  };

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4 backdrop-blur-sm"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-lg rounded-3xl border border-indigo-500/30 bg-[#0b0d19] p-6 shadow-2xl shadow-indigo-950/60">
        <div className="flex items-start justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="flex h-10 w-10 items-center justify-center rounded-2xl border border-indigo-500/30 bg-indigo-500/15 text-indigo-300">
              <KeyRound className="h-5 w-5" />
            </div>
            <div>
              <h2 className="text-base font-bold text-white">Connect the cockpit</h2>
              <p className="text-xs text-slate-400">
                The server requires an API key for replay controls, retraining, and the live WebSocket.
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="rounded-lg p-1.5 text-slate-500 transition-colors hover:bg-slate-800 hover:text-white"
            aria-label="Close API key dialog"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        <div className="mt-5 space-y-3">
          <label className="block text-[11px] font-semibold uppercase tracking-wider text-slate-400">
            API key
          </label>
          <div className="flex items-center gap-2 rounded-2xl border border-slate-700 bg-slate-950 px-3 py-2 focus-within:border-indigo-500">
            <input
              autoFocus
              type={reveal ? "text" : "password"}
              value={value}
              onChange={(event) => setValue(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter") handleSave();
                if (event.key === "Escape") onClose();
              }}
              placeholder="Paste AETURNUS_API_KEY_OPERATOR"
              className="min-w-0 flex-1 bg-transparent font-mono text-sm text-white outline-none placeholder:text-slate-600"
              autoComplete="off"
              spellCheck={false}
            />
            <button
              onClick={() => setReveal((current) => !current)}
              className="rounded-lg p-1.5 text-slate-500 hover:bg-slate-800 hover:text-slate-200"
              title={reveal ? "Hide key" : "Show key"}
              type="button"
            >
              {reveal ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
            </button>
          </div>

          <div className="rounded-2xl border border-cyan-500/20 bg-cyan-500/5 p-3 text-[11px] leading-relaxed text-cyan-100/80">
            <div className="mb-1 flex items-center gap-1.5 font-semibold text-cyan-200">
              <ShieldCheck className="h-3.5 w-3.5" />
              Where this key comes from
            </div>
            The app does not generate keys. The deployer sets a secret such as{" "}
            <code className="rounded bg-black/30 px-1 py-0.5 font-mono">AETURNUS_API_KEY_OPERATOR</code>{" "}
            in the server environment (for Render: Environment → Environment Variables), then pastes
            the same value here. Use the viewer key for read-only telemetry, or the operator key for
            replay and retraining controls.
          </div>

          <p className="text-[11px] text-slate-500">
            The key is stored only in this browser's <code className="font-mono">localStorage</code>.
            It is not written to the server or included in the build.
          </p>
        </div>

        <div className="mt-6 flex flex-wrap items-center justify-between gap-3">
          <button
            onClick={handleClear}
            className="flex items-center gap-1.5 rounded-xl border border-rose-500/20 px-3 py-2 text-xs font-semibold text-rose-300 transition-colors hover:bg-rose-500/10"
          >
            <Trash2 className="h-3.5 w-3.5" />
            Clear stored key
          </button>
          <div className="flex items-center gap-2">
            <button
              onClick={onClose}
              className="rounded-xl border border-slate-700 px-4 py-2 text-xs font-semibold text-slate-300 transition-colors hover:bg-slate-800"
            >
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!value.trim()}
              className="rounded-xl bg-gradient-to-r from-indigo-600 to-purple-600 px-4 py-2 text-xs font-bold text-white shadow-lg shadow-indigo-950/40 transition-opacity disabled:cursor-not-allowed disabled:opacity-40"
            >
              Save &amp; reconnect
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
