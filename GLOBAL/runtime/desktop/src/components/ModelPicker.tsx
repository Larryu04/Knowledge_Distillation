import { openPath } from "@tauri-apps/plugin-opener";
import { Check, ChevronDown, FolderOpen, RefreshCw } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import clsx from "clsx";

import { formatBytes } from "../lib/format";
import { useStore } from "../store";

const DOT: Record<string, string> = {
  ready: "bg-emerald-400",
  starting: "bg-amber-400 animate-pulse",
  downloading: "bg-sky-400 animate-pulse",
  failed: "bg-red-400",
  stopped: "bg-zinc-500",
  noModel: "bg-zinc-500",
};

export function ModelPicker() {
  const engine = useStore((s) => s.engine);
  const models = useStore((s) => s.models);
  const modelsDir = useStore((s) => s.modelsDir);
  const streaming = useStore((s) => s.stream !== null);
  const refreshModels = useStore((s) => s.refreshModels);
  const loadModel = useStore((s) => s.loadModel);
  const [open, setOpen] = useState(false);
  const root = useRef<HTMLDivElement>(null);

  // Close on outside click / Escape.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => !root.current?.contains(e.target as Node) && setOpen(false);
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    window.addEventListener("mousedown", onDown);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const activeId = engine.state === "ready" || engine.state === "starting" ? engine.modelId : null;
  const label = engine.state === "ready" || engine.state === "starting" ? engine.model : "No model loaded";

  const toggle = () => {
    if (!open) void refreshModels();
    setOpen(!open);
  };

  return (
    <div ref={root} className="relative">
      <button
        onClick={toggle}
        disabled={streaming}
        title={streaming ? "Wait for the answer to finish" : "Choose model"}
        className="flex items-center gap-2 rounded-lg border border-border px-3 py-1.5 text-sm hover:bg-white/5 disabled:opacity-50"
      >
        <span className={clsx("size-2 rounded-full", DOT[engine.state])} />
        <span className="max-w-56 truncate">{label}</span>
        <ChevronDown className="size-4 text-muted" />
      </button>

      {open && (
        <div className="absolute end-0 z-20 mt-2 w-80 overflow-hidden rounded-xl border border-border bg-panel shadow-2xl shadow-black/40">
          <div className="px-3 pt-3 pb-1 text-[11px] font-medium uppercase tracking-wider text-muted">Models</div>
          <div className="max-h-72 overflow-y-auto p-1.5">
            {models.length === 0 && (
              <p className="px-3 py-4 text-center text-sm text-muted">No .gguf files in the models folder.</p>
            )}
            {models.map((m) => (
              <button
                key={m.id}
                onClick={() => {
                  setOpen(false);
                  if (m.id !== activeId) void loadModel(m.id);
                }}
                className="flex w-full items-start gap-2 rounded-lg px-2.5 py-2 text-start hover:bg-white/5"
              >
                <Check className={clsx("mt-0.5 size-4 shrink-0 text-accent", m.id !== activeId && "invisible")} />
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm">{m.displayName}</span>
                  <span className="block truncate text-xs text-muted">
                    {m.id} · {formatBytes(m.sizeBytes)}
                  </span>
                </span>
                {m.hasProfile && (
                  <span
                    title="Ships a CoreCS profile (training prompt and sampling settings)"
                    className="mt-0.5 shrink-0 rounded bg-accent/15 px-1.5 py-0.5 text-[10px] font-medium text-accent"
                  >
                    TUNED
                  </span>
                )}
              </button>
            ))}
          </div>
          <div className="flex items-center gap-1 border-t border-border p-1.5 text-xs">
            <button
              onClick={() => openPath(modelsDir)}
              className="flex flex-1 items-center gap-1.5 rounded-md px-2.5 py-1.5 text-muted hover:bg-white/5 hover:text-zinc-200"
            >
              <FolderOpen className="size-3.5" /> Open models folder
            </button>
            <button
              onClick={() => refreshModels()}
              className="flex items-center gap-1.5 rounded-md px-2.5 py-1.5 text-muted hover:bg-white/5 hover:text-zinc-200"
            >
              <RefreshCw className="size-3.5" /> Refresh
            </button>
          </div>
          {engine.state === "ready" && engine.device && (
            <div className="border-t border-border px-3 py-2 text-[11px] text-muted">Running on {engine.device}</div>
          )}
        </div>
      )}
    </div>
  );
}
