import { openPath } from "@tauri-apps/plugin-opener";
import { FolderOpen, HardDrive, LoaderCircle, Power, RefreshCw, TriangleAlert } from "lucide-react";
import { useState } from "react";

import { useStore } from "../store";

const buttonClass =
  "flex items-center gap-1.5 rounded-md border border-current/25 px-2.5 py-1 text-xs font-medium hover:bg-white/5";

/** Shows the model engine's state whenever it is not ready. */
export function EngineBanner() {
  const engine = useStore((s) => s.engine);
  const modelsDir = useStore((s) => s.modelsDir);
  const retryEngine = useStore((s) => s.retryEngine);
  const [details, setDetails] = useState(false);

  if (engine.state === "ready") return null;

  if (engine.state === "starting") {
    return (
      <div className="flex items-center gap-2 border-b border-border bg-accent/10 px-5 py-2.5 text-sm text-accent">
        <LoaderCircle className="size-4 animate-spin" />
        Loading <span className="font-medium">{engine.model}</span>… this takes a few seconds.
      </div>
    );
  }

  if (engine.state === "failed") {
    const [headline, ...rest] = engine.message.split("\n");
    return (
      <div className="border-b border-border bg-red-500/10 px-5 py-2.5 text-sm text-red-300">
        <div className="flex items-center gap-2">
          <TriangleAlert className="size-4 shrink-0" />
          <span className="flex-1">
            The model engine stopped: <span className="text-red-200">{headline}</span>
          </span>
          {rest.length > 0 && (
            <button onClick={() => setDetails(!details)} className="text-xs underline-offset-2 hover:underline">
              {details ? "Hide details" : "Details"}
            </button>
          )}
          <button onClick={retryEngine} className={buttonClass}>
            <RefreshCw className="size-3.5" /> Retry
          </button>
        </div>
        {details && (
          <pre className="mt-2 max-h-48 overflow-auto rounded-md bg-black/30 p-2 font-mono text-[11px] leading-snug whitespace-pre-wrap text-red-200/80">
            {rest.join("\n")}
          </pre>
        )}
      </div>
    );
  }

  if (engine.state === "stopped") {
    return (
      <div className="flex items-center gap-2 border-b border-border bg-panel px-5 py-2.5 text-sm text-muted">
        <Power className="size-4" />
        <span className="flex-1">The model is stopped.</span>
        <button onClick={retryEngine} className={buttonClass}>
          <Power className="size-3.5" /> Start
        </button>
      </div>
    );
  }

  // noModel
  return (
    <div className="flex items-center gap-2 border-b border-border bg-amber-500/10 px-5 py-2.5 text-sm text-amber-200">
      <HardDrive className="size-4 shrink-0" />
      <span className="flex-1">
        No model installed. Copy a <code className="font-mono">.gguf</code> file into the models folder, then click
        Refresh.
      </span>
      <button onClick={() => openPath(modelsDir)} className={buttonClass}>
        <FolderOpen className="size-3.5" /> Open folder
      </button>
      <button onClick={retryEngine} className={buttonClass}>
        <RefreshCw className="size-3.5" /> Refresh
      </button>
    </div>
  );
}
