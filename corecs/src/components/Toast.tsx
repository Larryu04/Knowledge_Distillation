import { TriangleAlert, X } from "lucide-react";
import { useEffect } from "react";

import { useStore } from "../store";

const DISMISS_MS = 6000;

export function Toast() {
  const toast = useStore((s) => s.toast);
  const showToast = useStore((s) => s.showToast);

  useEffect(() => {
    if (!toast) return;
    const t = setTimeout(() => showToast(null), DISMISS_MS);
    return () => clearTimeout(t);
  }, [toast, showToast]);

  if (!toast) return null;
  return (
    <div className="fixed bottom-24 left-1/2 z-30 flex max-w-lg -translate-x-1/2 items-start gap-2 rounded-xl border border-red-500/30 bg-panel px-4 py-3 text-sm text-red-200 shadow-2xl shadow-black/50">
      <TriangleAlert className="mt-0.5 size-4 shrink-0 text-red-400" />
      <span className="flex-1">{toast}</span>
      <button onClick={() => showToast(null)} className="text-muted hover:text-zinc-200">
        <X className="size-4" />
      </button>
    </div>
  );
}
