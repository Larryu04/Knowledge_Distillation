import { Cpu } from "lucide-react";

import { textDir } from "../lib/bidi";
import { useStore } from "../store";

const EXAMPLES = [
  "מהו three-way handshake ב-TCP ולמה הוא נחוץ?",
  "הסבר את ההבדל בין תהליך (process) ל-thread",
  "What is the time complexity of Dijkstra's algorithm with a binary heap?",
  "Explain the pumping lemma for regular languages with an example",
];

export function EmptyState() {
  const ready = useStore((s) => s.engine.state === "ready");
  const send = useStore((s) => s.send);

  return (
    <div className="flex flex-1 flex-col items-center justify-center overflow-y-auto px-6 py-8">
      <div className="mb-4 flex size-14 items-center justify-center rounded-2xl bg-accent/15 text-accent">
        <Cpu className="size-7" />
      </div>
      <h2 className="text-2xl font-bold tracking-tight">What are we studying today?</h2>
      <p className="mt-2 max-w-md text-center text-sm text-muted">
        Your private Computer Science tutor. Ask in Hebrew or English — everything runs on this computer.
      </p>
      <div className="mt-8 grid w-full max-w-2xl grid-cols-1 gap-3 sm:grid-cols-2">
        {EXAMPLES.map((q) => (
          <button
            key={q}
            dir={textDir(q)}
            disabled={!ready}
            onClick={() => send(q)}
            className="rounded-xl border border-border bg-panel px-4 py-3 text-start text-sm text-zinc-300 transition-colors hover:border-accent/40 hover:text-zinc-100 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {q}
          </button>
        ))}
      </div>
    </div>
  );
}
