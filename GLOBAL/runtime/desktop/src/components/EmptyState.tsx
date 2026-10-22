import { Cpu } from "lucide-react";

import { textDir } from "../lib/bidi";
import { useStore } from "../store";

const EXAMPLES = [
  "Explain the key idea of this week's topic step by step",
  "What is a common mistake students make on this subject, and why is it wrong?",
  "Compare the two main approaches we learned and when to use each",
  "Walk me through a worked example of a typical exam question",
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
        Your private course tutor. Ask in your course language — everything runs on this computer.
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
