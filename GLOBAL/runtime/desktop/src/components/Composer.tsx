import { ArrowUp, Square } from "lucide-react";
import { useEffect, useLayoutEffect, useRef, useState, type KeyboardEvent } from "react";

import { textDir } from "../lib/bidi";
import { useStore } from "../store";

const MAX_HEIGHT = 208;

export function Composer() {
  const [text, setText] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);
  const activeId = useStore((s) => s.activeId);
  const streaming = useStore((s) => s.stream !== null);
  const ready = useStore((s) => s.engine.state === "ready");
  const send = useStore((s) => s.send);
  const stop = useStore((s) => s.stop);

  // Grow with the content up to MAX_HEIGHT, then scroll.
  useLayoutEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, MAX_HEIGHT)}px`;
  }, [text]);

  useEffect(() => {
    ref.current?.focus();
  }, [activeId]);

  const canSend = ready && !streaming && text.trim().length > 0;

  const submit = async () => {
    if (!canSend) return;
    const value = text;
    setText("");
    if (!(await send(value))) setText(value); // rejected: give the text back
  };

  const onKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && !e.nativeEvent.isComposing) {
      e.preventDefault();
      void submit();
    }
  };

  return (
    <div className="mx-auto w-full max-w-3xl px-6 pb-4">
      <div className="flex items-end gap-2 rounded-2xl border border-border bg-panel p-2 shadow-lg shadow-black/20 transition-colors focus-within:border-accent/50">
        <textarea
          ref={ref}
          dir={textDir(text)}
          rows={1}
          value={text}
          onChange={(e) => setText(e.target.value)}
          onKeyDown={onKeyDown}
          placeholder={ready ? "Ask a CS question…  ·  שאל שאלה…" : "Waiting for the model…"}
          className="flex-1 resize-none bg-transparent px-2 py-1.5 text-[15px] leading-relaxed outline-none placeholder:text-muted"
        />
        {streaming ? (
          <button
            onClick={stop}
            title="Stop generating"
            className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-zinc-100 text-zinc-900 hover:bg-white"
          >
            <Square className="size-3.5 fill-current" />
          </button>
        ) : (
          <button
            onClick={submit}
            disabled={!canSend}
            title="Send (Enter)"
            className="flex size-9 shrink-0 items-center justify-center rounded-xl bg-accent-strong text-white transition-colors hover:bg-accent disabled:bg-elevated disabled:text-muted"
          >
            <ArrowUp className="size-4" />
          </button>
        )}
      </div>
      <p className="mt-2 text-center text-[11px] text-muted">
        Runs 100% on this computer. CoreCS can make mistakes — double-check important answers.
      </p>
    </div>
  );
}
