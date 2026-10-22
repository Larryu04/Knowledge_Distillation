import { Check, Copy, Cpu, TriangleAlert } from "lucide-react";
import { memo, useState } from "react";

import type { Message } from "../lib/api";
import { textDir } from "../lib/bidi";
import { Markdown } from "./Markdown";

interface Props {
  message: Message;
  /** Live text while streaming, otherwise the saved content. */
  content: string;
  streaming: boolean;
  error?: string;
  tokensPerSecond?: number;
}

function TypingDots() {
  return (
    <div className="flex h-7 items-center gap-1" aria-label="Thinking">
      {[0, 1, 2].map((i) => (
        <span key={i} className="typing-dot size-1.5 rounded-full bg-muted" />
      ))}
    </div>
  );
}

export const MessageItem = memo(function MessageItem({ message, content, streaming, error, tokensPerSecond }: Props) {
  const [copied, setCopied] = useState(false);

  if (message.role === "user") {
    return (
      <div id={`msg-${message.id}`} className="flex justify-end">
        <div
          dir={textDir(content)}
          className="max-w-[85%] whitespace-pre-wrap rounded-2xl rounded-ee-md bg-elevated px-4 py-2.5 text-[15px] leading-relaxed"
        >
          {content}
        </div>
      </div>
    );
  }

  const copy = async () => {
    await navigator.clipboard.writeText(content);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div id={`msg-${message.id}`} className="group flex gap-3">
      <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-accent/15 text-accent">
        <Cpu className="size-4" />
      </div>
      <div className="min-w-0 flex-1">
        {content ? <Markdown text={content} streaming={streaming} /> : streaming ? <TypingDots /> : null}

        {message.status === "error" && (
          <div className="mt-2 flex items-start gap-2 rounded-lg border border-red-500/30 bg-red-500/10 px-3 py-2 text-sm text-red-300">
            <TriangleAlert className="mt-0.5 size-4 shrink-0" />
            <span className="whitespace-pre-wrap">{error ?? "This answer could not be completed."}</span>
          </div>
        )}

        {!streaming && (
          <div className="mt-2 flex h-6 items-center gap-3 text-xs text-muted">
            {message.status === "partial" && (
              <span className="rounded bg-amber-500/10 px-1.5 py-0.5 text-amber-300">Stopped</span>
            )}
            {content && (
              <button
                onClick={copy}
                className="flex items-center gap-1 opacity-0 transition-opacity group-hover:opacity-100 hover:text-zinc-200"
              >
                {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
                {copied ? "Copied" : "Copy"}
              </button>
            )}
            <span className="opacity-0 transition-opacity group-hover:opacity-100">
              {message.model}
              {tokensPerSecond ? ` · ${tokensPerSecond.toFixed(0)} tok/s` : ""}
            </span>
          </div>
        )}
      </div>
    </div>
  );
});
