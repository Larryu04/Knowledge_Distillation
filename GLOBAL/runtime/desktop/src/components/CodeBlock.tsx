import { Check, Copy } from "lucide-react";
import { useEffect, useState } from "react";

import { highlight } from "../lib/highlight";

interface Props {
  code: string;
  lang: string;
  /** While the answer is streaming the code changes every token: show it plain, highlight once done. */
  streaming: boolean;
}

export function CodeBlock({ code, lang, streaming }: Props) {
  const [html, setHtml] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (streaming) return;
    let alive = true;
    highlight(code, lang)
      .then((h) => alive && setHtml(h))
      .catch(() => alive && setHtml(null));
    return () => {
      alive = false;
    };
  }, [code, lang, streaming]);

  const copy = async () => {
    await navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  };

  return (
    <div dir="ltr" className="not-prose my-4 overflow-hidden rounded-xl border border-border bg-code">
      <div className="flex items-center justify-between border-b border-border px-3 py-1.5 text-xs text-muted">
        <span className="font-mono">{lang || "text"}</span>
        <button
          onClick={copy}
          className="flex items-center gap-1 rounded px-1.5 py-0.5 hover:bg-white/5 hover:text-zinc-200"
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      {html && !streaming ? (
        <div
          className="shiki-wrap overflow-x-auto font-mono text-[13px] leading-relaxed"
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="overflow-x-auto p-4 font-mono text-[13px] leading-relaxed text-zinc-200">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}
