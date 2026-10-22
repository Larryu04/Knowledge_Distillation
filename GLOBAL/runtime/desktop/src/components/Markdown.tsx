import { openUrl } from "@tauri-apps/plugin-opener";
import { Children, createElement, isValidElement, memo, useMemo, type ReactNode } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

import type { Nodes } from "hast";

import { hastText, textDir } from "../lib/bidi";
import { normalizeMath } from "../lib/format";
import { CodeBlock } from "./CodeBlock";

const remarkPlugins = [remarkGfm, remarkMath];
const rehypePlugins: [typeof rehypeKatex, { strict: false }][] = [[rehypeKatex, { strict: false }]];

// Each block picks its own direction, so Hebrew paragraphs render RTL and English ones LTR.
// List items and table cells inherit from their list/table so columns and bullets line up.
const AUTO_DIR_TAGS = ["p", "ul", "ol", "h1", "h2", "h3", "h4", "blockquote"] as const;

function makeComponents(streaming: boolean): Components {
  const components: Components = {
    pre({ children }) {
      const child = Children.toArray(children)[0];
      if (isValidElement<{ className?: string; children?: ReactNode }>(child)) {
        const lang = /language-([\w#+-]+)/.exec(child.props.className ?? "")?.[1] ?? "";
        const code = String(child.props.children ?? "").replace(/\n$/, "");
        return <CodeBlock code={code} lang={lang} streaming={streaming} />;
      }
      return <pre>{children}</pre>;
    },
    // Links open in the system browser, never inside the app window.
    a({ href, children }) {
      return (
        <a
          href={href}
          onClick={(e) => {
            e.preventDefault();
            if (href && /^(https?:|mailto:)/.test(href)) void openUrl(href);
          }}
        >
          {children}
        </a>
      );
    },
    table({ node, children }) {
      return (
        <div className="overflow-x-auto">
          <table dir={textDir(hastText(node))}>{children}</table>
        </div>
      );
    },
  };
  for (const tag of AUTO_DIR_TAGS) {
    (components as Record<string, unknown>)[tag] = autoDir(tag);
  }
  return components;
}

function autoDir(tag: string) {
  // `node` (the hast node react-markdown passes) must not reach the DOM.
  return function AutoDir({ node, ...props }: { node?: Nodes } & Record<string, unknown>) {
    return createElement(tag, { dir: textDir(hastText(node)), ...props });
  };
}

const staticComponents = makeComponents(false);
const streamingComponents = makeComponents(true);

interface Props {
  text: string;
  streaming?: boolean;
}

export const Markdown = memo(function Markdown({ text, streaming = false }: Props) {
  const source = useMemo(() => normalizeMath(text), [text]);
  return (
    <div
      className="markdown prose prose-invert max-w-none prose-headings:font-semibold prose-a:text-accent prose-strong:text-zinc-50 prose-li:my-0.5"
    >
      <ReactMarkdown
        remarkPlugins={remarkPlugins}
        rehypePlugins={rehypePlugins}
        components={streaming ? streamingComponents : staticComponents}
      >
        {source}
      </ReactMarkdown>
    </div>
  );
});
