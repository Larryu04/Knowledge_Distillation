// Syntax highlighting with Shiki, fully bundled (offline). Uses the JavaScript regex engine
// instead of Oniguruma WASM, so the strict CSP needs no 'wasm-unsafe-eval'.
import { createHighlighterCore, type HighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

const THEME = "github-dark-default";

let highlighter: Promise<HighlighterCore> | null = null;

function getHighlighter(): Promise<HighlighterCore> {
  highlighter ??= createHighlighterCore({
    themes: [import("shiki/themes/github-dark-default.mjs")],
    // Languages a CS curriculum actually uses; each is a lazily loaded chunk.
    langs: [
      import("shiki/langs/c.mjs"),
      import("shiki/langs/cpp.mjs"),
      import("shiki/langs/csharp.mjs"),
      import("shiki/langs/java.mjs"),
      import("shiki/langs/python.mjs"),
      import("shiki/langs/javascript.mjs"),
      import("shiki/langs/typescript.mjs"),
      import("shiki/langs/sql.mjs"),
      import("shiki/langs/shellscript.mjs"),
      import("shiki/langs/json.mjs"),
      import("shiki/langs/yaml.mjs"),
      import("shiki/langs/go.mjs"),
      import("shiki/langs/rust.mjs"),
      import("shiki/langs/asm.mjs"),
      import("shiki/langs/haskell.mjs"),
      import("shiki/langs/kotlin.mjs"),
      import("shiki/langs/html.mjs"),
      import("shiki/langs/css.mjs"),
      import("shiki/langs/xml.mjs"),
      import("shiki/langs/verilog.mjs"),
      import("shiki/langs/prolog.mjs"),
    ],
    engine: createJavaScriptRegexEngine({ forgiving: true }),
  });
  return highlighter;
}

const ALIASES: Record<string, string> = {
  "c++": "cpp",
  cc: "cpp",
  h: "c",
  hpp: "cpp",
  "c#": "csharp",
  cs: "csharp",
  py: "python",
  python3: "python",
  js: "javascript",
  ts: "typescript",
  sh: "shellscript",
  bash: "shellscript",
  shell: "shellscript",
  zsh: "shellscript",
  console: "shellscript",
  yml: "yaml",
  rs: "rust",
  golang: "go",
  assembly: "asm",
  nasm: "asm",
  x86asm: "asm",
  mysql: "sql",
  postgresql: "sql",
  sqlite: "sql",
  hs: "haskell",
  kt: "kotlin",
};

/** Returns highlighted HTML, or null for unknown languages (rendered as plain text). */
export async function highlight(code: string, lang: string): Promise<string | null> {
  const h = await getHighlighter();
  const key = lang.toLowerCase();
  const id = ALIASES[key] ?? key;
  if (!id || !h.getLoadedLanguages().includes(id)) return null;
  return h.codeToHtml(code, { lang: id, theme: THEME });
}
