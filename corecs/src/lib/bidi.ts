// Text direction for mixed Hebrew/English content.
//
// The browser's dir="auto" uses the FIRST strong character, so a Hebrew sentence that opens
// with an English term ("**TCP** הוא פרוטוקול...") renders left-to-right. Hebrew CS text is
// full of English terms, so we count words instead.
import type { Nodes } from "hast";

const HEBREW = /[֐-׿]/;
const LATIN = /[A-Za-z]/;

export type Dir = "rtl" | "ltr";

export function textDir(text: string): Dir {
  let hebrew = 0;
  let latin = 0;
  for (const word of text.split(/\s+/)) {
    if (HEBREW.test(word)) hebrew++;
    else if (LATIN.test(word)) latin++;
  }
  // Any real share of Hebrew words means a Hebrew sentence with embedded terms.
  return hebrew > 0 && hebrew * 3 >= latin ? "rtl" : "ltr";
}

/** Visible prose text of a markdown (hast) node, skipping code and rendered math. */
export function hastText(node: Nodes | undefined): string {
  if (!node) return "";
  if (node.type === "text") return node.value;
  if (node.type === "element") {
    const cls = node.properties?.className;
    if (node.tagName === "code" || (Array.isArray(cls) && cls.some((c) => String(c).startsWith("katex")))) {
      return "";
    }
  }
  return "children" in node ? node.children.map((c) => hastText(c as Nodes)).join(" ") : "";
}
