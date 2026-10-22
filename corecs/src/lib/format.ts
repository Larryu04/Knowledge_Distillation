import type { Conversation } from "./api";

export function formatBytes(n: number): string {
  if (n >= 1024 ** 3) return `${(n / 1024 ** 3).toFixed(1)} GB`;
  if (n >= 1024 ** 2) return `${(n / 1024 ** 2).toFixed(0)} MB`;
  return `${(n / 1024).toFixed(0)} KB`;
}

export interface ConversationGroup {
  label: string;
  items: Conversation[];
}

/** Groups conversations (already sorted newest first) into Today / Yesterday / ... buckets. */
export function groupByDate(conversations: Conversation[], now = new Date()): ConversationGroup[] {
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const day = 24 * 60 * 60 * 1000;
  const buckets: [string, number][] = [
    ["Today", startOfToday],
    ["Yesterday", startOfToday - day],
    ["Previous 7 days", startOfToday - 7 * day],
    ["Previous 30 days", startOfToday - 30 * day],
    ["Older", -Infinity],
  ];
  const groups: ConversationGroup[] = [];
  for (const c of conversations) {
    const label = buckets.find(([, since]) => c.updatedAt >= since)![0];
    const last = groups[groups.length - 1];
    if (last?.label === label) last.items.push(c);
    else groups.push({ label, items: [c] });
  }
  return groups;
}

/**
 * remark-math understands $...$ and $$...$$, but models often write \( ... \) and \[ ... \].
 * Converts those delimiters, leaving fenced and inline code untouched.
 */
export function normalizeMath(markdown: string): string {
  return markdown
    .split(/(```[\s\S]*?(?:```|$)|`[^`\n]*`)/)
    .map((part, i) =>
      i % 2 === 1
        ? part
        : part
            .replace(/\\\[([\s\S]+?)\\\]/g, (_, m: string) => `\n$$\n${m.trim()}\n$$\n`)
            .replace(/\\\(([\s\S]+?)\\\)/g, (_, m: string) => `$${m.trim()}$`),
    )
    .join("");
}

export function errorText(e: unknown): string {
  return typeof e === "string" ? e : e instanceof Error ? e.message : JSON.stringify(e);
}
