import { ask } from "@tauri-apps/plugin-dialog";
import { Cpu, MessageSquarePlus, Pencil, Search, ShieldCheck, Trash2, X } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import clsx from "clsx";

import { api, type Conversation, type SearchHit } from "../lib/api";
import { textDir } from "../lib/bidi";
import { groupByDate } from "../lib/format";
import { useStore } from "../store";

const SEARCH_DEBOUNCE_MS = 200;

/** Renders an FTS snippet, highlighting the \u0002...\u0003 marked matches. */
function Snippet({ text }: { text: string }) {
  return (
    <>
      {text.split(/(\u0002[^\u0003]*\u0003)/).map((part, i) =>
        part.startsWith("\u0002") ? (
          <mark key={i} className="rounded bg-accent/25 px-0.5 text-zinc-100">
            {part.slice(1, -1)}
          </mark>
        ) : (
          <span key={i}>{part}</span>
        ),
      )}
    </>
  );
}

function ConversationRow({ c, active, busy }: { c: Conversation; active: boolean; busy: boolean }) {
  const openConversation = useStore((s) => s.openConversation);
  const renameConversation = useStore((s) => s.renameConversation);
  const deleteConversation = useStore((s) => s.deleteConversation);
  const [editing, setEditing] = useState(false);
  const [title, setTitle] = useState(c.title);

  const save = async () => {
    setEditing(false);
    if (title.trim() && title.trim() !== c.title) await renameConversation(c.id, title.trim());
    else setTitle(c.title);
  };

  const remove = async () => {
    const ok = await ask("Delete this conversation? This cannot be undone.", {
      title: "Delete conversation",
      kind: "warning",
    });
    if (ok) await deleteConversation(c.id);
  };

  if (editing) {
    return (
      <input
        autoFocus
        dir={textDir(title)}
        value={title}
        onChange={(e) => setTitle(e.target.value)}
        onBlur={save}
        onKeyDown={(e) => {
          if (e.key === "Enter") void save();
          if (e.key === "Escape") {
            setTitle(c.title);
            setEditing(false);
          }
        }}
        className="w-full rounded-lg border border-accent/50 bg-elevated px-3 py-2 text-sm outline-none"
      />
    );
  }

  return (
    <div
      onClick={() => openConversation(c.id)}
      className={clsx(
        "group flex cursor-pointer items-center gap-1 rounded-lg px-3 py-2 text-sm",
        active ? "bg-elevated text-zinc-50" : "text-zinc-300 hover:bg-white/5",
      )}
    >
      <span dir={textDir(c.title)} className="flex-1 truncate text-start">
        {c.title || "New chat"}
      </span>
      {!busy && (
        <span className="hidden shrink-0 items-center gap-0.5 group-hover:flex">
          <button
            title="Rename"
            onClick={(e) => {
              e.stopPropagation();
              setTitle(c.title);
              setEditing(true);
            }}
            className="rounded p-1 text-muted hover:bg-white/10 hover:text-zinc-100"
          >
            <Pencil className="size-3.5" />
          </button>
          <button
            title="Delete"
            onClick={(e) => {
              e.stopPropagation();
              void remove();
            }}
            className="rounded p-1 text-muted hover:bg-white/10 hover:text-red-400"
          >
            <Trash2 className="size-3.5" />
          </button>
        </span>
      )}
    </div>
  );
}

export function Sidebar() {
  const conversations = useStore((s) => s.conversations);
  const activeId = useStore((s) => s.activeId);
  const streamingId = useStore((s) => s.stream?.conversationId ?? null);
  const newChat = useStore((s) => s.newChat);
  const openConversation = useStore((s) => s.openConversation);

  const [query, setQuery] = useState("");
  const [hits, setHits] = useState<SearchHit[] | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!query.trim()) {
      setHits(null);
      return;
    }
    let alive = true;
    const t = setTimeout(() => {
      api.searchMessages(query).then((h) => alive && setHits(h), () => alive && setHits([]));
    }, SEARCH_DEBOUNCE_MS);
    return () => {
      alive = false;
      clearTimeout(t);
    };
  }, [query]);

  // Ctrl/Cmd+K focuses search.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, []);

  const groups = useMemo(() => groupByDate(conversations), [conversations]);

  return (
    <aside className="flex w-72 shrink-0 flex-col border-e border-border bg-sidebar">
      <div className="flex items-center gap-2 px-4 pt-4 pb-3">
        <div className="flex size-8 items-center justify-center rounded-lg bg-accent/15 text-accent">
          <Cpu className="size-4.5" />
        </div>
        <span className="text-lg font-bold tracking-tight">CoreCS</span>
      </div>

      <div className="space-y-2 px-3">
        <button
          onClick={newChat}
          className="flex w-full items-center gap-2 rounded-lg border border-border px-3 py-2 text-sm hover:bg-white/5"
        >
          <MessageSquarePlus className="size-4" />
          New chat
        </button>
        <div className="flex items-center gap-2 rounded-lg bg-panel px-3 py-2 text-sm focus-within:ring-1 focus-within:ring-accent/50">
          <Search className="size-4 shrink-0 text-muted" />
          <input
            ref={searchRef}
            dir={textDir(query)}
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={(e) => e.key === "Escape" && setQuery("")}
            placeholder="Search chats  (Ctrl+K)"
            className="min-w-0 flex-1 bg-transparent outline-none placeholder:text-muted"
          />
          {query && (
            <button onClick={() => setQuery("")} className="text-muted hover:text-zinc-200">
              <X className="size-3.5" />
            </button>
          )}
        </div>
      </div>

      <nav className="mt-3 flex-1 overflow-y-auto px-3 pb-3">
        {hits !== null ? (
          hits.length === 0 ? (
            <p className="px-3 py-6 text-center text-sm text-muted">No matches</p>
          ) : (
            <div className="space-y-1">
              {hits.map((h) => (
                <button
                  key={h.messageId}
                  onClick={() => openConversation(h.conversationId, h.messageId)}
                  className="block w-full rounded-lg px-3 py-2 text-start hover:bg-white/5"
                >
                  <div dir={textDir(h.conversationTitle)} className="truncate text-sm text-zinc-200">
                    {h.conversationTitle || "New chat"}
                  </div>
                  <div dir={textDir(h.snippet)} className="line-clamp-2 text-xs text-muted">
                    <Snippet text={h.snippet} />
                  </div>
                </button>
              ))}
            </div>
          )
        ) : conversations.length === 0 ? (
          <p className="px-3 py-6 text-center text-sm text-muted">No conversations yet</p>
        ) : (
          groups.map((g) => (
            <div key={g.label} className="mb-3">
              <div className="px-3 pb-1 text-[11px] font-medium uppercase tracking-wider text-muted">{g.label}</div>
              {g.items.map((c) => (
                <ConversationRow key={c.id} c={c} active={c.id === activeId} busy={c.id === streamingId} />
              ))}
            </div>
          ))
        )}
      </nav>

      <div className="flex items-center gap-2 border-t border-border px-4 py-3 text-xs text-emerald-400/90">
        <ShieldCheck className="size-3.5" />
        100% local · nothing leaves this computer
      </div>
    </aside>
  );
}
