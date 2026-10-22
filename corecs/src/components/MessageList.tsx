import { useEffect, useLayoutEffect, useRef } from "react";

import { useStore } from "../store";
import { MessageItem } from "./MessageItem";

/** Distance from the bottom (px) within which the view keeps following new tokens. */
const STICK_THRESHOLD = 80;

export function MessageList() {
  const activeId = useStore((s) => s.activeId);
  const messages = useStore((s) => s.messages);
  const stream = useStore((s) => s.stream);
  const errors = useStore((s) => s.errors);
  const speeds = useStore((s) => s.speeds);
  const focusMessageId = useStore((s) => s.focusMessageId);
  const clearFocus = useStore((s) => s.clearFocus);

  const scroller = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    stick.current = true;
  }, [activeId]);

  // Follow the answer as it streams, unless the user scrolled up to read.
  useLayoutEffect(() => {
    const el = scroller.current;
    if (el && stick.current && focusMessageId === null) el.scrollTop = el.scrollHeight;
  }, [messages, stream?.text, focusMessageId]);

  // Jump to a message chosen from search results.
  useEffect(() => {
    if (focusMessageId === null) return;
    const el = document.getElementById(`msg-${focusMessageId}`);
    if (!el) return;
    stick.current = false;
    el.scrollIntoView({ block: "center" });
    el.classList.add("flash");
    const t = setTimeout(() => el.classList.remove("flash"), 2000);
    clearFocus();
    return () => clearTimeout(t);
  }, [focusMessageId, messages, clearFocus]);

  const onScroll = () => {
    const el = scroller.current;
    if (el) stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < STICK_THRESHOLD;
  };

  const liveStream = stream && stream.conversationId === activeId ? stream : null;

  return (
    <div ref={scroller} onScroll={onScroll} className="flex-1 overflow-y-auto">
      <div className="mx-auto flex max-w-3xl flex-col gap-6 px-6 py-8">
        {messages.map((m) => {
          const streaming = liveStream?.assistantId === m.id;
          return (
            <MessageItem
              key={m.id}
              message={m}
              content={streaming ? liveStream.text : m.content}
              streaming={streaming}
              error={errors[m.id]}
              tokensPerSecond={speeds[m.id]}
            />
          );
        })}
      </div>
    </div>
  );
}
