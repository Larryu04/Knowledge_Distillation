import { useEffect } from "react";

import { Composer } from "./components/Composer";
import { EmptyState } from "./components/EmptyState";
import { EngineBanner } from "./components/EngineBanner";
import { MessageList } from "./components/MessageList";
import { ModelPicker } from "./components/ModelPicker";
import { Sidebar } from "./components/Sidebar";
import { Toast } from "./components/Toast";
import { textDir } from "./lib/bidi";
import { errorText } from "./lib/format";
import { useStore } from "./store";

function ChatHeader() {
  const title = useStore((s) => s.conversations.find((c) => c.id === s.activeId)?.title);
  return (
    <header className="flex h-14 shrink-0 items-center justify-between gap-4 border-b border-border px-5">
      <h1 dir={textDir(title ?? "")} className="min-w-0 truncate font-medium text-zinc-200">
        {title || "New chat"}
      </h1>
      <ModelPicker />
    </header>
  );
}

export default function App() {
  const init = useStore((s) => s.init);
  const showToast = useStore((s) => s.showToast);
  const isNewChat = useStore((s) => s.activeId === null && s.messages.length === 0);

  useEffect(() => {
    init().catch((e) => showToast(`Failed to start: ${errorText(e)}`));
  }, [init, showToast]);

  return (
    <div className="flex h-full">
      <Sidebar />
      <main className="flex min-w-0 flex-1 flex-col">
        <ChatHeader />
        <EngineBanner />
        {isNewChat ? <EmptyState /> : <MessageList />}
        <Composer />
      </main>
      <Toast />
    </div>
  );
}
