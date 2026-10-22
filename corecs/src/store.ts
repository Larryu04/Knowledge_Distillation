import { create } from "zustand";

import { api, type Conversation, type EngineStatus, type Message, type ModelInfo } from "./lib/api";
import { errorText } from "./lib/format";

/** The reply currently being generated. Its text lives here (not in `messages`) while streaming. */
export interface Stream {
  conversationId: number;
  assistantId: number;
  text: string;
}

interface ChatStore {
  conversations: Conversation[];
  /** null = a new, not yet saved chat (created on the first message). */
  activeId: number | null;
  messages: Message[];
  stream: Stream | null;
  /** Session-only details for failed replies and generation speed, keyed by message id. */
  errors: Record<number, string>;
  speeds: Record<number, number>;
  /** Message to scroll to and highlight (from search). */
  focusMessageId: number | null;

  engine: EngineStatus;
  models: ModelInfo[];
  modelsDir: string;
  /** Last model the engine started, used by "Retry". */
  lastModelId: string | null;
  toast: string | null;

  init(): Promise<void>;
  refreshConversations(): Promise<void>;
  openConversation(id: number, focusMessageId?: number): Promise<void>;
  newChat(): void;
  renameConversation(id: number, title: string): Promise<void>;
  deleteConversation(id: number): Promise<void>;
  /** Resolves false if the message was rejected before streaming began. */
  send(text: string): Promise<boolean>;
  stop(): void;
  refreshModels(): Promise<ModelInfo[]>;
  loadModel(id: string): Promise<void>;
  retryEngine(): Promise<void>;
  showToast(message: string | null): void;
  clearFocus(): void;
}

let initialized = false;

export const useStore = create<ChatStore>()((set, get) => ({
  conversations: [],
  activeId: null,
  messages: [],
  stream: null,
  errors: {},
  speeds: {},
  focusMessageId: null,
  engine: { state: "noModel" },
  models: [],
  modelsDir: "",
  lastModelId: null,
  toast: null,

  async init() {
    if (initialized) return; // React StrictMode runs effects twice in development
    initialized = true;
    // Subscribe before reading the current status so no transition is missed.
    await api.onEngineStatus((engine) => {
      set((s) => ({
        engine,
        lastModelId: engine.state === "starting" || engine.state === "ready" ? engine.modelId : s.lastModelId,
      }));
    });
    const [engine, modelsDir] = await Promise.all([api.engineStatus(), api.modelsDir()]);
    set((s) => ({
      engine,
      modelsDir,
      lastModelId: engine.state === "starting" || engine.state === "ready" ? engine.modelId : s.lastModelId,
    }));
    await Promise.all([get().refreshConversations(), get().refreshModels()]);
  },

  async refreshConversations() {
    set({ conversations: await api.listConversations() });
  },

  async openConversation(id, focusMessageId) {
    set({ activeId: id, focusMessageId: focusMessageId ?? null });
    const messages = await api.getMessages(id);
    if (get().activeId === id) set({ messages });
  },

  newChat() {
    set({ activeId: null, messages: [], focusMessageId: null });
  },

  async renameConversation(id, title) {
    await api.renameConversation(id, title);
    await get().refreshConversations();
  },

  async deleteConversation(id) {
    await api.deleteConversation(id);
    if (get().activeId === id) get().newChat();
    await get().refreshConversations();
  },

  async send(raw) {
    const content = raw.trim();
    if (!content || get().stream) return false;
    if (get().engine.state !== "ready") {
      get().showToast("The model is not loaded yet.");
      return false;
    }

    let conversationId = get().activeId;
    try {
      if (conversationId === null) {
        const c = await api.createConversation();
        conversationId = c.id;
        set((s) => ({ conversations: [c, ...s.conversations], activeId: c.id, messages: [] }));
      }
    } catch (e) {
      get().showToast(errorText(e));
      return false;
    }
    const convId = conversationId;

    // Optimistic placeholders; real ids arrive with the "started" event.
    const now = Date.now();
    const tempUser: Message = {
      id: -now,
      conversationId: convId,
      role: "user",
      content,
      status: "complete",
      model: null,
      createdAt: now,
    };
    const tempAssistant: Message = { ...tempUser, id: -now - 1, role: "assistant", content: "", status: "streaming" };
    set((s) => ({
      messages: [...s.messages, tempUser, tempAssistant],
      stream: { conversationId: convId, assistantId: tempAssistant.id, text: "" },
    }));

    let text = "";
    let assistantId = tempAssistant.id;
    let finished = false;
    const finish = (status: Message["status"]) => {
      finished = true;
      set((s) => ({
        stream: null,
        messages:
          s.activeId === convId
            ? s.messages.map((m) => (m.id === assistantId ? { ...m, content: text, status } : m))
            : s.messages,
      }));
    };

    try {
      await api.sendMessage(convId, content, (ev) => {
        switch (ev.event) {
          case "started":
            assistantId = ev.data.assistantMessageId;
            set((s) => ({
              messages: s.messages.map((m) =>
                m.id === tempUser.id
                  ? { ...m, id: ev.data.userMessageId }
                  : m.id === tempAssistant.id
                    ? { ...m, id: assistantId }
                    : m,
              ),
              stream: s.stream && { ...s.stream, assistantId },
            }));
            break;
          case "token":
            text += ev.data.text;
            set((s) => (s.stream ? { stream: { ...s.stream, text } } : {}));
            break;
          case "done":
            if (ev.data.tokensPerSecond) {
              set((s) => ({ speeds: { ...s.speeds, [ev.data.messageId]: ev.data.tokensPerSecond! } }));
            }
            finish(ev.data.status);
            break;
          case "error":
            set((s) => ({ errors: { ...s.errors, [ev.data.messageId]: ev.data.message } }));
            finish("error");
            break;
        }
      });
    } catch (e) {
      // Rejected before anything was saved (engine not ready, busy, ...): undo the placeholders.
      set((s) => ({
        stream: null,
        messages: s.messages.filter((m) => m.id !== tempUser.id && m.id !== tempAssistant.id),
      }));
      get().showToast(errorText(e));
      return false;
    } finally {
      void get().refreshConversations(); // picks up the auto-generated title
    }

    // Channel events can trail the command's response; if "done" never arrives, resync from the DB.
    if (!finished) {
      await new Promise((r) => setTimeout(r, 300));
      if (!finished) {
        set({ stream: null });
        if (get().activeId === convId) set({ messages: await api.getMessages(convId) });
      }
    }
    return true;
  },

  stop() {
    void api.cancelGeneration();
  },

  async refreshModels() {
    const models = await api.listModels();
    set({ models });
    return models;
  },

  async loadModel(id) {
    set({ lastModelId: id });
    try {
      await api.loadModel(id);
    } catch {
      // The failure is reported through the engine status banner.
    }
  },

  async retryEngine() {
    const models = await get().refreshModels();
    const id = models.find((m) => m.id === get().lastModelId)?.id ?? models[0]?.id;
    if (id) await get().loadModel(id);
    else get().showToast("No model found in the models folder.");
  },

  showToast(message) {
    set({ toast: message });
  },

  clearFocus() {
    set({ focusMessageId: null });
  },
}));
