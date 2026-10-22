// Typed wrapper over the Rust core's IPC commands (src-tauri/src/commands.rs).
import { Channel, invoke } from "@tauri-apps/api/core";
import { listen, type UnlistenFn } from "@tauri-apps/api/event";

export interface Conversation {
  id: number;
  title: string;
  createdAt: number;
  updatedAt: number;
}

export type MessageStatus = "complete" | "streaming" | "partial" | "error";

export interface Message {
  id: number;
  conversationId: number;
  role: "user" | "assistant";
  content: string;
  status: MessageStatus;
  model: string | null;
  createdAt: number;
}

export interface SearchHit {
  messageId: number;
  conversationId: number;
  conversationTitle: string;
  /** Matches are wrapped in \u0002 ... \u0003. */
  snippet: string;
}

export interface ModelInfo {
  id: string;
  path: string;
  displayName: string;
  sizeBytes: number;
  hasProfile: boolean;
}

export type EngineStatus =
  | { state: "noModel" }
  | { state: "starting"; model: string; modelId: string }
  | { state: "ready"; model: string; modelId: string; device: string | null }
  | { state: "stopped" }
  | { state: "failed"; message: string };

export type ChatEvent =
  | { event: "started"; data: { userMessageId: number; assistantMessageId: number } }
  | { event: "token"; data: { text: string } }
  | {
      event: "done";
      data: {
        messageId: number;
        status: MessageStatus;
        finishReason: string | null;
        tokensPerSecond: number | null;
      };
    }
  | { event: "error"; data: { messageId: number; message: string } };

export const api = {
  listConversations: () => invoke<Conversation[]>("list_conversations"),
  createConversation: (title?: string) => invoke<Conversation>("create_conversation", { title }),
  renameConversation: (conversationId: number, title: string) =>
    invoke<void>("rename_conversation", { conversationId, title }),
  deleteConversation: (conversationId: number) => invoke<void>("delete_conversation", { conversationId }),
  getMessages: (conversationId: number) => invoke<Message[]>("get_messages", { conversationId }),
  searchMessages: (query: string) => invoke<SearchHit[]>("search_messages", { query }),

  /** Streams the reply to `onEvent`; resolves with the assistant message id when finished. */
  sendMessage: (conversationId: number, content: string, onEvent: (e: ChatEvent) => void) => {
    const channel = new Channel<ChatEvent>();
    channel.onmessage = onEvent;
    return invoke<number>("send_message", { conversationId, content, onEvent: channel });
  },
  cancelGeneration: () => invoke<void>("cancel_generation"),

  engineStatus: () => invoke<EngineStatus>("engine_status"),
  onEngineStatus: (cb: (s: EngineStatus) => void): Promise<UnlistenFn> =>
    listen<EngineStatus>("engine://status", (e) => cb(e.payload)),
  listModels: () => invoke<ModelInfo[]>("list_models"),
  modelsDir: () => invoke<string>("models_dir"),
  loadModel: (modelId: string) => invoke<void>("load_model", { modelId }),
  stopEngine: () => invoke<void>("stop_engine"),
};
