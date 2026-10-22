//! The IPC surface the React UI calls via `invoke(...)`. Arguments arrive camelCased
//! from JS (`conversationId`) and are mapped to snake_case here by Tauri.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use tauri::ipc::Channel;
use tauri::State;

use crate::chat::{self, ChatEvent, Generation};
use crate::db::{Conversation, Db, Message, SearchHit};
use crate::engine::{Engine, EngineStatus};
use crate::error::{Error, Result};
use crate::models::{self, ModelInfo};

pub const SETTING_ACTIVE_MODEL: &str = "engine.active_model";
pub const SETTING_DEVICE: &str = "engine.device";

pub struct AppState {
    pub db: Mutex<Db>,
    pub engine: Arc<Engine>,
    pub generation: Generation,
    pub models_dir: PathBuf,
}

impl AppState {
    /// Model to load at startup: $CORECS_MODEL (a .gguf path, handy in development), else the
    /// last model the user picked, else the newest model in the models folder.
    pub fn startup_model(&self) -> Result<Option<ModelInfo>> {
        if let Ok(path) = std::env::var("CORECS_MODEL") {
            return models::load(&PathBuf::from(path)).map(Some);
        }
        let available = models::scan(&self.models_dir)?;
        let preferred = self.db.lock().unwrap().setting(SETTING_ACTIVE_MODEL)?;
        let chosen = preferred.and_then(|id| available.iter().find(|m| m.id == id).cloned());
        Ok(chosen.or_else(|| available.into_iter().next()))
    }

    pub fn device_override(&self) -> Option<String> {
        self.db.lock().unwrap().setting(SETTING_DEVICE).ok().flatten()
    }
}

// ---- conversations -------------------------------------------------------------------

#[tauri::command]
pub fn list_conversations(state: State<'_, AppState>) -> Result<Vec<Conversation>> {
    state.db.lock().unwrap().list_conversations()
}

#[tauri::command]
pub fn create_conversation(state: State<'_, AppState>, title: Option<String>) -> Result<Conversation> {
    state.db.lock().unwrap().create_conversation(title.as_deref().unwrap_or(""))
}

#[tauri::command]
pub fn rename_conversation(state: State<'_, AppState>, conversation_id: i64, title: String) -> Result<()> {
    state.db.lock().unwrap().rename_conversation(conversation_id, title.trim())
}

#[tauri::command]
pub fn delete_conversation(state: State<'_, AppState>, conversation_id: i64) -> Result<()> {
    state.db.lock().unwrap().delete_conversation(conversation_id)
}

#[tauri::command]
pub fn get_messages(state: State<'_, AppState>, conversation_id: i64) -> Result<Vec<Message>> {
    state.db.lock().unwrap().messages(conversation_id)
}

#[tauri::command]
pub fn search_messages(state: State<'_, AppState>, query: String) -> Result<Vec<SearchHit>> {
    if query.trim().is_empty() {
        return Ok(vec![]);
    }
    state.db.lock().unwrap().search(query.trim(), 50)
}

// ---- chat ----------------------------------------------------------------------------

/// Streams the reply over `on_event`; resolves with the assistant message id once done.
#[tauri::command]
pub async fn send_message(
    state: State<'_, AppState>,
    conversation_id: i64,
    content: String,
    on_event: Channel<ChatEvent>,
) -> Result<i64> {
    chat::send_message(&state.db, &state.engine, &state.generation, conversation_id, content, |e| {
        let _ = on_event.send(e);
    })
    .await
}

#[tauri::command]
pub fn cancel_generation(state: State<'_, AppState>) {
    state.generation.cancel();
}

// ---- engine / models -----------------------------------------------------------------

#[tauri::command]
pub fn engine_status(state: State<'_, AppState>) -> EngineStatus {
    state.engine.status()
}

#[tauri::command]
pub fn list_models(state: State<'_, AppState>) -> Result<Vec<ModelInfo>> {
    models::scan(&state.models_dir)
}

#[tauri::command]
pub fn models_dir(state: State<'_, AppState>) -> String {
    state.models_dir.display().to_string()
}

/// Loads a model from the models folder by id (file name) and remembers the choice.
#[tauri::command]
pub async fn load_model(state: State<'_, AppState>, model_id: String) -> Result<()> {
    let model = models::scan(&state.models_dir)?
        .into_iter()
        .find(|m| m.id == model_id)
        .ok_or_else(|| Error::msg(format!("model {model_id} not found")))?;
    state.db.lock().unwrap().set_setting(SETTING_ACTIVE_MODEL, &model.id)?;
    state.engine.start(model, state.device_override()).await
}

#[tauri::command]
pub async fn stop_engine(state: State<'_, AppState>) -> Result<()> {
    state.generation.cancel();
    state.engine.stop().await;
    Ok(())
}
