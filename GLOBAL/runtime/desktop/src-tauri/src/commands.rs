//! The IPC surface the React UI calls via `invoke(...)`. Arguments arrive camelCased
//! from JS (`conversationId`) and are mapped to snake_case here by Tauri.

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use tauri::ipc::Channel;
use tauri::State;

use crate::chat::{self, ChatEvent, Generation};
use crate::db::{Conversation, Db, Message, SearchHit};
use crate::engine::{Engine, EngineStatus};
use crate::error::{Error, Result};
use crate::models::{self, ModelInfo};
use crate::provision::{self, ModelManifest};

pub const SETTING_ACTIVE_MODEL: &str = "engine.active_model";
pub const SETTING_DEVICE: &str = "engine.device";
/// `ModelManifest::key()` of the last model pulled on first launch.
pub const SETTING_PROVISIONED: &str = "provision.manifest";

pub struct AppState {
    pub db: Mutex<Db>,
    pub engine: Arc<Engine>,
    pub generation: Generation,
    pub models_dir: PathBuf,
    pub provisioning: AtomicBool,
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

    /// Where a lecturer's `corecs-model.json` can be: next to the executable (the installer
    /// copies it into the install folder) or in the app data folder (for a manual drop-in).
    fn manifest_dirs(&self) -> Vec<PathBuf> {
        let exe_dir = std::env::current_exe().ok().and_then(|e| e.parent().map(PathBuf::from));
        exe_dir.into_iter().chain(self.models_dir.parent().map(PathBuf::from)).collect()
    }

    /// The model still to be pulled, if any (see `provision` for the rules).
    fn pending_download(&self) -> Result<Option<ModelManifest>> {
        if std::env::var_os("CORECS_MODEL").is_some() {
            return Ok(None);
        }
        let available = models::scan(&self.models_dir)?;
        if let Some(m) = provision::find_custom(&self.manifest_dirs()) {
            let db = self.db.lock().unwrap();
            if db.setting(SETTING_PROVISIONED)?.as_deref() == Some(m.key().as_str()) {
                return Ok(None); // pulled on an earlier launch; the student may have switched since
            }
            if available.iter().any(|a| a.id == m.gguf) {
                // Already there (copied by hand, or a reinstall): just make it the active model.
                db.set_setting(SETTING_ACTIVE_MODEL, &m.gguf)?;
                db.set_setting(SETTING_PROVISIONED, &m.key())?;
                return Ok(None);
            }
            return Ok(Some(m));
        }
        Ok(available.is_empty().then(provision::default_manifest))
    }

    /// Pulls the lecturer's (or the default) model if needed and makes it the active model.
    /// Progress is reported through the engine status (`Downloading`).
    pub async fn provision(&self) -> Result<()> {
        let Some(m) = self.pending_download()? else { return Ok(()) };
        if self.provisioning.swap(true, Ordering::SeqCst) {
            return Err(Error::msg("the model is already being downloaded"));
        }
        let model = m.label();
        let result = provision::download(&m, &self.models_dir, |received, total| {
            self.engine.set_status(EngineStatus::Downloading { model: model.clone(), received, total });
        })
        .await;
        self.provisioning.store(false, Ordering::SeqCst);
        if let Err(e) = result {
            self.engine.set_status(EngineStatus::Failed { message: format!("Could not download {model}: {e}") });
            return Err(e);
        }
        let db = self.db.lock().unwrap();
        db.set_setting(SETTING_ACTIVE_MODEL, &m.gguf)?;
        db.set_setting(SETTING_PROVISIONED, &m.key())?;
        Ok(())
    }

    /// Startup and "Retry" with an empty models folder: pull a model if needed, then load one.
    pub async fn boot(&self) -> Result<()> {
        self.provision().await?;
        match self.startup_model()? {
            Some(model) => self.engine.start(model, self.device_override()).await,
            None => Ok(()), // status stays NoModel -> UI shows model setup
        }
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

/// Pulls the first-launch model (lecturer's or default) if none is installed yet, then loads one.
#[tauri::command]
pub async fn provision_model(state: State<'_, AppState>) -> Result<()> {
    state.boot().await
}

#[tauri::command]
pub async fn stop_engine(state: State<'_, AppState>) -> Result<()> {
    state.generation.cancel();
    state.engine.stop().await;
    Ok(())
}
