pub mod chat;
pub mod commands;
pub mod db;
pub mod engine;
pub mod error;
pub mod models;

use std::sync::{Arc, Mutex};

use tauri::{Emitter, Manager, RunEvent};

use commands::AppState;
use engine::{Engine, EngineStatus};

/// Event the UI listens to for engine state changes (see `EngineStatus`).
pub const ENGINE_STATUS_EVENT: &str = "engine://status";

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let data_dir = app.path().app_data_dir()?;
            std::fs::create_dir_all(data_dir.join("models"))?;
            let db = db::Db::open(data_dir.join("corecs.db"))?;

            let handle = app.handle().clone();
            let engine = Arc::new(Engine::new(
                Engine::default_server_exe()?,
                Arc::new(move |s: &EngineStatus| {
                    let _ = handle.emit(ENGINE_STATUS_EVENT, s);
                }),
            ));

            app.manage(AppState {
                db: Mutex::new(db),
                engine: engine.clone(),
                generation: Default::default(),
                models_dir: data_dir.join("models"),
            });

            // Load the model in the background; the UI follows along via ENGINE_STATUS_EVENT.
            let handle = app.handle().clone();
            tauri::async_runtime::spawn(async move {
                let state = handle.state::<AppState>();
                match state.startup_model() {
                    Ok(Some(model)) => {
                        let _ = engine.start(model, state.device_override()).await;
                    }
                    Ok(None) => {} // status stays NoModel -> UI shows model setup
                    Err(e) => eprintln!("[startup] {e}"),
                }
            });
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            commands::list_conversations,
            commands::create_conversation,
            commands::rename_conversation,
            commands::delete_conversation,
            commands::get_messages,
            commands::search_messages,
            commands::send_message,
            commands::cancel_generation,
            commands::engine_status,
            commands::list_models,
            commands::models_dir,
            commands::load_model,
            commands::stop_engine,
        ])
        .build(tauri::generate_context!())
        .expect("error while building tauri application");

    app.run(|handle, event| {
        if let RunEvent::Exit = event {
            let engine = handle.state::<AppState>().engine.clone();
            tauri::async_runtime::block_on(engine.stop());
        }
    });
}
