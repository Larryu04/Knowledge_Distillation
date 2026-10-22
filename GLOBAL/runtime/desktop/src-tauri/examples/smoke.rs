//! End-to-end check of the Rust core without the UI: launches llama-server, streams replies,
//! exercises multi-turn, Hebrew, cancellation and persistence.
//!
//!   cargo run --example smoke -- <path\to\model.gguf>
//!
//! Uses the sidecar Tauri copies into target\debug (run `cargo build` once first).

use std::io::Write;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use corecs_lib::chat::{self, ChatEvent, Generation};
use corecs_lib::db::Db;
use corecs_lib::engine::{Engine, EngineStatus};
use corecs_lib::models;

fn print_event(e: ChatEvent) {
    match e {
        ChatEvent::Token { text } => {
            print!("{text}");
            let _ = std::io::stdout().flush();
        }
        other => println!("\n  <{other:?}>"),
    }
}

#[tokio::main]
async fn main() {
    let model_path = std::env::args().nth(1).expect("usage: smoke <model.gguf>");
    let model = models::load(model_path.as_ref()).expect("load model");
    println!("model: {} (profile: {})", model.display_name, model.has_profile);

    let exe = std::env::current_exe().unwrap().parent().unwrap().parent().unwrap().join("llama-server.exe");
    let engine = Engine::new(exe, Arc::new(|s: &EngineStatus| println!("[status] {s:?}")));
    let db_path = std::env::temp_dir().join("corecs-smoke.db");
    let _ = std::fs::remove_file(&db_path);
    let db = Mutex::new(Db::open(&db_path).unwrap());
    let generation = Generation::default();

    let t = Instant::now();
    engine.start(model, None).await.expect("engine start");
    println!("engine ready in {:.1}s", t.elapsed().as_secs_f32());

    let conv = db.lock().unwrap().create_conversation("").unwrap();
    for q in ["What is a hash table?", "Give a one-line Python example of one.", "מהו מבנה נתונים מסוג מחסנית?"] {
        println!("\n> {q}");
        chat::send_message(&db, &engine, &generation, conv.id, q.into(), print_event).await.unwrap();
    }

    // Cancel mid-stream, after 20 tokens, like pressing "Stop" in the UI.
    println!("\n> (cancel test) Write a long essay about sorting algorithms.");
    let tokens = AtomicUsize::new(0);
    let q = "Write a long essay about sorting algorithms.".to_string();
    chat::send_message(&db, &engine, &generation, conv.id, q, |e| {
        if matches!(e, ChatEvent::Token { .. }) && tokens.fetch_add(1, Ordering::SeqCst) == 19 {
            generation.cancel();
        }
        print_event(e);
    })
    .await
    .unwrap();

    println!("\n--- persisted ---");
    let db = db.lock().unwrap();
    println!("title: {:?}", db.conversation(conv.id).unwrap().unwrap().title);
    for m in db.messages(conv.id).unwrap() {
        let preview: String = m.content.chars().take(50).collect();
        println!("#{} {:<9} {:<8} {preview:?}", m.id, m.role, m.status);
    }
    println!("search 'hash' hits: {}", db.search("hash", 10).unwrap().len());
    drop(db);

    engine.stop().await;
    println!("final status: {:?}", engine.status());
}
