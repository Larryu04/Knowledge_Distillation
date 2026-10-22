//! Streaming chat: persists the turn, builds the prompt, streams tokens from llama-server.

use std::sync::Mutex;
use std::time::{Duration, Instant};

use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use tokio_util::sync::CancellationToken;

use crate::db::{Db, Message};
use crate::engine::Engine;
use crate::error::{Error, Result};
use crate::models::ModelProfile;

/// How often a streaming reply is flushed to SQLite, so a crash loses at most this much.
const FLUSH_INTERVAL: Duration = Duration::from_millis(750);
/// Rough chars-per-token used to keep history inside the context window. Deliberately
/// low: Hebrew often tokenizes worse than English.
const CHARS_PER_TOKEN: usize = 2;
const TITLE_CHARS: usize = 60;

#[derive(Serialize, Clone, Debug)]
#[serde(tag = "event", content = "data", rename_all = "camelCase", rename_all_fields = "camelCase")]
pub enum ChatEvent {
    Started { user_message_id: i64, assistant_message_id: i64 },
    Token { text: String },
    Done { message_id: i64, status: String, finish_reason: Option<String>, tokens_per_second: Option<f64> },
    Error { message_id: i64, message: String },
}

/// One generation at a time (the server runs a single slot).
#[derive(Default)]
pub struct Generation {
    active: Mutex<Option<CancellationToken>>,
}

struct ActiveGuard<'a>(&'a Generation);

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        *self.0.active.lock().unwrap() = None;
    }
}

impl Generation {
    fn begin(&self) -> Result<(CancellationToken, ActiveGuard<'_>)> {
        let mut active = self.active.lock().unwrap();
        if active.is_some() {
            return Err(Error::msg("a reply is already being generated"));
        }
        let token = CancellationToken::new();
        *active = Some(token.clone());
        Ok((token, ActiveGuard(self)))
    }

    /// Stops the current reply; the text so far is kept with status `partial`.
    pub fn cancel(&self) {
        if let Some(t) = self.active.lock().unwrap().as_ref() {
            t.cancel();
        }
    }
}

/// Sends `content` as the next user turn and streams the reply through `emit`.
/// Returns the assistant message id. Engine/stream failures are reported via
/// `ChatEvent::Error` and saved on the message (status `error`), not returned as `Err`.
pub async fn send_message(
    db: &Mutex<Db>,
    engine: &Engine,
    generation: &Generation,
    conversation_id: i64,
    content: String,
    emit: impl Fn(ChatEvent),
) -> Result<i64> {
    let content = content.trim().to_string();
    if content.is_empty() {
        return Err(Error::msg("message is empty"));
    }
    let endpoint = engine.endpoint().await.ok_or_else(|| Error::msg("the model is not loaded yet"))?;
    let (cancel, _guard) = generation.begin()?;
    let profile = &endpoint.model.profile;

    let (user_msg, assistant_msg, history) = {
        let db = db.lock().unwrap();
        let conversation =
            db.conversation(conversation_id)?.ok_or_else(|| Error::msg("conversation not found"))?;
        let user_msg = db.insert_message(conversation_id, "user", &content, "complete", None)?;
        if conversation.title.is_empty() {
            db.rename_conversation(conversation_id, &make_title(&content))?;
        }
        let history = db.messages(conversation_id)?;
        let assistant_msg = db.insert_message(
            conversation_id,
            "assistant",
            "",
            "streaming",
            Some(&endpoint.model.display_name),
        )?;
        (user_msg, assistant_msg, history)
    };
    emit(ChatEvent::Started { user_message_id: user_msg.id, assistant_message_id: assistant_msg.id });

    let mut body = json!({ "messages": build_messages(&history, profile), "stream": true, "cache_prompt": true });
    for (k, v) in &profile.sampling {
        body[k] = v.clone();
    }
    body["stream"] = json!(true);

    let mut text = String::new();
    let outcome = stream_reply(engine, &endpoint.base_url, &endpoint.api_key, &body, &cancel, |delta| {
        text.push_str(delta);
        emit(ChatEvent::Token { text: delta.to_string() });
    }, |partial| {
        let _ = db.lock().unwrap().update_message(assistant_msg.id, partial, "streaming");
    })
    .await;

    let db = db.lock().unwrap();
    match outcome {
        Ok(done) => {
            let status = if done.cancelled { "partial" } else { "complete" };
            db.update_message(assistant_msg.id, &text, status)?;
            emit(ChatEvent::Done {
                message_id: assistant_msg.id,
                status: status.into(),
                finish_reason: done.finish_reason,
                tokens_per_second: done.tokens_per_second,
            });
        }
        Err(e) => {
            db.update_message(assistant_msg.id, &text, "error")?;
            emit(ChatEvent::Error { message_id: assistant_msg.id, message: e.to_string() });
        }
    }
    Ok(assistant_msg.id)
}

struct StreamOutcome {
    cancelled: bool,
    finish_reason: Option<String>,
    tokens_per_second: Option<f64>,
}

/// POSTs an OpenAI-style streaming chat request and feeds text deltas to `on_delta`.
/// `on_flush` receives the accumulated text every FLUSH_INTERVAL for persistence.
async fn stream_reply(
    engine: &Engine,
    base_url: &str,
    api_key: &str,
    body: &Value,
    cancel: &CancellationToken,
    mut on_delta: impl FnMut(&str),
    on_flush: impl Fn(&str),
) -> Result<StreamOutcome> {
    let resp = engine
        .http()
        .post(format!("{base_url}/v1/chat/completions"))
        .bearer_auth(api_key)
        .json(body)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let detail = resp.text().await.unwrap_or_default();
        return Err(Error::msg(format!("engine returned {status}: {}", error_message(&detail))));
    }

    let mut out = StreamOutcome { cancelled: false, finish_reason: None, tokens_per_second: None };
    let mut sse = SseParser::default();
    let mut text = String::new();
    let mut last_flush = Instant::now();
    let mut stream = resp.bytes_stream();

    loop {
        let chunk = tokio::select! {
            _ = cancel.cancelled() => {
                out.cancelled = true; // dropping the response closes the connection; the server stops generating
                break;
            }
            chunk = stream.next() => chunk,
        };
        let Some(chunk) = chunk else { break };
        for event in sse.push(&chunk?) {
            if event == "[DONE]" {
                return Ok(out);
            }
            let Ok(v) = serde_json::from_str::<Value>(&event) else { continue };
            if let Some(err) = v.get("error") {
                return Err(Error::msg(error_message(&err.to_string())));
            }
            let choice = &v["choices"][0];
            if let Some(delta) = choice["delta"]["content"].as_str().filter(|d| !d.is_empty()) {
                text.push_str(delta);
                on_delta(delta);
            }
            if let Some(reason) = choice["finish_reason"].as_str() {
                out.finish_reason = Some(reason.into());
            }
            if let Some(tps) = v["timings"]["predicted_per_second"].as_f64() {
                out.tokens_per_second = Some(tps);
            }
        }
        if last_flush.elapsed() >= FLUSH_INTERVAL {
            on_flush(&text);
            last_flush = Instant::now();
        }
    }
    Ok(out)
}

fn error_message(detail: &str) -> String {
    serde_json::from_str::<Value>(detail)
        .ok()
        .and_then(|v| v["error"]["message"].as_str().or(v["message"].as_str()).map(String::from))
        .unwrap_or_else(|| detail.chars().take(300).collect())
}

/// Incremental Server-Sent-Events parser. Buffers raw bytes and only decodes complete
/// lines, so multi-byte UTF-8 (Hebrew) split across network chunks is never corrupted.
#[derive(Default)]
struct SseParser {
    buf: Vec<u8>,
}

impl SseParser {
    /// Returns the `data:` payloads completed by this chunk.
    fn push(&mut self, chunk: &[u8]) -> Vec<String> {
        self.buf.extend_from_slice(chunk);
        let mut events = vec![];
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let line: Vec<u8> = self.buf.drain(..=pos).collect();
            let line = String::from_utf8_lossy(&line);
            if let Some(data) = line.trim_end_matches(['\r', '\n']).strip_prefix("data:") {
                events.push(data.trim_start().to_string());
            }
        }
        events
    }
}

/// Turns stored history into chat messages the model's template accepts:
/// strictly alternating user/assistant, starting and ending with a user turn,
/// trimmed (oldest turns first) to fit the context window, with the model profile's
/// preamble folded into the first user turn.
pub fn build_messages(history: &[Message], profile: &ModelProfile) -> Vec<Value> {
    // Keep everything the user saw; drop failed/empty assistant turns.
    let mut turns: Vec<(&str, String)> = vec![];
    for m in history.iter().filter(|m| !(m.role == "assistant" && m.content.trim().is_empty())) {
        match turns.last_mut() {
            // Merge consecutive same-role turns (e.g. a question whose answer failed).
            Some((role, text)) if *role == m.role => {
                text.push_str("\n\n");
                text.push_str(&m.content);
            }
            _ => turns.push((m.role.as_str(), m.content.clone())),
        }
    }
    while turns.first().is_some_and(|(r, _)| *r == "assistant") {
        turns.remove(0);
    }

    let preamble_chars = profile.prompt.as_ref().map_or(0, |p| p.preamble.chars().count() + p.separator.len());
    let budget_tokens = (profile.context_length() as u64).saturating_sub(profile.max_tokens()) as usize;
    let budget_chars = (budget_tokens * CHARS_PER_TOKEN).saturating_sub(preamble_chars);
    let total = |t: &[(&str, String)]| t.iter().map(|(_, c)| c.chars().count()).sum::<usize>();
    // Drop the oldest user+assistant pair while over budget; always keep the newest turn.
    while turns.len() > 2 && total(&turns) > budget_chars {
        turns.drain(..2);
    }

    if let (Some(p), Some((_, first))) = (&profile.prompt, turns.first_mut()) {
        *first = format!("{}{}{}", p.preamble, p.separator, first);
    }
    turns.into_iter().map(|(role, content)| json!({ "role": role, "content": content })).collect()
}

fn make_title(content: &str) -> String {
    let line = content.lines().next().unwrap_or("").trim();
    if line.chars().count() <= TITLE_CHARS {
        line.to_string()
    } else {
        line.chars().take(TITLE_CHARS).collect::<String>().trim_end().to_string() + "…"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::PromptProfile;

    fn msg(role: &str, content: &str) -> Message {
        Message {
            id: 0,
            conversation_id: 1,
            role: role.into(),
            content: content.into(),
            status: "complete".into(),
            model: None,
            created_at: 0,
        }
    }

    fn profile(preamble: Option<&str>, ctx: u32) -> ModelProfile {
        let mut p = ModelProfile::default();
        p.context_length = Some(ctx);
        p.sampling.insert("max_tokens".into(), json!(100));
        p.prompt = preamble.map(|s| PromptProfile { preamble: s.into(), separator: "\n\n".into() });
        p
    }

    #[test]
    fn preamble_goes_into_first_user_turn_only() {
        let h = [msg("user", "q1"), msg("assistant", "a1"), msg("user", "q2")];
        let m = build_messages(&h, &profile(Some("אתה מתרגל"), 4096));
        assert_eq!(m.len(), 3);
        assert_eq!(m[0]["content"], "אתה מתרגל\n\nq1");
        assert_eq!(m[2]["content"], "q2");
        assert!(m.iter().all(|x| x["role"] != "system"));
    }

    #[test]
    fn no_profile_means_plain_messages() {
        let m = build_messages(&[msg("user", "hi")], &ModelProfile::default());
        assert_eq!(m, vec![json!({"role": "user", "content": "hi"})]);
    }

    #[test]
    fn keeps_strict_alternation_after_failed_reply() {
        let h = [msg("user", "q1"), msg("assistant", ""), msg("user", "q1 again")];
        let m = build_messages(&h, &profile(None, 4096));
        assert_eq!(m.len(), 1);
        assert_eq!(m[0]["content"], "q1\n\nq1 again");
    }

    #[test]
    fn trims_oldest_turns_to_fit_context() {
        let long = "x".repeat(300);
        let h = [msg("user", &long), msg("assistant", &long), msg("user", "latest")];
        // (200 ctx - 100 max_tokens) * 2 chars = 200 chars budget
        let m = build_messages(&h, &profile(None, 200));
        assert_eq!(m.len(), 1);
        assert_eq!(m[0]["content"], "latest");
    }

    #[test]
    fn sse_parser_handles_split_utf8_and_crlf() {
        let payload = "data: {\"t\":\"שלום\"}\r\n\r\n".as_bytes();
        let mut p = SseParser::default();
        // Split in the middle of a multi-byte Hebrew character.
        let cut = payload.iter().position(|&b| b >= 0x80).unwrap() + 1;
        assert!(p.push(&payload[..cut]).is_empty());
        assert_eq!(p.push(&payload[cut..]), vec!["{\"t\":\"שלום\"}".to_string()]);
    }

    #[test]
    fn titles_are_first_line_truncated() {
        assert_eq!(make_title("short\nsecond line"), "short");
        assert_eq!(make_title(&"א".repeat(80)).chars().count(), TITLE_CHARS + 1);
    }
}
