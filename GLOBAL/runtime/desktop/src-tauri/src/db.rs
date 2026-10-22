//! Local chat history: a single SQLite file in the app-data folder.

use std::path::Path;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::Serialize;

use crate::error::Result;

/// Append-only: never edit a shipped migration, add a new one.
const MIGRATIONS: &[&str] = &[r#"
CREATE TABLE conversations (
    id          INTEGER PRIMARY KEY,
    title       TEXT    NOT NULL DEFAULT '',
    created_at  INTEGER NOT NULL,
    updated_at  INTEGER NOT NULL
);
CREATE TABLE messages (
    id              INTEGER PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT    NOT NULL CHECK (role IN ('user', 'assistant')),
    content         TEXT    NOT NULL,
    status          TEXT    NOT NULL DEFAULT 'complete'
                            CHECK (status IN ('complete', 'streaming', 'partial', 'error')),
    model           TEXT,
    created_at      INTEGER NOT NULL
);
CREATE INDEX idx_messages_conversation ON messages(conversation_id, id);
CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE VIRTUAL TABLE messages_fts USING fts5(content, content='messages', content_rowid='id');
CREATE TRIGGER messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
CREATE TRIGGER messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
END;
CREATE TRIGGER messages_au AFTER UPDATE OF content ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES ('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;
"#];

pub fn now_ms() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis() as i64).unwrap_or(0)
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Conversation {
    pub id: i64,
    pub title: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct Message {
    pub id: i64,
    pub conversation_id: i64,
    pub role: String,
    pub content: String,
    /// complete | streaming | partial (stopped or interrupted) | error
    pub status: String,
    pub model: Option<String>,
    pub created_at: i64,
}

#[derive(Serialize, Clone, Debug)]
#[serde(rename_all = "camelCase")]
pub struct SearchHit {
    pub message_id: i64,
    pub conversation_id: i64,
    pub conversation_title: String,
    /// Matches are wrapped in \u{2} ... \u{3} (control chars never found in real text,
    /// unlike brackets, which are common in code).
    pub snippet: String,
}

fn conversation_from_row(r: &Row) -> rusqlite::Result<Conversation> {
    Ok(Conversation { id: r.get(0)?, title: r.get(1)?, created_at: r.get(2)?, updated_at: r.get(3)? })
}

fn message_from_row(r: &Row) -> rusqlite::Result<Message> {
    Ok(Message {
        id: r.get(0)?,
        conversation_id: r.get(1)?,
        role: r.get(2)?,
        content: r.get(3)?,
        status: r.get(4)?,
        model: r.get(5)?,
        created_at: r.get(6)?,
    })
}

const MESSAGE_COLS: &str = "id, conversation_id, role, content, status, model, created_at";

pub struct Db {
    conn: Connection,
}

impl Db {
    /// Opens (or creates) the database, applies pending migrations, and marks any reply that
    /// was still streaming when the app last exited as `partial`.
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode = WAL; PRAGMA foreign_keys = ON;")?;
        let mut db = Db { conn };
        db.migrate()?;
        db.conn.execute("UPDATE messages SET status = 'partial' WHERE status = 'streaming'", [])?;
        Ok(db)
    }

    fn migrate(&mut self) -> Result<()> {
        let version: i64 = self.conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;
        for (i, sql) in MIGRATIONS.iter().enumerate().skip(version as usize) {
            let tx = self.conn.transaction()?;
            tx.execute_batch(sql)?;
            tx.execute_batch(&format!("PRAGMA user_version = {}", i + 1))?;
            tx.commit()?;
        }
        Ok(())
    }

    pub fn create_conversation(&self, title: &str) -> Result<Conversation> {
        let now = now_ms();
        self.conn.execute(
            "INSERT INTO conversations (title, created_at, updated_at) VALUES (?1, ?2, ?2)",
            params![title, now],
        )?;
        Ok(Conversation { id: self.conn.last_insert_rowid(), title: title.into(), created_at: now, updated_at: now })
    }

    pub fn conversation(&self, id: i64) -> Result<Option<Conversation>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id, title, created_at, updated_at FROM conversations WHERE id = ?1",
                [id],
                conversation_from_row,
            )
            .optional()?)
    }

    pub fn list_conversations(&self) -> Result<Vec<Conversation>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, title, created_at, updated_at FROM conversations ORDER BY updated_at DESC")?;
        let rows = stmt.query_map([], conversation_from_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn rename_conversation(&self, id: i64, title: &str) -> Result<()> {
        self.conn.execute("UPDATE conversations SET title = ?2 WHERE id = ?1", params![id, title])?;
        Ok(())
    }

    pub fn delete_conversation(&self, id: i64) -> Result<()> {
        self.conn.execute("DELETE FROM conversations WHERE id = ?1", [id])?;
        Ok(())
    }

    pub fn messages(&self, conversation_id: i64) -> Result<Vec<Message>> {
        let mut stmt = self.conn.prepare(&format!(
            "SELECT {MESSAGE_COLS} FROM messages WHERE conversation_id = ?1 ORDER BY id"
        ))?;
        let rows = stmt.query_map([conversation_id], message_from_row)?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn insert_message(
        &self,
        conversation_id: i64,
        role: &str,
        content: &str,
        status: &str,
        model: Option<&str>,
    ) -> Result<Message> {
        let now = now_ms();
        self.conn.execute(
            "INSERT INTO messages (conversation_id, role, content, status, model, created_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![conversation_id, role, content, status, model, now],
        )?;
        let id = self.conn.last_insert_rowid();
        self.conn.execute("UPDATE conversations SET updated_at = ?2 WHERE id = ?1", params![conversation_id, now])?;
        Ok(Message {
            id,
            conversation_id,
            role: role.into(),
            content: content.into(),
            status: status.into(),
            model: model.map(Into::into),
            created_at: now,
        })
    }

    pub fn update_message(&self, id: i64, content: &str, status: &str) -> Result<()> {
        self.conn.execute(
            "UPDATE messages SET content = ?2, status = ?3 WHERE id = ?1",
            params![id, content, status],
        )?;
        Ok(())
    }

    pub fn search(&self, query: &str, limit: u32) -> Result<Vec<SearchHit>> {
        // Quote as a single FTS5 phrase so user input can't be parsed as query syntax.
        let phrase = format!("\"{}\"", query.replace('"', "\"\""));
        let mut stmt = self.conn.prepare(
            "SELECT m.id, m.conversation_id, c.title, snippet(messages_fts, 0, char(2), char(3), '…', 12)
             FROM messages_fts
             JOIN messages m ON m.id = messages_fts.rowid
             JOIN conversations c ON c.id = m.conversation_id
             WHERE messages_fts MATCH ?1
             ORDER BY rank LIMIT ?2",
        )?;
        let rows = stmt.query_map(params![phrase, limit], |r| {
            Ok(SearchHit {
                message_id: r.get(0)?,
                conversation_id: r.get(1)?,
                conversation_title: r.get(2)?,
                snippet: r.get(3)?,
            })
        })?;
        Ok(rows.collect::<rusqlite::Result<_>>()?)
    }

    pub fn setting(&self, key: &str) -> Result<Option<String>> {
        Ok(self
            .conn
            .query_row("SELECT value FROM settings WHERE key = ?1", [key], |r| r.get(0))
            .optional()?)
    }

    pub fn set_setting(&self, key: &str, value: &str) -> Result<()> {
        self.conn.execute(
            "INSERT INTO settings (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key, value],
        )?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_search_and_cascade() {
        let db = Db::open(":memory:").unwrap();
        let c = db.create_conversation("").unwrap();
        db.insert_message(c.id, "user", "מהו three-way handshake?", "complete", None).unwrap();
        let a = db.insert_message(c.id, "assistant", "", "streaming", Some("m")).unwrap();
        db.update_message(a.id, "SYN, SYN-ACK, ACK", "complete").unwrap();

        let msgs = db.messages(c.id).unwrap();
        assert_eq!(msgs.len(), 2);
        assert_eq!(msgs[1].content, "SYN, SYN-ACK, ACK");

        let hits = db.search("handshake", 10).unwrap();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].snippet.contains("\u{2}handshake\u{3}"), "{:?}", hits[0].snippet);
        assert_eq!(db.search("SYN-ACK", 10).unwrap().len(), 1);
        assert_eq!(db.search("ACK\" OR x", 10).unwrap().len(), 0); // no query-syntax injection

        db.delete_conversation(c.id).unwrap();
        assert!(db.messages(c.id).unwrap().is_empty());
        assert!(db.search("handshake", 10).unwrap().is_empty());
    }

    #[test]
    fn settings_upsert() {
        let db = Db::open(":memory:").unwrap();
        assert_eq!(db.setting("k").unwrap(), None);
        db.set_setting("k", "a").unwrap();
        db.set_setting("k", "b").unwrap();
        assert_eq!(db.setting("k").unwrap().as_deref(), Some("b"));
    }
}
