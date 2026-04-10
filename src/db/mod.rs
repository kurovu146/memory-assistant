use rusqlite::{Connection, params};
use std::sync::Mutex;
use tracing::info;

pub struct Database {
    conn: Mutex<Connection>,
}

impl Database {
    pub fn open(path: &str) -> Result<Self, rusqlite::Error> {
        let conn = Connection::open(path)?;

        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000; PRAGMA foreign_keys=ON;")?;

        // Memory facts table
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                fact TEXT NOT NULL,
                category TEXT NOT NULL DEFAULT 'general',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed_at TEXT
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_facts_fts USING fts5(
                fact,
                content='memory_facts',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS memory_facts_ai AFTER INSERT ON memory_facts BEGIN
                INSERT INTO memory_facts_fts(rowid, fact) VALUES (new.id, new.fact);
            END;

            CREATE TRIGGER IF NOT EXISTS memory_facts_ad AFTER DELETE ON memory_facts BEGIN
                INSERT INTO memory_facts_fts(memory_facts_fts, rowid, fact) VALUES('delete', old.id, old.fact);
            END;
            "
        )?;

        // Add embedding column to memory_facts (idempotent — .ok() ignores "duplicate column" error)
        conn.execute_batch(
            "ALTER TABLE memory_facts ADD COLUMN embedding BLOB;"
        ).ok();

        // Knowledge documents table
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS knowledge_documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                tags TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_docs_fts USING fts5(
                title, content,
                content='knowledge_documents',
                content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS knowledge_docs_ai AFTER INSERT ON knowledge_documents BEGIN
                INSERT INTO knowledge_docs_fts(rowid, title, content) VALUES (new.id, new.title, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_docs_ad AFTER DELETE ON knowledge_documents BEGIN
                INSERT INTO knowledge_docs_fts(knowledge_docs_fts, rowid, title, content) VALUES('delete', old.id, old.title, old.content);
            END;
            "
        )?;

        // Entities & knowledge graph
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, name, entity_type)
            );

            CREATE TABLE IF NOT EXISTS entity_mentions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER NOT NULL REFERENCES entities(id),
                source_type TEXT NOT NULL,
                source_id INTEGER NOT NULL,
                context TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            "
        )?;

        // Knowledge chunks (for semantic search)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS knowledge_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                chunk_index INTEGER NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding BLOB,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(doc_id, chunk_index)
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS knowledge_chunks_fts USING fts5(
                content, content='knowledge_chunks', content_rowid='id'
            );

            CREATE TRIGGER IF NOT EXISTS knowledge_chunks_ai AFTER INSERT ON knowledge_chunks BEGIN
                INSERT INTO knowledge_chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_chunks_ad AFTER DELETE ON knowledge_chunks BEGIN
                INSERT INTO knowledge_chunks_fts(knowledge_chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS knowledge_chunks_au AFTER UPDATE OF content ON knowledge_chunks BEGIN
                INSERT INTO knowledge_chunks_fts(knowledge_chunks_fts, rowid, content) VALUES('delete', old.id, old.content);
                INSERT INTO knowledge_chunks_fts(rowid, content) VALUES (new.id, new.content);
            END;
            "
        )?;

        // Categories (dynamic, per-user)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS categories (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, name)
            );"
        )?;

        // Memory ↔ KB links
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS memory_kb_links (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id INTEGER NOT NULL REFERENCES memory_facts(id) ON DELETE CASCADE,
                doc_id INTEGER NOT NULL REFERENCES knowledge_documents(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(fact_id, doc_id)
            );"
        )?;

        // Fact-to-fact relations (semantic similarity links)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS fact_relations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_id_1 INTEGER NOT NULL REFERENCES memory_facts(id) ON DELETE CASCADE,
                fact_id_2 INTEGER NOT NULL REFERENCES memory_facts(id) ON DELETE CASCADE,
                similarity REAL NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(fact_id_1, fact_id_2),
                CHECK(fact_id_1 < fact_id_2)
            );

            CREATE INDEX IF NOT EXISTS idx_fact_relations_1 ON fact_relations(fact_id_1);
            CREATE INDEX IF NOT EXISTS idx_fact_relations_2 ON fact_relations(fact_id_2);"
        )?;

        // Pending approval queue (for non-whitelisted users' write requests in groups)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS pending_items (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                scope_id INTEGER NOT NULL,
                requested_by INTEGER NOT NULL,
                tool_name TEXT NOT NULL,
                args_json TEXT NOT NULL,
                summary TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );"
        )?;

        // Usage log (cost tracking per model)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS usage_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model TEXT NOT NULL,
                provider TEXT NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cache_creation_tokens INTEGER NOT NULL DEFAULT 0,
                cache_read_tokens INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );"
        )?;

        // User preferences
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS user_preferences (
                user_id INTEGER PRIMARY KEY,
                model TEXT NOT NULL DEFAULT 'claude-haiku-4-5-20251001',
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            );"
        )?;

        // Conversation sessions
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                last_active_at TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS session_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id),
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            "
        )?;

        info!("Database initialized: {path}");
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    // --- Memory ---

    pub fn save_fact(&self, user_id: u64, fact: &str, category: &str) -> Result<i64, String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO memory_facts (user_id, fact, category) VALUES (?1, ?2, ?3)",
            params![user_id as i64, fact, category],
        )
        .map_err(|e| e.to_string())?;
        Ok(conn.last_insert_rowid())
    }

    pub fn search_facts(&self, user_id: u64, keyword: &str) -> Result<Vec<(i64, String, String)>, String> {
        let conn = self.conn.lock().unwrap();

        // Try FTS5 first, fall back to LIKE
        let results: Vec<(i64, String, String)> = conn
            .prepare(
                "SELECT mf.id, mf.fact, mf.category FROM memory_facts mf
                 JOIN memory_facts_fts fts ON mf.id = fts.rowid
                 WHERE fts.fact MATCH ?1 AND mf.user_id = ?2
                 ORDER BY rank LIMIT 20"
            )
            .and_then(|mut stmt| {
                let rows = stmt.query_map(params![keyword, user_id as i64], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?;
                rows.collect()
            })
            .unwrap_or_else(|_| {
                // Fallback to LIKE
                conn.prepare(
                    "SELECT id, fact, category FROM memory_facts
                     WHERE user_id = ?1 AND fact LIKE '%' || ?2 || '%'
                     ORDER BY created_at DESC LIMIT 20"
                )
                .and_then(|mut stmt| {
                    let rows = stmt.query_map(params![user_id as i64, keyword], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                    })?;
                    rows.collect()
                })
                .unwrap_or_default()
            });

        // Update access count
        for (id, _, _) in &results {
            let _ = conn.execute(
                "UPDATE memory_facts SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?1",
                params![id],
            );
        }

        Ok(results)
    }

    pub fn list_facts(&self, user_id: u64, category: Option<&str>) -> Result<Vec<(i64, String, String)>, String> {
        let conn = self.conn.lock().unwrap();
        let (sql, p): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = match category {
            Some(cat) => (
                "SELECT id, fact, category FROM memory_facts WHERE user_id = ?1 AND category = ?2 ORDER BY created_at DESC LIMIT 30",
                vec![Box::new(user_id as i64), Box::new(cat.to_string())],
            ),
            None => (
                "SELECT id, fact, category FROM memory_facts WHERE user_id = ?1 ORDER BY created_at DESC LIMIT 30",
                vec![Box::new(user_id as i64)],
            ),
        };

        let mut stmt = conn.prepare(sql).map_err(|e| e.to_string())?;
        let params_refs: Vec<&dyn rusqlite::types::ToSql> = p.iter().map(|b| b.as_ref()).collect();
        let rows = stmt
            .query_map(params_refs.as_slice(), |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })
            .map_err(|e| e.to_string())?;

        rows.collect::<Result<Vec<_>, _>>().map_err(|e| e.to_string())
    }

    pub fn update_fact(&self, user_id: u64, fact_id: i64, new_fact: &str) -> Result<bool, String> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute(
                "UPDATE memory_facts SET fact = ?1 WHERE id = ?2 AND user_id = ?3",
                params![new_fact, fact_id, user_id as i64],
            )
            .map_err(|e| e.to_string())?;
        if rows > 0 {
            // Re-index FTS
            let _ = conn.execute(
                "INSERT INTO memory_facts_fts(memory_facts_fts, rowid, fact) VALUES('delete', ?1, '')",
                params![fact_id],
            );
            let _ = conn.execute(
                "INSERT INTO memory_facts_fts(rowid, fact) VALUES (?1, ?2)",
                params![fact_id, new_fact],
            );
        }
        Ok(rows > 0)
    }

    pub fn delete_fact(&self, user_id: u64, fact_id: i64) -> Result<bool, String> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute(
                "DELETE FROM memory_facts WHERE id = ?1 AND user_id = ?2",
                params![fact_id, user_id as i64],
            )
            .map_err(|e| e.to_string())?;
        Ok(rows > 0)
    }

    // --- Fact Embeddings & Relations ---

    pub fn update_fact_embedding(&self, fact_id: i64, embedding: &[u8]) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "UPDATE memory_facts SET embedding = ?1 WHERE id = ?2",
            params![embedding, fact_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Load all fact embeddings for a user (for cosine similarity search).
    /// Returns (fact_id, fact_text, category, embedding_bytes).
    pub fn load_all_fact_embeddings(
        &self,
        user_id: u64,
    ) -> Result<Vec<(i64, String, String, Vec<u8>)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT id, fact, category, embedding FROM memory_facts
             WHERE user_id = ?1 AND embedding IS NOT NULL"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![user_id as i64], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    /// Link two facts with a similarity score. Enforces fact_id_1 < fact_id_2.
    pub fn link_facts(&self, id_a: i64, id_b: i64, similarity: f32) -> Result<(), String> {
        let (lo, hi) = if id_a < id_b { (id_a, id_b) } else { (id_b, id_a) };
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR REPLACE INTO fact_relations (fact_id_1, fact_id_2, similarity) VALUES (?1, ?2, ?3)",
            params![lo, hi, similarity as f64],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Get facts related to a given fact_id. Returns (related_fact_id, fact_text, category, similarity).
    pub fn get_related_facts(&self, fact_id: i64) -> Result<Vec<(i64, String, String, f64)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT mf.id, mf.fact, mf.category, fr.similarity
             FROM fact_relations fr
             JOIN memory_facts mf ON mf.id = CASE
                 WHEN fr.fact_id_1 = ?1 THEN fr.fact_id_2
                 ELSE fr.fact_id_1
             END
             WHERE fr.fact_id_1 = ?1 OR fr.fact_id_2 = ?1
             ORDER BY fr.similarity DESC"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![fact_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    /// Delete all relations for a given fact_id (used before recomputing after edit).
    pub fn delete_fact_relations(&self, fact_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "DELETE FROM fact_relations WHERE fact_id_1 = ?1 OR fact_id_2 = ?1",
            params![fact_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Batch-link multiple fact pairs in a single transaction.
    pub fn link_facts_batch(&self, links: &[(i64, i64, f32)]) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch("BEGIN").map_err(|e| e.to_string())?;
        for &(id_a, id_b, similarity) in links {
            let (lo, hi) = if id_a < id_b { (id_a, id_b) } else { (id_b, id_a) };
            if let Err(e) = conn.execute(
                "INSERT OR REPLACE INTO fact_relations (fact_id_1, fact_id_2, similarity) VALUES (?1, ?2, ?3)",
                params![lo, hi, similarity as f64],
            ) {
                let _ = conn.execute_batch("ROLLBACK");
                return Err(e.to_string());
            }
        }
        conn.execute_batch("COMMIT").map_err(|e| e.to_string())?;
        Ok(())
    }

    /// Get facts that don't have embeddings yet. Returns (id, fact_text).
    pub fn get_unembedded_facts(&self, user_id: u64) -> Result<Vec<(i64, String)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT id, fact FROM memory_facts WHERE user_id = ?1 AND embedding IS NULL"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![user_id as i64], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    /// Get all user_ids that have facts (for migration across all users).
    pub fn get_fact_user_ids(&self) -> Result<Vec<u64>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare("SELECT DISTINCT user_id FROM memory_facts")
            .and_then(|mut stmt| {
                let rows = stmt.query_map([], |row| {
                    let uid: i64 = row.get(0)?;
                    Ok(uid as u64)
                })?;
                rows.collect()
            })
            .map_err(|e| e.to_string())
    }

    // --- Categories ---

    const DEFAULT_CATEGORIES: &'static [&'static str] = &[
        "preference", "decision", "personal", "technical", "project", "workflow", "general",
    ];

    pub fn ensure_default_categories(&self, user_id: u64) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        for cat in Self::DEFAULT_CATEGORIES {
            conn.execute(
                "INSERT OR IGNORE INTO categories (user_id, name) VALUES (?1, ?2)",
                params![user_id as i64, cat],
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    pub fn list_categories(&self, user_id: u64) -> Result<Vec<String>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare("SELECT name FROM categories WHERE user_id = ?1 ORDER BY name")
            .and_then(|mut stmt| {
                let rows = stmt.query_map(params![user_id as i64], |row| row.get(0))?;
                rows.collect()
            })
            .map_err(|e| e.to_string())
    }

    pub fn add_category(&self, user_id: u64, name: &str) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO categories (user_id, name) VALUES (?1, ?2)",
            params![user_id as i64, name],
        )
        .map_err(|e| {
            if e.to_string().contains("UNIQUE") {
                format!("Category '{name}' already exists.")
            } else {
                e.to_string()
            }
        })?;
        Ok(())
    }

    pub fn delete_category(&self, user_id: u64, name: &str) -> Result<bool, String> {
        let conn = self.conn.lock().unwrap();
        // Check if any facts use this category
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM memory_facts WHERE user_id = ?1 AND category = ?2",
                params![user_id as i64, name],
                |row| row.get(0),
            )
            .map_err(|e| e.to_string())?;
        if count > 0 {
            return Err(format!(
                "Cannot delete category '{name}': {count} fact(s) still use it."
            ));
        }
        let rows = conn
            .execute(
                "DELETE FROM categories WHERE user_id = ?1 AND name = ?2",
                params![user_id as i64, name],
            )
            .map_err(|e| e.to_string())?;
        Ok(rows > 0)
    }

    // --- Memory ↔ KB Links ---

    pub fn link_fact_to_doc(&self, fact_id: i64, doc_id: i64) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO memory_kb_links (fact_id, doc_id) VALUES (?1, ?2)",
            params![fact_id, doc_id],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    pub fn get_fact_links(&self, fact_id: i64) -> Result<Vec<(i64, String)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT kd.id, kd.title FROM memory_kb_links mkl
             JOIN knowledge_documents kd ON mkl.doc_id = kd.id
             WHERE mkl.fact_id = ?1"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![fact_id], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    pub fn get_doc_linked_facts(&self, doc_id: i64) -> Result<Vec<(i64, String, String)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT mf.id, mf.fact, mf.category FROM memory_kb_links mkl
             JOIN memory_facts mf ON mkl.fact_id = mf.id
             WHERE mkl.doc_id = ?1"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![doc_id], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    // --- Memory context for system prompt ---

    pub fn build_memory_context(&self, user_id: u64) -> String {
        let facts = self.list_facts(user_id, None).unwrap_or_default();
        if facts.is_empty() {
            return String::new();
        }

        let mut grouped: std::collections::HashMap<String, Vec<String>> = std::collections::HashMap::new();
        for (_, fact, category) in &facts {
            grouped
                .entry(category.clone())
                .or_default()
                .push(fact.clone());
        }

        let mut ctx = String::new();

        // "preference" category loaded first — communication rules, conventions, identity
        if let Some(prefs) = grouped.remove("preference") {
            ctx.push_str("\n--- CORE PREFERENCES (always active, never delete this category) ---\n");
            for item in &prefs {
                ctx.push_str(&format!("- {item}\n"));
            }
            ctx.push_str("--- END CORE PREFERENCES ---\n");
        }

        // Other categories
        if !grouped.is_empty() {
            ctx.push_str("\n--- MEMORY ---\n");
            for (cat, items) in &grouped {
                ctx.push_str(&format!("\n[{cat}]\n"));
                for item in items {
                    ctx.push_str(&format!("- {item}\n"));
                }
            }
            ctx.push_str("\n--- END MEMORY ---\n");
        }
        ctx
    }

    // --- Conversation history ---

    /// Get or create the active session for a user. Returns session_id.
    pub fn get_or_create_session(&self, user_id: u64) -> String {
        let conn = self.conn.lock().unwrap();
        let existing: Option<String> = conn
            .query_row(
                "SELECT id FROM sessions WHERE user_id = ?1 ORDER BY last_active_at DESC LIMIT 1",
                params![user_id as i64],
                |row| row.get(0),
            )
            .ok();

        if let Some(id) = existing {
            let _ = conn.execute(
                "UPDATE sessions SET last_active_at = datetime('now') WHERE id = ?1",
                params![&id],
            );
            return id;
        }

        let id = format!("{}-{}", user_id, chrono::Utc::now().timestamp());
        let _ = conn.execute(
            "INSERT INTO sessions (id, user_id) VALUES (?1, ?2)",
            params![&id, user_id as i64],
        );
        id
    }

    /// Load recent conversation history for a session (last N user+assistant message pairs).
    pub fn load_history(&self, session_id: &str, max_pairs: usize) -> Vec<(String, String)> {
        let conn = self.conn.lock().unwrap();
        let limit = (max_pairs * 2) as i64;
        let mut stmt = match conn.prepare(
            "SELECT role, content FROM session_messages WHERE session_id = ?1 ORDER BY id DESC LIMIT ?2"
        ) {
            Ok(s) => s,
            Err(_) => return vec![],
        };
        let rows: Vec<(String, String)> = stmt
            .query_map(params![session_id, limit], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })
            .map(|r| r.collect::<Result<Vec<_>, _>>().unwrap_or_default())
            .unwrap_or_default();

        let mut result = rows;
        result.reverse();
        result
    }


    /// Append a message to the session history.
    pub fn append_message(&self, session_id: &str, role: &str, content: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO session_messages (session_id, role, content) VALUES (?1, ?2, ?3)",
            params![session_id, role, content],
        );
    }

    // --- Knowledge Documents ---

    pub fn save_document(
        &self,
        user_id: u64,
        title: &str,
        content: &str,
        source: Option<&str>,
        tags: Option<&str>,
    ) -> Result<i64, String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO knowledge_documents (user_id, title, content, source, tags) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![user_id as i64, title, content, source, tags],
        )
        .map_err(|e| e.to_string())?;
        Ok(conn.last_insert_rowid())
    }

    pub fn search_documents(
        &self,
        user_id: u64,
        query: &str,
    ) -> Result<Vec<(i64, String, String, Option<String>)>, String> {
        let conn = self.conn.lock().unwrap();

        // FTS5 search with snippet
        let results = conn
            .prepare(
                "SELECT kd.id, kd.title, snippet(knowledge_docs_fts, 1, '**', '**', '...', 40), kd.source
                 FROM knowledge_documents kd
                 JOIN knowledge_docs_fts fts ON kd.id = fts.rowid
                 WHERE knowledge_docs_fts MATCH ?1 AND kd.user_id = ?2
                 ORDER BY rank LIMIT 10"
            )
            .and_then(|mut stmt| {
                let rows = stmt.query_map(params![query, user_id as i64], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                })?;
                rows.collect()
            })
            .unwrap_or_else(|_| {
                // Fallback to LIKE
                conn.prepare(
                    "SELECT id, title, substr(content, 1, 200), source
                     FROM knowledge_documents
                     WHERE user_id = ?1 AND (title LIKE '%' || ?2 || '%' OR content LIKE '%' || ?2 || '%')
                     ORDER BY created_at DESC LIMIT 10"
                )
                .and_then(|mut stmt| {
                    let rows = stmt.query_map(params![user_id as i64, query], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
                    })?;
                    rows.collect()
                })
                .unwrap_or_default()
            });

        Ok(results)
    }

    // --- Entities ---

    pub fn save_entity(&self, user_id: u64, name: &str, entity_type: &str) -> Result<i64, String> {
        let conn = self.conn.lock().unwrap();
        // INSERT OR IGNORE for unique constraint, then get the id
        conn.execute(
            "INSERT OR IGNORE INTO entities (user_id, name, entity_type) VALUES (?1, ?2, ?3)",
            params![user_id as i64, name, entity_type],
        )
        .map_err(|e| e.to_string())?;

        conn.query_row(
            "SELECT id FROM entities WHERE user_id = ?1 AND name = ?2 AND entity_type = ?3",
            params![user_id as i64, name, entity_type],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())
    }

    pub fn add_entity_mention(
        &self,
        entity_id: i64,
        source_type: &str,
        source_id: i64,
        context: Option<&str>,
    ) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO entity_mentions (entity_id, source_type, source_id, context) VALUES (?1, ?2, ?3, ?4)",
            params![entity_id, source_type, source_id, context],
        )
        .map_err(|e| e.to_string())?;
        Ok(())
    }

    /// List all knowledge documents for a user. Returns (id, title, source, created_at, chunk_count).
    pub fn list_documents(&self, user_id: u64) -> Result<Vec<(i64, String, Option<String>, String, i64)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT kd.id, kd.title, kd.source, kd.created_at,
                    (SELECT COUNT(*) FROM knowledge_chunks kc WHERE kc.doc_id = kd.id) as chunk_count
             FROM knowledge_documents kd
             WHERE kd.user_id = ?1
             ORDER BY kd.created_at DESC LIMIT 50"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![user_id as i64], |row| {
                Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    // --- Knowledge Chunks ---

    pub fn save_chunks(
        &self,
        doc_id: i64,
        chunks: &[(usize, usize, usize, &str)], // (chunk_index, start_line, end_line, content)
    ) -> Result<Vec<i64>, String> {
        let conn = self.conn.lock().unwrap();
        let mut ids = Vec::with_capacity(chunks.len());
        for &(chunk_index, start_line, end_line, content) in chunks {
            conn.execute(
                "INSERT OR REPLACE INTO knowledge_chunks (doc_id, chunk_index, start_line, end_line, content) VALUES (?1, ?2, ?3, ?4, ?5)",
                params![doc_id, chunk_index as i64, start_line as i64, end_line as i64, content],
            )
            .map_err(|e| e.to_string())?;
            ids.push(conn.last_insert_rowid());
        }
        Ok(ids)
    }

    pub fn update_chunk_embeddings(
        &self,
        chunk_ids: &[i64],
        embeddings: &[Vec<u8>],
    ) -> Result<(), String> {
        let conn = self.conn.lock().unwrap();
        for (id, blob) in chunk_ids.iter().zip(embeddings.iter()) {
            conn.execute(
                "UPDATE knowledge_chunks SET embedding = ?1 WHERE id = ?2",
                params![blob, id],
            )
            .map_err(|e| e.to_string())?;
        }
        Ok(())
    }

    /// FTS5 search on knowledge_chunks. Returns (chunk_id, doc_id, title, content, start_line, end_line, source, rank).
    pub fn search_chunks_fts(
        &self,
        user_id: u64,
        query: &str,
    ) -> Result<Vec<(i64, i64, String, String, i64, i64, Option<String>, f64)>, String> {
        let conn = self.conn.lock().unwrap();
        // Escape FTS5 special chars by wrapping in double quotes
        let escaped = format!("\"{}\"", query.replace('"', "\"\""));
        conn.prepare(
            "SELECT kc.id, kc.doc_id, kd.title, kc.content, kc.start_line, kc.end_line, kd.source, fts.rank
             FROM knowledge_chunks kc
             JOIN knowledge_chunks_fts fts ON kc.id = fts.rowid
             JOIN knowledge_documents kd ON kc.doc_id = kd.id
             WHERE knowledge_chunks_fts MATCH ?1 AND kd.user_id = ?2
             ORDER BY fts.rank LIMIT 20"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![escaped, user_id as i64], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    /// Load all chunk embeddings for a user (for brute-force cosine similarity).
    /// Returns (chunk_id, doc_id, title, content, start_line, end_line, source, embedding_bytes).
    pub fn load_all_embeddings(
        &self,
        user_id: u64,
    ) -> Result<Vec<(i64, i64, String, String, i64, i64, Option<String>, Vec<u8>)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT kc.id, kc.doc_id, kd.title, kc.content, kc.start_line, kc.end_line, kd.source, kc.embedding
             FROM knowledge_chunks kc
             JOIN knowledge_documents kd ON kc.doc_id = kd.id
             WHERE kd.user_id = ?1 AND kc.embedding IS NOT NULL"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![user_id as i64], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get(2)?,
                    row.get(3)?,
                    row.get(4)?,
                    row.get(5)?,
                    row.get(6)?,
                    row.get(7)?,
                ))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    /// Get document IDs that have no chunks yet (for migration).
    pub fn get_unchunked_doc_ids(&self) -> Result<Vec<(i64, String)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT kd.id, kd.content FROM knowledge_documents kd
             WHERE NOT EXISTS (SELECT 1 FROM knowledge_chunks kc WHERE kc.doc_id = kd.id)"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map([], |row| {
                Ok((row.get(0)?, row.get(1)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    pub fn get_chunk_content(&self, chunk_id: i64) -> Result<String, String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT content FROM knowledge_chunks WHERE id = ?1",
            params![chunk_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())
    }

    /// Patch document + affected chunks: find/replace text.
    /// Returns (updated_doc_content, affected_chunk_ids).
    pub fn patch_document(
        &self,
        user_id: u64,
        doc_id: i64,
        old_text: &str,
        new_text: &str,
    ) -> Result<Vec<i64>, String> {
        let conn = self.conn.lock().unwrap();

        // 1. Update document content
        let doc_content: String = conn
            .query_row(
                "SELECT content FROM knowledge_documents WHERE id = ?1 AND user_id = ?2",
                params![doc_id, user_id as i64],
                |row| row.get(0),
            )
            .map_err(|_| format!("Document #{doc_id} not found."))?;

        if !doc_content.contains(old_text) {
            return Err(format!("Text \"{old_text}\" not found in document #{doc_id}."));
        }

        let new_content = doc_content.replace(old_text, new_text);
        conn.execute(
            "UPDATE knowledge_documents SET content = ?1 WHERE id = ?2",
            params![&new_content, doc_id],
        )
        .map_err(|e| e.to_string())?;

        // Also update FTS for the document
        let _ = conn.execute(
            "INSERT INTO knowledge_docs_fts(knowledge_docs_fts, rowid, title, content) VALUES('delete', ?1, '', ?2)",
            params![doc_id, &doc_content],
        );
        let title: String = conn
            .query_row("SELECT title FROM knowledge_documents WHERE id = ?1", params![doc_id], |row| row.get(0))
            .unwrap_or_default();
        let _ = conn.execute(
            "INSERT INTO knowledge_docs_fts(rowid, title, content) VALUES (?1, ?2, ?3)",
            params![doc_id, &title, &new_content],
        );

        // 2. Find and update affected chunks (trigger handles FTS re-index)
        let chunk_ids: Vec<i64> = conn
            .prepare(
                "SELECT id FROM knowledge_chunks WHERE doc_id = ?1 AND content LIKE '%' || ?2 || '%'",
            )
            .and_then(|mut stmt| {
                let rows = stmt.query_map(params![doc_id, old_text], |row| row.get(0))?;
                rows.collect()
            })
            .map_err(|e| e.to_string())?;

        for chunk_id in &chunk_ids {
            let old_chunk: String = conn
                .query_row(
                    "SELECT content FROM knowledge_chunks WHERE id = ?1",
                    params![chunk_id],
                    |row| row.get(0),
                )
                .map_err(|e| e.to_string())?;
            let new_chunk = old_chunk.replace(old_text, new_text);
            conn.execute(
                "UPDATE knowledge_chunks SET content = ?1, embedding = NULL WHERE id = ?2",
                params![&new_chunk, chunk_id],
            )
            .map_err(|e| e.to_string())?;
        }

        Ok(chunk_ids)
    }

    // --- Chat Preferences ---

    pub fn get_chat_model(&self, scope_id: u64) -> String {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT model FROM user_preferences WHERE user_id = ?1",
            params![scope_id as i64],
            |row| row.get(0),
        )
        .unwrap_or_else(|_| crate::provider::model_registry::DEFAULT_MODEL.to_string())
    }

    pub fn set_chat_model(&self, scope_id: u64, model_id: &str) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO user_preferences (user_id, model, updated_at) VALUES (?1, ?2, datetime('now'))
             ON CONFLICT(user_id) DO UPDATE SET model = ?2, updated_at = datetime('now')",
            params![scope_id as i64, model_id],
        );
    }

    /// Get full content of a knowledge document by ID.
    pub fn get_document(&self, user_id: u64, doc_id: i64) -> Result<(String, String, Option<String>, Option<String>), String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT title, content, source, tags FROM knowledge_documents WHERE id = ?1 AND user_id = ?2",
            params![doc_id, user_id as i64],
            |row| Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?)),
        )
        .map_err(|e| e.to_string())
    }

    /// Delete a knowledge document (chunks cascade-deleted automatically).
    pub fn delete_document(&self, user_id: u64, doc_id: i64) -> Result<bool, String> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute(
                "DELETE FROM knowledge_documents WHERE id = ?1 AND user_id = ?2",
                params![doc_id, user_id as i64],
            )
            .map_err(|e| e.to_string())?;
        Ok(rows > 0)
    }

    // --- Pending Approval Queue ---

    pub fn save_pending(
        &self,
        scope_id: u64,
        requested_by: u64,
        tool_name: &str,
        args_json: &str,
        summary: &str,
    ) -> Result<i64, String> {
        let conn = self.conn.lock().unwrap();
        conn.execute(
            "INSERT INTO pending_items (scope_id, requested_by, tool_name, args_json, summary) VALUES (?1, ?2, ?3, ?4, ?5)",
            params![scope_id as i64, requested_by as i64, tool_name, args_json, summary],
        )
        .map_err(|e| e.to_string())?;
        Ok(conn.last_insert_rowid())
    }

    pub fn list_pending(&self, scope_id: u64) -> Result<Vec<(i64, u64, String, String, String)>, String> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT id, requested_by, tool_name, summary, created_at FROM pending_items WHERE scope_id = ?1 ORDER BY created_at ASC"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map(params![scope_id as i64], |row| {
                let rb: i64 = row.get(1)?;
                Ok((row.get(0)?, rb as u64, row.get(2)?, row.get(3)?, row.get(4)?))
            })?;
            rows.collect()
        })
        .map_err(|e| e.to_string())
    }

    pub fn get_pending(&self, id: i64) -> Result<(u64, u64, String, String, String), String> {
        let conn = self.conn.lock().unwrap();
        conn.query_row(
            "SELECT scope_id, requested_by, tool_name, args_json, summary FROM pending_items WHERE id = ?1",
            params![id],
            |row| {
                let s: i64 = row.get(0)?;
                let r: i64 = row.get(1)?;
                Ok((s as u64, r as u64, row.get(2)?, row.get(3)?, row.get(4)?))
            },
        )
        .map_err(|e| e.to_string())
    }

    pub fn delete_pending(&self, id: i64) -> Result<bool, String> {
        let conn = self.conn.lock().unwrap();
        let rows = conn
            .execute("DELETE FROM pending_items WHERE id = ?1", params![id])
            .map_err(|e| e.to_string())?;
        Ok(rows > 0)
    }

    // --- Entities ---

    // --- Usage Log ---

    pub fn log_usage(
        &self,
        model: &str,
        provider: &str,
        prompt_tokens: u32,
        completion_tokens: u32,
        cache_creation_tokens: u32,
        cache_read_tokens: u32,
    ) {
        let conn = self.conn.lock().unwrap();
        let _ = conn.execute(
            "INSERT INTO usage_log (model, provider, prompt_tokens, completion_tokens, cache_creation_tokens, cache_read_tokens) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            params![model, provider, prompt_tokens, completion_tokens, cache_creation_tokens, cache_read_tokens],
        );
    }

    /// Get usage summary grouped by model for the current month.
    pub fn get_monthly_usage(&self) -> Vec<(String, String, u64, u64, u64, u64, u64)> {
        let conn = self.conn.lock().unwrap();
        conn.prepare(
            "SELECT model, provider,
                    SUM(prompt_tokens), SUM(completion_tokens),
                    SUM(cache_creation_tokens), SUM(cache_read_tokens),
                    COUNT(*)
             FROM usage_log
             WHERE created_at >= strftime('%Y-%m-01', 'now')
             GROUP BY model
             ORDER BY SUM(prompt_tokens) + SUM(completion_tokens) DESC"
        )
        .and_then(|mut stmt| {
            let rows = stmt.query_map([], |row| {
                Ok((
                    row.get(0)?,
                    row.get(1)?,
                    row.get::<_, i64>(2)? as u64,
                    row.get::<_, i64>(3)? as u64,
                    row.get::<_, i64>(4)? as u64,
                    row.get::<_, i64>(5)? as u64,
                    row.get::<_, i64>(6)? as u64,
                ))
            })?;
            rows.collect()
        })
        .unwrap_or_default()
    }

    pub fn search_entities(
        &self,
        user_id: u64,
        query: &str,
    ) -> Result<Vec<(String, String, Vec<(String, i64, Option<String>)>)>, String> {
        let conn = self.conn.lock().unwrap();

        // Find matching entities
        let entities: Vec<(i64, String, String)> = conn
            .prepare(
                "SELECT id, name, entity_type FROM entities
                 WHERE user_id = ?1 AND name LIKE '%' || ?2 || '%'
                 ORDER BY name LIMIT 20"
            )
            .and_then(|mut stmt| {
                let rows = stmt.query_map(params![user_id as i64, query], |row| {
                    Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                })?;
                rows.collect()
            })
            .map_err(|e| e.to_string())?;

        // For each entity, get its mentions
        let mut results = Vec::new();
        for (entity_id, name, entity_type) in entities {
            let mentions: Vec<(String, i64, Option<String>)> = conn
                .prepare(
                    "SELECT source_type, source_id, context FROM entity_mentions
                     WHERE entity_id = ?1 ORDER BY created_at DESC LIMIT 10"
                )
                .and_then(|mut stmt| {
                    let rows = stmt.query_map(params![entity_id], |row| {
                        Ok((row.get(0)?, row.get(1)?, row.get(2)?))
                    })?;
                    rows.collect()
                })
                .unwrap_or_default();
            results.push((name, entity_type, mentions));
        }

        Ok(results)
    }
}
