use serde_json::json;

use crate::provider::{ToolDef, FunctionDef, ProviderPool};
use crate::tools;

/// Output from a tool execution — either plain text or text + image.
pub enum ToolOutput {
    Text(String),
    Image {
        text: String,
        image_base64: String,
        media_type: String,
    },
}

/// Registry of all available tools with definitions and executor
pub struct ToolRegistry;

impl ToolRegistry {
    /// Get tool definitions to send to LLM
    pub fn definitions() -> Vec<ToolDef> {
        vec![
            // --- Memory ---
            tool_def("memory_save",
                "Save an important fact to long-term memory for future conversations.",
                json!({
                    "type": "object",
                    "properties": {
                        "fact": { "type": "string", "description": "The fact to remember" },
                        "category": {
                            "type": "string",
                            "description": "Category of the fact (use category_list to see available categories)"
                        }
                    },
                    "required": ["fact"]
                }),
            ),
            tool_def("memory_search",
                "Search long-term memory for previously saved facts.",
                json!({
                    "type": "object",
                    "properties": {
                        "keyword": { "type": "string", "description": "Keyword to search for" }
                    },
                    "required": ["keyword"]
                }),
            ),
            tool_def("memory_list",
                "List all saved facts from long-term memory.",
                json!({
                    "type": "object",
                    "properties": {
                        "category": { "type": "string", "description": "Optional category filter" }
                    }
                }),
            ),
            tool_def("memory_delete",
                "Delete a fact from long-term memory by its ID.",
                json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer", "description": "The fact ID to delete" }
                    },
                    "required": ["id"]
                }),
            ),
            // --- Categories ---
            tool_def("category_list",
                "List all available memory categories for this user.",
                json!({ "type": "object", "properties": {} }),
            ),
            tool_def("category_add",
                "Create a new custom memory category.",
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Category name to create" }
                    },
                    "required": ["name"]
                }),
            ),
            tool_def("category_delete",
                "Delete an empty memory category. Fails if facts still use it.",
                json!({
                    "type": "object",
                    "properties": {
                        "name": { "type": "string", "description": "Category name to delete" }
                    },
                    "required": ["name"]
                }),
            ),
            // --- Knowledge ---
            tool_def("knowledge_save",
                "Save a document, article, note, or bookmark to the knowledge base. Entities (people, projects, technologies) are auto-extracted.",
                json!({
                    "type": "object",
                    "properties": {
                        "title": { "type": "string", "description": "Title of the document" },
                        "content": { "type": "string", "description": "Full content/text of the document" },
                        "source": { "type": "string", "description": "Source URL or reference (optional)" },
                        "tags": { "type": "string", "description": "Comma-separated tags (optional)" }
                    },
                    "required": ["title", "content"]
                }),
            ),
            tool_def("knowledge_search",
                "Search the knowledge base using hybrid semantic + keyword search. Returns relevant chunks with line numbers for citation.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query" }
                    },
                    "required": ["query"]
                }),
            ),
            tool_def("knowledge_list",
                "List all saved documents in the knowledge base. Shows titles, sources, chunk counts, and save dates.",
                json!({ "type": "object", "properties": {} }),
            ),
            tool_def("knowledge_get",
                "Get full content of a knowledge document by its ID. Use knowledge_list or knowledge_search to find the ID first.",
                json!({
                    "type": "object",
                    "properties": {
                        "doc_id": { "type": "integer", "description": "Document ID" }
                    },
                    "required": ["doc_id"]
                }),
            ),
            tool_def("knowledge_patch",
                "Patch a knowledge document by replacing specific text. Only affected chunks are re-indexed. Use this instead of delete+save when updating existing docs.",
                json!({
                    "type": "object",
                    "properties": {
                        "doc_id": { "type": "integer", "description": "Document ID to patch" },
                        "old_text": { "type": "string", "description": "Text to find in the document" },
                        "new_text": { "type": "string", "description": "Replacement text" }
                    },
                    "required": ["doc_id", "old_text", "new_text"]
                }),
            ),
            tool_def("knowledge_delete",
                "Delete a knowledge document and all its chunks by ID. Use knowledge_list to find the ID first.",
                json!({
                    "type": "object",
                    "properties": {
                        "doc_id": { "type": "integer", "description": "Document ID to delete" }
                    },
                    "required": ["doc_id"]
                }),
            ),
            tool_def("entity_search",
                "Search for entities (people, projects, technologies, concepts) in the knowledge graph. Shows which documents and facts mention them.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Entity name to search for" }
                    },
                    "required": ["query"]
                }),
            ),
            // --- Datetime ---
            tool_def("get_datetime",
                "Get current date and time in UTC and common timezones (Vietnam, US Eastern).",
                json!({ "type": "object", "properties": {} }),
            ),
            // --- System ---
            tool_def("bash",
                "Execute a bash command on the server. Use for git, npm, cargo, system info, or any terminal operation. Returns stdout + stderr.",
                json!({
                    "type": "object",
                    "properties": {
                        "command": { "type": "string", "description": "The bash command to execute" },
                        "timeout": { "type": "integer", "description": "Timeout in seconds (default 30, max 120)" }
                    },
                    "required": ["command"]
                }),
            ),
            tool_def("file_read",
                "Read a file from the filesystem. Returns content with line numbers. Use offset/limit for large files.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute or relative file path" },
                        "offset": { "type": "integer", "description": "Start reading from this line (0-indexed, optional)" },
                        "limit": { "type": "integer", "description": "Max number of lines to read (optional)" }
                    },
                    "required": ["path"]
                }),
            ),
            tool_def("file_write",
                "Write content to a file. Creates parent directories if needed. Overwrites existing file.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Absolute or relative file path" },
                        "content": { "type": "string", "description": "Content to write" }
                    },
                    "required": ["path", "content"]
                }),
            ),
            tool_def("file_list",
                "List directory contents. Shows files with sizes and subdirectories.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Directory path (default: home directory ~)" },
                        "recursive": { "type": "boolean", "description": "List recursively (default: false, max depth 5)" }
                    }
                }),
            ),
            tool_def("grep",
                "Search file contents using regex pattern (via ripgrep). Returns matching lines with line numbers.",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Regex pattern to search for" },
                        "path": { "type": "string", "description": "Directory or file to search in (default: home directory ~)" },
                        "include": { "type": "string", "description": "Glob pattern to filter files (e.g. '*.rs', '*.json')" },
                        "context": { "type": "integer", "description": "Lines of context around matches (default: 0)" }
                    },
                    "required": ["pattern"]
                }),
            ),
            tool_def("glob",
                "Find files by name pattern. Supports wildcards: * (any chars), ? (single char). Returns matching paths sorted by modification time (newest first).",
                json!({
                    "type": "object",
                    "properties": {
                        "pattern": { "type": "string", "description": "Glob pattern to match files (e.g. '*.json', '*.rs', 'kuro-*')" },
                        "path": { "type": "string", "description": "Directory to search in (default: home directory ~)" }
                    },
                    "required": ["pattern"]
                }),
            ),
            tool_def("image_read",
                "Read an image file from disk and analyze it using vision. Use this for image files (jpg, png, gif, webp) stored in ~/documents/. Returns the image for visual analysis.",
                json!({
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the image file" }
                    },
                    "required": ["path"]
                }),
            ),
        ]
    }

    /// Tools that modify memory/knowledge — restricted to whitelisted users in group chats.
    const WRITE_TOOLS: &'static [&'static str] = &[
        "memory_save", "memory_delete",
        "category_add", "category_delete",
        "knowledge_save", "knowledge_patch", "knowledge_delete",
    ];

    /// Execute a tool by name with given arguments.
    /// `kb_owner_id` is the knowledge base owner (user_id in private, chat_id in groups).
    /// `allowed_users` is used to restrict write tools in group chats.
    pub async fn execute(
        tool_name: &str,
        args_json: &str,
        user_id: u64,
        kb_owner_id: u64,
        db: &crate::db::Database,
        pool: &ProviderPool,
        embedding_client: Option<&crate::tools::EmbeddingClient>,
        allowed_users: &[u64],
    ) -> ToolOutput {
        // Non-whitelisted users: save write requests to pending queue for approval
        if Self::WRITE_TOOLS.contains(&tool_name)
            && !allowed_users.is_empty()
            && !allowed_users.contains(&user_id)
        {
            let args_parsed: serde_json::Value = serde_json::from_str(args_json).unwrap_or_default();
            let summary = Self::build_pending_summary(tool_name, &args_parsed);
            match db.save_pending(kb_owner_id, user_id, tool_name, args_json, &summary) {
                Ok(id) => return ToolOutput::Text(
                    format!("Request #{id} queued for approval: {summary}\nA whitelisted user can /approve {id} or /reject {id}.")
                ),
                Err(e) => return ToolOutput::Text(format!("Error saving request: {e}")),
            }
        }

        let args: serde_json::Value = serde_json::from_str(args_json).unwrap_or_default();

        let text_result = match tool_name {
            "memory_save" => {
                let fact = args["fact"].as_str().unwrap_or("");
                let category = args["category"].as_str().unwrap_or("general");
                tools::memory_save(db, kb_owner_id, fact, category).await
            }
            "memory_search" => {
                let keyword = args["keyword"].as_str().unwrap_or("");
                tools::memory_search(db, kb_owner_id, keyword).await
            }
            "memory_list" => {
                let category = args["category"].as_str();
                tools::memory_list(db, kb_owner_id, category).await
            }
            "memory_delete" => {
                let id = args["id"].as_i64().unwrap_or(0);
                match db.delete_fact(kb_owner_id, id) {
                    Ok(true) => format!("Deleted memory #{id}."),
                    Ok(false) => format!("Memory #{id} not found."),
                    Err(e) => format!("Error: {e}"),
                }
            }
            "category_list" => {
                let _ = db.ensure_default_categories(kb_owner_id);
                match db.list_categories(kb_owner_id) {
                    Ok(cats) if cats.is_empty() => "No categories found.".into(),
                    Ok(cats) => cats.join(", "),
                    Err(e) => format!("Error: {e}"),
                }
            }
            "category_add" => {
                let name = args["name"].as_str().unwrap_or("");
                if name.is_empty() {
                    "Error: name cannot be empty".into()
                } else {
                    let _ = db.ensure_default_categories(kb_owner_id);
                    match db.add_category(kb_owner_id, name) {
                        Ok(()) => format!("Category '{name}' created."),
                        Err(e) => format!("Error: {e}"),
                    }
                }
            }
            "category_delete" => {
                let name = args["name"].as_str().unwrap_or("");
                if name.is_empty() {
                    "Error: name cannot be empty".into()
                } else if name == "preference" {
                    "Error: category 'preference' is protected and cannot be deleted.".into()
                } else {
                    match db.delete_category(kb_owner_id, name) {
                        Ok(true) => format!("Category '{name}' deleted."),
                        Ok(false) => format!("Category '{name}' not found."),
                        Err(e) => format!("Error: {e}"),
                    }
                }
            }
            "knowledge_save" => {
                let title = args["title"].as_str().unwrap_or("");
                let content = args["content"].as_str().unwrap_or("");
                let source = args["source"].as_str();
                let tags = args["tags"].as_str();

                match tools::knowledge_save(db, kb_owner_id, title, content, source, tags, embedding_client).await {
                    Ok((doc_id, msg)) => {
                        // Auto-extract entities in background
                        let text = format!("{title}\n\n{content}");
                        let entity_count = tools::extract_and_link_entities(
                            pool, db, kb_owner_id, "document", doc_id, &text,
                        ).await;
                        if entity_count > 0 {
                            format!("{msg}\nExtracted {entity_count} entities.")
                        } else {
                            msg
                        }
                    }
                    Err(e) => format!("Error: {e}"),
                }
            }
            "knowledge_search" => {
                let query = args["query"].as_str().unwrap_or("");
                tools::knowledge_search(db, kb_owner_id, query, embedding_client).await
            }
            "knowledge_list" => {
                tools::knowledge_list(db, kb_owner_id).await
            }
            "knowledge_get" => {
                let doc_id = args["doc_id"].as_i64().unwrap_or(0);
                match db.get_document(kb_owner_id, doc_id) {
                    Ok((title, content, source, tags)) => {
                        let src = source.as_deref().unwrap_or("none");
                        let tgs = tags.as_deref().unwrap_or("none");
                        let mut out = format!("# {title}\nSource: {src}\nTags: {tgs}\n\n{content}");
                        // Append linked memory facts
                        if let Ok(linked) = db.get_doc_linked_facts(doc_id) {
                            if !linked.is_empty() {
                                out.push_str("\n\nLinked memories:");
                                for (fid, fact, _cat) in &linked {
                                    out.push_str(&format!("\n- [{fid}] {fact}"));
                                }
                            }
                        }
                        out
                    }
                    Err(_) => format!("Document #{doc_id} not found."),
                }
            }
            "knowledge_patch" => {
                let doc_id = args["doc_id"].as_i64().unwrap_or(0);
                let old_text = args["old_text"].as_str().unwrap_or("");
                let new_text = args["new_text"].as_str().unwrap_or("");
                if old_text.is_empty() {
                    "Error: old_text cannot be empty".into()
                } else {
                    tools::knowledge_patch(db, kb_owner_id, doc_id, old_text, new_text, embedding_client).await
                }
            }
            "knowledge_delete" => {
                let doc_id = args["doc_id"].as_i64().unwrap_or(0);
                match db.delete_document(kb_owner_id, doc_id) {
                    Ok(true) => format!("Deleted document #{doc_id} and all its chunks."),
                    Ok(false) => format!("Document #{doc_id} not found."),
                    Err(e) => format!("Error: {e}"),
                }
            }
            "entity_search" => {
                let query = args["query"].as_str().unwrap_or("");
                tools::entity_search(db, kb_owner_id, query).await
            }
            "get_datetime" => tools::get_datetime().await,
            "bash" => {
                let command = args["command"].as_str().unwrap_or("");
                let timeout = args["timeout"].as_u64().map(|t| t.min(120));
                tools::bash_exec(command, timeout).await
            }
            "file_read" => {
                let path = args["path"].as_str().unwrap_or("");
                let offset = args["offset"].as_u64().map(|v| v as usize);
                let limit = args["limit"].as_u64().map(|v| v as usize);
                tools::file_read(path, offset, limit).await
            }
            "file_write" => {
                let path = args["path"].as_str().unwrap_or("");
                let content = args["content"].as_str().unwrap_or("");
                tools::file_write(path, content).await
            }
            "file_list" => {
                let path = args["path"].as_str().unwrap_or("");
                let recursive = args["recursive"].as_bool().unwrap_or(false);
                tools::file_list(path, recursive).await
            }
            "grep" => {
                let pattern = args["pattern"].as_str().unwrap_or("");
                let path = args["path"].as_str();
                let include = args["include"].as_str();
                let context = args["context"].as_u64().map(|v| v as usize);
                tools::grep_search(pattern, path, include, context).await
            }
            "glob" => {
                let pattern = args["pattern"].as_str().unwrap_or("");
                let path = args["path"].as_str();
                tools::glob_search(pattern, path).await
            }
            "image_read" => {
                let path = args["path"].as_str().unwrap_or("");
                return Self::read_image(path).await;
            }
            _ => format!("Unknown tool: {tool_name}"),
        };

        ToolOutput::Text(text_result)
    }

    /// Build a human-readable summary for a pending write request.
    fn build_pending_summary(tool_name: &str, args: &serde_json::Value) -> String {
        match tool_name {
            "memory_save" => {
                let fact = args["fact"].as_str().unwrap_or("");
                let cat = args["category"].as_str().unwrap_or("general");
                let preview: String = fact.chars().take(100).collect();
                format!("[memory_save] [{cat}] {preview}")
            }
            "memory_delete" => {
                let id = args["id"].as_i64().unwrap_or(0);
                format!("[memory_delete] #{id}")
            }
            "knowledge_save" => {
                let title = args["title"].as_str().unwrap_or("");
                format!("[knowledge_save] \"{title}\"")
            }
            "knowledge_patch" => {
                let doc_id = args["doc_id"].as_i64().unwrap_or(0);
                format!("[knowledge_patch] doc #{doc_id}")
            }
            "knowledge_delete" => {
                let doc_id = args["doc_id"].as_i64().unwrap_or(0);
                format!("[knowledge_delete] doc #{doc_id}")
            }
            "category_add" => {
                let name = args["name"].as_str().unwrap_or("");
                format!("[category_add] \"{name}\"")
            }
            "category_delete" => {
                let name = args["name"].as_str().unwrap_or("");
                format!("[category_delete] \"{name}\"")
            }
            _ => format!("[{tool_name}]"),
        }
    }

    /// Read an image file from disk, return as ToolOutput::Image for vision analysis.
    async fn read_image(path: &str) -> ToolOutput {
        let p = if path.starts_with('~') {
            let home = std::env::var("HOME").unwrap_or_default();
            std::path::PathBuf::from(home).join(&path[2..])
        } else {
            std::path::PathBuf::from(path)
        };

        if !p.exists() {
            return ToolOutput::Text(format!("Error: file not found: {path}"));
        }

        // Check file size (max 20MB for Claude vision)
        match std::fs::metadata(&p) {
            Ok(meta) if meta.len() > 20 * 1024 * 1024 => {
                return ToolOutput::Text(format!(
                    "Error: image too large ({} bytes, max 20MB)",
                    meta.len()
                ));
            }
            Err(e) => return ToolOutput::Text(format!("Error: {e}")),
            _ => {}
        }

        let bytes = match tokio::fs::read(&p).await {
            Ok(b) => b,
            Err(e) => return ToolOutput::Text(format!("Error reading file: {e}")),
        };

        // Detect media type from extension
        let ext = p.extension()
            .map(|e| e.to_string_lossy().to_lowercase())
            .unwrap_or_default();
        let media_type = match ext.as_str() {
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "jpg" | "jpeg" => "image/jpeg",
            _ => {
                return ToolOutput::Text(format!(
                    "Error: unsupported image format: .{ext}. Supported: jpg, png, gif, webp"
                ));
            }
        };

        use base64::Engine;
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
        let file_name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();

        ToolOutput::Image {
            text: format!("Image file: {file_name} ({} bytes)", bytes.len()),
            image_base64,
            media_type: media_type.to_string(),
        }
    }
}

fn tool_def(name: &str, description: &str, parameters: serde_json::Value) -> ToolDef {
    ToolDef {
        tool_type: "function".into(),
        function: FunctionDef {
            name: name.into(),
            description: description.into(),
            parameters,
        },
    }
}
