use serde_json::json;

use crate::provider::{ToolDef, FunctionDef, ProviderPool};
use crate::tools;

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
                            "enum": ["preference", "decision", "personal", "technical", "project", "workflow", "general"],
                            "description": "Category of the fact"
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
                "Delete a specific memory by its ID.",
                json!({
                    "type": "object",
                    "properties": {
                        "id": { "type": "integer", "description": "The memory fact ID to delete" }
                    },
                    "required": ["id"]
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
                "Search the knowledge base for documents using full-text search.",
                json!({
                    "type": "object",
                    "properties": {
                        "query": { "type": "string", "description": "Search query" }
                    },
                    "required": ["query"]
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
                        "path": { "type": "string", "description": "Directory path (default: current dir)" },
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
                        "path": { "type": "string", "description": "Directory or file to search in (default: current dir)" },
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
                        "path": { "type": "string", "description": "Directory to search in (default: current dir)" }
                    },
                    "required": ["pattern"]
                }),
            ),
        ]
    }

    /// Execute a tool by name with given arguments
    pub async fn execute(
        tool_name: &str,
        args_json: &str,
        user_id: u64,
        db: &crate::db::Database,
        pool: &ProviderPool,
    ) -> String {
        let args: serde_json::Value = serde_json::from_str(args_json).unwrap_or_default();

        match tool_name {
            "memory_save" => {
                let fact = args["fact"].as_str().unwrap_or("");
                let category = args["category"].as_str().unwrap_or("general");
                tools::memory_save(db, user_id, fact, category).await
            }
            "memory_search" => {
                let keyword = args["keyword"].as_str().unwrap_or("");
                tools::memory_search(db, user_id, keyword).await
            }
            "memory_list" => {
                let category = args["category"].as_str();
                tools::memory_list(db, user_id, category).await
            }
            "memory_delete" => {
                let id = args["id"].as_i64().unwrap_or(0);
                tools::memory_delete(db, user_id, id).await
            }
            "knowledge_save" => {
                let title = args["title"].as_str().unwrap_or("");
                let content = args["content"].as_str().unwrap_or("");
                let source = args["source"].as_str();
                let tags = args["tags"].as_str();

                match tools::knowledge_save(db, user_id, title, content, source, tags).await {
                    Ok((doc_id, msg)) => {
                        // Auto-extract entities in background
                        let text = format!("{title}\n\n{content}");
                        let entity_count = tools::extract_and_link_entities(
                            pool, db, user_id, "document", doc_id, &text,
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
                tools::knowledge_search(db, user_id, query).await
            }
            "entity_search" => {
                let query = args["query"].as_str().unwrap_or("");
                tools::entity_search(db, user_id, query).await
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
            _ => format!("Unknown tool: {tool_name}"),
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
