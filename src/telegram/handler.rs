use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use teloxide::prelude::*;
use teloxide::types::{BotCommand, ChatAction, ParseMode};
use tracing::{error, info};

use crate::agent::{AgentLoop, AgentProgress};
use crate::config::Config;
use crate::db::Database;
use crate::provider::{Message, MessageContent, ProviderPool, Role};
use crate::skills;
use crate::tools::EmbeddingClient;

use super::formatter;

struct AppState {
    pool: ProviderPool,
    db: Database,
    config: Config,
    base_prompt: String,
    telegram_token: String,
    embedding_client: Option<EmbeddingClient>,
}

pub async fn run_bot(config: Config) {
    let bot = Bot::new(&config.telegram_bot_token);

    let pool = ProviderPool::new(config.claude_keys.clone());

    let db = Database::open("memory-assistant.db").expect("Failed to open database");

    // Init embedding client if VOYAGE_API_KEY is set
    let embedding_client = config.voyage_api_key.as_ref().map(|key| {
        info!("Voyage AI embedding client initialized (model: {})", config.voyage_model);
        EmbeddingClient::new(key.clone(), config.voyage_model.clone())
    });

    let embedding_status = if embedding_client.is_some() {
        format!("Voyage AI ({}) — ACTIVE", config.voyage_model)
    } else {
        "DISABLED (no API key)".to_string()
    };

    let base_prompt = format!("\
Private Second Brain — Telegram Knowledge Assistant.
Always loyal to your owner. Never expose secrets in output.
Vietnamese by default, English if user writes in English.
Keep responses concise (Telegram format).

## SYSTEM INFO
- Embedding: {embedding_status}
- Knowledge DB: SQLite + FTS5 full-text search + semantic vector search (hybrid)
- Documents: chunked (~500 chars), embedded, searchable by meaning
- Files on disk: ~/documents/{{USER_ID}}/
- Khi hỏi về hệ thống → dùng bash/knowledge_list để kiểm tra thực tế, KHÔNG đoán.

## AUTO-RAG CONTEXT
Hệ thống TỰ ĐỘNG search knowledge base + memory cho mỗi câu hỏi. Kết quả nằm ở cuối system prompt trong section \"--- AUTO-RAG ---\".

CÁCH DÙNG:
1. Nếu AUTO-RAG tìm thấy kết quả → dùng NGAY, kèm trích dẫn \"(Theo [title], dòng X-Y)\".
2. Nếu cần thêm chi tiết hoặc AUTO-RAG chưa đủ → gọi thêm tools: knowledge_search (từ khóa khác), file_list, grep, bash (pdftotext), file_read.
3. NGHIÊM CẤM trả lời \"không tìm thấy\" hoặc \"em không có thông tin\" mà chưa gọi tools search bổ sung.
4. Khi search file gốc trên disk: CHỈ search trong ~/documents/{{USER_ID}}. KHÔNG truy cập thư mục khác.
   file_list ~/documents/{{USER_ID}}, pdftotext + grep -n cho PDF, grep cho text files.
5. LUÔN trích dẫn: \"(Theo [tên file/document], dòng X-Y)\" rồi quote nội dung gốc.

## KHI NHẬN FILE / TÀI LIỆU
1. Đọc và tóm tắt nội dung chính của file.
2. Hỏi xác nhận: \"Em hiểu đây là [tóm tắt]. Anh muốn em lưu vào knowledge không?\"
3. Chỉ gọi knowledge_save SAU KHI anh xác nhận. Không tự động lưu.
4. Khi lưu, đặt title mô tả rõ ràng để dễ tìm lại sau.

## ENTITY & CONTEXT MAPPING
- Khi câu hỏi liên quan đến người/dự án/tổ chức → dùng entity_search để tìm tất cả mentions liên quan.
- Cross-reference: nếu fact A nói \"X là sếp\" và fact B nói \"X thích cà phê\" → hỏi \"sở thích của sếp\" phải chain 2 facts và trả lời được.
- Khi lưu thông tin về người → ghi rõ mối quan hệ (sếp, đồng nghiệp, bạn...) trong fact.

## TOOLS
- memory_save: facts ngắn gọn | knowledge_save: tài liệu/nội dung dài (entities tự động extract).
- memory_search + knowledge_search: dùng khi cần search thêm ngoài AUTO-RAG.
- knowledge_list: liệt kê tất cả documents đã lưu (dùng khi hỏi \"lưu gì rồi\", \"có bao nhiêu tài liệu\").
- entity_search: dùng khi hỏi về người/dự án/tổ chức cụ thể.
- file_read/file_write/file_list, grep, glob: thao tác file hệ thống.
- bash: chỉ cho shell commands (git, cargo, npm, pdftotext...).
- get_datetime: lấy ngày giờ hiện tại.");

    let state = Arc::new(AppState {
        pool,
        db,
        config: config.clone(),
        base_prompt,
        telegram_token: config.telegram_bot_token.clone(),
        embedding_client,
    });

    // Migrate existing documents: chunk + embed unchunked docs
    {
        let state_clone = state.clone();
        tokio::spawn(async move {
            migrate_unchunked_docs(&state_clone).await;
        });
    }

    info!(
        "Memory Assistant bot started. Allowed users: {:?}",
        config.allowed_users
    );

    // Register bot commands
    let commands = vec![
        BotCommand::new("start", "Bot info & status"),
        BotCommand::new("help", "Show available commands"),
        BotCommand::new("memory", "View saved memories"),
    ];
    if let Err(e) = bot.set_my_commands(commands).await {
        error!("Failed to set bot commands: {e}");
    } else {
        info!("Bot commands menu registered");
    }

    let handler = Update::filter_message().endpoint(handle_message);

    Dispatcher::builder(bot, handler)
        .dependencies(dptree::deps![state])
        .build()
        .dispatch()
        .await;
}

/// Edit a Telegram message, trying Markdown first then falling back to plain text.
async fn safe_edit(bot: &Bot, chat_id: ChatId, msg_id: i32, text: &str) {
    #[allow(deprecated)]
    let md_result = bot
        .edit_message_text(chat_id, teloxide::types::MessageId(msg_id), text)
        .parse_mode(ParseMode::Markdown)
        .await;
    if md_result.is_err() {
        let _ = bot
            .edit_message_text(chat_id, teloxide::types::MessageId(msg_id), text)
            .await;
    }
}

/// Save uploaded file to ~/documents/{user_id}/{file_name}.
/// Creates directories if needed. Skips if identical content already saved.
/// Adds _1, _2... suffix if same name but different content.
async fn save_file_to_disk(user_id: u64, file_name: &str, data: &[u8]) -> Option<PathBuf> {
    let home = match std::env::var("HOME") {
        Ok(h) => h,
        Err(_) => return None,
    };
    let dir = PathBuf::from(&home).join("documents").join(user_id.to_string());

    // Dedup: hash content, use .checksums/ dir with atomic create_new
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    data.hash(&mut hasher);
    let hash_hex = format!("{:016x}", hasher.finish());

    let checksums_dir = dir.join(".checksums");
    if let Err(e) = tokio::fs::create_dir_all(&checksums_dir).await {
        error!("Failed to create checksums dir: {e}");
        return None;
    }

    match tokio::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(checksums_dir.join(&hash_hex))
        .await
    {
        Ok(_) => {} // New content, proceed to save
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            info!("Duplicate file skipped: {file_name} (hash {hash_hex})");
            return None;
        }
        Err(_) => {} // Can't check, save anyway
    }

    let path = Path::new(file_name);
    let stem = path.file_stem().unwrap_or_default().to_string_lossy().to_string();
    let ext = path.extension().map(|e| format!(".{}", e.to_string_lossy())).unwrap_or_default();

    let mut dest = dir.join(file_name);
    let mut counter = 1u32;
    loop {
        match tokio::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&dest)
            .await
        {
            Ok(file) => {
                use tokio::io::AsyncWriteExt;
                let mut file = file;
                if let Err(e) = file.write_all(data).await {
                    error!("Failed to write file {}: {e}", dest.display());
                    return None;
                }
                info!("File saved: {}", dest.display());
                return Some(dest);
            }
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                dest = dir.join(format!("{stem}_{counter}{ext}"));
                counter += 1;
            }
            Err(e) => {
                error!("Failed to create file {}: {e}", dest.display());
                return None;
            }
        }
    }
}

async fn handle_message(
    msg: teloxide::types::Message,
    bot: Bot,
    state: Arc<AppState>,
) -> ResponseResult<()> {
    let user_id = msg.from.as_ref().map(|u| u.id.0).unwrap_or(0);

    // Auth check
    if !state.config.allowed_users.is_empty()
        && !state.config.allowed_users.contains(&user_id)
    {
        bot.send_message(msg.chat.id, "Unauthorized.").await?;
        return Ok(());
    }

    // Check for photo
    if let Some(photos) = msg.photo() {
        if !photos.is_empty() {
            let caption = msg.caption().unwrap_or("Analyze this image");
            return handle_photo(&msg, &bot, &state, photos, caption, user_id).await;
        }
    }

    // Check for document/file
    if let Some(doc) = msg.document() {
        let caption = msg.caption().unwrap_or("Analyze this file");
        return handle_document(&msg, &bot, &state, doc, caption, user_id).await;
    }

    // Get text content
    let text = match msg.text() {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return Ok(()),
    };

    // Handle commands
    if text.starts_with('/') {
        return handle_command(&msg, &bot, &state, &text, user_id).await;
    }

    // Run agent with text
    let content = MessageContent::Text(text.clone());
    run_agent_and_respond(&msg, &bot, &state, user_id, &text, content).await
}

/// Handle photo messages: download image → base64 → send to Claude vision
async fn handle_photo(
    msg: &teloxide::types::Message,
    bot: &Bot,
    state: &AppState,
    photos: &[teloxide::types::PhotoSize],
    caption: &str,
    user_id: u64,
) -> ResponseResult<()> {
    // Get highest resolution photo (last in array)
    let photo = &photos[photos.len() - 1];

    // Download the photo
    let file = bot.get_file(&photo.file.id).await?;
    let file_path = &file.path;

    let url = format!(
        "https://api.telegram.org/file/bot{}/{}",
        state.telegram_token, file_path
    );

    let image_bytes = match reqwest::get(&url).await {
        Ok(resp) => match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                bot.send_message(msg.chat.id, &format!("Download error: {e}")).await?;
                return Ok(());
            }
        },
        Err(e) => {
            bot.send_message(msg.chat.id, &format!("Download error: {e}")).await?;
            return Ok(());
        }
    };

    // Save to disk
    let photo_name = file_path.rsplit('/').next().unwrap_or("photo.jpg");
    save_file_to_disk(user_id, photo_name, &image_bytes).await;

    // Detect media type from file extension
    let media_type = if file_path.ends_with(".png") {
        "image/png"
    } else if file_path.ends_with(".gif") {
        "image/gif"
    } else if file_path.ends_with(".webp") {
        "image/webp"
    } else {
        "image/jpeg"
    };

    // Encode to base64
    use base64::Engine;
    let image_base64 = base64::engine::general_purpose::STANDARD.encode(&image_bytes);

    info!("Photo received: {} bytes, {media_type}", image_bytes.len());

    let history_text = format!("[Image: {media_type}] {caption}");
    let content = MessageContent::ImageWithText {
        text: caption.to_string(),
        image_base64,
        media_type: media_type.to_string(),
    };

    run_agent_with_direct_content(msg, bot, state, user_id, &history_text, content).await
}

/// Handle document/file messages: download → read text → send to Claude
async fn handle_document(
    msg: &teloxide::types::Message,
    bot: &Bot,
    state: &AppState,
    doc: &teloxide::types::Document,
    caption: &str,
    user_id: u64,
) -> ResponseResult<()> {
    let file_name = doc.file_name.as_deref().unwrap_or("unknown");
    let mime = doc.mime_type.as_ref().map(|m| m.to_string()).unwrap_or_default();

    // Categorize file type
    let is_text = mime.starts_with("text/")
        || mime.contains("json")
        || mime.contains("xml")
        || mime.contains("javascript")
        || mime.contains("typescript")
        || mime.contains("yaml")
        || mime.contains("toml")
        || mime.contains("markdown")
        || mime.contains("csv")
        || file_name.ends_with(".rs")
        || file_name.ends_with(".go")
        || file_name.ends_with(".py")
        || file_name.ends_with(".ts")
        || file_name.ends_with(".js")
        || file_name.ends_with(".md")
        || file_name.ends_with(".txt")
        || file_name.ends_with(".json")
        || file_name.ends_with(".yaml")
        || file_name.ends_with(".yml")
        || file_name.ends_with(".toml")
        || file_name.ends_with(".csv")
        || file_name.ends_with(".log")
        || file_name.ends_with(".sql")
        || file_name.ends_with(".sh")
        || file_name.ends_with(".env.example");

    let is_image = mime.starts_with("image/");

    let is_document = file_name.ends_with(".pdf")
        || file_name.ends_with(".docx")
        || file_name.ends_with(".xlsx")
        || file_name.ends_with(".xls")
        || file_name.ends_with(".doc")
        || mime.contains("pdf")
        || mime.contains("wordprocessingml")
        || mime.contains("spreadsheetml");

    if !is_text && !is_image && !is_document {
        bot.send_message(
            msg.chat.id,
            &format!("Unsupported file type: {mime}\nSupported: text, code, images, PDF, DOCX, XLSX"),
        ).await?;
        return Ok(());
    }

    // Download file
    let file = bot.get_file(&doc.file.id).await?;
    let file_path = &file.path;

    let url = format!(
        "https://api.telegram.org/file/bot{}/{}",
        state.telegram_token, file_path
    );

    let file_bytes = match reqwest::get(&url).await {
        Ok(resp) => match resp.bytes().await {
            Ok(b) => b,
            Err(e) => {
                bot.send_message(msg.chat.id, &format!("Download error: {e}")).await?;
                return Ok(());
            }
        },
        Err(e) => {
            bot.send_message(msg.chat.id, &format!("Download error: {e}")).await?;
            return Ok(());
        }
    };

    // Save to disk
    save_file_to_disk(user_id, file_name, &file_bytes).await;

    if is_image {
        // Handle as image
        let media_type = mime.clone();
        use base64::Engine;
        let image_base64 = base64::engine::general_purpose::STANDARD.encode(&file_bytes);

        info!("Image file received: {file_name}, {} bytes", file_bytes.len());

        let history_text = format!("[File: {file_name}] {caption}");
        let content = MessageContent::ImageWithText {
            text: format!("File: {file_name}\n\n{caption}"),
            image_base64,
            media_type,
        };
        return run_agent_with_direct_content(msg, bot, state, user_id, &history_text, content).await;
    }

    // Handle document files (PDF, DOCX, XLSX)
    if is_document {
        use crate::tools::file_extract;

        info!("Document received: {file_name}, {} bytes, {mime}", file_bytes.len());

        let extracted = file_extract::extract_document(file_name, &file_bytes);
        match extracted {
            Ok(text) => {
                // Text extraction succeeded
                let truncated = if text.chars().count() > 15000 {
                    let cut: String = text.chars().take(15000).collect();
                    format!("{cut}...\n\n(truncated, {} chars total)", text.chars().count())
                } else {
                    text
                };

                info!("Document extracted: {file_name}, {} chars", truncated.len());

                let prompt = format!("File: {file_name}\n\n```\n{truncated}\n```\n\n{caption}");
                let history_text = format!("[File: {file_name}] {caption}");
                let content = MessageContent::Text(prompt);

                return run_agent_with_direct_content(msg, bot, state, user_id, &history_text, content).await;
            }
            Err(e) => {
                // Text extraction failed — try PDF→image→Vision fallback
                if file_name.to_lowercase().ends_with(".pdf") {
                    info!("PDF text extraction failed ({e}), trying image fallback");
                    if let Some(images) = pdf_to_images(&file_bytes).await {
                        info!("PDF converted to {} page images", images.len());
                        use base64::Engine;
                        let image_base64 = base64::engine::general_purpose::STANDARD.encode(&images[0]);

                        let text = if images.len() > 1 {
                            format!("File: {file_name} (page 1 of {}, image-based PDF)\n\n{caption}", images.len())
                        } else {
                            format!("File: {file_name} (image-based PDF)\n\n{caption}")
                        };

                        let history_text = format!("[File: {file_name}] {caption}");
                        let content = MessageContent::ImageWithText {
                            text,
                            image_base64,
                            media_type: "image/png".to_string(),
                        };

                        return run_agent_with_direct_content(msg, bot, state, user_id, &history_text, content).await;
                    }
                }

                bot.send_message(msg.chat.id, &format!("Could not extract text from {file_name}: {e}")).await?;
                return Ok(());
            }
        }
    }

    // Handle as text file
    let file_content = match String::from_utf8(file_bytes.to_vec()) {
        Ok(s) => s,
        Err(_) => {
            bot.send_message(msg.chat.id, "Could not read file as text.").await?;
            return Ok(());
        }
    };

    // Truncate large files (char-safe for multibyte UTF-8)
    let truncated = if file_content.chars().count() > 15000 {
        let cut: String = file_content.chars().take(15000).collect();
        format!("{cut}...\n\n(truncated, {} chars total)", file_content.chars().count())
    } else {
        file_content
    };

    info!("Text file received: {file_name}, {} chars", truncated.len());

    let prompt = format!("File: {file_name}\n\n```\n{truncated}\n```\n\n{caption}");
    let history_text = format!("[File: {file_name}] {caption}");
    let content = MessageContent::Text(prompt);

    run_agent_with_direct_content(msg, bot, state, user_id, &history_text, content).await
}

/// Common function: run agent loop and send response back to Telegram
async fn run_agent_and_respond(
    msg: &teloxide::types::Message,
    bot: &Bot,
    state: &AppState,
    user_id: u64,
    history_text: &str,
    user_content: MessageContent,
) -> ResponseResult<()> {
    run_agent_and_respond_inner(msg, bot, state, user_id, history_text, user_content, false).await
}

/// Same as run_agent_and_respond but marks that direct content was provided (file/image).
/// This suppresses false "fabrication" warnings since the content is already in the prompt.
async fn run_agent_with_direct_content(
    msg: &teloxide::types::Message,
    bot: &Bot,
    state: &AppState,
    user_id: u64,
    history_text: &str,
    user_content: MessageContent,
) -> ResponseResult<()> {
    run_agent_and_respond_inner(msg, bot, state, user_id, history_text, user_content, true).await
}

async fn run_agent_and_respond_inner(
    msg: &teloxide::types::Message,
    bot: &Bot,
    state: &AppState,
    user_id: u64,
    history_text: &str,
    user_content: MessageContent,
    has_direct_content: bool,
) -> ResponseResult<()> {
    // Send initial progress message
    let _ = bot.send_chat_action(msg.chat.id, ChatAction::Typing).await;
    let progress_msg = bot.send_message(msg.chat.id, "Thinking...").await?;
    let progress_msg_id = progress_msg.id.0;

    // Typing indicator loop
    let bot_typing = bot.clone();
    let chat_id = msg.chat.id;
    let typing_active = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let typing_flag = typing_active.clone();
    let typing_handle = tokio::spawn(async move {
        while typing_flag.load(Ordering::Relaxed) {
            let _ = bot_typing.send_chat_action(chat_id, ChatAction::Typing).await;
            tokio::time::sleep(std::time::Duration::from_secs(4)).await;
        }
    });

    // Progress callback
    let bot_progress = bot.clone();
    let progress_chat_id = msg.chat.id;
    let last_edit = Arc::new(AtomicI64::new(0));

    let on_progress = move |progress: AgentProgress| {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_millis() as i64;
        let prev = last_edit.load(Ordering::Relaxed);

        if now - prev < 1500 {
            return;
        }

        let display_text = match &progress {
            AgentProgress::ToolUse(name) => formatter::format_progress(name),
            AgentProgress::Thinking => "Thinking...".to_string(),
        };

        last_edit.store(now, Ordering::Relaxed);
        let bot_inner = bot_progress.clone();
        let _ = tokio::task::spawn(async move {
            safe_edit(&bot_inner, progress_chat_id, progress_msg_id, &display_text).await;
        });
    };

    // Build system prompt with memory
    let memory_ctx = state.db.build_memory_context(user_id);
    let user_prompt = state.base_prompt.replace("{USER_ID}", &user_id.to_string());
    let mut system_prompt = skills::build_system_prompt(&user_prompt, &memory_ctx);

    // Auto-RAG: pre-search knowledge + memory before LLM call
    if !has_direct_content && !history_text.is_empty() {
        let (knowledge_results, memory_results) = tokio::join!(
            crate::tools::knowledge_search(
                &state.db,
                user_id,
                history_text,
                state.embedding_client.as_ref(),
            ),
            crate::tools::memory_search(&state.db, user_id, history_text),
        );

        let mut rag_ctx = String::new();
        if !knowledge_results.starts_with("No documents")
            && !knowledge_results.starts_with("Error")
        {
            rag_ctx.push_str("\n\n--- AUTO-RAG: KNOWLEDGE BASE ---\n");
            rag_ctx.push_str(&knowledge_results);
        }
        if !memory_results.starts_with("No facts")
            && !memory_results.starts_with("Error")
        {
            rag_ctx.push_str("\n\n--- AUTO-RAG: MEMORY ---\n");
            rag_ctx.push_str(&memory_results);
        }
        if !rag_ctx.is_empty() {
            system_prompt.push_str(&rag_ctx);
        }
    }

    // Load conversation history
    let session_id = state.db.get_or_create_session(user_id);
    let raw_history = state.db.load_history(&session_id, 6);
    let history: Vec<Message> = raw_history
        .into_iter()
        .filter_map(|(role, content)| {
            let r = match role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => return None,
            };
            Some(Message { role: r, content: MessageContent::Text(content) })
        })
        .collect();

    // Save user message to history (text representation)
    state.db.append_message(&session_id, "user", history_text);

    // Run agent loop
    let start = std::time::Instant::now();
    let result = AgentLoop::run(
        &state.pool,
        &system_prompt,
        user_content,
        user_id,
        &state.db,
        state.config.max_agent_turns,
        history,
        state.embedding_client.as_ref(),
        on_progress,
    )
    .await;

    let elapsed_secs = start.elapsed().as_secs_f64();

    // Stop typing indicator
    typing_active.store(false, Ordering::Relaxed);
    typing_handle.abort();

    match result {
        Ok(agent_result) => {
            let cleaned = if has_direct_content {
                formatter::clean_response_with_context(&agent_result.response, &agent_result.tools_used, true)
            } else {
                formatter::clean_response(&agent_result.response, &agent_result.tools_used)
            };

            // Save assistant response to history
            state.db.append_message(&session_id, "assistant", &cleaned);

            // Build final response with footer
            let footer = formatter::format_tools_footer(
                &agent_result.tools_used,
                &agent_result.tools_count,
                elapsed_secs,
                &agent_result.provider,
                agent_result.turns,
            );
            let full_response = format!("{cleaned}{footer}");

            let chunks = formatter::split_message(&full_response, 4096);

            if let Some(first) = chunks.first() {
                safe_edit(bot, msg.chat.id, progress_msg_id, first).await;
            }

            for chunk in chunks.iter().skip(1) {
                #[allow(deprecated)]
                let md_result = bot
                    .send_message(msg.chat.id, chunk)
                    .parse_mode(ParseMode::Markdown)
                    .await;
                if md_result.is_err() {
                    let _ = bot.send_message(msg.chat.id, chunk).await;
                }
            }
        }
        Err(err) => {
            error!("Agent error: {err}");
            safe_edit(bot, msg.chat.id, progress_msg_id, &format!("Error: {err}")).await;
        }
    }

    Ok(())
}

async fn handle_command(
    msg: &teloxide::types::Message,
    bot: &Bot,
    state: &AppState,
    text: &str,
    user_id: u64,
) -> ResponseResult<()> {
    match text.split_whitespace().next().unwrap_or("") {
        "/start" => {
            let key_count = state.config.claude_keys.len();
            bot.send_message(
                msg.chat.id,
                format!(
                    "Memory Assistant Bot\n\n\
                    Your private knowledge assistant.\n\
                    Send text, photos, or files to remember and analyze.\n\n\
                    API keys: {key_count} (round-robin)\n\
                    /help for commands"
                ),
            )
            .await?;
        }
        "/help" => {
            bot.send_message(
                msg.chat.id,
                "/start — Bot info\n\
                 /help — Show commands\n\
                 /memory — List saved memories\n\n\
                 Supported input:\n\
                 - Text messages\n\
                 - Photos (with optional caption)\n\
                 - Files: text, code, images\n\
                 - Documents: PDF, DOCX, XLSX\n\n\
                 System tools:\n\
                 - bash, file_read, file_write\n\
                 - file_list, grep, glob",
            )
            .await?;
        }
        "/memory" => {
            let facts = state.db.list_facts(user_id, None).unwrap_or_default();
            if facts.is_empty() {
                bot.send_message(msg.chat.id, "No memories saved yet.").await?;
            } else {
                let output: String = facts
                    .iter()
                    .map(|(id, fact, cat)| format!("[{id}] [{cat}] {fact}"))
                    .collect::<Vec<_>>()
                    .join("\n");
                for chunk in formatter::split_message(&output, 4096) {
                    bot.send_message(msg.chat.id, &chunk).await?;
                }
            }
        }
        _ => {
            bot.send_message(msg.chat.id, "Unknown command. /help")
                .await?;
        }
    }
    Ok(())
}

/// Convert PDF bytes to PNG images using pdftoppm (poppler).
/// Returns Vec of PNG bytes per page, or None if pdftoppm is not available.
async fn pdf_to_images(data: &[u8]) -> Option<Vec<Vec<u8>>> {
    let tmp_pdf = "/tmp/_pdf_to_img.pdf";
    let tmp_prefix = "/tmp/_pdf_page";

    tokio::fs::write(tmp_pdf, data).await.ok()?;

    // scale-to limits the longest dimension to 4000px (Claude max is 8000)
    let output = tokio::process::Command::new("pdftoppm")
        .args(["-png", "-scale-to", "4000", "-l", "3", tmp_pdf, tmp_prefix])
        .output()
        .await;

    let _ = tokio::fs::remove_file(tmp_pdf).await;

    let output = output.ok()?;
    if !output.status.success() {
        return None;
    }

    // Read generated page images (pdftoppm creates files like _pdf_page-1.png, _pdf_page-2.png)
    let mut images = Vec::new();
    for page in 1..=3 {
        let page_path = format!("{tmp_prefix}-{page}.png");
        if let Ok(img_data) = tokio::fs::read(&page_path).await {
            images.push(img_data);
            let _ = tokio::fs::remove_file(&page_path).await;
        }
    }

    if images.is_empty() {
        // Try single-page format: _pdf_page-01.png
        for page in 1..=3 {
            let page_path = format!("{tmp_prefix}-{page:02}.png");
            if let Ok(img_data) = tokio::fs::read(&page_path).await {
                images.push(img_data);
                let _ = tokio::fs::remove_file(&page_path).await;
            }
        }
    }

    if images.is_empty() { None } else { Some(images) }
}

/// Migrate existing documents that don't have chunks yet.
async fn migrate_unchunked_docs(state: &AppState) {
    use crate::tools::embedding::embedding_to_bytes;
    use crate::tools::knowledge::chunk_document;

    let docs = match state.db.get_unchunked_doc_ids() {
        Ok(d) => d,
        Err(e) => {
            error!("Migration: failed to get unchunked docs: {e}");
            return;
        }
    };

    if docs.is_empty() {
        return;
    }

    info!("Migration: chunking {} existing documents", docs.len());

    for (doc_id, content) in &docs {
        let chunks = chunk_document(content);
        let chunk_data: Vec<(usize, usize, usize, &str)> = chunks
            .iter()
            .map(|c| (c.chunk_index, c.start_line, c.end_line, c.content.as_str()))
            .collect();

        let chunk_ids = match state.db.save_chunks(*doc_id, &chunk_data) {
            Ok(ids) => ids,
            Err(e) => {
                error!("Migration: failed to save chunks for doc {doc_id}: {e}");
                continue;
            }
        };

        // Embed if client available
        if let Some(client) = &state.embedding_client {
            let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
            for (batch_start, batch_ids) in chunk_ids.chunks(128).enumerate() {
                let start = batch_start * 128;
                let end = (start + 128).min(texts.len());
                let batch_texts = &texts[start..end];

                match client.embed_batch(batch_texts, "document").await {
                    Ok(embeddings) => {
                        let blobs: Vec<Vec<u8>> =
                            embeddings.iter().map(|e| embedding_to_bytes(e)).collect();
                        if let Err(e) = state.db.update_chunk_embeddings(batch_ids, &blobs) {
                            error!("Migration: failed to save embeddings: {e}");
                        }
                    }
                    Err(e) => {
                        error!("Migration: embedding failed for doc {doc_id}: {e}");
                    }
                }

                // Small delay to respect rate limits
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
        }
    }

    info!("Migration: completed chunking {} documents", docs.len());
}
