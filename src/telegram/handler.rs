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

use super::formatter;

struct AppState {
    pool: ProviderPool,
    db: Database,
    config: Config,
    base_prompt: String,
    telegram_token: String,
}

pub async fn run_bot(config: Config) {
    let bot = Bot::new(&config.telegram_bot_token);

    let pool = ProviderPool::new(config.claude_keys.clone());

    let db = Database::open("memory-assistant.db").expect("Failed to open database");

    let base_prompt = "\
Private Knowledge Assistant. Telegram bot — keep responses concise.
Always loyal to your owner. Never lie or deceive. Never fabricate info.
Search memory before answering. Save facts user shares.
Vietnamese by default, English if user writes in English.

Tools: memory_save/search/list/delete, knowledge_save/search, entity_search, get_datetime, bash, file_read, file_write, file_list, grep, glob.
- Short facts → memory_save. Longer content → knowledge_save (entities auto-extracted).
- Use file_read/file_list/grep/glob for file ops, bash only for commands (git, cargo, npm, etc.).
- Never expose secrets in output.\
".to_string();

    let state = Arc::new(AppState {
        pool,
        db,
        config: config.clone(),
        base_prompt,
        telegram_token: config.telegram_bot_token.clone(),
    });

    info!(
        "Memory Assistant bot started. Allowed users: {:?}",
        config.allowed_users
    );

    // Register bot commands
    let commands = vec![
        BotCommand::new("start", "Bot info & status"),
        BotCommand::new("help", "Show available commands"),
        BotCommand::new("new", "Start new conversation"),
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

    run_agent_and_respond(msg, bot, state, user_id, &history_text, content).await
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

    // Only handle text-based files and images
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

    if !is_text && !is_image {
        bot.send_message(
            msg.chat.id,
            &format!("Unsupported file type: {mime}\nSupported: text files, code, images"),
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
        return run_agent_and_respond(msg, bot, state, user_id, &history_text, content).await;
    }

    // Handle as text file
    let file_content = match String::from_utf8(file_bytes.to_vec()) {
        Ok(s) => s,
        Err(_) => {
            bot.send_message(msg.chat.id, "Could not read file as text.").await?;
            return Ok(());
        }
    };

    // Truncate large files
    let truncated = if file_content.len() > 15000 {
        format!("{}...\n\n(truncated, {} bytes total)", &file_content[..15000], file_content.len())
    } else {
        file_content
    };

    info!("Text file received: {file_name}, {} chars", truncated.len());

    let prompt = format!("File: {file_name}\n\n```\n{truncated}\n```\n\n{caption}");
    let history_text = format!("[File: {file_name}] {caption}");
    let content = MessageContent::Text(prompt);

    run_agent_and_respond(msg, bot, state, user_id, &history_text, content).await
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
    let system_prompt = skills::build_system_prompt(&state.base_prompt, &memory_ctx);

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
        on_progress,
    )
    .await;

    let elapsed_secs = start.elapsed().as_secs_f64();

    // Stop typing indicator
    typing_active.store(false, Ordering::Relaxed);
    typing_handle.abort();

    match result {
        Ok(agent_result) => {
            let cleaned = formatter::clean_response(&agent_result.response, &agent_result.tools_used);

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
                 /new — Start new conversation\n\
                 /memory — List saved memories\n\n\
                 Supported input:\n\
                 - Text messages\n\
                 - Photos (with optional caption)\n\
                 - Files (text, code, images)\n\n\
                 System tools:\n\
                 - bash, file_read, file_write\n\
                 - file_list, grep, glob",
            )
            .await?;
        }
        "/new" => {
            state.db.clear_session(user_id);
            bot.send_message(msg.chat.id, "Session cleared. Starting fresh conversation.")
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
