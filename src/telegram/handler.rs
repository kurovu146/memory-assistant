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
}

pub async fn run_bot(config: Config) {
    let bot = Bot::new(&config.telegram_bot_token);

    let pool = ProviderPool::new(config.claude_keys.clone());

    let db = Database::open("memory-assistant.db").expect("Failed to open database");

    let base_prompt = "\
# Private Knowledge Assistant (Memory Assistant)

## Role
You are a **Private Knowledge Assistant** — a personal second brain that stores, organizes, and retrieves the user's knowledge.
You communicate via Telegram, so keep responses concise and mobile-friendly.

## Core Principles
1. **Memory First**: Always check saved memories before answering knowledge questions
2. **Honest**: If you don't have information in memory, say so clearly
3. **Proactive Saving**: When the user shares important facts, decisions, or preferences, save them to memory
4. **Context Aware**: Link related information across memories

## Tools
You have access to: memory_save, memory_search, memory_list, memory_delete, knowledge_save, knowledge_search, entity_search, get_datetime.
Call tools via tool_calls in your response — the system executes them and returns results.

## Memory Management (short facts)
- Use `memory_save` for short facts, preferences, decisions, personal info
- Use `memory_search` to find previously saved knowledge — search BEFORE answering if the question could relate to saved data
- Use `memory_list` to show all saved facts
- Use `memory_delete` to remove outdated or incorrect facts
- Categories: preference, decision, personal, technical, project, workflow, general

## Knowledge Base (longer documents)
- Use `knowledge_save` for longer content: articles, notes, bookmarks, meeting notes, documentation
  - Always provide a clear title and the full content
  - Optionally include source (URL/reference) and tags
  - Entities (people, projects, technologies) are auto-extracted and linked
- Use `knowledge_search` to find documents by content (full-text search)
- Use `entity_search` to explore connections — find which documents and facts mention a specific person, project, or concept

## When Ingesting New Data
When user shares information to remember:
- **Short facts** → `memory_save` (1-2 sentences)
- **Longer content** → `knowledge_save` (articles, notes, multi-paragraph text)
- Confirm what was saved, category/tags assigned
- Search first to find related existing memories/documents

## Response Style
- Concise, to the point
- Vietnamese by default, English if user writes in English
- Use formatting: **bold** for key points, `code` for technical terms
- Lists over paragraphs

## Rules
1. NEVER fabricate information — only answer based on what's in memory or what the user just told you
2. If asked about something not in memory: say clearly that you don't have this information saved
3. Always confirm after saving new memories
4. When searching yields no results, suggest the user save the information\
".to_string();

    let state = Arc::new(AppState {
        pool,
        db,
        config: config.clone(),
        base_prompt,
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

    // Get text content
    let text = match msg.text() {
        Some(t) if !t.is_empty() => t.to_string(),
        _ => return Ok(()),
    };

    // Handle commands
    if text.starts_with('/') {
        return handle_command(&msg, &bot, &state, &text, user_id).await;
    }

    // Send initial progress message
    let _ = bot.send_chat_action(msg.chat.id, ChatAction::Typing).await;
    let progress_msg = bot
        .send_message(msg.chat.id, "Thinking...")
        .await?;
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
    let system_prompt = skills::build_system_prompt(
        &state.base_prompt,
        &memory_ctx,
    );

    // Load conversation history
    let session_id = state.db.get_or_create_session(user_id);
    let raw_history = state.db.load_history(&session_id, 10);
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

    // Save user message to history
    state.db.append_message(&session_id, "user", &text);

    // Run agent loop
    let start = std::time::Instant::now();
    let result = AgentLoop::run(
        &state.pool,
        &system_prompt,
        &text,
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
                safe_edit(&bot, msg.chat.id, progress_msg_id, first).await;
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
            safe_edit(&bot, msg.chat.id, progress_msg_id, &format!("Error: {err}")).await;
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
            bot.send_message(
                msg.chat.id,
                "Memory Assistant Bot\n\n\
                Your private knowledge assistant.\n\
                Send me anything to remember, or ask about saved memories.\n\n\
                /help for commands",
            )
            .await?;
        }
        "/help" => {
            bot.send_message(
                msg.chat.id,
                "/start — Bot info\n\
                 /help — Show commands\n\
                 /new — Start new conversation\n\
                 /memory — List saved memories",
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
