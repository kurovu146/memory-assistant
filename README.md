# Memory Assistant

Private knowledge assistant - Telegram bot powered by Claude AI with persistent memory, document storage, and entity extraction.

## Features

- **Memory Management** - Save and search short facts with categories (preference, decision, personal, technical, project, workflow)
- **Knowledge Base** - Store longer documents/articles/notes with FTS5 full-text search
- **Entity Extraction** - Auto-extract entities (people, projects, technologies, concepts, organizations) from saved documents
- **Knowledge Graph** - Search entities and discover connections across documents and facts
- **Conversation History** - Session-based chat history for contextual responses
- **Telegram Interface** - Mobile-friendly responses with progress indicators

## Architecture

- **Runtime**: Rust + Tokio async
- **LLM**: Claude Haiku 4.5 (via Anthropic API) with round-robin key rotation
- **Database**: SQLite with FTS5 full-text search (WAL mode)
- **Telegram**: teloxide 0.13
- **Binary**: ~6.6MB (release, LTO + strip)

## Tools

| Tool | Description |
|------|-------------|
| `memory_save` | Save a short fact to long-term memory |
| `memory_search` | Search memories by keyword (FTS5) |
| `memory_list` | List all saved memories |
| `memory_delete` | Delete a memory by ID |
| `knowledge_save` | Save a document/article/note (auto-extracts entities) |
| `knowledge_search` | Full-text search across documents |
| `entity_search` | Search knowledge graph for entities and their mentions |
| `get_datetime` | Get current time in UTC, Vietnam, US Eastern |

## Setup

1. Clone and build:

```bash
git clone git@github.com:kurovu146/memory-assistant.git
cd memory-assistant
cargo build --release
```

2. Create `.env` (see `.env.example`):

```env
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_ALLOWED_USERS=your_telegram_user_id
CLAUDE_API_KEYS=sk-ant-xxx
MAX_AGENT_TURNS=5
RUST_LOG=info
```

3. Run:

```bash
./target/release/memory-assistant
```

## Database Schema

- `memory_facts` + FTS5 - Short facts with categories
- `knowledge_documents` + FTS5 - Longer documents with title, content, source, tags
- `entities` - Extracted named entities (person, project, technology, concept, organization)
- `entity_mentions` - Junction table linking entities to documents/facts
- `sessions` / `session_messages` - Conversation history

## Commands

- `/start` - Bot info
- `/help` - Show commands
- `/new` - Start fresh conversation
- `/memory` - List saved memories

## License

MIT
