use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct Config {
    pub telegram_bot_token: String,
    pub allowed_users: Vec<u64>,
    pub claude_keys: Vec<String>,
    pub max_agent_turns: usize,
    pub voyage_api_key: Option<String>,
    pub voyage_model: String,
    pub openai_api_key: Option<String>,
    pub gemini_api_key: Option<String>,
}

impl Config {
    pub fn from_env() -> Self {
        let env = load_dotenv();

        Self {
            telegram_bot_token: env
                .get("TELEGRAM_BOT_TOKEN")
                .cloned()
                .expect("TELEGRAM_BOT_TOKEN is required in .env"),
            allowed_users: env
                .get("TELEGRAM_ALLOWED_USERS")
                .map(|s| {
                    s.split(',')
                        .filter(|s| !s.is_empty())
                        .filter_map(|s| s.trim().parse().ok())
                        .collect()
                })
                .unwrap_or_default(),
            claude_keys: parse_keys(&env, "CLAUDE_API_KEYS"),
            max_agent_turns: env
                .get("MAX_AGENT_TURNS")
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
            voyage_api_key: env.get("VOYAGE_API_KEY").cloned().filter(|s| !s.is_empty()),
            voyage_model: env
                .get("VOYAGE_MODEL")
                .cloned()
                .unwrap_or_else(|| "voyage-4-lite".to_string()),
            openai_api_key: env.get("OPENAI_API_KEY").cloned().filter(|s| !s.is_empty()),
            gemini_api_key: env.get("GEMINI_API_KEY").cloned().filter(|s| !s.is_empty()),
        }
    }
}

/// Load .env file into a HashMap without polluting process environment.
fn load_dotenv() -> HashMap<String, String> {
    dotenvy::dotenv_iter()
        .map(|iter| iter.filter_map(|r| r.ok()).collect())
        .unwrap_or_default()
}

fn parse_keys(env: &HashMap<String, String>, key: &str) -> Vec<String> {
    env.get(key)
        .map(|s| {
            s.split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        })
        .unwrap_or_default()
}
