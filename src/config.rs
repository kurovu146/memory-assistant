use std::env;

#[derive(Debug, Clone)]
pub struct Config {
    pub telegram_bot_token: String,
    pub allowed_users: Vec<u64>,
    pub claude_keys: Vec<String>,
    pub max_agent_turns: usize,
}

impl Config {
    pub fn from_env() -> Self {
        dotenvy::from_path_override("/home/kuro/dev/memory-assistant/.env").ok();

        Self {
            telegram_bot_token: env::var("TELEGRAM_BOT_TOKEN")
                .expect("TELEGRAM_BOT_TOKEN is required"),
            allowed_users: env::var("TELEGRAM_ALLOWED_USERS")
                .unwrap_or_default()
                .split(',')
                .filter(|s| !s.is_empty())
                .filter_map(|s| s.trim().parse().ok())
                .collect(),
            claude_keys: parse_keys("CLAUDE_API_KEYS"),
            max_agent_turns: env::var("MAX_AGENT_TURNS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(5),
        }
    }
}

fn parse_keys(env_var: &str) -> Vec<String> {
    env::var(env_var)
        .unwrap_or_default()
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}
