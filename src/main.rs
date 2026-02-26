mod agent;
mod db;
mod provider;
mod skills;
mod tools;
mod config;
mod telegram;

use config::Config;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
        )
        .init();

    let config = Config::from_env();

    tracing::info!("Memory Assistant v{}", env!("CARGO_PKG_VERSION"));
    tracing::info!("Claude keys: {}", config.claude_keys.len());

    telegram::run_bot(config).await;
}
