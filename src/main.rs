use memory_assistant::config::Config;
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

    memory_assistant::telegram::run_bot(config).await;
}
