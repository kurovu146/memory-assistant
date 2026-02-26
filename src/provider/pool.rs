use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{info, warn};

use super::claude::ClaudeProvider;
use super::types::*;

struct KeyPool {
    keys: Vec<String>,
    index: AtomicUsize,
}

impl KeyPool {
    fn new(keys: Vec<String>) -> Self {
        Self {
            keys,
            index: AtomicUsize::new(0),
        }
    }

    fn next_key(&self) -> Option<&str> {
        if self.keys.is_empty() {
            return None;
        }
        let idx = self.index.fetch_add(1, Ordering::Relaxed) % self.keys.len();
        Some(&self.keys[idx])
    }

    fn len(&self) -> usize {
        self.keys.len()
    }
}

/// Claude-only provider pool with round-robin key rotation
pub struct ProviderPool {
    provider: ClaudeProvider,
    keys: KeyPool,
}

impl ProviderPool {
    pub fn new(claude_keys: Vec<String>) -> Self {
        info!("Provider pool: Claude with {} keys", claude_keys.len());
        Self {
            provider: ClaudeProvider::new(),
            keys: KeyPool::new(claude_keys),
        }
    }

    /// Send a chat request, trying all keys with rotation
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
    ) -> Result<(LlmResponse, String), ProviderError> {
        let num_keys = self.keys.len();
        if num_keys == 0 {
            return Err(ProviderError::NoKeys);
        }

        for _attempt in 0..num_keys {
            let key = match self.keys.next_key() {
                Some(k) => k,
                None => break,
            };

            info!("Trying Claude (key: {}...)", &key[..key.len().min(10)]);
            match self.provider.chat(messages, tools, key).await {
                Ok(response) => {
                    info!("Claude succeeded");
                    return Ok((response, "claude".to_string()));
                }
                Err(ProviderError::RateLimited) => {
                    warn!("Claude RATE LIMITED (key: {}...), trying next key", &key[..key.len().min(10)]);
                    continue;
                }
                Err(e) => {
                    warn!("Claude FAILED (key: {}...): {e}", &key[..key.len().min(10)]);
                    return Err(e);
                }
            }
        }

        Err(ProviderError::RequestError("All Claude keys exhausted".into()))
    }
}
