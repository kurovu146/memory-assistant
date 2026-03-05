use std::sync::atomic::{AtomicUsize, Ordering};
use tracing::{info, warn};

use super::claude::ClaudeProvider;
use super::model_registry::{self, ProviderType};
use super::openai_compat::OpenAICompatProvider;
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

/// Multi-provider pool with round-robin key rotation for Claude.
pub struct ProviderPool {
    claude: ClaudeProvider,
    openai_compat: OpenAICompatProvider,
    claude_keys: KeyPool,
    openai_key: Option<String>,
    gemini_key: Option<String>,
}

impl ProviderPool {
    pub fn new(
        claude_keys: Vec<String>,
        openai_key: Option<String>,
        gemini_key: Option<String>,
    ) -> Self {
        info!(
            "Provider pool: Claude ({} keys), OpenAI ({}), Gemini ({})",
            claude_keys.len(),
            if openai_key.is_some() { "configured" } else { "none" },
            if gemini_key.is_some() { "configured" } else { "none" },
        );
        Self {
            claude: ClaudeProvider::new(),
            openai_compat: OpenAICompatProvider::new(),
            claude_keys: KeyPool::new(claude_keys),
            openai_key,
            gemini_key,
        }
    }

    /// Check if a provider type has its API key configured.
    pub fn has_key_for(&self, provider: ProviderType) -> bool {
        match provider {
            ProviderType::Claude => self.claude_keys.len() > 0,
            ProviderType::OpenAI => self.openai_key.is_some(),
            ProviderType::Gemini => self.gemini_key.is_some(),
        }
    }

    /// Send a chat request, routing to the correct provider based on model.
    pub async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        model: &str,
    ) -> Result<(LlmResponse, String), ProviderError> {
        let model_info = model_registry::resolve_model(model);
        let provider_type = model_info
            .map(|m| m.provider)
            .unwrap_or(ProviderType::Claude);

        match provider_type {
            ProviderType::Claude => self.chat_claude(messages, tools, model).await,
            ProviderType::OpenAI => {
                let key = self
                    .openai_key
                    .as_deref()
                    .ok_or(ProviderError::NoKeys)?;
                let base_url = "https://api.openai.com/v1";
                info!("Trying OpenAI: model={model}");
                let response = self
                    .openai_compat
                    .chat(messages, tools, key, model, base_url)
                    .await?;
                info!("OpenAI succeeded");
                Ok((response, "openai".to_string()))
            }
            ProviderType::Gemini => {
                let key = self
                    .gemini_key
                    .as_deref()
                    .ok_or(ProviderError::NoKeys)?;
                let base_url = "https://generativelanguage.googleapis.com/v1beta/openai";
                info!("Trying Gemini: model={model}");
                let response = self
                    .openai_compat
                    .chat(messages, tools, key, model, base_url)
                    .await?;
                info!("Gemini succeeded");
                Ok((response, "gemini".to_string()))
            }
        }
    }

    /// Claude-specific routing with round-robin key rotation and retry logic.
    async fn chat_claude(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        model: &str,
    ) -> Result<(LlmResponse, String), ProviderError> {
        let num_keys = self.claude_keys.len();
        if num_keys == 0 {
            return Err(ProviderError::NoKeys);
        }

        let max_retries = 3;
        let retry_delays = [15, 30, 60];

        for retry in 0..=max_retries {
            for _attempt in 0..num_keys {
                let key = match self.claude_keys.next_key() {
                    Some(k) => k,
                    None => break,
                };

                info!("Trying Claude (key: {}...)", &key[..key.len().min(10)]);
                match self.claude.chat(messages, tools, key, model).await {
                    Ok(response) => {
                        info!("Claude succeeded");
                        return Ok((response, "claude".to_string()));
                    }
                    Err(ProviderError::RateLimited) => {
                        warn!(
                            "Claude RATE LIMITED (key: {}...), trying next key",
                            &key[..key.len().min(10)]
                        );
                        continue;
                    }
                    Err(e) => {
                        warn!("Claude FAILED (key: {}...): {e}", &key[..key.len().min(10)]);
                        return Err(e);
                    }
                }
            }

            if retry < max_retries {
                let delay = retry_delays[retry];
                warn!(
                    "All keys rate-limited, waiting {delay}s before retry {}/{}...",
                    retry + 1,
                    max_retries
                );
                tokio::time::sleep(std::time::Duration::from_secs(delay)).await;
            }
        }

        Err(ProviderError::RequestError(
            "All Claude keys exhausted after retries".into(),
        ))
    }
}
