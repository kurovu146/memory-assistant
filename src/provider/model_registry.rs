#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProviderType {
    Claude,
    OpenAI,
    Gemini,
    Kimi,
}

pub struct ModelInfo {
    pub id: &'static str,
    pub shortcut: &'static str,
    pub provider: ProviderType,
    pub label: &'static str,
    /// Pricing per 1M tokens: (input, output, cache_write, cache_read) in USD
    pub pricing: (f64, f64, f64, f64),
}

static MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "claude-haiku-4-5-20251001",
        shortcut: "haiku",
        provider: ProviderType::Claude,
        label: "Claude Haiku 4.5",
        // input, output, cache_write, cache_read per 1M tokens (USD)
        pricing: (0.80, 4.00, 1.00, 0.08),
    },
    ModelInfo {
        id: "claude-sonnet-4-6",
        shortcut: "sonnet",
        provider: ProviderType::Claude,
        label: "Claude Sonnet 4.6",
        pricing: (3.00, 15.00, 3.75, 0.30),
    },
    ModelInfo {
        id: "gpt-5-mini",
        shortcut: "gpt-5-mini",
        provider: ProviderType::OpenAI,
        label: "GPT-5 Mini",
        pricing: (0.40, 1.60, 0.00, 0.20),
    },
    ModelInfo {
        id: "gemini-3-flash-preview",
        shortcut: "gemini-pro",
        provider: ProviderType::Gemini,
        label: "Gemini 3 Flash",
        pricing: (0.10, 0.40, 0.025, 0.025),
    },
    ModelInfo {
        id: "kimi-k2.5",
        shortcut: "kimi",
        provider: ProviderType::Kimi,
        label: "Kimi K2.5",
        pricing: (0.20, 0.90, 0.00, 0.02),
    },
];

/// Resolve a model by shortcut or full ID. Returns None if not found.
pub fn resolve_model(input: &str) -> Option<&'static ModelInfo> {
    let lower = input.to_lowercase();
    MODELS
        .iter()
        .find(|m| m.shortcut == lower || m.id == lower)
}

/// List all available models.
pub fn list_models() -> &'static [ModelInfo] {
    MODELS
}

/// Calculate cost in USD for a given model and token counts.
pub fn calculate_cost(
    model_id: &str,
    prompt_tokens: u64,
    completion_tokens: u64,
    cache_creation_tokens: u64,
    cache_read_tokens: u64,
) -> (f64, f64, f64, f64) {
    let (pi, po, pw, pr) = resolve_model(model_id)
        .map(|m| m.pricing)
        .unwrap_or((0.0, 0.0, 0.0, 0.0));
    (
        prompt_tokens as f64 * pi / 1_000_000.0,
        completion_tokens as f64 * po / 1_000_000.0,
        cache_creation_tokens as f64 * pw / 1_000_000.0,
        cache_read_tokens as f64 * pr / 1_000_000.0,
    )
}

/// Default model ID.
pub const DEFAULT_MODEL: &str = "claude-haiku-4-5-20251001";
