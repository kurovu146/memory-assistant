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
}

static MODELS: &[ModelInfo] = &[
    ModelInfo {
        id: "claude-haiku-4-5-20251001",
        shortcut: "haiku",
        provider: ProviderType::Claude,
        label: "Claude Haiku 4.5",
    },
    ModelInfo {
        id: "claude-sonnet-4-6",
        shortcut: "sonnet",
        provider: ProviderType::Claude,
        label: "Claude Sonnet 4.6",
    },
    ModelInfo {
        id: "gpt-5-mini",
        shortcut: "gpt-5-mini",
        provider: ProviderType::OpenAI,
        label: "GPT-5 Mini",
    },
ModelInfo {
        id: "gemini-3.1-flash-lite-preview",
        shortcut: "gemini-flash",
        provider: ProviderType::Gemini,
        label: "Gemini 3.1 Flash Lite",
    },
    ModelInfo {
        id: "gemini-3-flash-preview",
        shortcut: "gemini-pro",
        provider: ProviderType::Gemini,
        label: "Gemini 3 Flash",
    },
    ModelInfo {
        id: "kimi-k2.5",
        shortcut: "kimi",
        provider: ProviderType::Kimi,
        label: "Kimi K2.5",
    },
    ModelInfo {
        id: "kimi-k2-turbo-preview",
        shortcut: "kimi-turbo",
        provider: ProviderType::Kimi,
        label: "Kimi K2 Turbo",
    },
    ModelInfo {
        id: "kimi-k2-thinking",
        shortcut: "kimi-think",
        provider: ProviderType::Kimi,
        label: "Kimi K2 Thinking",
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

/// Default model ID.
pub const DEFAULT_MODEL: &str = "claude-haiku-4-5-20251001";
