use reqwest::Client;
use serde_json::json;
use tracing::debug;

use super::types::*;

pub struct GeminiProvider {
    client: Client,
}

impl GeminiProvider {
    pub fn new() -> Self {
        Self {
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }

    pub async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        api_key: &str,
        model: &str,
    ) -> Result<LlmResponse, ProviderError> {
        let url = format!(
            "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        );

        let is_gemma = model.starts_with("gemma");
        let (contents, system_instruction) = build_google_contents(messages, is_gemma);

        let mut body = json!({ "contents": contents });

        if !is_gemma {
            if let Some(sys) = system_instruction {
                body["systemInstruction"] = sys;
            }
        }

        if !tools.is_empty() && !is_gemma {
            let decls: Vec<serde_json::Value> = tools
                .iter()
                .map(|t| {
                    json!({
                        "name": t.function.name,
                        "description": t.function.description,
                        "parameters": t.function.parameters,
                    })
                })
                .collect();
            body["tools"] = json!([{ "functionDeclarations": decls }]);
            body["toolConfig"] = json!({ "functionCallingConfig": { "mode": "AUTO" } });
        }

        debug!("Gemini native API request: model={model}");

        let resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", api_key.trim())
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::RequestError(e.to_string()))?;

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ProviderError::RateLimited);
        }
        if status == reqwest::StatusCode::UNAUTHORIZED || status == reqwest::StatusCode::FORBIDDEN {
            return Err(ProviderError::AuthError(format!("HTTP {status}")));
        }
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::RequestError(format!("HTTP {status}: {text}")));
        }

        parse_google_response(resp).await
    }
}

fn build_google_contents(
    messages: &[Message],
    is_gemma: bool,
) -> (Vec<serde_json::Value>, Option<serde_json::Value>) {
    let system_texts: Vec<String> = messages
        .iter()
        .filter(|m| m.role == Role::System)
        .map(|m| m.content.as_text().to_string())
        .filter(|s| !s.is_empty())
        .collect();

    let system_instruction = if !system_texts.is_empty() && !is_gemma {
        Some(json!({ "parts": [{ "text": system_texts.join("\n\n") }] }))
    } else {
        None
    };

    // Gemma has no systemInstruction field — fold system text into the first user turn.
    let gemma_system_prefix = if is_gemma && !system_texts.is_empty() {
        Some(format!("{}\n\n", system_texts.join("\n\n")))
    } else {
        None
    };
    let mut gemma_prefix_applied = false;

    let mut contents: Vec<serde_json::Value> = Vec::new();

    for m in messages {
        if m.role == Role::System {
            continue;
        }

        match &m.content {
            MessageContent::Text(text) => {
                let role = match m.role {
                    Role::User | Role::Tool => "user",
                    Role::Assistant => "model",
                    Role::System => unreachable!(),
                };
                let mut final_text = text.clone();
                if role == "user" && !gemma_prefix_applied {
                    if let Some(prefix) = &gemma_system_prefix {
                        final_text = format!("{prefix}{text}");
                        gemma_prefix_applied = true;
                    }
                }
                contents.push(json!({
                    "role": role,
                    "parts": [{ "text": final_text }]
                }));
            }
            MessageContent::ImageWithText {
                text,
                image_base64,
                media_type,
            } => {
                let mut parts: Vec<serde_json::Value> = vec![json!({
                    "inlineData": {
                        "mimeType": media_type,
                        "data": image_base64,
                    }
                })];
                let mut final_text = text.clone();
                if !gemma_prefix_applied {
                    if let Some(prefix) = &gemma_system_prefix {
                        final_text = format!("{prefix}{text}");
                        gemma_prefix_applied = true;
                    }
                }
                if !final_text.is_empty() {
                    parts.push(json!({ "text": final_text }));
                }
                contents.push(json!({ "role": "user", "parts": parts }));
            }
            MessageContent::MultiImageWithText { text, images } => {
                let mut parts: Vec<serde_json::Value> = Vec::new();
                for img in images {
                    parts.push(json!({
                        "inlineData": {
                            "mimeType": img.media_type,
                            "data": img.image_base64,
                        }
                    }));
                }
                let mut final_text = text.clone();
                if !gemma_prefix_applied {
                    if let Some(prefix) = &gemma_system_prefix {
                        final_text = format!("{prefix}{text}");
                        gemma_prefix_applied = true;
                    }
                }
                if !final_text.is_empty() {
                    parts.push(json!({ "text": final_text }));
                }
                contents.push(json!({ "role": "user", "parts": parts }));
            }
            MessageContent::ToolResult { name, content, .. } => {
                let response_value: serde_json::Value = serde_json::from_str(content)
                    .unwrap_or_else(|_| json!({ "result": content }));
                contents.push(json!({
                    "role": "user",
                    "parts": [{
                        "functionResponse": {
                            "name": name,
                            "response": response_value,
                        }
                    }]
                }));
            }
            MessageContent::ToolResultWithImage {
                name,
                text,
                image_base64,
                media_type,
                ..
            } => {
                let response_value = json!({ "result": text });
                contents.push(json!({
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": name,
                                "response": response_value,
                            }
                        },
                        {
                            "inlineData": {
                                "mimeType": media_type,
                                "data": image_base64,
                            }
                        }
                    ]
                }));
            }
            MessageContent::AssistantWithToolCalls {
                text, tool_calls, ..
            } => {
                let mut parts: Vec<serde_json::Value> = Vec::new();
                if let Some(t) = text {
                    if !t.is_empty() {
                        parts.push(json!({ "text": t }));
                    }
                }
                for tc in tool_calls {
                    let args: serde_json::Value =
                        serde_json::from_str(&tc.function.arguments).unwrap_or_else(|_| json!({}));
                    let mut part = json!({
                        "functionCall": {
                            "name": tc.function.name,
                            "args": args,
                        }
                    });
                    // Gemini 3 requires thoughtSignature to be echoed back for tool calls.
                    if let Some(sig) = &tc.thought_signature {
                        part["thoughtSignature"] = json!(sig);
                    }
                    parts.push(part);
                }
                contents.push(json!({ "role": "model", "parts": parts }));
            }
        }
    }

    (contents, system_instruction)
}

async fn parse_google_response(resp: reqwest::Response) -> Result<LlmResponse, ProviderError> {
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ProviderError::ParseError(e.to_string()))?;

    let candidate = body["candidates"]
        .get(0)
        .ok_or_else(|| ProviderError::ParseError("No candidates in response".into()))?;

    let parts = candidate["content"]["parts"]
        .as_array()
        .cloned()
        .unwrap_or_default();

    let mut text_parts: Vec<String> = Vec::new();
    let mut reasoning_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<ToolCall> = Vec::new();

    for (idx, part) in parts.iter().enumerate() {
        let is_thought = part["thought"].as_bool().unwrap_or(false);

        if let Some(text) = part["text"].as_str() {
            if text.is_empty() {
                continue;
            }
            if is_thought {
                reasoning_parts.push(text.to_string());
            } else {
                text_parts.push(text.to_string());
            }
        } else if let Some(fc) = part.get("functionCall") {
            let name = fc["name"].as_str().unwrap_or("").to_string();
            let args = fc.get("args").cloned().unwrap_or(json!({}));
            let arguments = serde_json::to_string(&args)
                .map_err(|e| ProviderError::ParseError(e.to_string()))?;
            let thought_signature = part["thoughtSignature"].as_str().map(|s| s.to_string());
            tool_calls.push(ToolCall {
                id: format!("call_{idx}"),
                function: ToolCallFunction { name, arguments },
                thought_signature,
            });
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };
    let reasoning_content = if reasoning_parts.is_empty() {
        None
    } else {
        Some(reasoning_parts.join(""))
    };

    let usage = Usage {
        prompt_tokens: body["usageMetadata"]["promptTokenCount"]
            .as_u64()
            .unwrap_or(0) as u32,
        completion_tokens: body["usageMetadata"]["candidatesTokenCount"]
            .as_u64()
            .unwrap_or(0) as u32,
        cache_creation_tokens: 0,
        cache_read_tokens: body["usageMetadata"]["cachedContentTokenCount"]
            .as_u64()
            .unwrap_or(0) as u32,
    };

    Ok(LlmResponse {
        content,
        tool_calls,
        usage,
        reasoning_content,
    })
}
