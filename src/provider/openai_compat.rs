use reqwest::Client;
use serde_json::json;
use tracing::debug;

use super::types::*;

pub struct OpenAICompatProvider {
    client: Client,
}

impl OpenAICompatProvider {
    pub fn new() -> Self {
        Self {
            client: Client::new(),
        }
    }

    pub async fn chat(
        &self,
        messages: &[Message],
        tools: &[ToolDef],
        api_key: &str,
        model: &str,
        base_url: &str,
    ) -> Result<LlmResponse, ProviderError> {
        let api_messages = build_openai_messages(messages);

        let mut body = json!({
            "model": model,
            "messages": api_messages,
        });

        if !tools.is_empty() {
            body["tools"] = json!(tools);
        }

        debug!("OpenAI-compat API request: model={model}, base_url={base_url}");

        let resp = self
            .client
            .post(format!("{base_url}/chat/completions"))
            .header("Authorization", format!("Bearer {api_key}"))
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| ProviderError::RequestError(e.to_string()))?;

        let status = resp.status();
        if status == reqwest::StatusCode::TOO_MANY_REQUESTS {
            return Err(ProviderError::RateLimited);
        }
        if status == reqwest::StatusCode::UNAUTHORIZED {
            return Err(ProviderError::AuthError("Invalid API key".into()));
        }
        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(ProviderError::RequestError(format!("HTTP {status}: {text}")));
        }

        parse_openai_response(resp).await
    }
}

/// Convert our generic Message format to OpenAI API format.
fn build_openai_messages(messages: &[Message]) -> Vec<serde_json::Value> {
    let mut api_messages: Vec<serde_json::Value> = Vec::new();

    for msg in messages {
        match (&msg.role, &msg.content) {
            (Role::System, MessageContent::Text(text)) => {
                api_messages.push(json!({
                    "role": "system",
                    "content": text,
                }));
            }
            (Role::User, MessageContent::Text(text)) => {
                api_messages.push(json!({
                    "role": "user",
                    "content": text,
                }));
            }
            (Role::User, MessageContent::ImageWithText { text, image_base64, media_type }) => {
                api_messages.push(json!({
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:{media_type};base64,{image_base64}"),
                            }
                        },
                        {
                            "type": "text",
                            "text": text,
                        }
                    ],
                }));
            }
            (Role::Assistant, MessageContent::Text(text)) => {
                api_messages.push(json!({
                    "role": "assistant",
                    "content": text,
                }));
            }
            (Role::Assistant, MessageContent::AssistantWithToolCalls { text, tool_calls, reasoning_content }) => {
                let mut msg_obj = json!({
                    "role": "assistant",
                });

                // Kimi K2 Thinking: pass back reasoning_content
                if let Some(rc) = reasoning_content {
                    msg_obj["reasoning_content"] = json!(rc);
                }

                if let Some(t) = text {
                    if !t.is_empty() {
                        msg_obj["content"] = json!(t);
                    }
                }

                let openai_tool_calls: Vec<serde_json::Value> = tool_calls
                    .iter()
                    .map(|tc| {
                        let mut call = json!({
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            }
                        });
                        // Gemini 3: pass back thought_signature
                        if let Some(sig) = &tc.thought_signature {
                            call["extra_content"] = json!({
                                "google": {
                                    "thought_signature": sig
                                }
                            });
                        }
                        call
                    })
                    .collect();

                if !openai_tool_calls.is_empty() {
                    msg_obj["tool_calls"] = json!(openai_tool_calls);
                }

                api_messages.push(msg_obj);
            }
            (Role::Tool, MessageContent::ToolResult { tool_call_id, content, .. }) => {
                api_messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": content,
                }));
            }
            // OpenAI doesn't support images in tool results — fallback to text only
            (Role::Tool, MessageContent::ToolResultWithImage { tool_call_id, text, .. }) => {
                api_messages.push(json!({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": text,
                }));
            }
            _ => {}
        }
    }

    api_messages
}

/// Parse OpenAI API response into our generic LlmResponse.
async fn parse_openai_response(resp: reqwest::Response) -> Result<LlmResponse, ProviderError> {
    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| ProviderError::ParseError(e.to_string()))?;

    let choice = body["choices"]
        .as_array()
        .and_then(|c| c.first())
        .ok_or_else(|| ProviderError::ParseError("No choices in response".into()))?;

    let message = &choice["message"];

    let content = message["content"].as_str().map(|s| s.to_string());
    let reasoning_content = message["reasoning_content"].as_str().map(|s| s.to_string());

    let mut tool_calls: Vec<ToolCall> = Vec::new();
    if let Some(tcs) = message["tool_calls"].as_array() {
        for tc in tcs {
            let id = tc["id"].as_str().unwrap_or("").to_string();
            let name = tc["function"]["name"].as_str().unwrap_or("").to_string();
            let arguments = tc["function"]["arguments"]
                .as_str()
                .unwrap_or("{}")
                .to_string();

            // Gemini 3: capture thought_signature from extra_content
            let thought_signature = tc["extra_content"]["google"]["thought_signature"]
                .as_str()
                .map(|s| s.to_string());

            tool_calls.push(ToolCall {
                id,
                function: ToolCallFunction { name, arguments },
                thought_signature,
            });
        }
    }

    let usage_obj = &body["usage"];
    // Gemini OpenAI compat: cache tokens in prompt_tokens_details.cached_tokens
    // and/or cache_read_input_tokens / cache_creation_input_tokens
    let cache_read = usage_obj["cache_read_input_tokens"].as_u64()
        .or_else(|| usage_obj["prompt_tokens_details"]["cached_tokens"].as_u64())
        .or_else(|| usage_obj["cached_tokens"].as_u64())
        .unwrap_or(0) as u32;
    let cache_creation = usage_obj["cache_creation_input_tokens"].as_u64()
        .unwrap_or(0) as u32;

    let usage = Usage {
        prompt_tokens: usage_obj["prompt_tokens"].as_u64().unwrap_or(0) as u32,
        completion_tokens: usage_obj["completion_tokens"].as_u64().unwrap_or(0) as u32,
        cache_creation_tokens: cache_creation,
        cache_read_tokens: cache_read,
    };

    Ok(LlmResponse {
        content,
        tool_calls,
        usage,
        reasoning_content,
    })
}
