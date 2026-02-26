use tracing::{debug, warn};

use crate::db::Database;
use crate::provider::{Message, MessageContent, ProviderPool, Role};

const EXTRACTION_PROMPT: &str = r#"Extract named entities from the following text. Return ONLY a JSON array of objects with "name" and "type" fields.

Valid types: person, project, technology, concept, organization

Rules:
- Only extract clearly named entities (proper nouns, specific names)
- Normalize names (capitalize properly)
- Skip generic terms
- Return empty array [] if no entities found
- Return ONLY the JSON array, no other text

Text:
"#;

/// Extract entities from text using Claude and link them to a source in the DB.
/// Returns the number of entities extracted.
pub async fn extract_and_link_entities(
    pool: &ProviderPool,
    db: &Database,
    user_id: u64,
    source_type: &str,
    source_id: i64,
    text: &str,
) -> usize {
    // Truncate text to avoid huge prompts
    let truncated = if text.len() > 3000 { &text[..3000] } else { text };

    let prompt = format!("{EXTRACTION_PROMPT}{truncated}");

    let messages = vec![Message {
        role: Role::User,
        content: MessageContent::Text(prompt),
    }];

    // Call Claude with no tools — just text extraction
    let response = match pool.chat(&messages, &[]).await {
        Ok((resp, _provider)) => resp,
        Err(e) => {
            warn!("Entity extraction failed: {e}");
            return 0;
        }
    };

    let response_text = match &response.content {
        Some(t) => t.clone(),
        None => return 0,
    };

    debug!("Entity extraction response: {response_text}");

    // Parse JSON array from response
    let entities = parse_entities(&response_text);
    let count = entities.len();

    // Save each entity and link to source
    for (name, entity_type) in &entities {
        match db.save_entity(user_id, name, entity_type) {
            Ok(entity_id) => {
                // Build a short context snippet
                let context = build_context_snippet(text, name);
                let _ = db.add_entity_mention(entity_id, source_type, source_id, context.as_deref());
            }
            Err(e) => {
                warn!("Failed to save entity '{name}': {e}");
            }
        }
    }

    if count > 0 {
        debug!("Extracted {count} entities from {source_type} #{source_id}");
    }
    count
}

fn parse_entities(text: &str) -> Vec<(String, String)> {
    // Find JSON array in response (may have surrounding text)
    let trimmed = text.trim();
    let json_str = if let Some(start) = trimmed.find('[') {
        if let Some(end) = trimmed.rfind(']') {
            &trimmed[start..=end]
        } else {
            return vec![];
        }
    } else {
        return vec![];
    };

    let parsed: Vec<serde_json::Value> = match serde_json::from_str(json_str) {
        Ok(v) => v,
        Err(_) => return vec![],
    };

    let valid_types = ["person", "project", "technology", "concept", "organization"];

    parsed
        .iter()
        .filter_map(|obj| {
            let name = obj["name"].as_str()?.trim().to_string();
            let entity_type = obj["type"].as_str()?.trim().to_lowercase();
            if name.is_empty() || !valid_types.contains(&entity_type.as_str()) {
                return None;
            }
            Some((name, entity_type))
        })
        .collect()
}

/// Extract a short context snippet around the entity name in the text.
fn build_context_snippet(text: &str, entity_name: &str) -> Option<String> {
    let lower_text = text.to_lowercase();
    let lower_name = entity_name.to_lowercase();

    let pos = lower_text.find(&lower_name)?;

    let start = pos.saturating_sub(50);
    let end = (pos + entity_name.len() + 50).min(text.len());

    // Align to char boundaries
    let start = text[..start]
        .rfind(char::is_whitespace)
        .map(|p| p + 1)
        .unwrap_or(start);
    let end = text[end..]
        .find(char::is_whitespace)
        .map(|p| end + p)
        .unwrap_or(end);

    Some(text[start..end].to_string())
}
