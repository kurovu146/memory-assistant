use crate::db::Database;

pub async fn knowledge_save(
    db: &Database,
    user_id: u64,
    title: &str,
    content: &str,
    source: Option<&str>,
    tags: Option<&str>,
) -> Result<(i64, String), String> {
    if title.is_empty() || content.is_empty() {
        return Err("Title and content are required".into());
    }
    let id = db.save_document(user_id, title, content, source, tags)?;
    Ok((id, format!("Saved document (ID: {id}): \"{title}\"")))
}

pub async fn knowledge_search(db: &Database, user_id: u64, query: &str) -> String {
    if query.is_empty() {
        return "Error: query cannot be empty".into();
    }
    match db.search_documents(user_id, query) {
        Ok(results) if results.is_empty() => "No documents found.".into(),
        Ok(results) => {
            let lines: Vec<String> = results
                .iter()
                .map(|(id, title, snippet, source)| {
                    let src = source.as_deref().unwrap_or("no source");
                    format!("[{id}] {title}\n  {snippet}\n  Source: {src}")
                })
                .collect();
            lines.join("\n\n")
        }
        Err(e) => format!("Error searching: {e}"),
    }
}

pub async fn entity_search(db: &Database, user_id: u64, query: &str) -> String {
    if query.is_empty() {
        return "Error: query cannot be empty".into();
    }
    match db.search_entities(user_id, query) {
        Ok(results) if results.is_empty() => "No entities found.".into(),
        Ok(results) => {
            let lines: Vec<String> = results
                .iter()
                .map(|(name, entity_type, mentions)| {
                    let mut line = format!("{name} [{entity_type}]");
                    if mentions.is_empty() {
                        line.push_str(" — no mentions");
                    } else {
                        for (src_type, src_id, context) in mentions {
                            let ctx = context.as_deref().unwrap_or("(no context)");
                            line.push_str(&format!("\n  - {src_type} #{src_id}: {ctx}"));
                        }
                    }
                    line
                })
                .collect();
            lines.join("\n\n")
        }
        Err(e) => format!("Error searching entities: {e}"),
    }
}
