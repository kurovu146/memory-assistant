use crate::db::Database;
use std::collections::BTreeSet;

pub async fn memory_save(db: &Database, user_id: u64, fact: &str, category: &str) -> String {
    if fact.is_empty() {
        return "Error: fact cannot be empty".into();
    }

    // Ensure default categories exist
    let _ = db.ensure_default_categories(user_id);

    // 1. Find and delete superseded facts (same category, FTS match) BEFORE saving
    let mut deleted = Vec::new();
    if let Ok(old_facts) = db.search_facts(user_id, fact) {
        for (old_id, old_fact, old_cat) in &old_facts {
            if old_cat != category {
                continue;
            }
            if db.delete_fact(user_id, *old_id).unwrap_or(false) {
                deleted.push(format!("#{old_id} {old_fact}"));
            }
            if deleted.len() >= 3 {
                break;
            }
        }
    }

    // 2. Save new fact
    let fact_id = match db.save_fact(user_id, fact, category) {
        Ok(id) => id,
        Err(e) => return format!("Error saving: {e}"),
    };

    let mut msg = format!("Saved (ID: {fact_id}): \"{fact}\" [{category}]");

    if !deleted.is_empty() {
        msg.push_str(&format!("\n🗑️ Superseded: {}", deleted.join(", ")));
    }

    // 3. Auto-link: search KB chunks for related docs
    if let Ok(chunks) = db.search_chunks_fts(user_id, fact) {
        let mut doc_ids = BTreeSet::new();
        let mut doc_titles: Vec<(i64, String)> = Vec::new();
        for (_chunk_id, doc_id, title, _, _, _, _, _) in &chunks {
            if doc_ids.insert(*doc_id) {
                doc_titles.push((*doc_id, title.clone()));
                if doc_ids.len() >= 3 {
                    break;
                }
            }
        }
        for (doc_id, _) in &doc_titles {
            let _ = db.link_fact_to_doc(fact_id, *doc_id);
        }
        if !doc_titles.is_empty() {
            let titles: Vec<String> = doc_titles
                .iter()
                .map(|(id, t)| format!("#{id} {t}"))
                .collect();
            msg.push_str(&format!("\n📚 Linked to: {}", titles.join(", ")));
            msg.push_str("\n⚠️ These KB docs may need updating — use knowledge_patch to update specific text.");
        }
    }

    msg
}

pub async fn memory_search(db: &Database, user_id: u64, keyword: &str) -> String {
    if keyword.is_empty() {
        return "Error: keyword cannot be empty".into();
    }
    match db.search_facts(user_id, keyword) {
        Ok(results) if results.is_empty() => "No facts found.".into(),
        Ok(results) => {
            let lines: Vec<String> = results
                .iter()
                .map(|(id, fact, cat)| format_fact_with_links(db, *id, fact, cat))
                .collect();
            lines.join("\n")
        }
        Err(e) => format!("Error searching: {e}"),
    }
}

pub async fn memory_list(db: &Database, user_id: u64, category: Option<&str>) -> String {
    match db.list_facts(user_id, category) {
        Ok(results) if results.is_empty() => "No facts saved yet.".into(),
        Ok(results) => {
            let lines: Vec<String> = results
                .iter()
                .map(|(id, fact, cat)| format_fact_with_links(db, *id, fact, cat))
                .collect();
            lines.join("\n")
        }
        Err(e) => format!("Error listing: {e}"),
    }
}

fn format_fact_with_links(db: &Database, id: i64, fact: &str, cat: &str) -> String {
    let mut line = format!("[{id}] [{cat}] {fact}");
    if let Ok(links) = db.get_fact_links(id) {
        if !links.is_empty() {
            let titles: Vec<String> = links.iter().map(|(did, t)| format!("#{did} {t}")).collect();
            line.push_str(&format!(" → 📚 {}", titles.join(", ")));
        }
    }
    line
}
