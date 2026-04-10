use crate::db::Database;
use std::collections::BTreeSet;

pub async fn memory_save(
    db: &Database,
    user_id: u64,
    fact: &str,
    category: &str,
    embedding_client: Option<&crate::tools::EmbeddingClient>,
) -> String {
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

    // 4. Embed fact + auto-link related facts
    if let Some(client) = embedding_client {
        match client.embed_batch(&[fact], "document").await {
            Ok(embeddings) if !embeddings.is_empty() => {
                let embedding = &embeddings[0];
                let blob = crate::tools::embedding::embedding_to_bytes(embedding);
                let _ = db.update_fact_embedding(fact_id, &blob);

                // Find similar existing facts and create links
                if let Ok(all_facts) = db.load_all_fact_embeddings(user_id) {
                    let mut similarities: Vec<(i64, f32)> = all_facts
                        .iter()
                        .filter(|(id, _, _, _)| *id != fact_id)
                        .map(|(id, _, _, emb_blob)| {
                            let emb = crate::tools::embedding::bytes_to_embedding(emb_blob);
                            let sim =
                                crate::tools::embedding::cosine_similarity(embedding, &emb);
                            (*id, sim)
                        })
                        .filter(|(_, sim)| *sim > 0.75)
                        .collect();

                    similarities.sort_by(|a, b| {
                        b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
                    });
                    let top_links = &similarities[..similarities.len().min(3)];

                    let mut linked_facts = Vec::new();
                    for (related_id, sim) in top_links {
                        if db.link_facts(fact_id, *related_id, *sim).is_ok() {
                            // Look up text from all_facts instead of extra DB query
                            if let Some((_, text, _, _)) = all_facts.iter().find(|(id, _, _, _)| *id == *related_id) {
                                linked_facts
                                    .push(format!("#{related_id} {text} ({sim:.2})"));
                            }
                        }
                    }

                    if !linked_facts.is_empty() {
                        msg.push_str(&format!(
                            "\n🔗 Related: {}",
                            linked_facts.join(", ")
                        ));
                    }
                }
            }
            Ok(_) => {}
            Err(e) => {
                tracing::warn!("Failed to embed fact: {e}");
            }
        }
    }

    msg
}

pub async fn memory_search(
    db: &Database,
    user_id: u64,
    keyword: &str,
    embedding_client: Option<&crate::tools::EmbeddingClient>,
) -> String {
    if keyword.is_empty() {
        return "Error: keyword cannot be empty".into();
    }

    use std::collections::HashMap;

    struct FactHit {
        id: i64,
        fact: String,
        category: String,
        fts_score: f64,
        vector_score: f64,
    }

    let mut hits: HashMap<i64, FactHit> = HashMap::new();

    // 1. FTS5 search
    if let Ok(results) = db.search_facts(user_id, keyword) {
        let count = results.len() as f64;
        for (i, (id, fact, cat)) in results.into_iter().enumerate() {
            let score = if count > 0.0 {
                1.0 - (i as f64 / count)
            } else {
                0.0
            };
            hits.insert(
                id,
                FactHit {
                    id,
                    fact,
                    category: cat,
                    fts_score: score,
                    vector_score: 0.0,
                },
            );
        }
    }

    // 2. Vector search (if embedding client available)
    if let Some(client) = embedding_client {
        if let Ok(query_emb) = client.embed_query(keyword).await {
            if let Ok(all_facts) = db.load_all_fact_embeddings(user_id) {
                let mut scored: Vec<(i64, String, String, f32)> = all_facts
                    .into_iter()
                    .map(|(id, fact, cat, blob)| {
                        let emb = crate::tools::embedding::bytes_to_embedding(&blob);
                        let sim = crate::tools::embedding::cosine_similarity(&query_emb, &emb);
                        (id, fact, cat, sim)
                    })
                    .collect();

                scored.sort_by(|a, b| {
                    b.3.partial_cmp(&a.3).unwrap_or(std::cmp::Ordering::Equal)
                });
                let top = &scored[..scored.len().min(20)];

                if let Some(max_sim) = top.first().map(|s| s.3) {
                    if max_sim > 0.0 {
                        for (id, fact, cat, sim) in top {
                            let normalized = *sim / max_sim;
                            hits.entry(*id)
                                .and_modify(|h| h.vector_score = normalized as f64)
                                .or_insert_with(|| FactHit {
                                    id: *id,
                                    fact: fact.clone(),
                                    category: cat.clone(),
                                    fts_score: 0.0,
                                    vector_score: normalized as f64,
                                });
                        }
                    }
                }
            }
        }
    }

    if hits.is_empty() {
        return "No facts found.".into();
    }

    // 3. Combine scores: 0.4 * fts + 0.6 * vector (FTS-only fallback)
    let has_vectors = hits.values().any(|h| h.vector_score > 0.0);
    let mut results: Vec<&FactHit> = hits.values().collect();
    results.sort_by(|a, b| {
        let score_a = if has_vectors {
            0.4 * a.fts_score + 0.6 * a.vector_score
        } else {
            a.fts_score
        };
        let score_b = if has_vectors {
            0.4 * b.fts_score + 0.6 * b.vector_score
        } else {
            b.fts_score
        };
        score_b
            .partial_cmp(&score_a)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // 4. Format top 20 results with related facts
    let lines: Vec<String> = results
        .iter()
        .take(20)
        .map(|h| format_fact_with_links(db, h.id, &h.fact, &h.category))
        .collect();
    lines.join("\n")
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

    // KB doc links
    if let Ok(links) = db.get_fact_links(id) {
        if !links.is_empty() {
            let titles: Vec<String> =
                links.iter().map(|(did, t)| format!("#{did} {t}")).collect();
            line.push_str(&format!(" -> KB: {}", titles.join(", ")));
        }
    }

    // Related facts
    if let Ok(related) = db.get_related_facts(id) {
        if !related.is_empty() {
            let related_strs: Vec<String> = related
                .iter()
                .take(3)
                .map(|(rid, rfact, _cat, sim)| {
                    let preview: String = rfact.chars().take(50).collect();
                    format!("#{rid} {preview}({sim:.2})")
                })
                .collect();
            line.push_str(&format!(" -> Related: {}", related_strs.join(", ")));
        }
    }

    line
}
