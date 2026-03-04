use std::collections::HashMap;

use crate::db::Database;
use crate::tools::embedding::{
    EmbeddingClient, bytes_to_embedding, cosine_similarity, embedding_to_bytes,
};

// --- Chunking ---

pub struct Chunk {
    pub chunk_index: usize,
    pub start_line: usize,
    pub end_line: usize,
    pub content: String,
}

const CHUNK_TARGET_CHARS: usize = 500;
const OVERLAP_LINES: usize = 3;

/// Split document into chunks by lines. Uses char count (not bytes) for sizing.
/// Prefers paragraph boundaries (\n\n), falls back to line boundaries.
pub fn chunk_document(content: &str) -> Vec<Chunk> {
    let char_count: usize = content.chars().count();
    if char_count < CHUNK_TARGET_CHARS {
        let line_count = content.lines().count().max(1);
        return vec![Chunk {
            chunk_index: 0,
            start_line: 1,
            end_line: line_count,
            content: content.to_string(),
        }];
    }

    // Work entirely with lines — no byte-offset slicing
    let lines: Vec<&str> = content.lines().collect();
    let mut chunks = Vec::new();
    let mut line_idx = 0;
    let mut chunk_index = 0;

    while line_idx < lines.len() {
        let start_line_idx = line_idx;
        let mut char_acc = 0usize;
        let mut end_line_idx = line_idx;

        // Accumulate lines until we reach ~CHUNK_TARGET_CHARS
        while end_line_idx < lines.len() && char_acc < CHUNK_TARGET_CHARS {
            char_acc += lines[end_line_idx].chars().count() + 1; // +1 for newline
            end_line_idx += 1;
        }

        // Try to extend to paragraph boundary (empty line) within next few lines
        let look_ahead = (end_line_idx + 5).min(lines.len());
        let mut found_para = false;
        for i in end_line_idx..look_ahead {
            if lines[i].trim().is_empty() {
                end_line_idx = i + 1;
                found_para = true;
                break;
            }
        }

        // If no paragraph break and we stopped mid-sentence, try sentence boundary
        if !found_para && end_line_idx > start_line_idx {
            // Check if last included line ends with sentence-ending punctuation
            let last = end_line_idx - 1;
            if last > start_line_idx && !lines[last].ends_with('.') && !lines[last].ends_with('?') && !lines[last].ends_with('!') {
                // Try to include one more line if it ends a sentence
                if end_line_idx < lines.len() && (lines[end_line_idx].ends_with('.') || lines[end_line_idx].ends_with('?') || lines[end_line_idx].ends_with('!')) {
                    end_line_idx += 1;
                }
            }
        }

        // Ensure we include at least one line
        if end_line_idx <= start_line_idx {
            end_line_idx = start_line_idx + 1;
        }

        let chunk_content = lines[start_line_idx..end_line_idx].join("\n");

        chunks.push(Chunk {
            chunk_index,
            start_line: start_line_idx + 1, // 1-based
            end_line: end_line_idx,          // 1-based (inclusive)
            content: chunk_content.trim().to_string(),
        });

        chunk_index += 1;

        // Advance with overlap (go back OVERLAP_LINES lines)
        if end_line_idx >= lines.len() {
            break;
        }
        line_idx = if end_line_idx > OVERLAP_LINES {
            end_line_idx - OVERLAP_LINES
        } else {
            end_line_idx
        };
    }

    chunks
}

// --- Knowledge Save ---

pub async fn knowledge_save(
    db: &Database,
    user_id: u64,
    title: &str,
    content: &str,
    source: Option<&str>,
    tags: Option<&str>,
    embedding_client: Option<&EmbeddingClient>,
) -> Result<(i64, String), String> {
    if title.is_empty() || content.is_empty() {
        return Err("Title and content are required".into());
    }
    let doc_id = db.save_document(user_id, title, content, source, tags)?;

    // Chunk the document
    let chunks = chunk_document(content);
    let chunk_data: Vec<(usize, usize, usize, &str)> = chunks
        .iter()
        .map(|c| (c.chunk_index, c.start_line, c.end_line, c.content.as_str()))
        .collect();
    let chunk_ids = db.save_chunks(doc_id, &chunk_data)?;
    let chunk_count = chunk_ids.len();

    // Embed if client available
    if let Some(client) = embedding_client {
        let texts: Vec<&str> = chunks.iter().map(|c| c.content.as_str()).collect();
        // Batch in groups of 128
        for (batch_start, batch_ids) in chunk_ids.chunks(128).enumerate() {
            let start = batch_start * 128;
            let end = (start + 128).min(texts.len());
            let batch_texts = &texts[start..end];

            match client.embed_batch(batch_texts, "document").await {
                Ok(embeddings) => {
                    let blobs: Vec<Vec<u8>> =
                        embeddings.iter().map(|e| embedding_to_bytes(e)).collect();
                    if let Err(e) = db.update_chunk_embeddings(batch_ids, &blobs) {
                        tracing::warn!("Failed to save embeddings: {e}");
                    }
                }
                Err(e) => {
                    tracing::warn!("Embedding failed (will retry later): {e}");
                }
            }
        }
    }

    Ok((
        doc_id,
        format!("Saved document (ID: {doc_id}): \"{title}\" — {chunk_count} chunks"),
    ))
}

// --- Hybrid Search ---

/// Search result combining FTS and vector scores.
struct SearchHit {
    _chunk_id: i64,
    doc_id: i64,
    title: String,
    content: String,
    start_line: i64,
    end_line: i64,
    source: Option<String>,
    fts_score: f64,
    vector_score: f64,
}

pub async fn knowledge_search(
    db: &Database,
    user_id: u64,
    query: &str,
    embedding_client: Option<&EmbeddingClient>,
) -> String {
    if query.is_empty() {
        return "Error: query cannot be empty".into();
    }

    let mut hits: HashMap<i64, SearchHit> = HashMap::new();

    // 1. FTS5 search on chunks
    match db.search_chunks_fts(user_id, query) {
        Ok(results) => {
            if !results.is_empty() {
                // Normalize FTS ranks (they're negative, more negative = better match)
                let max_rank = results
                    .iter()
                    .map(|r| r.7.abs())
                    .fold(f64::MIN, f64::max);
                for (chunk_id, doc_id, title, content, start_line, end_line, source, rank) in results
                {
                    let normalized = if max_rank > 0.0 {
                        rank.abs() / max_rank
                    } else {
                        0.0
                    };
                    hits.insert(
                        chunk_id,
                        SearchHit {
                            _chunk_id: chunk_id,
                            doc_id,
                            title,
                            content,
                            start_line,
                            end_line,
                            source,
                            fts_score: normalized,
                            vector_score: 0.0,
                        },
                    );
                }
            }
        }
        Err(e) => {
            tracing::warn!("FTS search failed: {e}");
            // Fall back to old document-level search
            return fallback_document_search(db, user_id, query);
        }
    }

    // 2. Vector search (if embedding client available)
    if let Some(client) = embedding_client {
        if let Ok(query_embedding) = client.embed_query(query).await {
            if let Ok(all_chunks) = db.load_all_embeddings(user_id) {
                let mut scored: Vec<(i64, i64, String, String, i64, i64, Option<String>, f32)> =
                    all_chunks
                        .into_iter()
                        .map(
                            |(chunk_id, doc_id, title, content, start_line, end_line, source, blob)| {
                                let emb = bytes_to_embedding(&blob);
                                let sim = cosine_similarity(&query_embedding, &emb);
                                (chunk_id, doc_id, title, content, start_line, end_line, source, sim)
                            },
                        )
                        .collect();

                scored.sort_by(|a, b| b.7.partial_cmp(&a.7).unwrap_or(std::cmp::Ordering::Equal));
                let top = &scored[..scored.len().min(20)];

                if let Some(max_sim) = top.first().map(|s| s.7) {
                    if max_sim > 0.0 {
                        for (chunk_id, doc_id, title, content, start_line, end_line, source, sim) in
                            top
                        {
                            let normalized = *sim / max_sim;
                            hits.entry(*chunk_id)
                                .and_modify(|h| h.vector_score = normalized as f64)
                                .or_insert_with(|| SearchHit {
                                    _chunk_id: *chunk_id,
                                    doc_id: *doc_id,
                                    title: title.clone(),
                                    content: content.clone(),
                                    start_line: *start_line,
                                    end_line: *end_line,
                                    source: source.clone(),
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
        return "No documents found.".into();
    }

    // 3. Combine scores: 0.4 * fts + 0.6 * vector (or FTS-only if no vectors)
    let has_vectors = hits.values().any(|h| h.vector_score > 0.0);
    let mut results: Vec<&SearchHit> = hits.values().collect();
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

    // Dedup by doc_id+line range, keep top 10
    let mut seen_ranges: Vec<(i64, i64, i64)> = Vec::new();
    let mut output_lines: Vec<String> = Vec::new();

    for hit in results {
        let key = (hit.doc_id, hit.start_line, hit.end_line);
        if seen_ranges.contains(&key) {
            continue;
        }
        seen_ranges.push(key);

        let src = hit.source.as_deref().unwrap_or("no source");
        let line_range = if hit.start_line == hit.end_line {
            format!("dòng {}", hit.start_line)
        } else {
            format!("dòng {}-{}", hit.start_line, hit.end_line)
        };

        output_lines.push(format!(
            "[{}] {} ({})\n  {}\n  Source: {}",
            hit.doc_id, hit.title, line_range, hit.content, src
        ));

        if output_lines.len() >= 10 {
            break;
        }
    }

    output_lines.join("\n\n")
}

fn fallback_document_search(db: &Database, user_id: u64, query: &str) -> String {
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

pub async fn knowledge_list(db: &Database, user_id: u64) -> String {
    match db.list_documents(user_id) {
        Ok(docs) if docs.is_empty() => "No documents saved yet.".into(),
        Ok(docs) => {
            let lines: Vec<String> = docs
                .iter()
                .map(|(id, title, source, created_at, chunk_count)| {
                    let src = source.as_deref().unwrap_or("no source");
                    format!("[{id}] {title}  ({chunk_count} chunks)\n  Source: {src}  |  Saved: {created_at}")
                })
                .collect();
            format!("{} documents:\n\n{}", docs.len(), lines.join("\n\n"))
        }
        Err(e) => format!("Error listing documents: {e}"),
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
