use serde::{Deserialize, Serialize};
use tracing::debug;

pub struct EmbeddingClient {
    client: reqwest::Client,
    api_key: String,
    model: String,
}

#[derive(Serialize)]
struct EmbeddingRequest<'a> {
    model: &'a str,
    input: &'a [&'a str],
    input_type: &'a str,
}

#[derive(Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

#[derive(Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

impl EmbeddingClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            api_key,
            model,
        }
    }

    /// Embed a batch of texts. `input_type` should be "document" or "query".
    pub async fn embed_batch(
        &self,
        texts: &[&str],
        input_type: &str,
    ) -> Result<Vec<Vec<f32>>, String> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        debug!(
            "Embedding {} texts (type={}, model={})",
            texts.len(),
            input_type,
            self.model
        );

        let body = EmbeddingRequest {
            model: &self.model,
            input: texts,
            input_type,
        };

        let resp = self
            .client
            .post("https://api.voyageai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await
            .map_err(|e| format!("Voyage API request failed: {e}"))?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Voyage API error {status}: {body}"));
        }

        let result: EmbeddingResponse = resp
            .json()
            .await
            .map_err(|e| format!("Voyage API parse error: {e}"))?;

        Ok(result.data.into_iter().map(|d| d.embedding).collect())
    }

    /// Embed a single query text.
    pub async fn embed_query(&self, text: &str) -> Result<Vec<f32>, String> {
        let results = self.embed_batch(&[text], "query").await?;
        results
            .into_iter()
            .next()
            .ok_or_else(|| "Empty embedding response".to_string())
    }
}

/// Convert f32 embedding to little-endian bytes for BLOB storage.
pub fn embedding_to_bytes(embedding: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(embedding.len() * 4);
    for &val in embedding {
        bytes.extend_from_slice(&val.to_le_bytes());
    }
    bytes
}

/// Convert little-endian bytes back to f32 embedding.
pub fn bytes_to_embedding(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom == 0.0 {
        0.0
    } else {
        dot / denom
    }
}
