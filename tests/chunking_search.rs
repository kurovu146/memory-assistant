use memory_assistant::db::Database;
use memory_assistant::tools::embedding::{cosine_similarity, embedding_to_bytes, bytes_to_embedding};
use memory_assistant::tools::knowledge::chunk_document;

// --- chunk_document tests ---

#[test]
fn chunk_short_document() {
    let content = "Đây là một tài liệu ngắn bằng tiếng Việt.";
    let chunks = chunk_document(content);
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].start_line, 1);
    assert_eq!(chunks[0].chunk_index, 0);
    assert_eq!(chunks[0].content, content);
}

#[test]
fn chunk_vietnamese_utf8_no_panic() {
    // 600+ chars of Vietnamese text — must NOT panic on UTF-8 boundaries
    let content = "Điều 1: Phạm vi và đối tượng áp dụng\n\
        Hợp đồng này được ký kết giữa Công ty TNHH ABC (bên A) và Ông Nguyễn Văn X (bên B).\n\
        Bên A đồng ý cung cấp dịch vụ tư vấn pháp lý cho bên B trong thời gian 12 tháng.\n\n\
        Điều 2: Quyền và nghĩa vụ của các bên\n\
        Bên A có trách nhiệm đảm bảo chất lượng dịch vụ theo tiêu chuẩn ISO 9001.\n\
        Bên B có nghĩa vụ thanh toán đầy đủ và đúng hạn theo quy định tại Điều 3.\n\n\
        Điều 3: Phương thức thanh toán\n\
        Bên B thanh toán cho bên A số tiền 50.000.000 VNĐ (năm mươi triệu đồng) mỗi quý.\n\
        Thanh toán bằng chuyển khoản ngân hàng vào tài khoản do bên A chỉ định.\n\
        Hạn thanh toán: ngày 15 của tháng đầu tiên mỗi quý.\n\n\
        Điều 4: Thời hạn hợp đồng\n\
        Hợp đồng có hiệu lực từ ngày ký và kết thúc sau 12 tháng.\n\
        Hai bên có thể gia hạn bằng phụ lục hợp đồng.";

    let chunks = chunk_document(content);
    assert!(chunks.len() >= 2, "Long doc should produce multiple chunks, got {}", chunks.len());

    // Verify line numbers are 1-based and non-overlapping start_line increases
    for (i, chunk) in chunks.iter().enumerate() {
        assert!(chunk.start_line >= 1, "chunk {} start_line should be >= 1", i);
        assert!(chunk.end_line >= chunk.start_line, "chunk {} end_line < start_line", i);
        assert!(!chunk.content.is_empty(), "chunk {} is empty", i);
    }

    // Verify no content is lost: every line should appear in at least one chunk
    for (line_num, line) in content.lines().enumerate() {
        let line_1based = line_num + 1;
        let found = chunks.iter().any(|c| c.start_line <= line_1based && c.end_line >= line_1based);
        assert!(found, "Line {} not covered by any chunk: {}", line_1based, line);
    }
}

#[test]
fn chunk_preserves_line_numbers() {
    let lines: Vec<String> = (1..=20).map(|i| format!("Line number {} content here.", i)).collect();
    let content = lines.join("\n");
    let chunks = chunk_document(&content);

    assert!(chunks.len() >= 1);
    assert_eq!(chunks[0].start_line, 1);

    // Last chunk should cover the last line
    let last = chunks.last().unwrap();
    assert!(last.end_line >= 20, "Last chunk should cover line 20, got end_line={}", last.end_line);
}

#[test]
fn chunk_single_line_document() {
    let content = "x".repeat(1000);
    let chunks = chunk_document(&content);
    // Single line, but > 500 chars — should still produce chunk(s)
    assert!(chunks.len() >= 1);
    assert_eq!(chunks[0].start_line, 1);
}

#[test]
fn chunk_emoji_and_special_chars() {
    let content = "🎉 Chào mừng! 🎊\n\
        Đây là dòng có emoji 🚀 và ký tự đặc biệt: é, ñ, ü, ß\n\
        日本語テスト\n\
        한국어 테스트\n"
        .repeat(30); // Make it long enough to chunk
    let chunks = chunk_document(&content);
    assert!(chunks.len() >= 1);
    for chunk in &chunks {
        assert!(!chunk.content.is_empty());
    }
}

// --- DB chunk tests ---

#[test]
fn db_save_and_search_chunks() {
    let db = Database::open(":memory:").expect("open in-memory db");
    let user_id = 1u64;

    // Save a document
    let doc_id = db.save_document(user_id, "Hợp đồng ABC", "full content here", None, None).unwrap();

    // Save chunks
    let chunks = vec![
        (0usize, 1usize, 5usize, "Điều 1: Phạm vi áp dụng cho hợp đồng"),
        (1, 6, 10, "Điều 2: Quyền nghĩa vụ các bên tham gia"),
        (2, 11, 15, "Điều 3: Phương thức thanh toán chuyển khoản"),
    ];
    let ids = db.save_chunks(doc_id, &chunks).unwrap();
    assert_eq!(ids.len(), 3);

    // FTS search
    let results = db.search_chunks_fts(user_id, "thanh toán").unwrap();
    assert!(!results.is_empty(), "FTS should find 'thanh toán'");

    let first = &results[0];
    assert_eq!(first.1, doc_id); // doc_id matches
    assert!(first.4 > 0); // start_line > 0
}

#[test]
fn db_unchunked_docs() {
    let db = Database::open(":memory:").expect("open in-memory db");
    let user_id = 1u64;

    let doc_id = db.save_document(user_id, "Test", "content", None, None).unwrap();
    let unchunked = db.get_unchunked_doc_ids().unwrap();
    assert_eq!(unchunked.len(), 1);
    assert_eq!(unchunked[0].0, doc_id);

    // After saving chunks, should not appear
    db.save_chunks(doc_id, &[(0, 1, 1, "content")]).unwrap();
    let unchunked = db.get_unchunked_doc_ids().unwrap();
    assert!(unchunked.is_empty());
}

#[test]
fn db_update_and_load_embeddings() {
    let db = Database::open(":memory:").expect("open in-memory db");
    let user_id = 1u64;

    let doc_id = db.save_document(user_id, "Test", "content", None, None).unwrap();
    let ids = db.save_chunks(doc_id, &[(0, 1, 1, "chunk text")]).unwrap();

    // Create fake embedding
    let fake_emb: Vec<f32> = (0..128).map(|i| i as f32 / 128.0).collect();
    let blob = embedding_to_bytes(&fake_emb);
    db.update_chunk_embeddings(&ids, &[blob]).unwrap();

    // Load and verify
    let loaded = db.load_all_embeddings(user_id).unwrap();
    assert_eq!(loaded.len(), 1);
    let recovered = bytes_to_embedding(&loaded[0].7);
    assert_eq!(recovered.len(), 128);
    assert!((recovered[0] - 0.0).abs() < 1e-6);
    assert!((recovered[1] - 1.0/128.0).abs() < 1e-6);
}

// --- Embedding utility tests ---

#[test]
fn cosine_similarity_identical() {
    let a = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&a, &a);
    assert!((sim - 1.0).abs() < 1e-5);
}

#[test]
fn cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0];
    let b = vec![0.0, 1.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-5);
}

#[test]
fn cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim - (-1.0)).abs() < 1e-5);
}

#[test]
fn embedding_bytes_roundtrip() {
    let original: Vec<f32> = vec![0.1, -0.5, 3.14, 0.0, f32::MAX, f32::MIN];
    let bytes = embedding_to_bytes(&original);
    let recovered = bytes_to_embedding(&bytes);
    assert_eq!(original, recovered);
}

#[test]
fn embedding_bytes_empty() {
    let bytes = embedding_to_bytes(&[]);
    assert!(bytes.is_empty());
    let recovered = bytes_to_embedding(&bytes);
    assert!(recovered.is_empty());
}
