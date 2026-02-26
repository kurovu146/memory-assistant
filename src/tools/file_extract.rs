//! Extract text content from binary document formats (PDF, DOCX, XLSX).

use std::io::Cursor;

/// Extract text from a PDF file's bytes.
pub fn extract_pdf(data: &[u8]) -> Result<String, String> {
    pdf_extract::extract_text_from_mem(data).map_err(|e| format!("PDF parse error: {e}"))
}

/// Extract text from a DOCX file's bytes.
/// DOCX is a ZIP archive containing XML files. We read word/document.xml
/// and strip XML tags to get plain text.
pub fn extract_docx(data: &[u8]) -> Result<String, String> {
    let cursor = Cursor::new(data);
    let mut archive = zip::ZipArchive::new(cursor).map_err(|e| format!("DOCX ZIP error: {e}"))?;

    let mut text = String::new();

    // Read word/document.xml (main content)
    let xml_content = {
        let mut file = archive
            .by_name("word/document.xml")
            .map_err(|e| format!("DOCX missing document.xml: {e}"))?;
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut file, &mut buf)
            .map_err(|e| format!("DOCX read error: {e}"))?;
        buf
    };

    // Simple XML text extraction: split on < and > to find text nodes
    // <w:t>text</w:t> and <w:t xml:space="preserve">text</w:t>
    let mut in_tag = false;
    let mut current_text = String::new();
    let mut last_was_paragraph_end = false;

    for ch in xml_content.chars() {
        match ch {
            '<' => {
                if !current_text.is_empty() {
                    text.push_str(&current_text);
                    current_text.clear();
                    last_was_paragraph_end = false;
                }
                in_tag = true;
                current_text.clear();
            }
            '>' => {
                // Check if this is a paragraph end tag
                let tag = &current_text;
                if tag.starts_with("/w:p") || tag.starts_with("w:br") {
                    if !last_was_paragraph_end {
                        text.push('\n');
                        last_was_paragraph_end = true;
                    }
                } else if tag.starts_with("/w:tr") {
                    // Table row end
                    text.push('\n');
                } else if tag.starts_with("/w:tc") {
                    // Table cell end
                    text.push('\t');
                }
                in_tag = false;
                current_text.clear();
            }
            _ => {
                if in_tag {
                    current_text.push(ch);
                } else {
                    current_text.push(ch);
                }
            }
        }
    }
    if !current_text.is_empty() {
        text.push_str(&current_text);
    }

    // Clean up excessive whitespace
    let text = text
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n");

    if text.is_empty() {
        return Err("DOCX: no text content found".to_string());
    }

    Ok(text)
}

/// Extract text from an XLSX file's bytes.
/// Uses calamine to read all sheets and format as tab-separated text.
pub fn extract_xlsx(data: &[u8]) -> Result<String, String> {
    use calamine::{Reader, Xlsx, Data};

    let cursor = Cursor::new(data);
    let mut workbook: Xlsx<_> =
        Xlsx::new(cursor).map_err(|e| format!("XLSX parse error: {e}"))?;

    let sheet_names: Vec<String> = workbook.sheet_names().to_vec();
    let mut output = String::new();

    for name in &sheet_names {
        let range = workbook
            .worksheet_range(name)
            .map_err(|e| format!("XLSX sheet '{name}' error: {e}"))?;

        if range.is_empty() {
            continue;
        }

        if sheet_names.len() > 1 {
            output.push_str(&format!("=== Sheet: {name} ===\n"));
        }

        for row in range.rows() {
            let cells: Vec<String> = row
                .iter()
                .map(|cell| match cell {
                    Data::Empty => String::new(),
                    Data::String(s) => s.clone(),
                    Data::Float(f) => {
                        if *f == (*f as i64) as f64 {
                            format!("{}", *f as i64)
                        } else {
                            format!("{f}")
                        }
                    }
                    Data::Int(i) => format!("{i}"),
                    Data::Bool(b) => format!("{b}"),
                    Data::DateTime(dt) => format!("{dt}"),
                    Data::DateTimeIso(s) => s.clone(),
                    Data::DurationIso(s) => s.clone(),
                    Data::Error(e) => format!("{e:?}"),
                })
                .collect();

            // Skip completely empty rows
            if cells.iter().all(|c| c.is_empty()) {
                continue;
            }

            output.push_str(&cells.join("\t"));
            output.push('\n');
        }

        output.push('\n');
    }

    if output.trim().is_empty() {
        return Err("XLSX: no data found".to_string());
    }

    Ok(output.trim().to_string())
}

/// Detect file type from extension and extract text.
/// Returns Ok(text) or Err(reason).
pub fn extract_document(file_name: &str, data: &[u8]) -> Result<String, String> {
    let lower = file_name.to_lowercase();

    if lower.ends_with(".pdf") {
        extract_pdf(data)
    } else if lower.ends_with(".docx") {
        extract_docx(data)
    } else if lower.ends_with(".xlsx") || lower.ends_with(".xls") {
        extract_xlsx(data)
    } else if lower.ends_with(".doc") {
        Err("Legacy .doc format not supported. Please convert to .docx".to_string())
    } else {
        Err(format!("Unsupported document format: {lower}"))
    }
}
