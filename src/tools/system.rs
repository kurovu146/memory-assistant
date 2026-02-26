use std::path::Path;
use tokio::process::Command;
use tracing::{info, warn};

/// Maximum output size for command results (chars).
const MAX_OUTPUT: usize = 30_000;
/// Maximum file read size (bytes).
const MAX_READ_BYTES: usize = 500_000;
/// Command timeout in seconds.
const CMD_TIMEOUT_SECS: u64 = 30;

// ---------- bash ----------

/// Execute a bash command and return stdout+stderr.
pub async fn bash_exec(command: &str, timeout_secs: Option<u64>) -> String {
    if command.is_empty() {
        return "Error: command cannot be empty".into();
    }

    // Safety: block obviously dangerous commands
    if is_dangerous_command(command) {
        warn!("Blocked dangerous command: {command}");
        return "Error: this command is blocked for safety reasons".into();
    }

    let timeout = timeout_secs.unwrap_or(CMD_TIMEOUT_SECS);

    info!("bash_exec: {command}");

    // Ensure common paths are available (some process managers strip PATH)
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/kuro".to_string());
    let current_path = std::env::var("PATH").unwrap_or_default();
    let full_path = format!("{home}/.cargo/bin:/usr/local/bin:/usr/bin:/bin:{current_path}");

    let result = tokio::time::timeout(
        std::time::Duration::from_secs(timeout),
        Command::new("bash")
            .arg("-c")
            .arg(command)
            .env("PATH", &full_path)
            .env("HOME", &home)
            .output(),
    )
    .await;

    match result {
        Ok(Ok(output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let exit_code = output.status.code().unwrap_or(-1);

            let mut result = String::new();
            if !stdout.is_empty() {
                result.push_str(&stdout);
            }
            if !stderr.is_empty() {
                if !result.is_empty() {
                    result.push_str("\n--- stderr ---\n");
                }
                result.push_str(&stderr);
            }
            if result.is_empty() {
                result = format!("(no output, exit code: {exit_code})");
            } else if exit_code != 0 {
                result.push_str(&format!("\n(exit code: {exit_code})"));
            }

            truncate_output(&result)
        }
        Ok(Err(e)) => format!("Error executing command: {e}"),
        Err(_) => format!("Error: command timed out after {timeout}s"),
    }
}

// ---------- file_read ----------

/// Read a file from the filesystem with optional line range.
pub async fn file_read(path: &str, offset: Option<usize>, limit: Option<usize>) -> String {
    if path.is_empty() {
        return "Error: path cannot be empty".into();
    }

    let p = Path::new(path);
    if !p.exists() {
        return format!("Error: file not found: {path}");
    }
    if p.is_dir() {
        return format!("Error: {path} is a directory, use file_list instead");
    }

    // Check file size
    match std::fs::metadata(p) {
        Ok(meta) if meta.len() > MAX_READ_BYTES as u64 => {
            return format!(
                "Error: file too large ({} bytes, max {}). Use offset/limit or grep instead.",
                meta.len(),
                MAX_READ_BYTES
            );
        }
        Err(e) => return format!("Error reading file metadata: {e}"),
        _ => {}
    }

    match tokio::fs::read_to_string(path).await {
        Ok(content) => {
            let lines: Vec<&str> = content.lines().collect();
            let start = offset.unwrap_or(0);
            let count = limit.unwrap_or(lines.len());

            if start >= lines.len() {
                return format!("Error: offset {start} exceeds file length ({} lines)", lines.len());
            }

            let end = (start + count).min(lines.len());
            let selected: Vec<String> = lines[start..end]
                .iter()
                .enumerate()
                .map(|(i, line)| format!("{:>5}\t{}", start + i + 1, line))
                .collect();

            let result = selected.join("\n");
            if end < lines.len() {
                format!("{result}\n\n... ({} more lines)", lines.len() - end)
            } else {
                result
            }
        }
        Err(e) => format!("Error reading file: {e}"),
    }
}

// ---------- file_write ----------

/// Write content to a file. Creates parent directories if needed.
pub async fn file_write(path: &str, content: &str) -> String {
    if path.is_empty() {
        return "Error: path cannot be empty".into();
    }

    // Safety: don't write to sensitive locations
    if is_sensitive_path(path) {
        warn!("Blocked write to sensitive path: {path}");
        return "Error: cannot write to this path for safety reasons".into();
    }

    let p = Path::new(path);

    // Create parent dirs if needed
    if let Some(parent) = p.parent() {
        if !parent.exists() {
            if let Err(e) = tokio::fs::create_dir_all(parent).await {
                return format!("Error creating directories: {e}");
            }
        }
    }

    info!("file_write: {path} ({} bytes)", content.len());

    match tokio::fs::write(path, content).await {
        Ok(()) => format!("Written {} bytes to {path}", content.len()),
        Err(e) => format!("Error writing file: {e}"),
    }
}

// ---------- file_list ----------

/// List directory contents with basic info (type, size).
pub async fn file_list(path: &str, recursive: bool) -> String {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/kuro".to_string());
    let dir = if path.is_empty() { &home } else { path };
    let p = Path::new(dir);

    if !p.exists() {
        return format!("Error: path not found: {dir}");
    }
    if !p.is_dir() {
        return format!("Error: {dir} is not a directory");
    }

    if recursive {
        list_recursive(p, p, 0)
    } else {
        list_flat(p)
    }
}

fn list_flat(dir: &Path) -> String {
    let mut entries = Vec::new();

    match std::fs::read_dir(dir) {
        Ok(rd) => {
            for entry in rd.flatten() {
                let meta = entry.metadata().ok();
                let name = entry.file_name().to_string_lossy().to_string();
                let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);

                if is_dir {
                    entries.push(format!("  {name}/"));
                } else {
                    entries.push(format!("  {name}  ({size} bytes)"));
                }
            }
        }
        Err(e) => return format!("Error reading directory: {e}"),
    }

    entries.sort();
    if entries.is_empty() {
        format!("{}/  (empty)", dir.display())
    } else {
        format!("{}/ ({} items)\n{}", dir.display(), entries.len(), entries.join("\n"))
    }
}

fn list_recursive(base: &Path, dir: &Path, depth: usize) -> String {
    if depth > 5 {
        return format!("{}  ... (max depth reached)", "  ".repeat(depth));
    }

    let mut result = Vec::new();
    let indent = "  ".repeat(depth);

    let dir_name = if depth == 0 {
        dir.display().to_string()
    } else {
        dir.file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| dir.display().to_string())
    };

    result.push(format!("{indent}{dir_name}/"));

    match std::fs::read_dir(dir) {
        Ok(rd) => {
            let mut entries: Vec<_> = rd.flatten().collect();
            entries.sort_by_key(|e| e.file_name());

            // Limit entries to avoid huge output
            let max_entries = 100;
            let total = entries.len();

            for entry in entries.iter().take(max_entries) {
                let meta = entry.metadata().ok();
                let is_dir = meta.as_ref().map(|m| m.is_dir()).unwrap_or(false);
                let name = entry.file_name().to_string_lossy().to_string();

                // Skip hidden dirs in recursive mode
                if is_dir && name.starts_with('.') {
                    continue;
                }

                if is_dir {
                    result.push(list_recursive(base, &entry.path(), depth + 1));
                } else {
                    let size = meta.as_ref().map(|m| m.len()).unwrap_or(0);
                    result.push(format!("{}  {name}  ({size} bytes)", "  ".repeat(depth + 1)));
                }
            }

            if total > max_entries {
                result.push(format!("{}  ... ({} more items)", "  ".repeat(depth + 1), total - max_entries));
            }
        }
        Err(e) => result.push(format!("{indent}  (error: {e})")),
    }

    result.join("\n")
}

// ---------- grep ----------

/// Search file contents using a pattern. Returns matching lines with context.
pub async fn grep_search(
    pattern: &str,
    path: Option<&str>,
    include_glob: Option<&str>,
    context_lines: Option<usize>,
) -> String {
    if pattern.is_empty() {
        return "Error: pattern cannot be empty".into();
    }

    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/kuro".to_string());
    let search_dir = path.unwrap_or(&home);
    let ctx = context_lines.unwrap_or(0);

    // Build grep/rg command
    let mut cmd_parts = vec!["rg".to_string(), "--color=never".to_string()];

    if ctx > 0 {
        cmd_parts.push(format!("-C{ctx}"));
    }

    cmd_parts.push("-n".to_string()); // line numbers
    cmd_parts.push("--max-count=50".to_string()); // limit matches per file
    cmd_parts.push("--max-filesize=1M".to_string());

    if let Some(glob) = include_glob {
        cmd_parts.push(format!("--glob={glob}"));
    }

    cmd_parts.push("--".to_string());
    cmd_parts.push(pattern.to_string());
    cmd_parts.push(search_dir.to_string());

    info!("grep: pattern={pattern}, path={search_dir}");

    // Try ripgrep first, fallback to grep
    let rg_cmd = cmd_parts.join(" ");
    let result = bash_exec(&rg_cmd, Some(15)).await;

    // If rg not found, fallback to grep
    if result.contains("not found") || result.contains("No such file") {
        let grep_cmd = format!(
            "grep -rn {} --include='{}' -- '{}' '{}'",
            if ctx > 0 { format!("-C{ctx}") } else { String::new() },
            include_glob.unwrap_or("*"),
            pattern.replace('\'', "'\\''"),
            search_dir,
        );
        bash_exec(&grep_cmd, Some(15)).await
    } else {
        result
    }
}

// ---------- glob ----------

/// Find files matching a glob pattern. Supports *, **, ?.
/// Returns matching file paths sorted by modification time (newest first).
pub async fn glob_search(pattern: &str, path: Option<&str>) -> String {
    if pattern.is_empty() {
        return "Error: pattern cannot be empty".into();
    }

    // Default to home directory so bot can search the entire user filesystem
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/kuro".to_string());
    let base_dir = path.unwrap_or(&home);
    let base = Path::new(base_dir);

    if !base.exists() {
        return format!("Error: path not found: {base_dir}");
    }

    info!("glob: pattern={pattern}, path={base_dir}");

    // Extract the filename pattern (last component)
    let name_pattern = if pattern.contains('/') || pattern.contains("**") {
        pattern.replace("**/", "*/")
            .rsplit('/')
            .next()
            .unwrap_or(pattern)
            .to_string()
    } else {
        pattern.to_string()
    };

    // Simple, cross-platform find command
    // Exclude heavy dirs via -not -path to avoid BSD/GNU find prune differences
    let excludes = [
        "*/Library/*", "*/node_modules/*", "*/.git/*", "*/.Trash/*",
        "*/.cache/*", "*/.npm/*", "*/.cargo/registry/*", "*/.rustup/*",
        "*/Pictures/*", "*/Movies/*", "*/Music/*",
    ];
    let exclude_expr: String = excludes
        .iter()
        .map(|e| format!("-not -path '{e}'"))
        .collect::<Vec<_>>()
        .join(" ");

    let cmd = format!(
        "find '{}' -type f -name '{}' {} 2>/dev/null | head -200",
        base_dir, name_pattern, exclude_expr
    );

    info!("glob cmd: {cmd}");

    let output = bash_exec(&cmd, Some(30)).await;
    info!("glob output ({} chars): {}", output.len(), &output[..output.len().min(200)]);

    if output.is_empty() || output.contains("(no output") {
        return format!("No files matching '{pattern}' found in {base_dir}");
    }

    // Collect and sort by modification time (newest first)
    let mut files: Vec<(String, u64)> = output
        .lines()
        .filter(|l| !l.is_empty() && !l.starts_with("Error") && !l.contains("--- stderr"))
        .filter_map(|path| {
            let meta = std::fs::metadata(path).ok()?;
            let mtime = meta
                .modified()
                .ok()?
                .duration_since(std::time::UNIX_EPOCH)
                .ok()?
                .as_secs();
            Some((path.to_string(), mtime))
        })
        .collect();

    files.sort_by(|a, b| b.1.cmp(&a.1)); // newest first

    if files.is_empty() {
        return format!("No files matching '{pattern}' found in {base_dir}");
    }

    let total = files.len();
    let result: Vec<String> = files
        .iter()
        .take(100)
        .map(|(path, _)| {
            let size = std::fs::metadata(path)
                .map(|m| m.len())
                .unwrap_or(0);
            format!("{path}  ({size} bytes)")
        })
        .collect();

    let mut output = format!("{total} files found:\n{}", result.join("\n"));
    if total > 100 {
        output.push_str(&format!("\n... ({} more files)", total - 100));
    }
    output
}

// ---------- Helpers ----------

fn truncate_output(s: &str) -> String {
    if s.len() > MAX_OUTPUT {
        let truncated = &s[..MAX_OUTPUT];
        format!("{truncated}\n\n... (output truncated, {} total chars)", s.len())
    } else {
        s.to_string()
    }
}

fn is_dangerous_command(cmd: &str) -> bool {
    let blocked = [
        "rm -rf /",
        "rm -rf /*",
        "mkfs.",
        "dd if=",
        "> /dev/sd",
        "chmod -R 777 /",
        ":(){ :|:& };:",
        "shutdown",
        "reboot",
        "init 0",
        "init 6",
        "halt",
        "poweroff",
    ];
    let lower = cmd.to_lowercase();
    blocked.iter().any(|b| lower.contains(b))
}

fn is_sensitive_path(path: &str) -> bool {
    let blocked_prefixes = [
        "/etc/passwd",
        "/etc/shadow",
        "/etc/sudoers",
        "/root/.ssh",
        "/boot/",
        "/proc/",
        "/sys/",
    ];
    blocked_prefixes.iter().any(|p| path.starts_with(p))
}
