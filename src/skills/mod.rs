/// Build the full system prompt with memory context
pub fn build_system_prompt(base_prompt: &str, memory_context: &str) -> String {
    let mut prompt = base_prompt.to_string();
    if !memory_context.is_empty() {
        prompt.push_str(memory_context);
    }
    prompt
}
