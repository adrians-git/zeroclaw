use async_trait::async_trait;
use serde::{Deserialize, Serialize};

// ── Shared message types for tool calling ────────────────────────

/// A message in the conversation history (user, assistant, tool).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallRequest>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: "system".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: "user".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: "assistant".into(),
            content: Some(content.into()),
            tool_calls: None,
            tool_call_id: None,
        }
    }
}

/// A tool call request from the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallRequest {
    pub id: String,
    pub name: String,
    /// JSON-encoded arguments string.
    pub arguments: String,
}

/// Structured response from an LLM (may contain tool calls).
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: Option<String>,
    pub tool_calls: Vec<ToolCallRequest>,
    /// Normalized finish reason: "stop", "tool_calls", "length", etc.
    pub finish_reason: String,
    pub usage: Option<UsageInfo>,
}

/// Token usage information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ── Model listing ──────────────────────────────────────────────

/// Metadata about an available model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInfo {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub owned_by: Option<String>,
}

// ── Legacy types (used by upstream conversation system) ─────────

/// A tool call requested by the LLM (legacy, used by ConversationMessage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    pub name: String,
    pub arguments: String,
}

/// An LLM response that may contain text, tool calls, or both (legacy).
#[derive(Debug, Clone)]
pub struct ChatResponse {
    /// Text content of the response (may be empty if only tool calls).
    pub text: Option<String>,
    /// Tool calls requested by the LLM.
    pub tool_calls: Vec<ToolCall>,
}

impl ChatResponse {
    /// True when the LLM wants to invoke at least one tool.
    pub fn has_tool_calls(&self) -> bool {
        !self.tool_calls.is_empty()
    }

    /// Convenience: return text content or empty string.
    pub fn text_or_empty(&self) -> &str {
        self.text.as_deref().unwrap_or("")
    }
}

/// A tool result to feed back to the LLM.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResultMessage {
    pub tool_call_id: String,
    pub content: String,
}

/// A message in a multi-turn conversation, including tool interactions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum ConversationMessage {
    /// Regular chat message (system, user, assistant).
    Chat(ChatMessage),
    /// Tool calls from the assistant (stored for history fidelity).
    AssistantToolCalls {
        text: Option<String>,
        tool_calls: Vec<ToolCall>,
    },
    /// Result of a tool execution, fed back to the LLM.
    ToolResult(ToolResultMessage),
}

// ── Provider trait ──────────────────────────────────────────────

#[async_trait]
pub trait Provider: Send + Sync {
    async fn chat(&self, message: &str, model: &str, temperature: f64) -> anyhow::Result<String> {
        self.chat_with_system(None, message, model, temperature)
            .await
    }

    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String>;

    /// Multi-turn conversation. Default implementation extracts the last user
    /// message and delegates to `chat_with_system`.
    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let system = messages
            .iter()
            .find(|m| m.role == "system")
            .and_then(|m| m.content.as_deref());
        let last_user = messages
            .iter()
            .rfind(|m| m.role == "user")
            .and_then(|m| m.content.as_deref())
            .unwrap_or("");
        self.chat_with_system(system, last_user, model, temperature)
            .await
    }

    /// Send a conversation with tool definitions and get a structured response.
    ///
    /// Default implementation delegates to `chat_with_system` for backward
    /// compatibility — providers that support native tool calling should override.
    async fn chat_with_tools(
        &self,
        system_prompt: Option<&str>,
        messages: &[ChatMessage],
        _tools: &[crate::tools::ToolSpec],
        model: &str,
        temperature: f64,
        _max_tokens: u32,
    ) -> anyhow::Result<LlmResponse> {
        // Flatten messages into a single string for providers that don't support tools.
        let combined: String = messages
            .iter()
            .filter_map(|m| m.content.as_deref())
            .collect::<Vec<_>>()
            .join("\n\n");

        let text = self
            .chat_with_system(system_prompt, &combined, model, temperature)
            .await?;

        Ok(LlmResponse {
            content: Some(text),
            tool_calls: vec![],
            finish_reason: "stop".to_string(),
            usage: None,
        })
    }

    /// Warm up the HTTP connection pool (TLS handshake, DNS, HTTP/2 setup).
    /// Default implementation is a no-op; providers with HTTP clients should override.
    async fn warmup(&self) -> anyhow::Result<()> {
        Ok(())
    }

    /// List available models from this provider.
    /// Default implementation returns an empty list (provider doesn't support model listing).
    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        Ok(vec![])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chat_message_constructors() {
        let sys = ChatMessage::system("Be helpful");
        assert_eq!(sys.role, "system");
        assert_eq!(sys.content.as_deref(), Some("Be helpful"));

        let user = ChatMessage::user("Hello");
        assert_eq!(user.role, "user");

        let asst = ChatMessage::assistant("Hi there");
        assert_eq!(asst.role, "assistant");
    }

    #[test]
    fn chat_response_helpers() {
        let empty = ChatResponse {
            text: None,
            tool_calls: vec![],
        };
        assert!(!empty.has_tool_calls());
        assert_eq!(empty.text_or_empty(), "");

        let with_tools = ChatResponse {
            text: Some("Let me check".into()),
            tool_calls: vec![ToolCall {
                id: "1".into(),
                name: "shell".into(),
                arguments: "{}".into(),
            }],
        };
        assert!(with_tools.has_tool_calls());
        assert_eq!(with_tools.text_or_empty(), "Let me check");
    }

    #[test]
    fn tool_call_serialization() {
        let tc = ToolCall {
            id: "call_123".into(),
            name: "file_read".into(),
            arguments: r#"{"path":"test.txt"}"#.into(),
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("call_123"));
        assert!(json.contains("file_read"));
    }

    #[test]
    fn conversation_message_variants() {
        let chat = ConversationMessage::Chat(ChatMessage::user("hi"));
        let json = serde_json::to_string(&chat).unwrap();
        assert!(json.contains("\"type\":\"Chat\""));

        let tool_result = ConversationMessage::ToolResult(ToolResultMessage {
            tool_call_id: "1".into(),
            content: "done".into(),
        });
        let json = serde_json::to_string(&tool_result).unwrap();
        assert!(json.contains("\"type\":\"ToolResult\""));
    }

    #[test]
    fn llm_response_basics() {
        let resp = LlmResponse {
            content: Some("hello".into()),
            tool_calls: vec![],
            finish_reason: "stop".into(),
            usage: None,
        };
        assert!(resp.tool_calls.is_empty());
        assert_eq!(resp.content.as_deref(), Some("hello"));
    }

    #[test]
    fn tool_call_request_serialization() {
        let tc = ToolCallRequest {
            id: "call_1".into(),
            name: "shell".into(),
            arguments: r#"{"cmd":"ls"}"#.into(),
        };
        let json = serde_json::to_string(&tc).unwrap();
        assert!(json.contains("call_1"));
        assert!(json.contains("shell"));
    }

    #[test]
    fn usage_info_serialization() {
        let usage = UsageInfo {
            prompt_tokens: 100,
            completion_tokens: 50,
            total_tokens: 150,
        };
        let json = serde_json::to_string(&usage).unwrap();
        assert!(json.contains("150"));
    }
}
