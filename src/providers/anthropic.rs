use crate::providers::traits::{
    ChatMessage, ContentPart, LlmResponse, ModelInfo, Provider, ToolCallRequest, UsageInfo,
};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct AnthropicProvider {
    credential: Option<String>,
    base_url: String,
    client: Client,
}

// ── Wire types for chat_with_system (simple string content) ─────

#[derive(Debug, Serialize)]
struct SimpleChatRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<SimpleMessage>,
    temperature: f64,
}

#[derive(Debug, Serialize)]
struct SimpleMessage {
    role: String,
    content: String,
}

// ── Wire types for chat_with_tools (polymorphic content) ────────

#[derive(Debug, Serialize)]
struct ToolChatRequest {
    model: String,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
    messages: Vec<ToolMessage>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
}

#[derive(Debug, Serialize)]
struct ToolMessage {
    role: String,
    content: MessageContent,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(untagged)]
enum MessageContent {
    Text(String),
    Blocks(Vec<ContentBlock>),
}

#[derive(Debug, Serialize, Deserialize, Clone)]
#[serde(tag = "type")]
enum ContentBlock {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image")]
    Image { source: ImageSource },
    #[serde(rename = "tool_use")]
    ToolUse {
        id: String,
        name: String,
        input: serde_json::Value,
    },
    #[serde(rename = "tool_result")]
    ToolResult {
        tool_use_id: String,
        content: String,
    },
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ImageSource {
    #[serde(rename = "type")]
    source_type: String,
    media_type: String,
    data: String,
}

#[derive(Debug, Serialize)]
struct ToolDefinition {
    name: String,
    description: String,
    input_schema: serde_json::Value,
}

// ── Response types ──────────────────────────────────────────────

#[derive(Debug, Deserialize)]
struct ChatResponse {
    content: Vec<ResponseBlock>,
    #[serde(default)]
    stop_reason: Option<String>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum ResponseBlock {
    ToolUse {
        #[allow(dead_code)]
        #[serde(rename = "type")]
        block_type: String,
        id: String,
        name: String,
        input: serde_json::Value,
    },
    Text {
        #[allow(dead_code)]
        #[serde(rename = "type")]
        block_type: String,
        text: String,
    },
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    #[serde(default)]
    input_tokens: u32,
    #[serde(default)]
    output_tokens: u32,
}

impl AnthropicProvider {
    pub fn new(api_key: Option<&str>) -> Self {
        Self::with_base_url(api_key, None)
    }

    pub fn with_base_url(api_key: Option<&str>, base_url: Option<&str>) -> Self {
        let base_url = base_url
            .map(|u| u.trim_end_matches('/'))
            .unwrap_or("https://api.anthropic.com")
            .to_string();
        Self {
            credential: api_key
                .map(str::trim)
                .filter(|k| !k.is_empty())
                .map(ToString::to_string),
            base_url,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }

    fn is_setup_token(token: &str) -> bool {
        token.starts_with("sk-ant-oat01-")
    }

    fn api_key(&self) -> anyhow::Result<&str> {
        self.credential.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "Anthropic API key not set. Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN."
            )
        })
    }
}

fn convert_tools(tools: &[ToolSpec]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|t| ToolDefinition {
            name: t.name.clone(),
            description: t.description.clone(),
            input_schema: t.parameters.clone(),
        })
        .collect()
}

fn convert_messages(messages: &[ChatMessage]) -> Vec<ToolMessage> {
    messages
        .iter()
        .map(|m| {
            if m.role == "tool" {
                // Tool result → Anthropic "user" role with tool_result content block
                ToolMessage {
                    role: "user".to_string(),
                    content: MessageContent::Blocks(vec![ContentBlock::ToolResult {
                        tool_use_id: m.tool_call_id.clone().unwrap_or_default(),
                        content: m.text_content_lossy().unwrap_or_default(),
                    }]),
                }
            } else if m.role == "assistant" && m.tool_calls.is_some() {
                // Assistant with tool calls → content blocks
                let mut blocks = Vec::new();
                if let Some(text) = m.text_content() {
                    if !text.is_empty() {
                        blocks.push(ContentBlock::Text {
                            text: text.to_string(),
                        });
                    }
                }
                for tc in m.tool_calls.as_ref().unwrap() {
                    let input: serde_json::Value =
                        serde_json::from_str(&tc.arguments).unwrap_or(serde_json::json!({}));
                    blocks.push(ContentBlock::ToolUse {
                        id: tc.id.clone(),
                        name: tc.name.clone(),
                        input,
                    });
                }
                ToolMessage {
                    role: "assistant".to_string(),
                    content: MessageContent::Blocks(blocks),
                }
            } else {
                // Regular user/assistant message — handle multimodal parts
                match &m.content {
                    Some(crate::providers::traits::MessageContent::Parts(parts)) => {
                        let blocks = parts
                            .iter()
                            .map(|p| match p {
                                ContentPart::Text { text } => {
                                    ContentBlock::Text { text: text.clone() }
                                }
                                ContentPart::Image { data, media_type } => {
                                    ContentBlock::Image {
                                        source: ImageSource {
                                            source_type: "base64".to_string(),
                                            media_type: media_type.clone(),
                                            data: data.clone(),
                                        },
                                    }
                                }
                            })
                            .collect();
                        ToolMessage {
                            role: m.role.clone(),
                            content: MessageContent::Blocks(blocks),
                        }
                    }
                    _ => ToolMessage {
                        role: m.role.clone(),
                        content: MessageContent::Text(
                            m.text_content_lossy().unwrap_or_default(),
                        ),
                    },
                }
            }
        })
        .collect()
}

fn parse_response(resp: ChatResponse) -> LlmResponse {
    let mut text_parts = Vec::new();
    let mut tool_calls = Vec::new();

    for block in &resp.content {
        match block {
            ResponseBlock::Text { text, .. } => {
                text_parts.push(text.clone());
            }
            ResponseBlock::ToolUse {
                id, name, input, ..
            } => {
                tool_calls.push(ToolCallRequest {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: serde_json::to_string(input).unwrap_or_else(|_| "{}".to_string()),
                });
            }
        }
    }

    let content = if text_parts.is_empty() {
        None
    } else {
        Some(text_parts.join(""))
    };

    let finish_reason = match resp.stop_reason.as_deref() {
        Some("tool_use") => "tool_calls".to_string(),
        Some("end_turn") | None if !tool_calls.is_empty() => "tool_calls".to_string(),
        Some("end_turn") | None => "stop".to_string(),
        Some(r) => r.to_string(),
    };

    let usage = resp.usage.map(|u| UsageInfo {
        prompt_tokens: u.input_tokens,
        completion_tokens: u.output_tokens,
        total_tokens: u.input_tokens + u.output_tokens,
    });

    LlmResponse {
        content,
        tool_calls,
        finish_reason,
        usage,
    }
}

#[async_trait]
impl Provider for AnthropicProvider {
    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let credential = self.credential.as_ref().ok_or_else(|| {
            anyhow::anyhow!(
                "Anthropic credentials not set. Set ANTHROPIC_API_KEY or ANTHROPIC_OAUTH_TOKEN (setup-token)."
            )
        })?;

        let request = SimpleChatRequest {
            model: model.to_string(),
            max_tokens: 4096,
            system: system_prompt.map(ToString::to_string),
            messages: vec![SimpleMessage {
                role: "user".to_string(),
                content: message.to_string(),
            }],
            temperature,
        };

        let mut request = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request);

        if Self::is_setup_token(credential) {
            request = request.header("Authorization", format!("Bearer {credential}"));
        } else {
            request = request.header("x-api-key", credential);
        }

        let response = request.send().await?;

        if !response.status().is_success() {
            return Err(super::api_error("Anthropic", response).await);
        }

        let chat_response: ChatResponse = response.json().await?;

        chat_response
            .content
            .into_iter()
            .find_map(|block| match block {
                ResponseBlock::Text { text, .. } => Some(text),
                ResponseBlock::ToolUse { .. } => None,
            })
            .ok_or_else(|| anyhow::anyhow!("No response from Anthropic"))
    }

    async fn chat_with_tools(
        &self,
        system_prompt: Option<&str>,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
        model: &str,
        temperature: f64,
        max_tokens: u32,
    ) -> anyhow::Result<LlmResponse> {
        let api_key = self.api_key()?;

        let wire_messages = convert_messages(messages);

        let tool_defs = if tools.is_empty() {
            None
        } else {
            Some(convert_tools(tools))
        };

        let request = ToolChatRequest {
            model: model.to_string(),
            max_tokens,
            system: system_prompt.map(ToString::to_string),
            messages: wire_messages,
            temperature,
            tools: tool_defs,
        };

        let response = self
            .client
            .post(format!("{}/v1/messages", self.base_url))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("Anthropic", response).await);
        }

        let chat_response: ChatResponse = response.json().await?;
        Ok(parse_response(chat_response))
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let api_key = self.api_key()?;
        let response = self
            .client
            .get(format!("{}/v1/models", self.base_url))
            .header("x-api-key", api_key)
            .header("anthropic-version", "2023-06-01")
            .send()
            .await?;
        if !response.status().is_success() {
            return Err(super::api_error("Anthropic", response).await);
        }
        let body: serde_json::Value = response.json().await?;
        let mut models: Vec<ModelInfo> = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        Some(ModelInfo {
                            id: m["id"].as_str()?.to_string(),
                            owned_by: None,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
        models.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(models)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_with_key() {
        let p = AnthropicProvider::new(Some("sk-ant-test123"));
        assert!(p.credential.is_some());
        assert_eq!(p.credential.as_deref(), Some("sk-ant-test123"));
        assert_eq!(p.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn creates_without_key() {
        let p = AnthropicProvider::new(None);
        assert!(p.credential.is_none());
        assert_eq!(p.base_url, "https://api.anthropic.com");
    }

    #[test]
    fn creates_with_empty_key() {
        let p = AnthropicProvider::new(Some(""));
        assert!(p.credential.is_none());
    }

    #[test]
    fn creates_with_whitespace_key() {
        let p = AnthropicProvider::new(Some("  sk-ant-test123  "));
        assert!(p.credential.is_some());
        assert_eq!(p.credential.as_deref(), Some("sk-ant-test123"));
    }

    #[test]
    fn creates_with_custom_base_url() {
        let p =
            AnthropicProvider::with_base_url(Some("sk-ant-test"), Some("https://api.example.com"));
        assert_eq!(p.base_url, "https://api.example.com");
        assert_eq!(p.credential.as_deref(), Some("sk-ant-test"));
    }

    #[test]
    fn custom_base_url_trims_trailing_slash() {
        let p = AnthropicProvider::with_base_url(None, Some("https://api.example.com/"));
        assert_eq!(p.base_url, "https://api.example.com");
    }

    #[test]
    fn default_base_url_when_none_provided() {
        let p = AnthropicProvider::with_base_url(None, None);
        assert_eq!(p.base_url, "https://api.anthropic.com");
    }

    #[tokio::test]
    async fn chat_fails_without_key() {
        let p = AnthropicProvider::new(None);
        let result = p
            .chat_with_system(None, "hello", "claude-3-opus", 0.7)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("credentials not set"),
            "Expected key error, got: {err}"
        );
    }

    #[test]
    fn setup_token_detection_works() {
        assert!(AnthropicProvider::is_setup_token("sk-ant-oat01-abcdef"));
        assert!(!AnthropicProvider::is_setup_token("sk-ant-api-key"));
    }

    #[tokio::test]
    async fn chat_with_system_fails_without_key() {
        let p = AnthropicProvider::new(None);
        let result = p
            .chat_with_system(Some("You are ZeroClaw"), "hello", "claude-3-opus", 0.7)
            .await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn chat_with_tools_fails_without_key() {
        let p = AnthropicProvider::new(None);
        let messages = vec![ChatMessage::user("hello")];
        let result = p
            .chat_with_tools(None, &messages, &[], "claude-3-opus", 0.7, 4096)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("API key not set"),
            "Expected key error, got: {err}"
        );
    }

    // ── Simple chat request serialization ────────────────────

    #[test]
    fn chat_request_serializes_without_system() {
        let req = SimpleChatRequest {
            model: "claude-3-opus".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![SimpleMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: 0.7,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(
            !json.contains("system"),
            "system field should be skipped when None"
        );
        assert!(json.contains("claude-3-opus"));
        assert!(json.contains("hello"));
    }

    #[test]
    fn chat_request_serializes_with_system() {
        let req = SimpleChatRequest {
            model: "claude-3-opus".to_string(),
            max_tokens: 4096,
            system: Some("You are ZeroClaw".to_string()),
            messages: vec![SimpleMessage {
                role: "user".to_string(),
                content: "hello".to_string(),
            }],
            temperature: 0.7,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"system\":\"You are ZeroClaw\""));
    }

    // ── Response deserialization ─────────────────────────────

    #[test]
    fn chat_response_deserializes() {
        let json = r#"{"content":[{"type":"text","text":"Hello there!"}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ResponseBlock::Text { text, .. } => assert_eq!(text, "Hello there!"),
            _ => panic!("Expected text block"),
        }
    }

    #[test]
    fn chat_response_empty_content() {
        let json = r#"{"content":[]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.content.is_empty());
    }

    #[test]
    fn chat_response_multiple_blocks() {
        let json =
            r#"{"content":[{"type":"text","text":"First"},{"type":"text","text":"Second"}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        match &resp.content[0] {
            ResponseBlock::Text { text, .. } => assert_eq!(text, "First"),
            _ => panic!("Expected text block"),
        }
        match &resp.content[1] {
            ResponseBlock::Text { text, .. } => assert_eq!(text, "Second"),
            _ => panic!("Expected text block"),
        }
    }

    #[test]
    fn temperature_range_serializes() {
        for temp in [0.0, 0.5, 1.0, 2.0] {
            let req = SimpleChatRequest {
                model: "claude-3-opus".to_string(),
                max_tokens: 4096,
                system: None,
                messages: vec![],
                temperature: temp,
            };
            let json = serde_json::to_string(&req).unwrap();
            assert!(json.contains(&format!("{temp}")));
        }
    }

    // ── Tool definition serialization ────────────────────────

    #[test]
    fn tool_definition_uses_input_schema() {
        let def = ToolDefinition {
            name: "shell".to_string(),
            description: "Run shell commands".to_string(),
            input_schema: serde_json::json!({
                "type": "object",
                "properties": {
                    "command": {"type": "string"}
                },
                "required": ["command"]
            }),
        };
        let json = serde_json::to_string(&def).unwrap();
        assert!(json.contains("\"input_schema\""));
        assert!(!json.contains("\"parameters\""));
        assert!(json.contains("\"name\":\"shell\""));
        assert!(json.contains("\"command\""));
    }

    #[test]
    fn convert_tools_maps_parameters_to_input_schema() {
        let specs = vec![ToolSpec {
            name: "read_file".to_string(),
            description: "Read a file".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}),
        }];
        let defs = convert_tools(&specs);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].name, "read_file");
        assert_eq!(defs[0].input_schema, serde_json::json!({"type": "object", "properties": {"path": {"type": "string"}}}));
    }

    #[test]
    fn tool_chat_request_serializes_with_tools() {
        let req = ToolChatRequest {
            model: "claude-3-opus".to_string(),
            max_tokens: 4096,
            system: Some("You are helpful".to_string()),
            messages: vec![ToolMessage {
                role: "user".to_string(),
                content: MessageContent::Text("hello".to_string()),
            }],
            temperature: 0.7,
            tools: Some(vec![ToolDefinition {
                name: "shell".to_string(),
                description: "Run commands".to_string(),
                input_schema: serde_json::json!({"type": "object"}),
            }]),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"input_schema\""));
        assert!(json.contains("\"name\":\"shell\""));
        assert!(json.contains("\"system\":\"You are helpful\""));
        assert!(json.contains("\"max_tokens\":4096"));
    }

    #[test]
    fn tool_chat_request_omits_tools_when_none() {
        let req = ToolChatRequest {
            model: "claude-3-opus".to_string(),
            max_tokens: 4096,
            system: None,
            messages: vec![],
            temperature: 0.7,
            tools: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
        assert!(!json.contains("system"));
    }

    // ── Response deserialization with tool_use blocks ─────────

    #[test]
    fn response_deserializes_with_tool_use() {
        let json = r#"{
            "content":[{
                "type":"tool_use",
                "id":"toolu_abc123",
                "name":"shell",
                "input":{"command":"ls -la"}
            }],
            "stop_reason":"tool_use",
            "usage":{"input_tokens":100,"output_tokens":50}
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 1);
        match &resp.content[0] {
            ResponseBlock::ToolUse { id, name, input, .. } => {
                assert_eq!(id, "toolu_abc123");
                assert_eq!(name, "shell");
                assert_eq!(input["command"], "ls -la");
            }
            _ => panic!("Expected tool_use block"),
        }
        assert_eq!(resp.stop_reason.as_deref(), Some("tool_use"));
        assert_eq!(resp.usage.as_ref().unwrap().input_tokens, 100);
        assert_eq!(resp.usage.as_ref().unwrap().output_tokens, 50);
    }

    #[test]
    fn response_deserializes_mixed_text_and_tool_use() {
        let json = r#"{
            "content":[
                {"type":"text","text":"Let me check that."},
                {"type":"tool_use","id":"toolu_1","name":"read_file","input":{"path":"/tmp/test"}}
            ],
            "stop_reason":"tool_use"
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.content.len(), 2);
        match &resp.content[0] {
            ResponseBlock::Text { text, .. } => assert_eq!(text, "Let me check that."),
            _ => panic!("Expected text block"),
        }
        match &resp.content[1] {
            ResponseBlock::ToolUse { name, .. } => assert_eq!(name, "read_file"),
            _ => panic!("Expected tool_use block"),
        }
    }

    // ── Tool result message conversion ──────────────────────

    #[test]
    fn convert_messages_maps_tool_result() {
        use crate::providers::traits::MessageContent as SharedContent;
        let msgs = vec![ChatMessage {
            role: "tool".to_string(),
            content: Some(SharedContent::Text("file list here".to_string())),
            tool_calls: None,
            tool_call_id: Some("toolu_abc".to_string()),
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
        match &wire[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolResult {
                        tool_use_id,
                        content,
                    } => {
                        assert_eq!(tool_use_id, "toolu_abc");
                        assert_eq!(content, "file list here");
                    }
                    _ => panic!("Expected ToolResult block"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    #[test]
    fn convert_messages_maps_tool_result_serialization() {
        use crate::providers::traits::MessageContent as SharedContent;
        let msgs = vec![ChatMessage {
            role: "tool".to_string(),
            content: Some(SharedContent::Text("output".to_string())),
            tool_calls: None,
            tool_call_id: Some("toolu_1".to_string()),
        }];
        let wire = convert_messages(&msgs);
        let json = serde_json::to_string(&wire[0]).unwrap();
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("\"type\":\"tool_result\""));
        assert!(json.contains("\"tool_use_id\":\"toolu_1\""));
        assert!(json.contains("\"content\":\"output\""));
    }

    #[test]
    fn convert_messages_maps_assistant_with_tool_calls() {
        use crate::providers::traits::MessageContent as SharedContent;
        let msgs = vec![ChatMessage {
            role: "assistant".to_string(),
            content: Some(SharedContent::Text("I'll run that.".to_string())),
            tool_calls: Some(vec![ToolCallRequest {
                id: "toolu_xyz".to_string(),
                name: "shell".to_string(),
                arguments: r#"{"command":"ls"}"#.to_string(),
            }]),
            tool_call_id: None,
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire[0].role, "assistant");
        match &wire[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 2);
                match &blocks[0] {
                    ContentBlock::Text { text } => assert_eq!(text, "I'll run that."),
                    _ => panic!("Expected Text block"),
                }
                match &blocks[1] {
                    ContentBlock::ToolUse { id, name, input } => {
                        assert_eq!(id, "toolu_xyz");
                        assert_eq!(name, "shell");
                        assert_eq!(input["command"], "ls");
                    }
                    _ => panic!("Expected ToolUse block"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    #[test]
    fn convert_messages_assistant_tool_calls_without_text() {
        let msgs = vec![ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCallRequest {
                id: "toolu_1".to_string(),
                name: "read_file".to_string(),
                arguments: r#"{"path":"/tmp"}"#.to_string(),
            }]),
            tool_call_id: None,
        }];
        let wire = convert_messages(&msgs);
        match &wire[0].content {
            MessageContent::Blocks(blocks) => {
                assert_eq!(blocks.len(), 1);
                match &blocks[0] {
                    ContentBlock::ToolUse { name, .. } => assert_eq!(name, "read_file"),
                    _ => panic!("Expected ToolUse block"),
                }
            }
            _ => panic!("Expected Blocks content"),
        }
    }

    #[test]
    fn convert_messages_preserves_plain_user_message() {
        let msgs = vec![ChatMessage::user("hello")];
        let wire = convert_messages(&msgs);
        assert_eq!(wire[0].role, "user");
        match &wire[0].content {
            MessageContent::Text(text) => assert_eq!(text, "hello"),
            _ => panic!("Expected Text content"),
        }
    }

    // ── parse_response tests ────────────────────────────────

    #[test]
    fn parse_response_text_only() {
        let resp = ChatResponse {
            content: vec![ResponseBlock::Text {
                block_type: "text".to_string(),
                text: "Hello!".to_string(),
            }],
            stop_reason: Some("end_turn".to_string()),
            usage: None,
        };
        let llm = parse_response(resp);
        assert_eq!(llm.content.as_deref(), Some("Hello!"));
        assert!(llm.tool_calls.is_empty());
        assert_eq!(llm.finish_reason, "stop");
    }

    #[test]
    fn parse_response_tool_use() {
        let resp = ChatResponse {
            content: vec![ResponseBlock::ToolUse {
                block_type: "tool_use".to_string(),
                id: "toolu_abc".to_string(),
                name: "shell".to_string(),
                input: serde_json::json!({"command": "ls"}),
            }],
            stop_reason: Some("tool_use".to_string()),
            usage: Some(WireUsage {
                input_tokens: 100,
                output_tokens: 50,
            }),
        };
        let llm = parse_response(resp);
        assert!(llm.content.is_none());
        assert_eq!(llm.tool_calls.len(), 1);
        assert_eq!(llm.tool_calls[0].id, "toolu_abc");
        assert_eq!(llm.tool_calls[0].name, "shell");
        assert_eq!(llm.tool_calls[0].arguments, r#"{"command":"ls"}"#);
        assert_eq!(llm.finish_reason, "tool_calls");
        let usage = llm.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[test]
    fn parse_response_mixed_text_and_tool_use() {
        let resp = ChatResponse {
            content: vec![
                ResponseBlock::Text {
                    block_type: "text".to_string(),
                    text: "Let me check.".to_string(),
                },
                ResponseBlock::ToolUse {
                    block_type: "tool_use".to_string(),
                    id: "toolu_1".to_string(),
                    name: "read_file".to_string(),
                    input: serde_json::json!({"path": "/tmp/test"}),
                },
            ],
            stop_reason: Some("tool_use".to_string()),
            usage: None,
        };
        let llm = parse_response(resp);
        assert_eq!(llm.content.as_deref(), Some("Let me check."));
        assert_eq!(llm.tool_calls.len(), 1);
        assert_eq!(llm.tool_calls[0].name, "read_file");
        assert_eq!(llm.finish_reason, "tool_calls");
    }

    #[test]
    fn parse_response_empty_content() {
        let resp = ChatResponse {
            content: vec![],
            stop_reason: Some("end_turn".to_string()),
            usage: None,
        };
        let llm = parse_response(resp);
        assert!(llm.content.is_none());
        assert!(llm.tool_calls.is_empty());
        assert_eq!(llm.finish_reason, "stop");
    }

    #[test]
    fn parse_response_normalizes_tool_use_to_tool_calls() {
        let resp = ChatResponse {
            content: vec![ResponseBlock::ToolUse {
                block_type: "tool_use".to_string(),
                id: "t1".to_string(),
                name: "test".to_string(),
                input: serde_json::json!({}),
            }],
            stop_reason: Some("tool_use".to_string()),
            usage: None,
        };
        let llm = parse_response(resp);
        assert_eq!(llm.finish_reason, "tool_calls");
    }

    #[test]
    fn parse_response_normalizes_end_turn_to_stop() {
        let resp = ChatResponse {
            content: vec![ResponseBlock::Text {
                block_type: "text".to_string(),
                text: "Done".to_string(),
            }],
            stop_reason: Some("end_turn".to_string()),
            usage: None,
        };
        let llm = parse_response(resp);
        assert_eq!(llm.finish_reason, "stop");
    }

    #[test]
    fn parse_response_infers_tool_calls_when_stop_reason_missing() {
        let resp = ChatResponse {
            content: vec![ResponseBlock::ToolUse {
                block_type: "tool_use".to_string(),
                id: "t1".to_string(),
                name: "test".to_string(),
                input: serde_json::json!({}),
            }],
            stop_reason: None,
            usage: None,
        };
        let llm = parse_response(resp);
        assert_eq!(llm.finish_reason, "tool_calls");
    }

    #[test]
    fn parse_response_preserves_max_tokens_reason() {
        let resp = ChatResponse {
            content: vec![ResponseBlock::Text {
                block_type: "text".to_string(),
                text: "Truncated".to_string(),
            }],
            stop_reason: Some("max_tokens".to_string()),
            usage: None,
        };
        let llm = parse_response(resp);
        assert_eq!(llm.finish_reason, "max_tokens");
    }

    #[test]
    fn parse_response_usage_maps_input_output_tokens() {
        let resp = ChatResponse {
            content: vec![ResponseBlock::Text {
                block_type: "text".to_string(),
                text: "Hi".to_string(),
            }],
            stop_reason: Some("end_turn".to_string()),
            usage: Some(WireUsage {
                input_tokens: 200,
                output_tokens: 75,
            }),
        };
        let llm = parse_response(resp);
        let usage = llm.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 200);
        assert_eq!(usage.completion_tokens, 75);
        assert_eq!(usage.total_tokens, 275);
    }

    // ── Content block serialization round-trips ─────────────

    #[test]
    fn content_block_text_serializes() {
        let block = ContentBlock::Text {
            text: "hello".to_string(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"text\""));
        assert!(json.contains("\"text\":\"hello\""));
    }

    #[test]
    fn content_block_tool_use_serializes() {
        let block = ContentBlock::ToolUse {
            id: "toolu_1".to_string(),
            name: "shell".to_string(),
            input: serde_json::json!({"cmd": "ls"}),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_use\""));
        assert!(json.contains("\"id\":\"toolu_1\""));
        assert!(json.contains("\"name\":\"shell\""));
        assert!(json.contains("\"cmd\":\"ls\""));
    }

    #[test]
    fn content_block_tool_result_serializes() {
        let block = ContentBlock::ToolResult {
            tool_use_id: "toolu_1".to_string(),
            content: "success".to_string(),
        };
        let json = serde_json::to_string(&block).unwrap();
        assert!(json.contains("\"type\":\"tool_result\""));
        assert!(json.contains("\"tool_use_id\":\"toolu_1\""));
        assert!(json.contains("\"content\":\"success\""));
    }

    #[test]
    fn message_content_text_serializes_as_string() {
        let content = MessageContent::Text("hello".to_string());
        let json = serde_json::to_string(&content).unwrap();
        assert_eq!(json, "\"hello\"");
    }

    #[test]
    fn message_content_blocks_serializes_as_array() {
        let content = MessageContent::Blocks(vec![ContentBlock::Text {
            text: "hello".to_string(),
        }]);
        let json = serde_json::to_string(&content).unwrap();
        assert!(json.starts_with('['));
        assert!(json.contains("\"type\":\"text\""));
    }
}
