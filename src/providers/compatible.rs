//! Generic OpenAI-compatible provider.
//! Most LLM APIs follow the same `/v1/chat/completions` format.
//! This module provides a single implementation that works for all of them.

use crate::providers::traits::{
    ChatMessage, LlmResponse, ModelInfo, Provider, ToolCallRequest, UsageInfo,
};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

/// A provider that speaks the OpenAI-compatible chat completions API.
/// Used by: Venice, Vercel AI Gateway, Cloudflare AI Gateway, Moonshot,
/// Synthetic, `OpenCode` Zen, `Z.AI`, `GLM`, `MiniMax`, Bedrock, Qianfan, Groq, Mistral, `xAI`, etc.
pub struct OpenAiCompatibleProvider {
    pub(crate) name: String,
    pub(crate) base_url: String,
    pub(crate) api_key: Option<String>,
    pub(crate) auth_header: AuthStyle,
    client: Client,
}

/// How the provider expects the API key to be sent.
#[derive(Debug, Clone)]
pub enum AuthStyle {
    /// `Authorization: Bearer <key>`
    Bearer,
    /// `x-api-key: <key>` (used by some Chinese providers)
    XApiKey,
    /// Custom header name
    Custom(String),
}

impl OpenAiCompatibleProvider {
    pub fn new(name: &str, base_url: &str, api_key: Option<&str>, auth_style: AuthStyle) -> Self {
        Self {
            name: name.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            api_key: api_key.map(ToString::to_string),
            auth_header: auth_style,
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }

    /// Build the full URL for chat completions, detecting if base_url already includes the path.
    /// This allows custom providers with non-standard endpoints (e.g., VolcEngine ARK uses
    /// `/api/coding/v3/chat/completions` instead of `/v1/chat/completions`).
    fn chat_completions_url(&self) -> String {
        // If base_url already contains "chat/completions", use it as-is
        if self.base_url.contains("chat/completions") {
            self.base_url.clone()
        } else {
            format!("{}/chat/completions", self.base_url)
        }
    }

    /// Build the full URL for responses API, detecting if base_url already includes the path.
    fn responses_url(&self) -> String {
        // If base_url already contains "responses", use it as-is
        if self.base_url.contains("responses") {
            self.base_url.clone()
        } else {
            format!("{}/v1/responses", self.base_url)
        }
    }
}

// ── Wire types (OpenAI format) ──────────────────────────────────

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    temperature: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
}

#[derive(Debug, Serialize, Deserialize)]
struct Message {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<WireToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct WireToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: WireFunction,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct WireFunction {
    name: String,
    arguments: String,
}

#[derive(Debug, Serialize)]
struct ToolDefinition {
    #[serde(rename = "type")]
    tool_type: String,
    function: FunctionDef,
}

#[derive(Debug, Serialize)]
struct FunctionDef {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct ApiChatResponse {
    choices: Vec<Choice>,
    #[serde(default)]
    usage: Option<WireUsage>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<ToolCall>>,
}

#[derive(Debug, Deserialize, Serialize)]
struct ToolCall {
    #[serde(default)]
    id: Option<String>,
    #[serde(rename = "type")]
    kind: Option<String>,
    function: Option<Function>,
}

#[derive(Debug, Deserialize, Serialize)]
struct Function {
    name: Option<String>,
    arguments: Option<String>,
}

#[derive(Debug, Serialize)]
struct ResponsesRequest {
    model: String,
    input: Vec<ResponsesInput>,
    #[serde(skip_serializing_if = "Option::is_none")]
    instructions: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stream: Option<bool>,
}

#[derive(Debug, Serialize)]
struct ResponsesInput {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct ResponsesResponse {
    #[serde(default)]
    output: Vec<ResponsesOutput>,
    #[serde(default)]
    output_text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponsesOutput {
    #[serde(default)]
    content: Vec<ResponsesContent>,
}

#[derive(Debug, Deserialize)]
struct ResponsesContent {
    #[serde(rename = "type")]
    kind: Option<String>,
    text: Option<String>,
}

fn first_nonempty(text: Option<&str>) -> Option<String> {
    text.and_then(|value| {
        let trimmed = value.trim();
        if trimmed.is_empty() {
            None
        } else {
            Some(trimmed.to_string())
        }
    })
}

fn extract_responses_text(response: ResponsesResponse) -> Option<String> {
    if let Some(text) = first_nonempty(response.output_text.as_deref()) {
        return Some(text);
    }

    for item in &response.output {
        for content in &item.content {
            if content.kind.as_deref() == Some("output_text") {
                if let Some(text) = first_nonempty(content.text.as_deref()) {
                    return Some(text);
                }
            }
        }
    }

    for item in &response.output {
        for content in &item.content {
            if let Some(text) = first_nonempty(content.text.as_deref()) {
                return Some(text);
            }
        }
    }

    None
}

impl OpenAiCompatibleProvider {
    fn apply_auth_header(
        &self,
        req: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder {
        match &self.auth_header {
            AuthStyle::Bearer => req.header("Authorization", format!("Bearer {api_key}")),
            AuthStyle::XApiKey => req.header("x-api-key", api_key),
            AuthStyle::Custom(header) => req.header(header, api_key),
        }
    }

    fn api_key(&self) -> anyhow::Result<&str> {
        self.api_key.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "{} API key not set. Run `zeroclaw onboard` or set the appropriate env var.",
                self.name
            )
        })
    }

    fn apply_auth(
        &self,
        req: reqwest::RequestBuilder,
        api_key: &str,
    ) -> reqwest::RequestBuilder {
        self.apply_auth_header(req, api_key)
    }

    async fn send_request(&self, request: &ChatRequest) -> anyhow::Result<ApiChatResponse> {
        let api_key = self.api_key()?;

        let url = self.chat_completions_url();
        let response = self
            .apply_auth_header(self.client.post(&url).json(request), api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("{} API error: {error}", self.name);
        }

        Ok(response.json().await?)
    }

    async fn chat_via_responses(
        &self,
        api_key: &str,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
    ) -> anyhow::Result<String> {
        let request = ResponsesRequest {
            model: model.to_string(),
            input: vec![ResponsesInput {
                role: "user".to_string(),
                content: message.to_string(),
            }],
            instructions: system_prompt.map(str::to_string),
            stream: Some(false),
        };

        let url = self.responses_url();

        let response = self
            .apply_auth_header(self.client.post(&url).json(&request), api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("{} Responses API error: {error}", self.name);
        }

        let responses: ResponsesResponse = response.json().await?;

        extract_responses_text(responses)
            .ok_or_else(|| anyhow::anyhow!("No response from {} Responses API", self.name))
    }
}

fn convert_tools(tools: &[ToolSpec]) -> Vec<ToolDefinition> {
    tools
        .iter()
        .map(|t| ToolDefinition {
            tool_type: "function".to_string(),
            function: FunctionDef {
                name: t.name.clone(),
                description: t.description.clone(),
                parameters: t.parameters.clone(),
            },
        })
        .collect()
}

fn convert_messages(messages: &[ChatMessage]) -> Vec<Message> {
    messages
        .iter()
        .map(|m| Message {
            role: m.role.clone(),
            content: m.content.clone(),
            tool_calls: m.tool_calls.as_ref().map(|calls| {
                calls
                    .iter()
                    .map(|tc| WireToolCall {
                        id: tc.id.clone(),
                        call_type: "function".to_string(),
                        function: WireFunction {
                            name: tc.name.clone(),
                            arguments: tc.arguments.clone(),
                        },
                    })
                    .collect()
            }),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect()
}

fn parse_response(resp: ApiChatResponse, provider_name: &str) -> anyhow::Result<LlmResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No response from {}", provider_name))?;

    let tool_calls: Vec<ToolCallRequest> = choice
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .filter_map(|tc| {
            let func = tc.function?;
            Some(ToolCallRequest {
                id: tc.id.unwrap_or_default(),
                name: func.name.unwrap_or_default(),
                arguments: func.arguments.unwrap_or_default(),
            })
        })
        .collect();

    let finish_reason = match choice.finish_reason.as_deref() {
        Some("tool_calls") => "tool_calls".to_string(),
        _ if !tool_calls.is_empty() => "tool_calls".to_string(),
        Some(r) => r.to_string(),
        None => "stop".to_string(),
    };

    let usage = resp.usage.map(|u| UsageInfo {
        prompt_tokens: u.prompt_tokens,
        completion_tokens: u.completion_tokens,
        total_tokens: u.prompt_tokens + u.completion_tokens,
    });

    Ok(LlmResponse {
        content: choice.message.content,
        tool_calls,
        finish_reason,
        usage,
    })
}

#[async_trait]
impl Provider for OpenAiCompatibleProvider {
    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let api_key = self.api_key()?;
        let mut messages = Vec::new();

        if let Some(sys) = system_prompt {
            messages.push(Message {
                role: "system".to_string(),
                content: Some(sys.to_string()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        messages.push(Message {
            role: "user".to_string(),
            content: Some(message.to_string()),
            tool_calls: None,
            tool_call_id: None,
        });

        let request = ChatRequest {
            model: model.to_string(),
            messages,
            temperature,
            tools: None,
            max_tokens: None,
        };

        let url = self.chat_completions_url();

        let response = self
            .apply_auth_header(self.client.post(&url).json(&request), api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let error = response.text().await?;
            let sanitized = super::sanitize_api_error(&error);

            if status == reqwest::StatusCode::NOT_FOUND {
                return self
                    .chat_via_responses(api_key, system_prompt, message, model)
                    .await
                    .map_err(|responses_err| {
                        anyhow::anyhow!(
                            "{} API error ({status}): {sanitized} (chat completions unavailable; responses fallback failed: {responses_err})",
                            self.name
                        )
                    });
            }

            anyhow::bail!("{} API error ({status}): {sanitized}", self.name);
        }

        let chat_response: ApiChatResponse = response.json().await?;

        chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| {
                // If tool_calls are present, serialize the full message as JSON
                // so parse_tool_calls can handle the OpenAI-style format
                if c.message.tool_calls.is_some()
                    && c.message
                        .tool_calls
                        .as_ref()
                        .map_or(false, |t| !t.is_empty())
                {
                    serde_json::to_string(&c.message)
                        .unwrap_or_else(|_| c.message.content.unwrap_or_default())
                } else {
                    // No tool calls, return content as-is
                    c.message.content.unwrap_or_default()
                }
            })
            .ok_or_else(|| anyhow::anyhow!("No response from {}", self.name))
    }

    async fn chat_with_history(
        &self,
        messages: &[ChatMessage],
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
        let api_key = self.api_key()?;

        let api_messages: Vec<Message> = messages
            .iter()
            .map(|m| Message {
                role: m.role.clone(),
                content: m.content.clone(),
                tool_calls: None,
                tool_call_id: None,
            })
            .collect();

        let request = ChatRequest {
            model: model.to_string(),
            messages: api_messages,
            temperature,
            tools: None,
            max_tokens: None,
        };

        let url = self.chat_completions_url();
        let response = self
            .apply_auth_header(self.client.post(&url).json(&request), api_key)
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();

            // Mirror chat_with_system: 404 may mean this provider uses the Responses API
            if status == reqwest::StatusCode::NOT_FOUND {
                // Extract system prompt and last user message for responses fallback
                let system = messages.iter().find(|m| m.role == "system");
                let last_user = messages.iter().rfind(|m| m.role == "user");
                if let Some(user_msg) = last_user {
                    return self
                        .chat_via_responses(
                            api_key,
                            system.and_then(|m| m.content.as_deref()),
                            user_msg.content.as_deref().unwrap_or(""),
                            model,
                        )
                        .await
                        .map_err(|responses_err| {
                            anyhow::anyhow!(
                                "{} API error (chat completions unavailable; responses fallback failed: {responses_err})",
                                self.name
                            )
                        });
                }
            }

            return Err(super::api_error(&self.name, response).await);
        }

        let chat_response: ApiChatResponse = response.json().await?;

        chat_response
            .choices
            .into_iter()
            .next()
            .map(|c| {
                // If tool_calls are present, serialize the full message as JSON
                // so parse_tool_calls can handle the OpenAI-style format
                if c.message.tool_calls.is_some()
                    && c.message
                        .tool_calls
                        .as_ref()
                        .map_or(false, |t| !t.is_empty())
                {
                    serde_json::to_string(&c.message)
                        .unwrap_or_else(|_| c.message.content.unwrap_or_default())
                } else {
                    // No tool calls, return content as-is
                    c.message.content.unwrap_or_default()
                }
            })
            .ok_or_else(|| anyhow::anyhow!("No response from {}", self.name))
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
        let mut wire_messages = Vec::new();

        if let Some(sys) = system_prompt {
            wire_messages.push(Message {
                role: "system".to_string(),
                content: Some(sys.to_string()),
                tool_calls: None,
                tool_call_id: None,
            });
        }

        wire_messages.extend(convert_messages(messages));

        let tool_defs = if tools.is_empty() {
            None
        } else {
            Some(convert_tools(tools))
        };

        let request = ChatRequest {
            model: model.to_string(),
            messages: wire_messages,
            temperature,
            tools: tool_defs,
            max_tokens: Some(max_tokens),
        };

        let chat_response = self.send_request(&request).await?;
        parse_response(chat_response, &self.name)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let url = format!("{}/v1/models", self.base_url);
        let req = if let Ok(api_key) = self.api_key() {
            self.apply_auth(self.client.get(&url), api_key)
        } else {
            self.client.get(&url)
        };
        let Ok(response) = req.send().await else {
            return Ok(vec![]);
        };
        if !response.status().is_success() {
            return Ok(vec![]);
        }
        let Ok(body) = response.json::<serde_json::Value>().await else {
            return Ok(vec![]);
        };
        let mut models: Vec<ModelInfo> = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        Some(ModelInfo {
                            id: m["id"].as_str()?.to_string(),
                            owned_by: m["owned_by"].as_str().map(ToString::to_string),
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

    fn make_provider(name: &str, url: &str, key: Option<&str>) -> OpenAiCompatibleProvider {
        OpenAiCompatibleProvider::new(name, url, key, AuthStyle::Bearer)
    }

    #[test]
    fn creates_with_key() {
        let p = make_provider("venice", "https://api.venice.ai", Some("vn-key"));
        assert_eq!(p.name, "venice");
        assert_eq!(p.base_url, "https://api.venice.ai");
        assert_eq!(p.api_key.as_deref(), Some("vn-key"));
    }

    #[test]
    fn creates_without_key() {
        let p = make_provider("test", "https://example.com", None);
        assert!(p.api_key.is_none());
    }

    #[test]
    fn strips_trailing_slash() {
        let p = make_provider("test", "https://example.com/", None);
        assert_eq!(p.base_url, "https://example.com");
    }

    #[tokio::test]
    async fn chat_fails_without_key() {
        let p = make_provider("Venice", "https://api.venice.ai", None);
        let result = p
            .chat_with_system(None, "hello", "llama-3.3-70b", 0.7)
            .await;
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Venice API key not set"));
    }

    #[test]
    fn request_serializes_correctly() {
        let req = ChatRequest {
            model: "llama-3.3-70b".to_string(),
            messages: vec![
                Message {
                    role: "system".to_string(),
                    content: Some("You are ZeroClaw".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
                Message {
                    role: "user".to_string(),
                    content: Some("hello".to_string()),
                    tool_calls: None,
                    tool_call_id: None,
                },
            ],
            temperature: 0.7,
            tools: None,
            max_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("llama-3.3-70b"));
        assert!(json.contains("system"));
        assert!(json.contains("user"));
    }

    #[test]
    fn response_deserializes() {
        let json = r#"{"choices":[{"message":{"content":"Hello from Venice!"}}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].message.content,
            Some("Hello from Venice!".to_string())
        );
    }

    #[test]
    fn response_empty_choices() {
        let json = r#"{"choices":[]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.choices.is_empty());
    }

    #[test]
    fn x_api_key_auth_style() {
        let p = OpenAiCompatibleProvider::new(
            "moonshot",
            "https://api.moonshot.cn",
            Some("ms-key"),
            AuthStyle::XApiKey,
        );
        assert!(matches!(p.auth_header, AuthStyle::XApiKey));
    }

    #[test]
    fn custom_auth_style() {
        let p = OpenAiCompatibleProvider::new(
            "custom",
            "https://api.example.com",
            Some("key"),
            AuthStyle::Custom("X-Custom-Key".into()),
        );
        assert!(matches!(p.auth_header, AuthStyle::Custom(_)));
    }

    #[tokio::test]
    async fn all_compatible_providers_fail_without_key() {
        let providers = vec![
            make_provider("Venice", "https://api.venice.ai", None),
            make_provider("Moonshot", "https://api.moonshot.cn", None),
            make_provider("GLM", "https://open.bigmodel.cn", None),
            make_provider("MiniMax", "https://api.minimax.chat", None),
            make_provider("Groq", "https://api.groq.com/openai", None),
            make_provider("Mistral", "https://api.mistral.ai", None),
            make_provider("xAI", "https://api.x.ai", None),
        ];

        for p in providers {
            let result = p.chat_with_system(None, "test", "model", 0.7).await;
            assert!(result.is_err(), "{} should fail without key", p.name);
            assert!(
                result.unwrap_err().to_string().contains("API key not set"),
                "{} error should mention key",
                p.name
            );
        }
    }

    #[test]
    fn responses_extracts_top_level_output_text() {
        let json = r#"{"output_text":"Hello from top-level","output":[]}"#;
        let response: ResponsesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            extract_responses_text(response).as_deref(),
            Some("Hello from top-level")
        );
    }

    #[test]
    fn responses_extracts_nested_output_text() {
        let json =
            r#"{"output":[{"content":[{"type":"output_text","text":"Hello from nested"}]}]}"#;
        let response: ResponsesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            extract_responses_text(response).as_deref(),
            Some("Hello from nested")
        );
    }

    #[test]
    fn responses_extracts_any_text_as_fallback() {
        let json = r#"{"output":[{"content":[{"type":"message","text":"Fallback text"}]}]}"#;
        let response: ResponsesResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            extract_responses_text(response).as_deref(),
            Some("Fallback text")
        );
    }

    // ══════════════════════════════════════════════════════════
    // Custom endpoint path tests (Issue #114)
    // ══════════════════════════════════════════════════════════

    #[test]
    fn chat_completions_url_standard_openai() {
        // Standard OpenAI-compatible providers get /chat/completions appended
        let p = make_provider("openai", "https://api.openai.com/v1", None);
        assert_eq!(
            p.chat_completions_url(),
            "https://api.openai.com/v1/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_trailing_slash() {
        // Trailing slash is stripped, then /chat/completions appended
        let p = make_provider("test", "https://api.example.com/v1/", None);
        assert_eq!(
            p.chat_completions_url(),
            "https://api.example.com/v1/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_volcengine_ark() {
        // VolcEngine ARK uses custom path - should use as-is
        let p = make_provider(
            "volcengine",
            "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions",
            None,
        );
        assert_eq!(
            p.chat_completions_url(),
            "https://ark.cn-beijing.volces.com/api/coding/v3/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_custom_full_endpoint() {
        // Custom provider with full endpoint path
        let p = make_provider(
            "custom",
            "https://my-api.example.com/v2/llm/chat/completions",
            None,
        );
        assert_eq!(
            p.chat_completions_url(),
            "https://my-api.example.com/v2/llm/chat/completions"
        );
    }

    #[test]
    fn responses_url_standard() {
        // Standard providers get /v1/responses appended
        let p = make_provider("test", "https://api.example.com", None);
        assert_eq!(p.responses_url(), "https://api.example.com/v1/responses");
    }

    #[test]
    fn responses_url_custom_full_endpoint() {
        // Custom provider with full responses endpoint
        let p = make_provider(
            "custom",
            "https://my-api.example.com/api/v2/responses",
            None,
        );
        assert_eq!(
            p.responses_url(),
            "https://my-api.example.com/api/v2/responses"
        );
    }

    #[test]
    fn chat_completions_url_without_v1() {
        // Provider configured without /v1 in base URL
        let p = make_provider("test", "https://api.example.com", None);
        assert_eq!(
            p.chat_completions_url(),
            "https://api.example.com/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_base_with_v1() {
        // Provider configured with /v1 in base URL
        let p = make_provider("test", "https://api.example.com/v1", None);
        assert_eq!(
            p.chat_completions_url(),
            "https://api.example.com/v1/chat/completions"
        );
    }

    // ══════════════════════════════════════════════════════════
    // Provider-specific endpoint tests (Issue #167)
    // ══════════════════════════════════════════════════════════

    #[test]
    fn chat_completions_url_zai() {
        // Z.AI uses /api/paas/v4 base path
        let p = make_provider("zai", "https://api.z.ai/api/paas/v4", None);
        assert_eq!(
            p.chat_completions_url(),
            "https://api.z.ai/api/paas/v4/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_glm() {
        // GLM (BigModel) uses /api/paas/v4 base path
        let p = make_provider("glm", "https://open.bigmodel.cn/api/paas/v4", None);
        assert_eq!(
            p.chat_completions_url(),
            "https://open.bigmodel.cn/api/paas/v4/chat/completions"
        );
    }

    #[test]
    fn chat_completions_url_opencode() {
        // OpenCode Zen uses /zen/v1 base path
        let p = make_provider("opencode", "https://opencode.ai/zen/v1", None);
        assert_eq!(
            p.chat_completions_url(),
            "https://opencode.ai/zen/v1/chat/completions"
        );
    }
}
