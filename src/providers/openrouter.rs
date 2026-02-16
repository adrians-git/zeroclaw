use crate::providers::traits::{
    ChatMessage, LlmResponse, ModelInfo, Provider, ToolCallRequest, UsageInfo,
};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OpenRouterProvider {
    api_key: Option<String>,
    client: Client,
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
struct Choice {
    message: ResponseMessage,
    #[serde(default)]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

#[derive(Debug, Deserialize)]
struct WireUsage {
    #[serde(default)]
    prompt_tokens: u32,
    #[serde(default)]
    completion_tokens: u32,
    #[serde(default)]
    total_tokens: u32,
}

impl OpenRouterProvider {
    pub fn new(api_key: Option<&str>) -> Self {
        Self {
            api_key: api_key.map(ToString::to_string),
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(120))
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }

    fn api_key(&self) -> anyhow::Result<&str> {
        self.api_key.as_deref().ok_or_else(|| {
            anyhow::anyhow!(
                "OpenRouter API key not set. Run `zeroclaw onboard` or set OPENROUTER_API_KEY env var."
            )
        })
    }

    async fn send_request(&self, request: &ChatRequest) -> anyhow::Result<ApiChatResponse> {
        let api_key = self.api_key()?;

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .header(
                "HTTP-Referer",
                "https://github.com/theonlyhennygod/zeroclaw",
            )
            .header("X-Title", "ZeroClaw")
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("OpenRouter API error: {error}");
        }

        Ok(response.json().await?)
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

fn parse_response(resp: ApiChatResponse) -> anyhow::Result<LlmResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No response from OpenRouter"))?;

    let tool_calls: Vec<ToolCallRequest> = choice
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .map(|tc| ToolCallRequest {
            id: tc.id,
            name: tc.function.name,
            arguments: tc.function.arguments,
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
        total_tokens: u.total_tokens,
    });

    Ok(LlmResponse {
        content: choice.message.content,
        tool_calls,
        finish_reason,
        usage,
    })
}

#[async_trait]
impl Provider for OpenRouterProvider {
    async fn warmup(&self) -> anyhow::Result<()> {
        // Hit a lightweight endpoint to establish TLS + HTTP/2 connection pool.
        // This prevents the first real chat request from timing out on cold start.
        if let Some(api_key) = self.api_key.as_ref() {
            self.client
                .get("https://openrouter.ai/api/v1/auth/key")
                .header("Authorization", format!("Bearer {api_key}"))
                .send()
                .await?
                .error_for_status()?;
        }
        Ok(())
    }

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

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .header(
                "HTTP-Referer",
                "https://github.com/theonlyhennygod/zeroclaw",
            )
            .header("X-Title", "ZeroClaw")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenRouter", response).await);
        }

        let chat_response: ApiChatResponse = response.json().await?;

        chat_response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenRouter"))
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

        let response = self
            .client
            .post("https://openrouter.ai/api/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .header(
                "HTTP-Referer",
                "https://github.com/theonlyhennygod/zeroclaw",
            )
            .header("X-Title", "ZeroClaw")
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenRouter", response).await);
        }

        let chat_response: ApiChatResponse = response.json().await?;

        chat_response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenRouter"))
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
        parse_response(chat_response)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let api_key = self.api_key()?;
        let response = self
            .client
            .get("https://openrouter.ai/api/v1/models")
            .header("Authorization", format!("Bearer {api_key}"))
            .send()
            .await?;
        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("OpenRouter models API error: {error}");
        }
        let body: serde_json::Value = response.json().await?;
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

    #[test]
    fn creates_with_key() {
        let p = OpenRouterProvider::new(Some("sk-or-test123"));
        assert!(p.api_key.is_some());
        assert_eq!(p.api_key.as_deref(), Some("sk-or-test123"));
    }

    #[test]
    fn creates_without_key() {
        let p = OpenRouterProvider::new(None);
        assert!(p.api_key.is_none());
    }

    #[tokio::test]
    async fn chat_fails_without_key() {
        let p = OpenRouterProvider::new(None);
        let result = p
            .chat_with_system(None, "hello", "test-model", 0.7)
            .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("API key not set"),
            "Expected key error, got: {err}"
        );
    }

    // ── Serialization tests ─────────────────────────────────

    #[test]
    fn request_serializes_with_tools() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some("hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: 0.7,
            tools: Some(vec![ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: "shell".to_string(),
                    description: "Run shell commands".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
            max_tokens: Some(4096),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"shell\""));
        assert!(json.contains("\"max_tokens\":4096"));
    }

    #[test]
    fn request_omits_tools_when_none() {
        let req = ChatRequest {
            model: "test".to_string(),
            messages: vec![],
            temperature: 0.7,
            tools: None,
            max_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("tools"));
        assert!(!json.contains("max_tokens"));
    }

    #[test]
    fn message_omits_null_fields() {
        let msg = Message {
            role: "user".to_string(),
            content: Some("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(!json.contains("tool_calls"));
        assert!(!json.contains("tool_call_id"));
    }

    #[test]
    fn message_includes_tool_calls_when_present() {
        let msg = Message {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![WireToolCall {
                id: "call_1".to_string(),
                call_type: "function".to_string(),
                function: WireFunction {
                    name: "shell".to_string(),
                    arguments: r#"{"command":"ls"}"#.to_string(),
                },
            }]),
            tool_call_id: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        assert!(json.contains("call_1"));
        assert!(json.contains("shell"));
    }

    // ── Response deserialization ─────────────────────────────

    #[test]
    fn response_deserializes_text_only() {
        let json = r#"{"choices":[{"message":{"content":"Hello!"},"finish_reason":"stop"}]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hello!"));
        assert!(resp.choices[0].message.tool_calls.is_none());
    }

    #[test]
    fn response_deserializes_with_tool_calls() {
        let json = r#"{
            "choices":[{
                "message":{
                    "content":null,
                    "tool_calls":[{
                        "id":"call_abc",
                        "type":"function",
                        "function":{"name":"shell","arguments":"{\"command\":\"ls\"}"}
                    }]
                },
                "finish_reason":"tool_calls"
            }],
            "usage":{"prompt_tokens":10,"completion_tokens":5,"total_tokens":15}
        }"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        let tc = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].id, "call_abc");
        assert_eq!(tc[0].function.name, "shell");
        assert!(resp.usage.is_some());
        assert_eq!(resp.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn response_empty_choices() {
        let json = r#"{"choices":[]}"#;
        let resp: ApiChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.choices.is_empty());
    }

    // ── parse_response tests ────────────────────────────────

    #[test]
    fn parse_response_text() {
        let resp = ApiChatResponse {
            choices: vec![Choice {
                message: ResponseMessage {
                    content: Some("Hello".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        let llm = parse_response(resp).unwrap();
        assert_eq!(llm.content.as_deref(), Some("Hello"));
        assert!(llm.tool_calls.is_empty());
        assert_eq!(llm.finish_reason, "stop");
    }

    #[test]
    fn parse_response_tool_calls() {
        let resp = ApiChatResponse {
            choices: vec![Choice {
                message: ResponseMessage {
                    content: None,
                    tool_calls: Some(vec![WireToolCall {
                        id: "call_1".to_string(),
                        call_type: "function".to_string(),
                        function: WireFunction {
                            name: "shell".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
                finish_reason: Some("tool_calls".to_string()),
            }],
            usage: Some(WireUsage {
                prompt_tokens: 100,
                completion_tokens: 50,
                total_tokens: 150,
            }),
        };
        let llm = parse_response(resp).unwrap();
        assert_eq!(llm.tool_calls.len(), 1);
        assert_eq!(llm.tool_calls[0].name, "shell");
        assert_eq!(llm.finish_reason, "tool_calls");
        assert_eq!(llm.usage.unwrap().total_tokens, 150);
    }

    #[test]
    fn parse_response_normalizes_finish_reason_when_tool_calls_present() {
        let resp = ApiChatResponse {
            choices: vec![Choice {
                message: ResponseMessage {
                    content: None,
                    tool_calls: Some(vec![WireToolCall {
                        id: "c1".to_string(),
                        call_type: "function".to_string(),
                        function: WireFunction {
                            name: "test".to_string(),
                            arguments: "{}".to_string(),
                        },
                    }]),
                },
                finish_reason: Some("stop".to_string()), // wrong but has tool_calls
            }],
            usage: None,
        };
        let llm = parse_response(resp).unwrap();
        assert_eq!(llm.finish_reason, "tool_calls");
    }

    #[test]
    fn parse_response_empty_choices_errors() {
        let resp = ApiChatResponse {
            choices: vec![],
            usage: None,
        };
        assert!(parse_response(resp).is_err());
    }

    // ── convert_tools ───────────────────────────────────────

    #[test]
    fn convert_tools_maps_correctly() {
        let specs = vec![ToolSpec {
            name: "shell".to_string(),
            description: "Run commands".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {}}),
        }];
        let defs = convert_tools(&specs);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].tool_type, "function");
        assert_eq!(defs[0].function.name, "shell");
    }

    // ── convert_messages ────────────────────────────────────

    #[test]
    fn convert_messages_preserves_tool_result() {
        let msgs = vec![ChatMessage {
            role: "tool".to_string(),
            content: Some("file list here".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire[0].role, "tool");
        assert_eq!(wire[0].tool_call_id.as_deref(), Some("call_1"));
        assert_eq!(wire[0].content.as_deref(), Some("file list here"));
    }
}
