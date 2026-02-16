use crate::providers::traits::{
    ChatMessage, LlmResponse, ModelInfo, Provider, ToolCallRequest, UsageInfo,
};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OpenAiProvider {
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
    #[serde(skip_serializing_if = "Option::is_none")]
    max_completion_tokens: Option<u32>,
}

/// Older models that still need `max_tokens` instead of `max_completion_tokens`.
fn uses_legacy_max_tokens(model: &str) -> bool {
    model.starts_with("gpt-3.5")
        || (model.starts_with("gpt-4-") && !model.contains("4o"))
        || model == "gpt-4"
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
struct ChatResponse {
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

impl OpenAiProvider {
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
            anyhow::anyhow!("OpenAI API key not set. Set OPENAI_API_KEY or edit config.toml.")
        })
    }

    async fn send_request(&self, request: &ChatRequest) -> anyhow::Result<ChatResponse> {
        let api_key = self.api_key()?;

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(request)
            .send()
            .await?;

        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("OpenAI API error: {error}");
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

fn parse_response(resp: ChatResponse) -> anyhow::Result<LlmResponse> {
    let choice = resp
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No response from OpenAI"))?;

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
impl Provider for OpenAiProvider {
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
            max_completion_tokens: None,
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {api_key}"))
            .json(&request)
            .send()
            .await?;

        if !response.status().is_success() {
            return Err(super::api_error("OpenAI", response).await);
        }

        let chat_response: ChatResponse = response.json().await?;

        chat_response
            .choices
            .into_iter()
            .next()
            .and_then(|c| c.message.content)
            .ok_or_else(|| anyhow::anyhow!("No response from OpenAI"))
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

        let (mt, mct) = if uses_legacy_max_tokens(model) {
            (Some(max_tokens), None)
        } else {
            (None, Some(max_tokens))
        };
        let request = ChatRequest {
            model: model.to_string(),
            messages: wire_messages,
            temperature,
            tools: tool_defs,
            max_tokens: mt,
            max_completion_tokens: mct,
        };

        let chat_response = self.send_request(&request).await?;
        parse_response(chat_response)
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        /// Prefixes of models that are NOT chat-compatible.
        const NON_CHAT_PREFIXES: &[&str] = &[
            "dall-e",
            "tts",
            "whisper",
            "text-embedding",
            "text-moderation",
            "davinci",
            "babbage",
            "codex-",
            "text-davinci",
            "text-babbage",
            "text-curie",
            "text-ada",
            "curie",
            "ada",
            "code-",
            "text-search-",
            "text-similarity-",
        ];

        let api_key = self.api_key()?;
        let response = self
            .client
            .get("https://api.openai.com/v1/models")
            .header("Authorization", format!("Bearer {api_key}"))
            .send()
            .await?;
        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!("OpenAI models API error: {error}");
        }
        let body: serde_json::Value = response.json().await?;
        let mut models: Vec<ModelInfo> = body["data"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        let id = m["id"].as_str()?;
                        let dominated = NON_CHAT_PREFIXES
                            .iter()
                            .any(|p| id.starts_with(p));
                        if dominated {
                            return None;
                        }
                        Some(ModelInfo {
                            id: id.to_string(),
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
        let p = OpenAiProvider::new(Some("sk-proj-abc123"));
        assert_eq!(p.api_key.as_deref(), Some("sk-proj-abc123"));
    }

    #[test]
    fn creates_without_key() {
        let p = OpenAiProvider::new(None);
        assert!(p.api_key.is_none());
    }

    #[test]
    fn creates_with_empty_key() {
        let p = OpenAiProvider::new(Some(""));
        assert_eq!(p.api_key.as_deref(), Some(""));
    }

    #[tokio::test]
    async fn chat_fails_without_key() {
        let p = OpenAiProvider::new(None);
        let result = p.chat_with_system(None, "hello", "gpt-4o", 0.7).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("API key not set"));
    }

    #[tokio::test]
    async fn chat_with_system_fails_without_key() {
        let p = OpenAiProvider::new(None);
        let result = p
            .chat_with_system(Some("You are ZeroClaw"), "test", "gpt-4o", 0.5)
            .await;
        assert!(result.is_err());
    }

    #[test]
    fn request_serializes_with_system_message() {
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
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
            max_completion_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"role\":\"system\""));
        assert!(json.contains("\"role\":\"user\""));
        assert!(json.contains("gpt-4o"));
    }

    #[test]
    fn request_serializes_without_system() {
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some("hello".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            temperature: 0.0,
            tools: None,
            max_tokens: None,
            max_completion_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("system"));
        assert!(json.contains("\"temperature\":0.0"));
    }

    #[test]
    fn response_deserializes_single_choice() {
        let json = r#"{"choices":[{"message":{"content":"Hi!"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 1);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("Hi!"));
    }

    #[test]
    fn response_deserializes_empty_choices() {
        let json = r#"{"choices":[]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.choices.is_empty());
    }

    #[test]
    fn response_deserializes_multiple_choices() {
        let json = r#"{"choices":[{"message":{"content":"A"}},{"message":{"content":"B"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.choices.len(), 2);
        assert_eq!(resp.choices[0].message.content.as_deref(), Some("A"));
    }

    #[test]
    fn response_with_unicode() {
        let json = r#"{"choices":[{"message":{"content":"こんにちは 🦀"}}]}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.choices[0].message.content.as_deref(),
            Some("こんにちは 🦀")
        );
    }

    #[test]
    fn response_with_long_content() {
        let long = "x".repeat(100_000);
        let json = format!(r#"{{"choices":[{{"message":{{"content":"{long}"}}}}]}}"#);
        let resp: ChatResponse = serde_json::from_str(&json).unwrap();
        assert_eq!(resp.choices[0].message.content.as_deref().unwrap().len(), 100_000);
    }

    // ── Tool calling tests ──────────────────────────────────

    #[test]
    fn request_serializes_with_tools() {
        let req = ChatRequest {
            model: "gpt-4o".to_string(),
            messages: vec![],
            temperature: 0.7,
            tools: Some(vec![ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: "shell".to_string(),
                    description: "Run commands".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
            max_tokens: Some(4096),
            max_completion_tokens: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"max_tokens\":4096"));
    }

    #[test]
    fn response_deserializes_with_tool_calls() {
        let json = r#"{
            "choices":[{
                "message":{
                    "tool_calls":[{
                        "id":"call_1",
                        "type":"function",
                        "function":{"name":"shell","arguments":"{}"}
                    }]
                },
                "finish_reason":"tool_calls"
            }]
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        let tc = resp.choices[0].message.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].function.name, "shell");
    }

    #[test]
    fn parse_response_text() {
        let resp = ChatResponse {
            choices: vec![Choice {
                message: ResponseMessage {
                    content: Some("Hi".to_string()),
                    tool_calls: None,
                },
                finish_reason: Some("stop".to_string()),
            }],
            usage: None,
        };
        let llm = parse_response(resp).unwrap();
        assert_eq!(llm.content.as_deref(), Some("Hi"));
        assert!(llm.tool_calls.is_empty());
        assert_eq!(llm.finish_reason, "stop");
    }

    #[test]
    fn parse_response_tool_calls() {
        let resp = ChatResponse {
            choices: vec![Choice {
                message: ResponseMessage {
                    content: None,
                    tool_calls: Some(vec![WireToolCall {
                        id: "c1".to_string(),
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
                prompt_tokens: 10,
                completion_tokens: 5,
                total_tokens: 15,
            }),
        };
        let llm = parse_response(resp).unwrap();
        assert_eq!(llm.tool_calls.len(), 1);
        assert_eq!(llm.finish_reason, "tool_calls");
        assert_eq!(llm.usage.unwrap().total_tokens, 15);
    }

    #[test]
    fn parse_response_empty_errors() {
        let resp = ChatResponse {
            choices: vec![],
            usage: None,
        };
        assert!(parse_response(resp).is_err());
    }
}
