use crate::providers::traits::{ChatMessage, LlmResponse, ModelInfo, Provider, ToolCallRequest};
use crate::tools::ToolSpec;
use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};

pub struct OllamaProvider {
    base_url: String,
    client: Client,
}

// ── Wire types (Ollama format — OpenAI-compatible for tools) ────

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    stream: bool,
    options: Options,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolDefinition>>,
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
    #[serde(default)]
    id: String,
    #[serde(rename = "type", default = "default_function_type")]
    call_type: String,
    function: WireFunction,
}

fn default_function_type() -> String {
    "function".to_string()
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct WireFunction {
    name: String,
    /// Ollama returns arguments as a JSON value (object), not a string like `OpenAI`.
    arguments: serde_json::Value,
}

#[derive(Debug, Serialize)]
struct Options {
    temperature: f64,
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
    message: ResponseMessage,
}

#[derive(Debug, Deserialize)]
struct ResponseMessage {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    tool_calls: Option<Vec<WireToolCall>>,
}

impl OllamaProvider {
    pub fn new(base_url: Option<&str>) -> Self {
        Self {
            base_url: base_url
                .unwrap_or("http://localhost:11434")
                .trim_end_matches('/')
                .to_string(),
            client: Client::builder()
                .timeout(std::time::Duration::from_secs(300)) // Ollama runs locally, may be slow
                .connect_timeout(std::time::Duration::from_secs(10))
                .build()
                .unwrap_or_else(|_| Client::new()),
        }
    }

    async fn send_request(&self, request: &ChatRequest) -> anyhow::Result<ChatResponse> {
        let url = format!("{}/api/chat", self.base_url);

        let response = self.client.post(&url).json(request).send().await?;

        if !response.status().is_success() {
            let error = response.text().await?;
            anyhow::bail!(
                "Ollama error: {error}. Is Ollama running? (brew install ollama && ollama serve)"
            );
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
                            arguments: serde_json::from_str(&tc.arguments)
                                .unwrap_or(serde_json::Value::Object(serde_json::Map::default())),
                        },
                    })
                    .collect()
            }),
            tool_call_id: m.tool_call_id.clone(),
        })
        .collect()
}

fn parse_response(resp: ChatResponse) -> LlmResponse {
    let tool_calls: Vec<ToolCallRequest> = resp
        .message
        .tool_calls
        .unwrap_or_default()
        .into_iter()
        .enumerate()
        .map(|(i, tc)| {
            let id = if tc.id.is_empty() {
                format!("ollama_call_{i}")
            } else {
                tc.id
            };
            ToolCallRequest {
                id,
                name: tc.function.name,
                arguments: serde_json::to_string(&tc.function.arguments)
                    .unwrap_or_else(|_| "{}".to_string()),
            }
        })
        .collect();

    let finish_reason = if tool_calls.is_empty() {
        "stop".to_string()
    } else {
        "tool_calls".to_string()
    };

    LlmResponse {
        content: resp.message.content.filter(|s| !s.is_empty()),
        tool_calls,
        finish_reason,
        usage: None, // Ollama doesn't return usage in the same format
    }
}

#[async_trait]
impl Provider for OllamaProvider {
    async fn chat_with_system(
        &self,
        system_prompt: Option<&str>,
        message: &str,
        model: &str,
        temperature: f64,
    ) -> anyhow::Result<String> {
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
            stream: false,
            options: Options { temperature },
            tools: None,
        };

        let chat_response = self.send_request(&request).await?;
        Ok(chat_response.message.content.unwrap_or_default())
    }

    async fn chat_with_tools(
        &self,
        system_prompt: Option<&str>,
        messages: &[ChatMessage],
        tools: &[ToolSpec],
        model: &str,
        temperature: f64,
        _max_tokens: u32,
    ) -> anyhow::Result<LlmResponse> {
        let mut wire_messages = Vec::new();

        if !response.status().is_success() {
            let err = super::api_error("Ollama", response).await;
            anyhow::bail!("{err}. Is Ollama running? (brew install ollama && ollama serve)");
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
            stream: false,
            options: Options { temperature },
            tools: tool_defs,
        };

        let chat_response = self.send_request(&request).await?;
        Ok(parse_response(chat_response))
    }

    async fn list_models(&self) -> anyhow::Result<Vec<ModelInfo>> {
        let url = format!("{}/api/tags", self.base_url);
        let Ok(response) = self.client.get(&url).send().await else {
            return Ok(vec![]);
        };
        if !response.status().is_success() {
            return Ok(vec![]);
        }
        let Ok(body) = response.json::<serde_json::Value>().await else {
            return Ok(vec![]);
        };
        let mut models: Vec<ModelInfo> = body["models"]
            .as_array()
            .map(|arr| {
                arr.iter()
                    .filter_map(|m| {
                        Some(ModelInfo {
                            id: m["name"].as_str()?.to_string(),
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
    fn default_url() {
        let p = OllamaProvider::new(None);
        assert_eq!(p.base_url, "http://localhost:11434");
    }

    #[test]
    fn custom_url_trailing_slash() {
        let p = OllamaProvider::new(Some("http://192.168.1.100:11434/"));
        assert_eq!(p.base_url, "http://192.168.1.100:11434");
    }

    #[test]
    fn custom_url_no_trailing_slash() {
        let p = OllamaProvider::new(Some("http://myserver:11434"));
        assert_eq!(p.base_url, "http://myserver:11434");
    }

    #[test]
    fn empty_url_uses_empty() {
        let p = OllamaProvider::new(Some(""));
        assert_eq!(p.base_url, "");
    }

    #[test]
    fn request_serializes_with_system() {
        let req = ChatRequest {
            model: "llama3".to_string(),
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
            stream: false,
            options: Options { temperature: 0.7 },
            tools: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"stream\":false"));
        assert!(json.contains("llama3"));
        assert!(json.contains("system"));
        assert!(json.contains("\"temperature\":0.7"));
        // tools should be omitted when None
        assert!(!json.contains("\"tools\""));
    }

    #[test]
    fn request_serializes_without_system() {
        let req = ChatRequest {
            model: "mistral".to_string(),
            messages: vec![Message {
                role: "user".to_string(),
                content: Some("test".to_string()),
                tool_calls: None,
                tool_call_id: None,
            }],
            stream: false,
            options: Options { temperature: 0.0 },
            tools: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(!json.contains("\"role\":\"system\""));
        assert!(json.contains("mistral"));
    }

    #[test]
    fn response_deserializes() {
        let json = r#"{"message":{"role":"assistant","content":"Hello from Ollama!"}}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(
            resp.message.content.as_deref(),
            Some("Hello from Ollama!")
        );
    }

    #[test]
    fn response_with_empty_content() {
        let json = r#"{"message":{"role":"assistant","content":""}}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.message.content.as_deref(), Some(""));
    }

    #[test]
    fn response_with_multiline() {
        let json = r#"{"message":{"role":"assistant","content":"line1\nline2\nline3"}}"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        assert!(resp.message.content.as_deref().unwrap().contains("line1"));
    }

    // ── Tool calling tests ──────────────────────────────────

    #[test]
    fn request_serializes_with_tools() {
        let req = ChatRequest {
            model: "llama3".to_string(),
            messages: vec![],
            stream: false,
            options: Options { temperature: 0.7 },
            tools: Some(vec![ToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionDef {
                    name: "shell".to_string(),
                    description: "Run commands".to_string(),
                    parameters: serde_json::json!({"type": "object"}),
                },
            }]),
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"type\":\"function\""));
        assert!(json.contains("\"name\":\"shell\""));
    }

    #[test]
    fn response_deserializes_with_tool_calls() {
        let json = r#"{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "function": {
                        "name": "shell",
                        "arguments": {"command": "ls -la"}
                    }
                }]
            }
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        let tc = resp.message.tool_calls.as_ref().unwrap();
        assert_eq!(tc.len(), 1);
        assert_eq!(tc[0].function.name, "shell");
        assert_eq!(tc[0].function.arguments["command"], "ls -la");
    }

    #[test]
    fn response_deserializes_with_tool_call_id() {
        let json = r#"{
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": {"path": "/tmp/test"}
                    }
                }]
            }
        }"#;
        let resp: ChatResponse = serde_json::from_str(json).unwrap();
        let tc = resp.message.tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].id, "call_123");
    }

    #[test]
    fn parse_response_text_only() {
        let resp = ChatResponse {
            message: ResponseMessage {
                content: Some("Hello!".to_string()),
                tool_calls: None,
            },
        };
        let llm = parse_response(resp);
        assert_eq!(llm.content.as_deref(), Some("Hello!"));
        assert!(llm.tool_calls.is_empty());
        assert_eq!(llm.finish_reason, "stop");
    }

    #[test]
    fn parse_response_empty_content_becomes_none() {
        let resp = ChatResponse {
            message: ResponseMessage {
                content: Some(String::new()),
                tool_calls: None,
            },
        };
        let llm = parse_response(resp);
        assert!(llm.content.is_none());
    }

    #[test]
    fn parse_response_tool_calls() {
        let resp = ChatResponse {
            message: ResponseMessage {
                content: None,
                tool_calls: Some(vec![WireToolCall {
                    id: "call_1".to_string(),
                    call_type: "function".to_string(),
                    function: WireFunction {
                        name: "shell".to_string(),
                        arguments: serde_json::json!({"command": "ls"}),
                    },
                }]),
            },
        };
        let llm = parse_response(resp);
        assert_eq!(llm.tool_calls.len(), 1);
        assert_eq!(llm.tool_calls[0].name, "shell");
        assert_eq!(llm.tool_calls[0].id, "call_1");
        assert_eq!(llm.finish_reason, "tool_calls");
        // Arguments should be serialized to a JSON string
        let args: serde_json::Value =
            serde_json::from_str(&llm.tool_calls[0].arguments).unwrap();
        assert_eq!(args["command"], "ls");
    }

    #[test]
    fn parse_response_tool_calls_generates_ids_when_empty() {
        let resp = ChatResponse {
            message: ResponseMessage {
                content: None,
                tool_calls: Some(vec![
                    WireToolCall {
                        id: String::new(),
                        call_type: "function".to_string(),
                        function: WireFunction {
                            name: "shell".to_string(),
                            arguments: serde_json::json!({}),
                        },
                    },
                    WireToolCall {
                        id: String::new(),
                        call_type: "function".to_string(),
                        function: WireFunction {
                            name: "read_file".to_string(),
                            arguments: serde_json::json!({"path": "/tmp"}),
                        },
                    },
                ]),
            },
        };
        let llm = parse_response(resp);
        assert_eq!(llm.tool_calls[0].id, "ollama_call_0");
        assert_eq!(llm.tool_calls[1].id, "ollama_call_1");
    }

    #[test]
    fn convert_tools_maps_spec() {
        let specs = vec![ToolSpec {
            name: "shell".to_string(),
            description: "Run commands".to_string(),
            parameters: serde_json::json!({"type": "object", "properties": {"command": {"type": "string"}}}),
        }];
        let defs = convert_tools(&specs);
        assert_eq!(defs.len(), 1);
        assert_eq!(defs[0].tool_type, "function");
        assert_eq!(defs[0].function.name, "shell");
    }

    #[test]
    fn convert_messages_preserves_user_message() {
        let msgs = vec![ChatMessage {
            role: "user".to_string(),
            content: Some("hello".to_string()),
            tool_calls: None,
            tool_call_id: None,
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire.len(), 1);
        assert_eq!(wire[0].role, "user");
        assert_eq!(wire[0].content.as_deref(), Some("hello"));
    }

    #[test]
    fn convert_messages_maps_tool_result() {
        let msgs = vec![ChatMessage {
            role: "tool".to_string(),
            content: Some("file list here".to_string()),
            tool_calls: None,
            tool_call_id: Some("call_1".to_string()),
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire[0].role, "tool");
        assert_eq!(wire[0].tool_call_id.as_deref(), Some("call_1"));
    }

    #[test]
    fn convert_messages_maps_assistant_with_tool_calls() {
        let msgs = vec![ChatMessage {
            role: "assistant".to_string(),
            content: None,
            tool_calls: Some(vec![ToolCallRequest {
                id: "call_1".to_string(),
                name: "shell".to_string(),
                arguments: r#"{"command":"ls"}"#.to_string(),
            }]),
            tool_call_id: None,
        }];
        let wire = convert_messages(&msgs);
        assert_eq!(wire[0].role, "assistant");
        let tc = wire[0].tool_calls.as_ref().unwrap();
        assert_eq!(tc[0].function.name, "shell");
        // Arguments should be deserialized from string to JSON value
        assert_eq!(tc[0].function.arguments["command"], "ls");
    }
}
