use crate::observability::{Observer, ObserverEvent};
use crate::providers::traits::{ChatMessage, Provider, UsageInfo};
use crate::security::SecurityPolicy;
use crate::tools::{Tool, ToolSpec};
use serde::Serialize;
use std::time::Instant;

/// Configuration for the tool calling loop.
pub struct ToolLoopConfig {
    /// Maximum number of LLM round-trips before forcing a text response.
    pub max_iterations: u32,
    /// Maximum tokens per LLM response.
    pub max_tokens: u32,
}

/// Record of a single tool invocation for the web UI.
#[derive(Debug, Clone, Serialize)]
pub struct ToolInvocationRecord {
    pub name: String,
    /// JSON arguments (truncated to 1000 chars).
    pub arguments: String,
    /// Result preview (truncated to 500 chars).
    pub result_preview: String,
    pub success: bool,
}

/// Result of a completed tool loop.
pub struct ToolLoopResult {
    /// Final text response from the LLM.
    pub final_content: String,
    /// Number of LLM round-trips taken.
    pub iterations: u32,
    /// Total number of individual tool calls executed.
    pub total_tool_calls: u32,
    /// Usage info from the last LLM response (if available).
    pub usage: Option<UsageInfo>,
    /// Detailed per-tool-call invocation records for the web UI.
    pub tool_invocations: Vec<ToolInvocationRecord>,
}

/// Run the tool calling loop: send messages to the LLM, execute any requested
/// tool calls, feed results back, and repeat until the LLM returns text or
/// the iteration limit is reached.
///
/// Follows the `PicoClaw` pattern:
/// - Errors from tool execution go back as messages (don't break the loop)
/// - At `max_iterations`, one final request with empty tools forces a text summary
/// - Security policy is checked before every tool execution
#[allow(clippy::too_many_arguments)]
pub async fn run_tool_loop(
    provider: &dyn Provider,
    system_prompt: Option<&str>,
    initial_message: &str,
    tools: &[Box<dyn Tool>],
    security: &SecurityPolicy,
    model: &str,
    temperature: f64,
    config: &ToolLoopConfig,
    observer: Option<&dyn Observer>,
) -> anyhow::Result<ToolLoopResult> {
    let tool_specs: Vec<ToolSpec> = tools.iter().map(|t| t.spec()).collect();

    let mut messages = vec![ChatMessage {
        role: "user".to_string(),
        content: Some(initial_message.to_string()),
        tool_calls: None,
        tool_call_id: None,
    }];

    let mut total_tool_calls = 0u32;
    let mut last_usage: Option<UsageInfo> = None;
    let mut tool_invocations: Vec<ToolInvocationRecord> = Vec::new();

    for iteration in 0..config.max_iterations {
        let response = provider
            .chat_with_tools(
                system_prompt,
                &messages,
                &tool_specs,
                model,
                temperature,
                config.max_tokens,
            )
            .await?;

        if let Some(usage) = &response.usage {
            last_usage = Some(usage.clone());
        }

        // No tool calls — LLM is done, return the text
        if response.tool_calls.is_empty() {
            return Ok(ToolLoopResult {
                final_content: response.content.unwrap_or_default(),
                iterations: iteration + 1,
                total_tool_calls,
                usage: last_usage,
                tool_invocations,
            });
        }

        // Append assistant message (with tool calls) to history
        messages.push(ChatMessage {
            role: "assistant".to_string(),
            content: response.content.clone(),
            tool_calls: Some(response.tool_calls.clone()),
            tool_call_id: None,
        });

        // Execute each tool call
        for tc in &response.tool_calls {
            total_tool_calls += 1;

            let result_text = execute_tool_call(tc, tools, security, observer).await;

            let success = !result_text.starts_with("Error");
            tool_invocations.push(ToolInvocationRecord {
                name: tc.name.clone(),
                arguments: truncate_str(&tc.arguments, 1000),
                result_preview: truncate_str(&result_text, 500),
                success,
            });

            // Append tool result message
            messages.push(ChatMessage {
                role: "tool".to_string(),
                content: Some(result_text),
                tool_calls: None,
                tool_call_id: Some(tc.id.clone()),
            });
        }
    }

    // Max iterations reached — send one final request with no tools to force text
    let final_response = provider
        .chat_with_tools(
            system_prompt,
            &messages,
            &[], // Empty tools forces a text response
            model,
            temperature,
            config.max_tokens,
        )
        .await?;

    if let Some(usage) = &final_response.usage {
        last_usage = Some(usage.clone());
    }

    Ok(ToolLoopResult {
        final_content: final_response.content.unwrap_or_else(|| {
            "I've reached the maximum number of tool iterations.".to_string()
        }),
        iterations: config.max_iterations,
        total_tool_calls,
        usage: last_usage,
        tool_invocations,
    })
}

/// Truncate a string to at most `max_len` characters.
fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        let mut end = max_len;
        while !s.is_char_boundary(end) {
            end -= 1;
        }
        format!("{}...", &s[..end])
    }
}

/// Execute a single tool call, returning the result text.
/// Security checks and error handling are done here — errors become
/// message text instead of breaking the loop.
async fn execute_tool_call(
    tc: &crate::providers::traits::ToolCallRequest,
    tools: &[Box<dyn Tool>],
    security: &SecurityPolicy,
    observer: Option<&dyn Observer>,
) -> String {
    // Security: check autonomy level
    if !security.can_act() {
        return "Error: Security policy is read-only — tool execution blocked.".to_string();
    }

    // Security: check rate limit
    if !security.record_action() {
        return "Error: Rate limit exceeded — too many tool calls this hour.".to_string();
    }

    // Find the tool by name
    let Some(tool) = tools.iter().find(|t| t.name() == tc.name) else {
        return format!("Error: Unknown tool '{}'", tc.name);
    };

    // Parse arguments
    let args: serde_json::Value = serde_json::from_str(&tc.arguments)
        .unwrap_or(serde_json::Value::Object(serde_json::Map::default()));

    // Execute
    let start = Instant::now();
    let result = tool.execute(args).await;
    let duration = start.elapsed();

    match result {
        Ok(tr) => {
            if let Some(obs) = observer {
                obs.record_event(&ObserverEvent::ToolCall {
                    tool: tc.name.clone(),
                    duration,
                    success: tr.success,
                });
            }
            if tr.success {
                tr.output
            } else {
                format!("Error: {}", tr.error.as_deref().unwrap_or(&tr.output))
            }
        }
        Err(e) => {
            if let Some(obs) = observer {
                obs.record_event(&ObserverEvent::ToolCall {
                    tool: tc.name.clone(),
                    duration,
                    success: false,
                });
            }
            format!("Error executing tool: {e}")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::providers::traits::{LlmResponse, ToolCallRequest};
    use crate::security::policy::AutonomyLevel;
    use crate::tools::ToolResult;
    use async_trait::async_trait;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    // ── Mock provider ──────────────────────────────────────

    /// A mock provider that returns scripted responses.
    struct MockToolProvider {
        /// Sequence of responses to return on each call.
        responses: Vec<LlmResponse>,
        /// Tracks how many calls have been made.
        call_count: AtomicUsize,
    }

    #[async_trait]
    impl Provider for MockToolProvider {
        async fn chat_with_system(
            &self,
            _system_prompt: Option<&str>,
            _message: &str,
            _model: &str,
            _temperature: f64,
        ) -> anyhow::Result<String> {
            Ok("mock".to_string())
        }

        async fn chat_with_tools(
            &self,
            _system_prompt: Option<&str>,
            _messages: &[ChatMessage],
            _tools: &[ToolSpec],
            _model: &str,
            _temperature: f64,
            _max_tokens: u32,
        ) -> anyhow::Result<LlmResponse> {
            let idx = self.call_count.fetch_add(1, Ordering::SeqCst);
            if idx < self.responses.len() {
                Ok(self.responses[idx].clone())
            } else {
                // Default: return stop
                Ok(LlmResponse {
                    content: Some("done".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: None,
                })
            }
        }
    }

    // ── Mock tool ──────────────────────────────────────────

    struct MockTool {
        tool_name: &'static str,
        response: &'static str,
        call_count: Arc<AtomicUsize>,
    }

    #[async_trait]
    impl Tool for MockTool {
        fn name(&self) -> &str {
            self.tool_name
        }

        fn description(&self) -> &str {
            "A mock tool for testing"
        }

        fn parameters_schema(&self) -> serde_json::Value {
            serde_json::json!({"type": "object", "properties": {}})
        }

        async fn execute(&self, _args: serde_json::Value) -> anyhow::Result<ToolResult> {
            self.call_count.fetch_add(1, Ordering::SeqCst);
            Ok(ToolResult {
                success: true,
                output: self.response.to_string(),
                error: None,
            })
        }
    }

    fn default_config() -> ToolLoopConfig {
        ToolLoopConfig {
            max_iterations: 10,
            max_tokens: 4096,
        }
    }

    fn default_security() -> SecurityPolicy {
        SecurityPolicy::default()
    }

    // ── Tests ──────────────────────────────────────────────

    #[tokio::test]
    async fn immediate_text_response() {
        let provider = MockToolProvider {
            responses: vec![LlmResponse {
                content: Some("Hello!".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: None,
            }],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            None,
            "hi",
            &[],
            &default_security(),
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.final_content, "Hello!");
        assert_eq!(result.iterations, 1);
        assert_eq!(result.total_tool_calls, 0);
    }

    #[tokio::test]
    async fn single_tool_call_then_response() {
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            tool_name: "shell",
            response: "file1.txt\nfile2.txt",
            call_count: Arc::clone(&tool_calls),
        })];

        let provider = MockToolProvider {
            responses: vec![
                // First: LLM requests a tool call
                LlmResponse {
                    content: Some("Let me check.".to_string()),
                    tool_calls: vec![ToolCallRequest {
                        id: "call_1".to_string(),
                        name: "shell".to_string(),
                        arguments: r#"{"command":"ls"}"#.to_string(),
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: None,
                },
                // Second: LLM returns text after seeing tool result
                LlmResponse {
                    content: Some("Here are your files: file1.txt, file2.txt".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: None,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            Some("You are helpful"),
            "List files",
            &tools,
            &default_security(),
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(
            result.final_content,
            "Here are your files: file1.txt, file2.txt"
        );
        assert_eq!(result.iterations, 2);
        assert_eq!(result.total_tool_calls, 1);
        assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn multiple_tool_calls_in_one_response() {
        let shell_calls = Arc::new(AtomicUsize::new(0));
        let read_calls = Arc::new(AtomicUsize::new(0));

        let tools: Vec<Box<dyn Tool>> = vec![
            Box::new(MockTool {
                tool_name: "shell",
                response: "ok",
                call_count: Arc::clone(&shell_calls),
            }),
            Box::new(MockTool {
                tool_name: "file_read",
                response: "content here",
                call_count: Arc::clone(&read_calls),
            }),
        ];

        let provider = MockToolProvider {
            responses: vec![
                LlmResponse {
                    content: None,
                    tool_calls: vec![
                        ToolCallRequest {
                            id: "c1".to_string(),
                            name: "shell".to_string(),
                            arguments: "{}".to_string(),
                        },
                        ToolCallRequest {
                            id: "c2".to_string(),
                            name: "file_read".to_string(),
                            arguments: "{}".to_string(),
                        },
                    ],
                    finish_reason: "tool_calls".to_string(),
                    usage: None,
                },
                LlmResponse {
                    content: Some("Done with both.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: None,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            None,
            "do stuff",
            &tools,
            &default_security(),
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.total_tool_calls, 2);
        assert_eq!(shell_calls.load(Ordering::SeqCst), 1);
        assert_eq!(read_calls.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn unknown_tool_returns_error_message() {
        let provider = MockToolProvider {
            responses: vec![
                LlmResponse {
                    content: None,
                    tool_calls: vec![ToolCallRequest {
                        id: "c1".to_string(),
                        name: "nonexistent".to_string(),
                        arguments: "{}".to_string(),
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: None,
                },
                LlmResponse {
                    content: Some("I see the tool doesn't exist.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: None,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            None,
            "test",
            &[],
            &default_security(),
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        // The loop should continue despite the unknown tool error
        assert_eq!(
            result.final_content,
            "I see the tool doesn't exist."
        );
        assert_eq!(result.total_tool_calls, 1);
    }

    #[tokio::test]
    async fn readonly_security_blocks_tool_execution() {
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            tool_name: "shell",
            response: "should not run",
            call_count: Arc::clone(&tool_calls),
        })];

        let security = SecurityPolicy {
            autonomy: AutonomyLevel::ReadOnly,
            ..SecurityPolicy::default()
        };

        let provider = MockToolProvider {
            responses: vec![
                LlmResponse {
                    content: None,
                    tool_calls: vec![ToolCallRequest {
                        id: "c1".to_string(),
                        name: "shell".to_string(),
                        arguments: "{}".to_string(),
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: None,
                },
                LlmResponse {
                    content: Some("Security blocked.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: None,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            None,
            "test",
            &tools,
            &security,
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        // Tool should NOT have been called
        assert_eq!(tool_calls.load(Ordering::SeqCst), 0);
        assert_eq!(result.final_content, "Security blocked.");
    }

    #[tokio::test]
    async fn rate_limit_blocks_after_exhaustion() {
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            tool_name: "shell",
            response: "ok",
            call_count: Arc::clone(&tool_calls),
        })];

        let security = SecurityPolicy {
            max_actions_per_hour: 1,
            ..SecurityPolicy::default()
        };

        let provider = MockToolProvider {
            responses: vec![
                // Two tool calls in one response — second should be rate limited
                LlmResponse {
                    content: None,
                    tool_calls: vec![
                        ToolCallRequest {
                            id: "c1".to_string(),
                            name: "shell".to_string(),
                            arguments: "{}".to_string(),
                        },
                        ToolCallRequest {
                            id: "c2".to_string(),
                            name: "shell".to_string(),
                            arguments: "{}".to_string(),
                        },
                    ],
                    finish_reason: "tool_calls".to_string(),
                    usage: None,
                },
                LlmResponse {
                    content: Some("Rate limited.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: None,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            None,
            "test",
            &tools,
            &security,
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        // Only the first tool call should execute
        assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
        assert_eq!(result.total_tool_calls, 2);
    }

    #[tokio::test]
    async fn max_iterations_forces_text_response() {
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let tools: Vec<Box<dyn Tool>> = vec![Box::new(MockTool {
            tool_name: "shell",
            response: "ok",
            call_count: Arc::clone(&tool_calls),
        })];

        // Provider always returns tool calls — will hit iteration limit
        let provider = MockToolProvider {
            responses: vec![
                LlmResponse {
                    content: None,
                    tool_calls: vec![ToolCallRequest {
                        id: "c1".to_string(),
                        name: "shell".to_string(),
                        arguments: "{}".to_string(),
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: None,
                },
                LlmResponse {
                    content: None,
                    tool_calls: vec![ToolCallRequest {
                        id: "c2".to_string(),
                        name: "shell".to_string(),
                        arguments: "{}".to_string(),
                    }],
                    finish_reason: "tool_calls".to_string(),
                    usage: None,
                },
                // This should be the final forced-text response
                LlmResponse {
                    content: Some("Summary after iterations.".to_string()),
                    tool_calls: vec![],
                    finish_reason: "stop".to_string(),
                    usage: None,
                },
            ],
            call_count: AtomicUsize::new(0),
        };

        let config = ToolLoopConfig {
            max_iterations: 2,
            max_tokens: 4096,
        };

        let result = run_tool_loop(
            &provider,
            None,
            "test",
            &tools,
            &default_security(),
            "test",
            0.0,
            &config,
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.final_content, "Summary after iterations.");
        assert_eq!(result.iterations, 2);
        assert_eq!(result.total_tool_calls, 2);
    }

    #[tokio::test]
    async fn usage_info_from_last_response() {
        let provider = MockToolProvider {
            responses: vec![LlmResponse {
                content: Some("Hi".to_string()),
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: Some(UsageInfo {
                    prompt_tokens: 100,
                    completion_tokens: 50,
                    total_tokens: 150,
                }),
            }],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            None,
            "test",
            &[],
            &default_security(),
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        let usage = result.usage.unwrap();
        assert_eq!(usage.prompt_tokens, 100);
        assert_eq!(usage.completion_tokens, 50);
        assert_eq!(usage.total_tokens, 150);
    }

    #[tokio::test]
    async fn empty_content_on_stop_returns_empty_string() {
        let provider = MockToolProvider {
            responses: vec![LlmResponse {
                content: None,
                tool_calls: vec![],
                finish_reason: "stop".to_string(),
                usage: None,
            }],
            call_count: AtomicUsize::new(0),
        };

        let result = run_tool_loop(
            &provider,
            None,
            "test",
            &[],
            &default_security(),
            "test",
            0.0,
            &default_config(),
            None,
        )
        .await
        .unwrap();

        assert_eq!(result.final_content, "");
    }
}
