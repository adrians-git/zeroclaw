use crate::security::SecurityPolicy;
use crate::tools::traits::{Tool, ToolResult};
use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use super::Skill;

/// Maximum execution time for HTTP skill requests.
const HTTP_TIMEOUT_SECS: u64 = 120;
/// Maximum execution time for shell skill commands.
const SHELL_TIMEOUT_SECS: u64 = 60;
/// Maximum response body size (2 MB).
const MAX_RESPONSE_BYTES: usize = 2 * 1024 * 1024;
/// Environment variables safe to pass to shell commands.
const SAFE_ENV_VARS: &[&str] = &[
    "PATH", "HOME", "TERM", "LANG", "LC_ALL", "LC_CTYPE", "USER", "SHELL", "TMPDIR",
];

/// Bridge that turns a `SkillTool` definition into an executable `Tool`.
pub struct SkillToolBridge {
    tool_name: String,
    description: String,
    kind: String,
    command_template: String,
    default_args: HashMap<String, String>,
    schema: serde_json::Value,
    security: Arc<SecurityPolicy>,
    // HTTP-specific
    http_method: String,
    http_headers: HashMap<String, String>,
    body_template: Option<String>,
    client: reqwest::Client,
}

impl std::fmt::Debug for SkillToolBridge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SkillToolBridge")
            .field("tool_name", &self.tool_name)
            .field("kind", &self.kind)
            .finish()
    }
}

/// Build executable `Tool` instances from loaded skills.
pub fn skill_tools_from_skills(
    skills: &[Skill],
    security: Arc<SecurityPolicy>,
) -> Vec<Box<dyn Tool>> {
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(HTTP_TIMEOUT_SECS))
        .build()
        .unwrap_or_default();

    let mut tools: Vec<Box<dyn Tool>> = Vec::new();

    for skill in skills {
        for st in &skill.tools {
            if st.kind == "script" {
                // Scripts not yet supported
                continue;
            }

            let tool_name = format!("{}_{}", skill.name, st.name);
            let schema = build_schema(&st.command, st.body_template.as_deref(), &st.args);

            tools.push(Box::new(SkillToolBridge {
                tool_name,
                description: st.description.clone(),
                kind: st.kind.clone(),
                command_template: st.command.clone(),
                default_args: st.args.clone(),
                schema,
                security: security.clone(),
                http_method: st.method.clone(),
                http_headers: st.headers.clone(),
                body_template: st.body_template.clone(),
                client: client.clone(),
            }));
        }
    }

    tools
}

// ── Variable substitution ──────────────────────────────────────────

/// Substitute `${var}`, `${var:default}`, and `${env:VAR}` in a template.
fn substitute(template: &str, args: &HashMap<String, String>) -> String {
    let mut result = String::with_capacity(template.len());
    let mut chars = template.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut var_expr = String::new();
            let mut depth = 1;
            for c in chars.by_ref() {
                if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                var_expr.push(c);
            }

            if let Some(stripped) = var_expr.strip_prefix("env:") {
                // Environment variable lookup
                result.push_str(&std::env::var(stripped).unwrap_or_default());
            } else if let Some((name, default)) = var_expr.split_once(':') {
                // Variable with default
                if let Some(val) = args.get(name) {
                    result.push_str(val);
                } else {
                    result.push_str(default);
                }
            } else {
                // Plain variable
                if let Some(val) = args.get(&var_expr) {
                    result.push_str(val);
                }
                // If not found, substitute as empty string
            }
        } else {
            result.push(ch);
        }
    }

    result
}

/// Shell-escape a value to prevent injection.
fn shell_escape(s: &str) -> String {
    if s.is_empty() {
        return "''".to_string();
    }
    // If all chars are safe, return as-is
    if s.chars()
        .all(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/' | ',' | ':'))
    {
        return s.to_string();
    }
    // Wrap in single quotes, escaping existing single quotes
    format!("'{}'", s.replace('\'', "'\\''"))
}

/// Substitute variables in a shell command with shell-escaped values.
fn substitute_shell(template: &str, args: &HashMap<String, String>) -> String {
    let escaped_args: HashMap<String, String> = args
        .iter()
        .map(|(k, v)| (k.clone(), shell_escape(v)))
        .collect();
    substitute(template, &escaped_args)
}

// ── Schema generation ──────────────────────────────────────────────

/// Extract `${var}` names from templates and build a JSON schema.
fn build_schema(
    command_template: &str,
    body_template: Option<&str>,
    default_args: &HashMap<String, String>,
) -> serde_json::Value {
    let mut vars: Vec<String> = Vec::new();

    // Parse variables from all templates
    for template in std::iter::once(command_template).chain(body_template) {
        extract_vars(template, &mut vars);
    }

    // Deduplicate
    vars.sort();
    vars.dedup();

    // Remove env: vars (those are resolved at runtime, not user-provided)
    vars.retain(|v| !v.starts_with("env:"));

    let mut properties = serde_json::Map::new();
    let mut required = Vec::new();

    for var in &vars {
        // Strip default from var name
        let name = var.split(':').next().unwrap_or(var);
        let has_default = var.contains(':') || default_args.contains_key(name);

        properties.insert(
            name.to_string(),
            serde_json::json!({
                "type": "string",
                "description": format!("Parameter: {name}")
            }),
        );

        if !has_default {
            required.push(serde_json::Value::String(name.to_string()));
        }
    }

    serde_json::json!({
        "type": "object",
        "properties": properties,
        "required": required
    })
}

/// Extract `${...}` variable expressions from a template string.
fn extract_vars(template: &str, out: &mut Vec<String>) {
    let mut chars = template.chars().peekable();
    while let Some(ch) = chars.next() {
        if ch == '$' && chars.peek() == Some(&'{') {
            chars.next(); // consume '{'
            let mut expr = String::new();
            let mut depth = 1;
            for c in chars.by_ref() {
                if c == '{' {
                    depth += 1;
                } else if c == '}' {
                    depth -= 1;
                    if depth == 0 {
                        break;
                    }
                }
                expr.push(c);
            }
            if !expr.is_empty() {
                out.push(expr);
            }
        }
    }
}

// ── Tool trait implementation ──────────────────────────────────────

#[async_trait]
impl Tool for SkillToolBridge {
    fn name(&self) -> &str {
        &self.tool_name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn parameters_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    async fn execute(&self, args: serde_json::Value) -> anyhow::Result<ToolResult> {
        // Merge provided args with defaults
        let mut merged = self.default_args.clone();
        if let Some(obj) = args.as_object() {
            for (k, v) in obj {
                if let Some(s) = v.as_str() {
                    merged.insert(k.clone(), s.to_string());
                } else {
                    merged.insert(k.clone(), v.to_string());
                }
            }
        }

        match self.kind.as_str() {
            "http" => self.execute_http(&merged).await,
            "shell" => self.execute_shell(&merged).await,
            other => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Unsupported skill tool kind: {other}")),
            }),
        }
    }
}

impl SkillToolBridge {
    async fn execute_http(&self, args: &HashMap<String, String>) -> anyhow::Result<ToolResult> {
        let url = substitute(&self.command_template, args);

        let method = match self.http_method.to_uppercase().as_str() {
            "GET" => reqwest::Method::GET,
            "POST" => reqwest::Method::POST,
            "PUT" => reqwest::Method::PUT,
            "PATCH" => reqwest::Method::PATCH,
            "DELETE" => reqwest::Method::DELETE,
            other => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Unsupported HTTP method: {other}")),
                });
            }
        };

        let mut request = self.client.request(method, &url);

        // Apply headers with env var substitution
        for (key, val_template) in &self.http_headers {
            let val = substitute(val_template, args);
            request = request.header(key.as_str(), val);
        }

        // Apply body if present
        if let Some(body_tmpl) = &self.body_template {
            let body = substitute(body_tmpl, args);
            request = request.body(body);
        }

        let response = match request.send().await {
            Ok(r) => r,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("HTTP request failed: {e}")),
                });
            }
        };

        let status = response.status();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .unwrap_or("")
            .to_string();

        let is_image = content_type.starts_with("image/");

        let bytes = match response.bytes().await {
            Ok(b) => b,
            Err(e) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(format!("Failed to read response body: {e}")),
                });
            }
        };

        if bytes.len() > MAX_RESPONSE_BYTES {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Response too large: {} bytes (max {MAX_RESPONSE_BYTES})",
                    bytes.len()
                )),
            });
        }

        let output = if is_image {
            // Return as data URI so the tool loop can detect it
            use base64::Engine;
            let b64 = base64::engine::general_purpose::STANDARD.encode(&bytes);
            format!("data:{content_type};base64,{b64}")
        } else {
            // Try to parse as JSON and extract useful text
            let text = String::from_utf8_lossy(&bytes).to_string();

            // For OpenAI image generation API, extract the b64_json field
            if let Ok(json) = serde_json::from_str::<serde_json::Value>(&text) {
                if let Some(b64) = json
                    .get("data")
                    .and_then(|d| d.as_array())
                    .and_then(|arr| arr.first())
                    .and_then(|item| item.get("b64_json"))
                    .and_then(|v| v.as_str())
                {
                    // Detect format from response or default to png
                    let mime = json
                        .get("data")
                        .and_then(|d| d.as_array())
                        .and_then(|arr| arr.first())
                        .and_then(|item| item.get("content_type"))
                        .and_then(|v| v.as_str())
                        .unwrap_or("image/png");
                    format!("data:{mime};base64,{b64}")
                } else {
                    text
                }
            } else {
                text
            }
        };

        Ok(ToolResult {
            success: status.is_success(),
            output,
            error: if status.is_success() {
                None
            } else {
                Some(format!("HTTP {status}"))
            },
        })
    }

    async fn execute_shell(&self, args: &HashMap<String, String>) -> anyhow::Result<ToolResult> {
        let command = substitute_shell(&self.command_template, args);

        // Validate via security policy
        match self
            .security
            .validate_command_execution(&command, false)
        {
            Ok(_) => {}
            Err(reason) => {
                return Ok(ToolResult {
                    success: false,
                    output: String::new(),
                    error: Some(reason),
                });
            }
        }

        if !self.security.record_action() {
            return Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some("Rate limit exceeded".into()),
            });
        }

        let mut cmd = tokio::process::Command::new("sh");
        cmd.arg("-c").arg(&command);
        cmd.current_dir(&self.security.workspace_dir);
        cmd.env_clear();

        for var in SAFE_ENV_VARS {
            if let Ok(val) = std::env::var(var) {
                cmd.env(var, val);
            }
        }

        let result =
            tokio::time::timeout(Duration::from_secs(SHELL_TIMEOUT_SECS), cmd.output()).await;

        match result {
            Ok(Ok(output)) => {
                let stdout = String::from_utf8_lossy(&output.stdout).to_string();
                let stderr = String::from_utf8_lossy(&output.stderr).to_string();
                Ok(ToolResult {
                    success: output.status.success(),
                    output: stdout,
                    error: if stderr.is_empty() {
                        None
                    } else {
                        Some(stderr)
                    },
                })
            }
            Ok(Err(e)) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!("Failed to execute command: {e}")),
            }),
            Err(_) => Ok(ToolResult {
                success: false,
                output: String::new(),
                error: Some(format!(
                    "Command timed out after {SHELL_TIMEOUT_SECS}s"
                )),
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn substitute_plain_var() {
        let mut args = HashMap::new();
        args.insert("name".to_string(), "world".to_string());
        assert_eq!(substitute("hello ${name}", &args), "hello world");
    }

    #[test]
    fn substitute_var_with_default() {
        let args = HashMap::new();
        assert_eq!(
            substitute("size=${size:1024x1024}", &args),
            "size=1024x1024"
        );
    }

    #[test]
    fn substitute_var_override_default() {
        let mut args = HashMap::new();
        args.insert("size".to_string(), "512x512".to_string());
        assert_eq!(substitute("size=${size:1024x1024}", &args), "size=512x512");
    }

    #[test]
    fn substitute_env_var() {
        std::env::set_var("ZEROCLAW_TEST_BRIDGE_VAR", "secret123");
        let args = HashMap::new();
        let result = substitute("Bearer ${env:ZEROCLAW_TEST_BRIDGE_VAR}", &args);
        assert_eq!(result, "Bearer secret123");
        std::env::remove_var("ZEROCLAW_TEST_BRIDGE_VAR");
    }

    #[test]
    fn substitute_missing_var_becomes_empty() {
        let args = HashMap::new();
        assert_eq!(substitute("x=${missing}y", &args), "x=y");
    }

    #[test]
    fn substitute_multiple_vars() {
        let mut args = HashMap::new();
        args.insert("a".to_string(), "1".to_string());
        args.insert("b".to_string(), "2".to_string());
        assert_eq!(substitute("${a}+${b}", &args), "1+2");
    }

    #[test]
    fn shell_escape_safe_string() {
        assert_eq!(shell_escape("hello"), "hello");
        assert_eq!(shell_escape("path/to/file.txt"), "path/to/file.txt");
    }

    #[test]
    fn shell_escape_empty() {
        assert_eq!(shell_escape(""), "''");
    }

    #[test]
    fn shell_escape_dangerous_chars() {
        assert_eq!(shell_escape("hello world"), "'hello world'");
        assert_eq!(shell_escape("$(rm -rf /)"), "'$(rm -rf /)'");
        assert_eq!(shell_escape("a;b"), "'a;b'");
    }

    #[test]
    fn shell_escape_single_quotes() {
        assert_eq!(shell_escape("it's"), "'it'\\''s'");
    }

    #[test]
    fn build_schema_extracts_vars() {
        let schema = build_schema(
            "https://api.example.com/${endpoint}",
            Some(r#"{"prompt":"${prompt}","size":"${size:1024x1024}"}"#),
            &HashMap::new(),
        );
        let props = schema["properties"].as_object().unwrap();
        assert!(props.contains_key("endpoint"));
        assert!(props.contains_key("prompt"));
        assert!(props.contains_key("size"));

        let required = schema["required"].as_array().unwrap();
        // endpoint and prompt have no default → required
        assert!(required.contains(&serde_json::json!("endpoint")));
        assert!(required.contains(&serde_json::json!("prompt")));
        // size has default → not required
        assert!(!required.contains(&serde_json::json!("size")));
    }

    #[test]
    fn build_schema_skips_env_vars() {
        let schema = build_schema(
            "https://api.example.com/v1",
            Some(r#"{"key":"${env:API_KEY}"}"#),
            &HashMap::new(),
        );
        let props = schema["properties"].as_object().unwrap();
        assert!(!props.contains_key("env:API_KEY"));
    }

    #[test]
    fn build_schema_default_args_make_optional() {
        let mut defaults = HashMap::new();
        defaults.insert("quality".to_string(), "medium".to_string());

        let schema = build_schema(
            "https://api.example.com",
            Some(r#"{"quality":"${quality}"}"#),
            &defaults,
        );
        let required = schema["required"].as_array().unwrap();
        assert!(!required.contains(&serde_json::json!("quality")));
    }

    #[test]
    fn skill_tools_from_skills_creates_tools() {
        let skills = vec![Skill {
            name: "test-skill".to_string(),
            description: "A test".to_string(),
            version: "1.0.0".to_string(),
            author: None,
            tags: vec![],
            tools: vec![
                super::super::SkillTool {
                    name: "greet".to_string(),
                    description: "Says hello".to_string(),
                    kind: "shell".to_string(),
                    command: "echo hello ${name}".to_string(),
                    args: HashMap::new(),
                    method: "GET".to_string(),
                    headers: HashMap::new(),
                    body_template: None,
                },
                super::super::SkillTool {
                    name: "fetch".to_string(),
                    description: "Fetches data".to_string(),
                    kind: "http".to_string(),
                    command: "https://api.example.com/${path}".to_string(),
                    args: HashMap::new(),
                    method: "GET".to_string(),
                    headers: HashMap::new(),
                    body_template: None,
                },
                super::super::SkillTool {
                    name: "run_script".to_string(),
                    description: "Runs a script".to_string(),
                    kind: "script".to_string(),
                    command: "script.py".to_string(),
                    args: HashMap::new(),
                    method: "GET".to_string(),
                    headers: HashMap::new(),
                    body_template: None,
                },
            ],
            prompts: vec![],
            location: None,
        }];

        let security = Arc::new(SecurityPolicy::default());
        let tools = skill_tools_from_skills(&skills, security);

        // script kind is skipped
        assert_eq!(tools.len(), 2);
        assert_eq!(tools[0].name(), "test-skill_greet");
        assert_eq!(tools[1].name(), "test-skill_fetch");
    }

    #[test]
    fn skill_tools_empty_skills() {
        let security = Arc::new(SecurityPolicy::default());
        let tools = skill_tools_from_skills(&[], security);
        assert!(tools.is_empty());
    }

    #[test]
    fn extract_vars_from_template() {
        let mut vars = Vec::new();
        extract_vars(
            r#"{"model":"gpt-4","prompt":"${prompt}","n":1,"size":"${size:1024x1024}"}"#,
            &mut vars,
        );
        assert!(vars.contains(&"prompt".to_string()));
        assert!(vars.contains(&"size:1024x1024".to_string()));
    }

    #[test]
    fn substitute_shell_escapes_values() {
        let mut args = HashMap::new();
        args.insert("input".to_string(), "hello; rm -rf /".to_string());
        let result = substitute_shell("echo ${input}", &args);
        assert_eq!(result, "echo 'hello; rm -rf /'");
    }
}
