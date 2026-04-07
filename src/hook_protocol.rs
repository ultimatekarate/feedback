//! Claude Code hook protocol types.
//!
//! Dictionary layer: inert nouns. Data transfer objects for the
//! PreToolUse hook JSON protocol.

use serde::{Deserialize, Serialize};

/// Input received from Claude Code on a PreToolUse hook invocation.
#[derive(Debug, Clone, Deserialize)]
pub struct HookInput {
    /// Name of the tool being invoked (e.g., "Read", "Write", "Bash").
    pub tool_name: String,
    /// The tool's input arguments as raw JSON.
    pub tool_input: serde_json::Value,
    /// Session identifier.
    #[serde(default)]
    pub session_id: String,
}

/// Output returned to Claude Code from the hook.
#[derive(Debug, Clone, Serialize)]
pub struct HookOutput {
    /// Hook-specific output conforming to Claude Code's protocol.
    #[serde(rename = "hookSpecificOutput")]
    pub hook_specific_output: HookDecision,
}

/// The decision payload within the hook output.
#[derive(Debug, Clone, Serialize)]
pub struct HookDecision {
    /// Always "PreToolUse" for this hook type.
    #[serde(rename = "hookEventName")]
    pub hook_event_name: String,
    /// "allow", "deny", or "ask".
    #[serde(rename = "permissionDecision")]
    pub permission_decision: String,
    /// Explanation sent to Claude — this is how the governor
    /// communicates guidance to the agent.
    #[serde(rename = "permissionDecisionReason")]
    pub permission_decision_reason: String,
    /// Modified tool input, if the governor wants to adjust the call.
    #[serde(rename = "updatedInput", skip_serializing_if = "Option::is_none")]
    pub updated_input: Option<serde_json::Value>,
}

impl HookOutput {
    /// Construct an "allow" response.
    pub fn allow() -> Self {
        Self {
            hook_specific_output: HookDecision {
                hook_event_name: "PreToolUse".into(),
                permission_decision: "allow".into(),
                permission_decision_reason: String::new(),
                updated_input: None,
            },
        }
    }

    /// Construct a "deny" response with a reason the agent will see.
    pub fn deny(reason: &str) -> Self {
        Self {
            hook_specific_output: HookDecision {
                hook_event_name: "PreToolUse".into(),
                permission_decision: "deny".into(),
                permission_decision_reason: reason.to_string(),
                updated_input: None,
            },
        }
    }

    /// Construct an "allow" response with modified input.
    pub fn allow_with_modification(reason: &str, updated_input: serde_json::Value) -> Self {
        Self {
            hook_specific_output: HookDecision {
                hook_event_name: "PreToolUse".into(),
                permission_decision: "allow".into(),
                permission_decision_reason: reason.to_string(),
                updated_input: Some(updated_input),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allow_serializes_correctly() {
        let output = HookOutput::allow();
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"permissionDecision\":\"allow\""));
    }

    #[test]
    fn deny_includes_reason() {
        let output = HookOutput::deny("Read cascade limit reached");
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("Read cascade limit reached"));
        assert!(json.contains("\"permissionDecision\":\"deny\""));
    }

    #[test]
    fn hook_input_deserializes() {
        let json = r#"{"tool_name":"Read","tool_input":{"file_path":"/foo/bar.rs"}}"#;
        let input: HookInput = serde_json::from_str(json).unwrap();
        assert_eq!(input.tool_name, "Read");
    }
}
