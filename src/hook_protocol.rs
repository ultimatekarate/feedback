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
    /// Free-form guidance the agent will see alongside the decision.
    /// Independent of `permission_decision_reason` — the latter is
    /// coupled to deny/modify verdicts, while `additional_context` is
    /// used to nudge the agent even on pure allows (e.g., compaction
    /// watchdog warnings).
    #[serde(rename = "additionalContext", skip_serializing_if = "Option::is_none")]
    pub additional_context: Option<String>,
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
                additional_context: None,
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
                additional_context: None,
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
                additional_context: None,
            },
        }
    }

    /// Construct an "allow" response with a warning reason but no
    /// input modification. Used for the warn-zone middle tier where
    /// the governor wants to surface guidance to the agent without
    /// actually patching the tool call (e.g., the channel is in the
    /// warning zone but no useful patch exists for this tool name).
    pub fn allow_with_warning(reason: &str) -> Self {
        Self {
            hook_specific_output: HookDecision {
                hook_event_name: "PreToolUse".into(),
                permission_decision: "allow".into(),
                permission_decision_reason: reason.to_string(),
                updated_input: None,
                additional_context: None,
            },
        }
    }

    /// Attach `additionalContext` guidance to an existing response.
    /// Used by the compaction watchdog to surface advice without
    /// changing the underlying permission decision — an allow remains
    /// an allow, a modify remains a modify, but the agent sees the
    /// extra context alongside.
    pub fn with_additional_context(mut self, context: String) -> Self {
        self.hook_specific_output.additional_context = Some(context);
        self
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

    #[test]
    fn allow_with_warning_carries_reason_without_modification() {
        let output = HookOutput::allow_with_warning("context at 72% of critical");
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"permissionDecision\":\"allow\""));
        assert!(json.contains("context at 72% of critical"));
        // No updatedInput field — must be omitted, not null,
        // because skip_serializing_if = "Option::is_none".
        assert!(!json.contains("updatedInput"));
    }

    #[test]
    fn additional_context_is_omitted_when_none() {
        // Default builders must not emit additionalContext at all —
        // an empty/null field would make Claude Code surface an empty
        // note to the agent on every single allow.
        for output in [
            HookOutput::allow(),
            HookOutput::deny("x"),
            HookOutput::allow_with_warning("y"),
        ] {
            let json = serde_json::to_string(&output).unwrap();
            assert!(
                !json.contains("additionalContext"),
                "default output must not serialize additionalContext: {json}"
            );
        }
    }

    #[test]
    fn with_additional_context_attaches_guidance() {
        let output = HookOutput::allow().with_additional_context(
            "You compacted twice recently — consider breaking work into chunks.".into(),
        );
        let json = serde_json::to_string(&output).unwrap();
        assert!(json.contains("\"additionalContext\""));
        assert!(json.contains("compacted twice recently"));
        // The underlying decision is still an allow — watchdog guidance
        // never flips the verdict.
        assert!(json.contains("\"permissionDecision\":\"allow\""));
    }
}
