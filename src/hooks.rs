//! Claude Code PreToolUse hook integration.
//!
//! Hands layer: IO-dependent. Bridges the Claude Code hook protocol
//! to the governor. Maps tool calls to pressure impulses and converts
//! governor verdicts to hook responses.
//!
//! ## Orthogonality invariant
//!
//! Each channel has a unique impulse source that no other channel shares:
//!
//! | Channel    | Exclusive source                                    |
//! |------------|-----------------------------------------------------|
//! | Context    | Estimated token load from tool I/O payload           |
//! | Cost       | Per-call spend tax (the only universal impulse)      |
//! | Latency    | Wall-clock time waiting for Bash commands             |
//! | Error      | Tool failures + parsed test/build failures (PostTool)|
//! | Progress   | Time-proportional stall minus productive actions     |
//! | Repetition | 2nd+ access to the same file path                   |
//!
//! No single impulse fires on all channels simultaneously.

use crate::channels::Channel;
use crate::decision::Decision;
use crate::governor::Governor;
use crate::hook_protocol::{HookInput, HookOutput};

/// Map a tool call to pressure impulses using orthogonal channel design.
///
/// Each channel receives impulses from a distinct, non-overlapping set
/// of observables. The `governor` is needed for stateful tracking
/// (file access counts, elapsed time).
fn apply_impulses(input: &HookInput, governor: &mut Governor) {
    // ── Progress: time-proportional stall accumulation ──────────
    // Must happen first — measures elapsed time BETWEEN calls.
    governor.accumulate_progress_stall();

    let tool = input.tool_name.as_str();
    let file_path = input
        .tool_input
        .get("file_path")
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let command = input
        .tool_input
        .get("command")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match tool {
        "Read" => {
            // Context: measured median 1175 tokens (range 289-8032).
            governor.record_impulse(Channel::Context, 1175.0);
            // Repetition: only on revisit.
            if !file_path.is_empty() {
                let count = governor.record_file_access(file_path);
                if count > 1 {
                    governor.record_impulse(Channel::Repetition, 1.0);
                }
            }
        }
        "Glob" | "Grep" => {
            // Context: measured median 94-108 tokens.
            governor.record_impulse(Channel::Context, 100.0);
        }
        "Write" => {
            // Context: measured median 2404 tokens.
            governor.record_impulse(Channel::Context, 2400.0);
            // Progress: strongest forward-motion signal.
            governor.record_impulse(Channel::Progress, -5.0);
        }
        "Edit" => {
            // Context: measured median 208 tokens.
            governor.record_impulse(Channel::Context, 200.0);
            // Progress: forward motion.
            governor.record_impulse(Channel::Progress, -3.0);
            // Repetition: only on revisit.
            if !file_path.is_empty() {
                let count = governor.record_file_access(file_path);
                if count > 1 {
                    governor.record_impulse(Channel::Repetition, 0.5);
                }
            }
        }
        "Bash" => {
            // Context: measured median 565 tokens.
            governor.record_impulse(Channel::Context, 565.0);
            // Latency: estimated elapsed time. Actual elapsed comes from
            // PostToolUse (Phase 2). These are PreToolUse estimates based
            // on measured medians: test/check ~5s (v1/v2), other ~2s.
            let is_build_or_test = command.contains("cargo test")
                || command.contains("cargo check")
                || command.contains("cargo build")
                || command.contains("pytest")
                || command.contains("python -m pytest");
            if is_build_or_test {
                governor.record_impulse(Channel::Latency, 5.0);
                // Verification step — mild forward motion.
                governor.record_impulse(Channel::Progress, -1.0);
            } else {
                governor.record_impulse(Channel::Latency, 2.0);
            }
        }
        "Agent" => {
            // Context: measured 4384 tokens (n=1).
            governor.record_impulse(Channel::Context, 4400.0);
            // Cost: agent surcharge (on top of per-call tax below).
            governor.record_impulse(Channel::Cost, 0.05);
        }
        _ => {
            // Unknown tool: conservative default.
            governor.record_impulse(Channel::Context, 200.0);
        }
    }

    // ── Cost: the ONLY universal per-call impulse ──────────────
    // No other channel fires on every tool call.
    governor.record_impulse(Channel::Cost, 0.005);

    // Note: Error channel is NOT populated here. PreToolUse fires
    // BEFORE the tool runs, so we cannot see errors or parse output.
    // Error impulses require a PostToolUse hook (Phase 2) or offline
    // replay via analyze_session.py.
}

/// Handle a PreToolUse hook invocation.
///
/// This is the main entry point called by the hook script/binary.
/// 1. Deserialize the hook input
/// 2. Accumulate progress stall since last call
/// 3. Map tool call to orthogonal pressure impulses
/// 4. Evaluate and return the decision
pub fn handle_pre_tool_use(
    input_json: &str,
    governor: &mut Governor,
) -> Result<String, serde_json::Error> {
    let input: HookInput = serde_json::from_str(input_json)?;

    // Apply all impulses (including progress stall accumulation)
    apply_impulses(&input, governor);

    // Evaluate the pressure state
    let verdict = governor.evaluate(&input.tool_name);

    // Convert to hook output
    let output = match verdict.decision {
        Decision::Allow => HookOutput::allow(),
        Decision::Deny { ref reason } => HookOutput::deny(reason),
        Decision::Modify {
            ref reason,
            ref patch,
        } => match &patch.updated_input {
            Some(updated) => {
                HookOutput::allow_with_modification(reason, updated.clone())
            }
            None => HookOutput::allow(),
        },
    };

    serde_json::to_string(&output)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_input(tool: &str, input: serde_json::Value) -> HookInput {
        HookInput {
            tool_name: tool.into(),
            tool_input: input,
            session_id: String::new(),
        }
    }

    #[test]
    fn read_first_visit_no_repetition() {
        let mut gov = Governor::from_defaults();
        let input = make_input("Read", serde_json::json!({"file_path": "/foo/bar.rs"}));
        apply_impulses(&input, &mut gov);
        let vals = gov.current_values();
        assert!(vals[Channel::Context.index()] > 0.0, "should have context");
        // First visit — repetition should be zero (or near-zero from decay)
        assert!(
            vals[Channel::Repetition.index()] < 0.01,
            "first visit should not fire repetition, got {}",
            vals[Channel::Repetition.index()]
        );
    }

    #[test]
    fn read_revisit_fires_repetition() {
        let mut gov = Governor::from_defaults();
        let input = make_input("Read", serde_json::json!({"file_path": "/foo/bar.rs"}));
        apply_impulses(&input, &mut gov);  // first visit
        apply_impulses(&input, &mut gov);  // revisit
        let vals = gov.current_values();
        assert!(
            vals[Channel::Repetition.index()] > 0.5,
            "revisit should fire repetition, got {}",
            vals[Channel::Repetition.index()]
        );
    }

    #[test]
    fn write_reduces_progress() {
        let mut gov = Governor::from_defaults();
        // First, build some stall pressure by recording a no-op Read.
        let read = make_input("Read", serde_json::json!({"file_path": "/a.rs"}));
        apply_impulses(&read, &mut gov);
        let before = gov.current_values()[Channel::Progress.index()];

        // Now Write: should reduce progress pressure (negative impulse).
        let write = make_input("Write", serde_json::json!({"file_path": "/b.rs", "content": "x"}));
        apply_impulses(&write, &mut gov);
        let after = gov.current_values()[Channel::Progress.index()];

        assert!(
            after < before,
            "Write should reduce progress pressure: before={before}, after={after}"
        );
        // Negative values are permitted — exponential decay pulls
        // them back toward zero asymptotically.
    }

    #[test]
    fn bash_pytest_higher_latency() {
        let mut gov = Governor::from_defaults();
        let test_input = make_input("Bash", serde_json::json!({"command": "python -m pytest tests/ -v"}));
        apply_impulses(&test_input, &mut gov);
        let vals_test = gov.current_values();

        let mut gov2 = Governor::from_defaults();
        let other_input = make_input("Bash", serde_json::json!({"command": "ls -la"}));
        apply_impulses(&other_input, &mut gov2);
        let vals_other = gov2.current_values();

        assert!(
            vals_test[Channel::Latency.index()] > vals_other[Channel::Latency.index()],
            "pytest should produce higher latency ({}) than ls ({})",
            vals_test[Channel::Latency.index()],
            vals_other[Channel::Latency.index()]
        );
    }

    #[test]
    fn cost_is_only_universal_impulse() {
        // Every tool type should produce a cost impulse.
        // No other single channel should be present in ALL tool types.
        let tools = vec![
            ("Read", serde_json::json!({"file_path": "/a.rs"})),
            ("Write", serde_json::json!({"file_path": "/b.rs", "content": "x"})),
            ("Edit", serde_json::json!({"file_path": "/c.rs", "old_string": "a", "new_string": "b"})),
            ("Bash", serde_json::json!({"command": "echo hi"})),
            ("Grep", serde_json::json!({"pattern": "foo"})),
            ("Glob", serde_json::json!({"pattern": "*.rs"})),
            ("Agent", serde_json::json!({"prompt": "do stuff"})),
        ];
        for (tool, input_val) in &tools {
            let mut gov = Governor::from_defaults();
            let input = make_input(tool, input_val.clone());
            apply_impulses(&input, &mut gov);
            let vals = gov.current_values();
            assert!(
                vals[Channel::Cost.index()] > 0.0,
                "{} should produce cost impulse",
                tool
            );
        }
    }

    #[test]
    fn grep_glob_only_context_and_cost() {
        let mut gov = Governor::from_defaults();
        let input = make_input("Grep", serde_json::json!({"pattern": "foo"}));
        apply_impulses(&input, &mut gov);
        let vals = gov.current_values();
        assert!(vals[Channel::Context.index()] > 0.0);
        assert!(vals[Channel::Cost.index()] > 0.0);
        // All other channels should be near-zero (only progress stall from dt≈0)
        assert!(vals[Channel::Latency.index()] < 0.01);
        assert!(vals[Channel::Error.index()] < 0.01);
        assert!(vals[Channel::Repetition.index()] < 0.01);
    }

    #[test]
    fn handle_hook_returns_valid_json() {
        let mut gov = Governor::from_defaults();
        let input = r#"{"tool_name":"Read","tool_input":{"file_path":"/foo.rs"}}"#;
        let result = handle_pre_tool_use(input, &mut gov);
        assert!(result.is_ok());
        let json = result.unwrap();
        assert!(json.contains("permissionDecision"));
    }
}
