//! Real-time spectral monitoring and decision logic.
//!
//! Laboratory layer: pure math. Takes a pressure snapshot, builds the
//! Jacobian at the current operating point, runs spectral analysis,
//! and produces a Decision. No IO.

use volterra_stability::config::{ImpulseRates, OperatingPoint};
use volterra_stability::eigenvalues::analyze_stability;
use volterra_stability::scaler::composite_stress;
use volterra_stability::spectral::analyze_spectral_gap;

use crate::channels::{Channel, ChannelThresholds};
use crate::decision::{Decision, PressureSnapshot, ToolPatch};
use crate::pressure_model::AgentCouplingModel;

/// Build a PressureSnapshot from the current integral values.
///
/// Computes the Jacobian at the current operating point, runs eigenvalue
/// analysis for stability, and spectral gap analysis for robustness.
pub fn check_stability(
    model: &AgentCouplingModel,
    values: &[f64],
    rates: &ImpulseRates,
) -> PressureSnapshot {
    use volterra_stability::coupling::CouplingModel;

    let cfg = model.config();
    let criticals = cfg.criticals();
    let weights = &cfg.stress_weights;

    let op = OperatingPoint {
        vals: values.to_vec(),
    };

    let j = model.build_jacobian(rates, &op);

    let stability = analyze_stability("live", &j, None, None);
    let spectral = analyze_spectral_gap("live", &j);

    let stress = composite_stress(values, &criticals, weights);

    PressureSnapshot {
        values: values.to_vec(),
        prev_values: None, // populated by Governor after construction
        composite_stress: stress,
        spectral_gap: spectral.spectral_gap_gamma1,
        is_stable: stability.is_stable,
    }
}

/// Pure decision logic: map a pressure snapshot to a governor Decision.
///
/// Four levels of intervention, checked in order:
/// 1. System spectrally unstable                       → Deny (cascading)
/// 2. Any channel above its deny threshold             → Deny (channel-specific)
///    **Exception: `Channel::Context` is never denied here.** Context
///    pressure is handled by Claude Code's own auto-compactor, which
///    summarises the transcript before the window actually fills; the
///    governor's role on this channel is to surface guidance, not to
///    block tool calls the agent needs in order to reach the point
///    where Claude Code would have compacted anyway. The watchdog in
///    `feedback-hook` catches the pathological "compaction thrashing"
///    case out-of-band via `additionalContext` injection.
/// 3. Spectral gap below 1e-6 (instability boundary)   → Deny (margin)
/// 4. Any channel in the warn zone (warn < n ≤ deny)   → Modify
///    - If a tool-specific patch exists for the
///      (channel, tool_name) pair AND tool_input is
///      supplied, the Modify carries an updated_input
///      that conservatively narrows the call (e.g.,
///      cap a Read's `limit`, cap a Bash's `timeout`).
///    - Otherwise the Modify carries only an
///      `additional_context` warning so the agent
///      sees the guidance even with no input patch.
///    Context is also included at this tier — if its normalised value
///    is anywhere above the warn threshold (including what would have
///    been the deny zone), it produces a Modify with a Read/Grep
///    narrowing patch instead of blocking the call.
/// 5. Otherwise                                        → Allow
///
/// `tool_input` is the raw JSON arguments the agent supplied for the
/// tool call. It's optional because some callers (e.g., the Coalition
/// MCP server's `feedback_evaluate` tool) only know the tool name and
/// have no actual input to patch. When `None`, warn-zone Modify
/// verdicts will only carry `additional_context`, never `updated_input`.
pub fn evaluate_decision(
    snapshot: &PressureSnapshot,
    thresholds: &ChannelThresholds,
    tool_name: &str,
    tool_input: Option<&serde_json::Value>,
) -> Decision {
    let cfg = crate::channels::FeedbackConfig::from_defaults();
    let criticals = cfg.system.criticals();

    // 1. Check for instability — immediate deny
    if !snapshot.is_stable {
        return Decision::Deny {
            reason: format!(
                "System is spectrally unstable (gamma1={:.4}). \
                 Pressure is cascading — stop and reassess your approach.",
                snapshot.spectral_gap
            ),
        };
    }

    // 2. Check per-channel deny thresholds. Context is intentionally
    //    excluded — see the docstring above. If Context crosses its
    //    nominal deny threshold, it falls through to the warn-zone
    //    loop below and becomes a Modify verdict with narrowing
    //    patches, so the agent keeps making progress while the
    //    auto-compactor takes care of the actual window pressure.
    for ch in Channel::ALL {
        if ch == Channel::Context {
            continue;
        }
        let i = ch.index();
        let normalized = snapshot.values[i] / criticals[i];
        if normalized > thresholds.deny[i] {
            let pct = normalized * 100.0;
            let prev_pct = prev_normalized_pct(snapshot, ch);
            let traj = trajectory_label(pct, prev_pct);
            return Decision::Deny {
                reason: format!(
                    "{} pressure at {:.0}% of critical{}. {}",
                    ch.name(),
                    pct,
                    traj,
                    deny_guidance(ch, tool_name),
                ),
            };
        }
    }

    // 3. Check spectral gap — narrow gap is a hidden instability.
    if snapshot.spectral_gap < 1e-6 && snapshot.is_stable {
        return Decision::Deny {
            reason: format!(
                "Spectral gap dangerously narrow (gamma1={:.6}). \
                 The system is at the instability boundary. \
                 Try a different approach or ask the user for guidance.",
                snapshot.spectral_gap
            ),
        };
    }

    // 4. Check per-channel warn thresholds. The first channel in
    //    priority order whose normalized value lands in the warn band
    //    (warn < normalized ≤ deny) yields a Modify verdict.
    for ch in Channel::ALL {
        let i = ch.index();
        let normalized = snapshot.values[i] / criticals[i];
        if normalized > thresholds.warn[i] {
            return build_warn_modify(ch, tool_name, tool_input, normalized, snapshot);
        }
    }

    // 5. All clear.
    Decision::Allow
}

/// Build a `Decision::Modify` verdict for a warn-zone channel.
///
/// When `tool_input` is supplied AND the (channel, tool_name) pair has
/// a known conservative patch, the resulting `ToolPatch` carries an
/// `updated_input` value the hook layer can use as the replacement
/// arguments. Otherwise the patch carries only `additional_context`,
/// which the hook layer surfaces via `HookOutput::allow_with_warning`.
fn build_warn_modify(
    channel: Channel,
    tool_name: &str,
    tool_input: Option<&serde_json::Value>,
    normalized: f64,
    snapshot: &PressureSnapshot,
) -> Decision {
    let pct = normalized * 100.0;
    let prev_pct = prev_normalized_pct(snapshot, channel);
    let traj = trajectory_label(pct, prev_pct);
    let warning = format!(
        "{} pressure at {:.0}% of critical{} (warn zone). {}",
        channel.name(),
        pct,
        traj,
        warn_guidance(channel, tool_name),
    );

    let updated_input = tool_input.and_then(|input| patch_for(channel, tool_name, input));

    Decision::Modify {
        reason: warning.clone(),
        patch: ToolPatch {
            updated_input,
            additional_context: Some(warning),
        },
    }
}

/// Construct a conservatively-patched copy of `tool_input` for known
/// (channel, tool_name) pairs. Returns `None` if there's no patch
/// recipe for this combination — in which case the Modify verdict will
/// rely on `additional_context` alone to surface the warning.
///
/// The recipes only narrow scope; they never widen it. If the agent
/// already specified a more conservative bound (e.g., `limit: 50`
/// when the cap is 200), the existing bound is preserved.
fn patch_for(
    channel: Channel,
    tool_name: &str,
    tool_input: &serde_json::Value,
) -> Option<serde_json::Value> {
    match (channel, tool_name) {
        // Context pressure → narrow Read scope to first 200 lines.
        (Channel::Context, "Read") => {
            let mut patched = tool_input.clone();
            cap_or_set_u64(&mut patched, "limit", 200);
            ensure_field_u64(&mut patched, "offset", 0);
            Some(patched)
        }
        // Context pressure → cap Grep output at 50 lines.
        (Channel::Context, "Grep") => {
            let mut patched = tool_input.clone();
            cap_or_set_u64(&mut patched, "head_limit", 50);
            Some(patched)
        }
        // Latency pressure → cap Bash timeout at 30s to fail fast.
        (Channel::Latency, "Bash") => {
            let mut patched = tool_input.clone();
            cap_or_set_u64(&mut patched, "timeout", 30_000);
            Some(patched)
        }
        // No recipe — Modify will fall back to additional_context only.
        _ => None,
    }
}

/// Set a non-negative integer field on a JSON object, but only widen
/// the constraint when no value exists or when the existing value is
/// looser than `cap`. If the existing value is already at or below
/// `cap`, leave it alone — the agent's constraint is more restrictive
/// than ours and we don't want to override it.
fn cap_or_set_u64(input: &mut serde_json::Value, field: &str, cap: u64) {
    let Some(obj) = input.as_object_mut() else {
        return;
    };
    let needs_cap = match obj.get(field).and_then(|v| v.as_u64()) {
        Some(existing) => existing > cap,
        None => true,
    };
    if needs_cap {
        obj.insert(field.to_string(), serde_json::json!(cap));
    }
}

/// Insert a default value for a field iff the field is absent.
/// Used for `offset: 0` so a Read patch always has both bounds set.
fn ensure_field_u64(input: &mut serde_json::Value, field: &str, default: u64) {
    let Some(obj) = input.as_object_mut() else {
        return;
    };
    if !obj.contains_key(field) {
        obj.insert(field.to_string(), serde_json::json!(default));
    }
}

/// Channel-specific guidance for the warn-zone tier. Mirrors
/// `deny_guidance` but with softer language and concrete suggestions
/// the agent can act on without abandoning the call entirely.
fn warn_guidance(channel: Channel, tool_name: &str) -> String {
    match channel {
        Channel::Context => format!(
            "Context is filling up — narrowing the scope of '{}'. \
             Consider summarizing what you've already learned before \
             reading more files.",
            tool_name
        ),
        Channel::Latency => format!(
            "Recent calls have been slow — capping the timeout on '{}'. \
             Consider whether the operation is necessary or could be \
             replaced with something cheaper.",
            tool_name
        ),
        Channel::Error => format!(
            "Unresolved errors are accumulating. Consider focusing on \
             fixing the existing errors before invoking '{}'.",
            tool_name
        ),
        Channel::Repetition => format!(
            "You've repeated similar tool calls recently. Consider \
             whether '{}' is the right next step or if a different \
             approach would make more progress.",
            tool_name
        ),
        Channel::Progress => format!(
            "Forward progress is stalling. Consider breaking the task \
             into smaller pieces before invoking '{}'.",
            tool_name
        ),
    }
}

/// Compute a trajectory label from current and previous normalized
/// pressure. Returns " (falling)", " (rising)", " (stable)", or ""
/// depending on whether the pressure is trending down, up, holding
/// steady, or unknown (no previous value). Uses a 2% dead band so
/// sub-unit fluctuations from decay don't flip the label.
///
/// The trajectory label helps the agent distinguish transient pressure
/// (attrition through low-cost retries is viable) from escalating
/// pressure (a genuine strategy change is needed).
fn trajectory_label(current_pct: f64, prev_pct: Option<f64>) -> &'static str {
    match prev_pct {
        Some(prev) if current_pct < prev - 2.0 => " (falling)",
        Some(prev) if current_pct > prev + 2.0 => " (rising)",
        Some(_) => " (stable)",
        None => "",
    }
}

/// Compute the previous normalized percentage for a channel from a
/// snapshot's `prev_values`, if available.
fn prev_normalized_pct(snapshot: &PressureSnapshot, channel: Channel) -> Option<f64> {
    let cfg = crate::channels::FeedbackConfig::from_defaults();
    let criticals = cfg.system.criticals();
    snapshot.prev_values.as_ref().map(|prev| {
        prev[channel.index()] / criticals[channel.index()] * 100.0
    })
}

/// Generate channel-specific guidance when denying a tool call.
/// Context is handled via Modify, not Deny, so it is unreachable here —
/// `evaluate_decision` skips `Channel::Context` in the deny loop.
fn deny_guidance(channel: Channel, tool_name: &str) -> String {
    match channel {
        Channel::Context => {
            // Unreachable in practice. Keep a harmless fallback rather
            // than panicking: if a future caller ever routes Context
            // through the deny path we should surface *something*
            // instead of crashing the hook.
            "Context pressure is high.".into()
        }
        Channel::Error => {
            "Too many unresolved errors. Focus on fixing the current error \
             before attempting new work."
                .into()
        }
        Channel::Repetition => {
            format!(
                "You've accessed the same files repeatedly. \
                 Your current approach to '{}' isn't working — try a different strategy.",
                tool_name
            )
        }
        Channel::Progress => {
            "No forward progress detected. Break the current task into \
             smaller steps or ask the user for clarification."
                .into()
        }
        Channel::Latency => {
            "Response latency is very high. Reduce request complexity \
             by working with smaller files or simpler queries."
                .into()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a snapshot with one channel held at the given fraction
    /// of its critical threshold and everything else at zero. Used by
    /// the warn-zone tests to land precisely in the warn band without
    /// accidentally triggering deny on a different channel.
    fn snapshot_with_channel_at(channel: Channel, fraction: f64) -> PressureSnapshot {
        let cfg = crate::channels::FeedbackConfig::from_defaults();
        let criticals = cfg.system.criticals();
        let mut values = vec![0.0; Channel::DIM];
        values[channel.index()] = criticals[channel.index()] * fraction;
        PressureSnapshot {
            values,
            prev_values: None,
            composite_stress: fraction,
            spectral_gap: 0.5,
            is_stable: true,
        }
    }

    #[test]
    fn allow_when_all_channels_low() {
        let snapshot = PressureSnapshot {
            values: vec![0.0; Channel::DIM],
            prev_values: None,
            composite_stress: 0.0,
            spectral_gap: 1.0,
            is_stable: true,
        };
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", None);
        assert!(matches!(decision, Decision::Allow));
    }

    #[test]
    fn deny_when_unstable() {
        let snapshot = PressureSnapshot {
            values: vec![0.0; Channel::DIM],
            prev_values: None,
            composite_stress: 0.0,
            spectral_gap: 0.001,
            is_stable: false,
        };
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", None);
        assert!(matches!(decision, Decision::Deny { .. }));
    }

    #[test]
    fn deny_when_channel_exceeds_threshold() {
        // Push error channel to 90% of critical (above 85% deny threshold).
        let snapshot = snapshot_with_channel_at(Channel::Error, 0.90);
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", None);
        assert!(matches!(decision, Decision::Deny { .. }));
    }

    // ── Warn-zone tier (Decision::Modify) ──────────────────────────

    #[test]
    fn warn_zone_with_no_tool_input_returns_modify_with_only_context() {
        // Error channel at 70% of critical lands in the warn band
        // (warn=0.60, deny=0.85). With tool_input=None, the Modify
        // verdict must carry only additional_context, not updated_input.
        let snapshot = snapshot_with_channel_at(Channel::Error, 0.70);
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Bash", None);
        match decision {
            Decision::Modify { reason, patch } => {
                assert!(reason.contains("error"), "reason should mention channel: {reason}");
                assert!(reason.contains("warn zone"), "reason should mark the tier: {reason}");
                assert!(patch.updated_input.is_none(), "no input → no patched input");
                assert!(patch.additional_context.is_some(), "warning text must be present");
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[test]
    fn warn_zone_context_pressure_caps_read_limit() {
        // Context channel at 70% of critical, agent calling Read with
        // no offset/limit. Patched input must add `limit: 200` and
        // `offset: 0`, preserving file_path.
        let snapshot = snapshot_with_channel_at(Channel::Context, 0.70);
        let thresholds = ChannelThresholds::default();
        let tool_input = serde_json::json!({"file_path": "/foo/bar.rs"});
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", Some(&tool_input));
        match decision {
            Decision::Modify { patch, .. } => {
                let updated = patch.updated_input.expect("Read patch must produce updated_input");
                assert_eq!(updated["file_path"], "/foo/bar.rs", "file_path preserved");
                assert_eq!(updated["limit"], 200, "limit capped to 200");
                assert_eq!(updated["offset"], 0, "offset defaulted to 0");
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[test]
    fn warn_zone_preserves_existing_tighter_constraint() {
        // Agent already specified limit=50, which is more conservative
        // than the cap of 200. The patch must NOT widen it.
        let snapshot = snapshot_with_channel_at(Channel::Context, 0.70);
        let thresholds = ChannelThresholds::default();
        let tool_input = serde_json::json!({"file_path": "/foo/bar.rs", "limit": 50});
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", Some(&tool_input));
        match decision {
            Decision::Modify { patch, .. } => {
                let updated = patch.updated_input.expect("must produce updated_input");
                assert_eq!(updated["limit"], 50, "existing tighter limit must be preserved");
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[test]
    fn warn_zone_caps_existing_looser_constraint() {
        // Agent specified limit=5000, which is looser than the cap.
        // The patch must narrow it to 200.
        let snapshot = snapshot_with_channel_at(Channel::Context, 0.70);
        let thresholds = ChannelThresholds::default();
        let tool_input = serde_json::json!({"file_path": "/foo/bar.rs", "limit": 5000});
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", Some(&tool_input));
        match decision {
            Decision::Modify { patch, .. } => {
                let updated = patch.updated_input.expect("must produce updated_input");
                assert_eq!(updated["limit"], 200, "loose limit must be narrowed to cap");
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[test]
    fn warn_zone_latency_pressure_caps_bash_timeout() {
        // Latency at 65% of critical, calling Bash with no timeout.
        // Patched input must add `timeout: 30000`.
        let snapshot = snapshot_with_channel_at(Channel::Latency, 0.65);
        let thresholds = ChannelThresholds::default();
        let tool_input = serde_json::json!({"command": "cargo build"});
        let decision = evaluate_decision(&snapshot, &thresholds, "Bash", Some(&tool_input));
        match decision {
            Decision::Modify { patch, .. } => {
                let updated = patch.updated_input.expect("must produce updated_input");
                assert_eq!(updated["command"], "cargo build", "command preserved");
                assert_eq!(updated["timeout"], 30_000, "timeout capped to 30s");
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[test]
    fn warn_zone_grep_caps_head_limit() {
        let snapshot = snapshot_with_channel_at(Channel::Context, 0.70);
        let thresholds = ChannelThresholds::default();
        let tool_input = serde_json::json!({"pattern": "TODO"});
        let decision = evaluate_decision(&snapshot, &thresholds, "Grep", Some(&tool_input));
        match decision {
            Decision::Modify { patch, .. } => {
                let updated = patch.updated_input.expect("must produce updated_input");
                assert_eq!(updated["pattern"], "TODO", "pattern preserved");
                assert_eq!(updated["head_limit"], 50, "head_limit capped to 50");
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[test]
    fn warn_zone_unknown_tool_falls_back_to_context_only() {
        // Channel pressure in warn zone but the (channel, tool_name)
        // pair has no patch recipe. The Modify verdict must still fire
        // (the warning is valuable) but without an updated_input.
        let snapshot = snapshot_with_channel_at(Channel::Context, 0.70);
        let thresholds = ChannelThresholds::default();
        let tool_input = serde_json::json!({"some": "thing"});
        let decision =
            evaluate_decision(&snapshot, &thresholds, "WeirdCustomTool", Some(&tool_input));
        match decision {
            Decision::Modify { patch, reason } => {
                assert!(patch.updated_input.is_none(), "no recipe → no patched input");
                assert!(patch.additional_context.is_some());
                assert!(reason.contains("context"), "reason mentions channel: {reason}");
            }
            other => panic!("expected Modify, got {other:?}"),
        }
    }

    #[test]
    fn context_above_deny_threshold_becomes_modify_not_deny() {
        // Context at 95% of critical — normally would be solidly in
        // the deny zone (deny=0.85). Context is demoted: the verdict
        // must be a Modify that narrows the Read, never a Deny that
        // stalls the agent before Claude Code's own auto-compactor
        // can fire.
        let snapshot = snapshot_with_channel_at(Channel::Context, 0.95);
        let thresholds = ChannelThresholds::default();
        let tool_input = serde_json::json!({"file_path": "/foo/bar.rs"});
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", Some(&tool_input));
        match decision {
            Decision::Modify { patch, reason } => {
                assert!(
                    reason.contains("context"),
                    "reason should mention context: {reason}"
                );
                let updated = patch.updated_input.expect("Read patch must produce updated_input");
                assert_eq!(updated["limit"], 200, "limit capped to 200 even in deny zone");
            }
            other => panic!("expected Modify (context is demoted), got {other:?}"),
        }
    }

    #[test]
    fn context_above_deny_threshold_does_not_shadow_other_channels() {
        // Context at 95% AND Error at 90% — previously Context would
        // have won the deny loop (first in Channel::ALL order), now
        // it's skipped so Error correctly surfaces as the deny reason.
        let cfg = crate::channels::FeedbackConfig::from_defaults();
        let criticals = cfg.system.criticals();
        let mut values = vec![0.0; Channel::DIM];
        values[Channel::Context.index()] = criticals[Channel::Context.index()] * 0.95;
        values[Channel::Error.index()] = criticals[Channel::Error.index()] * 0.90;
        let snapshot = PressureSnapshot {
            values,
            prev_values: None,
            composite_stress: 0.92,
            spectral_gap: 0.5,
            is_stable: true,
        };
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", None);
        match decision {
            Decision::Deny { reason } => {
                assert!(
                    reason.contains("error"),
                    "error must surface through; context is demoted: {reason}"
                );
            }
            other => panic!("expected Deny on error channel, got {other:?}"),
        }
    }

    #[test]
    fn deny_takes_priority_over_warn() {
        // Error at 90% (deny zone) AND context at 70% (warn zone).
        // Deny must win.
        let cfg = crate::channels::FeedbackConfig::from_defaults();
        let criticals = cfg.system.criticals();
        let mut values = vec![0.0; Channel::DIM];
        values[Channel::Context.index()] = criticals[Channel::Context.index()] * 0.70;
        values[Channel::Error.index()] = criticals[Channel::Error.index()] * 0.90;
        let snapshot = PressureSnapshot {
            values,
            prev_values: None,
            composite_stress: 0.85,
            spectral_gap: 0.5,
            is_stable: true,
        };
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read", None);
        match decision {
            Decision::Deny { reason } => {
                assert!(reason.contains("error"), "deny must surface error, not context: {reason}");
            }
            other => panic!("expected Deny, got {other:?}"),
        }
    }
}
