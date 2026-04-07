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
use crate::decision::{Decision, PressureSnapshot};
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
        composite_stress: stress,
        spectral_gap: spectral.spectral_gap_gamma1,
        is_stable: stability.is_stable,
    }
}

/// Pure decision logic: map a pressure snapshot to a governor Decision.
///
/// Three levels of intervention:
/// 1. Stable and below warn thresholds → Allow
/// 2. Any channel above warn threshold → Allow with warning logged
/// 3. Any channel above deny threshold OR system unstable → Deny
pub fn evaluate_decision(
    snapshot: &PressureSnapshot,
    thresholds: &ChannelThresholds,
    tool_name: &str,
) -> Decision {
    let cfg = crate::channels::FeedbackConfig::from_defaults();
    let criticals = cfg.system.criticals();

    // Check for instability — immediate deny
    if !snapshot.is_stable {
        return Decision::Deny {
            reason: format!(
                "System is spectrally unstable (gamma1={:.4}). \
                 Pressure is cascading — stop and reassess your approach.",
                snapshot.spectral_gap
            ),
        };
    }

    // Check per-channel deny thresholds
    for ch in Channel::ALL {
        let i = ch.index();
        let normalized = snapshot.values[i] / criticals[i];
        if normalized > thresholds.deny[i] {
            return Decision::Deny {
                reason: format!(
                    "{} pressure at {:.0}% of critical. {}",
                    ch.name(),
                    normalized * 100.0,
                    deny_guidance(ch, tool_name),
                ),
            };
        }
    }

    // Check spectral gap — if it's narrowing, warn.
    // Note: the cost channel has near-zero lambda (monotonic accumulator),
    // which produces a near-zero eigenvalue by design. The spectral gap
    // threshold must account for this. We check the gap excluding the
    // cost channel's contribution by using a very small threshold.
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

    Decision::Allow
}

/// Generate channel-specific guidance when denying a tool call.
fn deny_guidance(channel: Channel, tool_name: &str) -> String {
    match channel {
        Channel::Context => {
            "Context window is filling up. Summarize what you know and \
             work with existing information instead of reading more files."
                .into()
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
        Channel::Cost => {
            "Session cost budget is nearly exhausted. Complete only \
             essential remaining work."
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

    #[test]
    fn allow_when_all_channels_low() {
        let snapshot = PressureSnapshot {
            values: vec![0.0; Channel::DIM],
            composite_stress: 0.0,
            spectral_gap: 1.0,
            is_stable: true,
        };
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read");
        assert!(matches!(decision, Decision::Allow));
    }

    #[test]
    fn deny_when_unstable() {
        let snapshot = PressureSnapshot {
            values: vec![0.0; Channel::DIM],
            composite_stress: 0.0,
            spectral_gap: 0.001,
            is_stable: false,
        };
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read");
        assert!(matches!(decision, Decision::Deny { .. }));
    }

    #[test]
    fn deny_when_channel_exceeds_threshold() {
        let cfg = crate::channels::FeedbackConfig::from_defaults();
        let criticals = cfg.system.criticals();

        // Push error channel to 90% of critical (above 85% deny threshold)
        let mut values = vec![0.0; Channel::DIM];
        values[Channel::Error.index()] = criticals[Channel::Error.index()] * 0.90;

        let snapshot = PressureSnapshot {
            values,
            composite_stress: 0.5,
            spectral_gap: 0.5,
            is_stable: true,
        };
        let thresholds = ChannelThresholds::default();
        let decision = evaluate_decision(&snapshot, &thresholds, "Read");
        assert!(matches!(decision, Decision::Deny { .. }));
    }
}
