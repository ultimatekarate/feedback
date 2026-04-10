//! Pressure channel definitions and system configuration.
//!
//! Dictionary layer: inert nouns. Maps the six agentic pressure channels
//! to a volterra-stability SystemConfig.

use serde::{Deserialize, Serialize};
use volterra_stability::config::{ChannelConfig, SystemConfig};

/// The five pressure channels of an agentic workflow.
///
/// Cost is tracked as a standalone metric (cumulative dollar spend),
/// not a pressure channel. It's redundant with every other channel —
/// every tool call fires cost AND at least one other channel — and its
/// near-zero decay rate distorts spectral analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Channel {
    /// Token accumulation in the context window.
    Context,
    /// Response time per interaction.
    Latency,
    /// Unresolved failure count (compile errors, test failures, tool failures).
    Error,
    /// Stall detection: inverse of forward progress.
    Progress,
    /// Same-file access patterns (re-reading, re-editing).
    Repetition,
}

impl Channel {
    /// Total number of pressure channels.
    pub const DIM: usize = 5;

    /// All channels in index order.
    pub const ALL: [Channel; 5] = [
        Channel::Context,
        Channel::Latency,
        Channel::Error,
        Channel::Progress,
        Channel::Repetition,
    ];

    /// Map channel to its index in the integral bank.
    pub fn index(self) -> usize {
        match self {
            Channel::Context => 0,
            Channel::Latency => 1,
            Channel::Error => 2,
            Channel::Progress => 3,
            Channel::Repetition => 4,
        }
    }

    /// Channel name for reporting.
    pub fn name(self) -> &'static str {
        match self {
            Channel::Context => "context",
            Channel::Latency => "latency",
            Channel::Error => "error",
            Channel::Progress => "progress",
            Channel::Repetition => "repetition",
        }
    }
}

/// Per-channel thresholds for governor intervention.
///
/// Distinct from the critical values in the scaler (those define the
/// mathematical saturation point). These define policy — when the
/// governor should warn vs deny.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelThresholds {
    /// Pressure level that triggers a warning (logged, not blocked).
    pub warn: Vec<f64>,
    /// Pressure level that triggers denial (tool call blocked).
    pub deny: Vec<f64>,
}

impl Default for ChannelThresholds {
    fn default() -> Self {
        Self {
            // Warn at 60% of critical, deny at 85%.
            warn: vec![0.60, 0.60, 0.60, 0.60, 0.60],
            deny: vec![0.85, 0.85, 0.85, 0.85, 0.85],
        }
    }
}

/// Complete configuration for the Feedback governor.
#[derive(Debug, Clone)]
pub struct FeedbackConfig {
    /// The underlying volterra-stability system config.
    pub system: SystemConfig,
    /// Governor intervention thresholds.
    pub thresholds: ChannelThresholds,
}

impl FeedbackConfig {
    /// Default 5-channel configuration calibrated from empirical session data.
    ///
    /// Calibration source: three honeypot sessions (Phase 0).
    ///   v1 bug-fix: 38 calls / 5 min / healthy → peaks at ~30%
    ///   v2 bug-fix: 45 calls / 6 min / healthy → peaks at ~30%
    ///   v3 feature-on-bugs: 47 calls / 24 min / struggling → progress 70%, latency 55%
    ///
    /// Decay rates (lambda) encode the memory horizon — how far back the
    /// integral looks. Calibrated so pressure accumulates over the relevant
    /// behavioral timescale for each channel.
    ///
    /// Critical thresholds are set so healthy sessions peak at ~30% and
    /// struggling sessions cross the 60% warn threshold on the channels
    /// that distinguish them.
    ///
    /// Cost is tracked as a standalone metric in the Governor, not as a
    /// pressure channel. It's redundant (every tool call fires cost AND
    /// another channel) and its near-zero lambda distorts spectral analysis.
    pub fn from_defaults() -> Self {
        let channels = vec![
            ChannelConfig {
                name: "context".into(),
                // Working set window: how long a piece of context remains
                // relevant to the current task. Derived from the inter-reference
                // time of files in the agent's working set (~6 min cycle).
                // lambda = ln(2)/360 ≈ 0.0019
                lambda: 0.0019,
                // Impulses are measured token counts (chars/4 estimate from
                // JSONL data): Read=1175, Edit=200, Write=2400, Bash=565,
                // Grep/Glob=100, Agent=4400. Healthy sessions peak at
                // ~11K-12K token-units. 4x healthy peak = 50K.
                critical: 50_000.0,
            },
            ChannelConfig {
                name: "latency".into(),
                // EWMA of response time over ~15 recent tool calls. At one
                // call per ~8s, that's a window of ~130s.
                // lambda = ln(2)/130 ≈ 0.0053
                lambda: 0.0053,
                // Units: actual elapsed seconds from Bash commands.
                // Sustained 2-min Bash calls at 1 per 60s: rate=2.0,
                // steady=2.0/0.0053=377. Critical at 350 = catastrophic
                // build times. Healthy sessions ~0.5%, v3 peaks at ~88%.
                critical: 350.0,
            },
            ChannelConfig {
                name: "error".into(),
                // Fix cycle duration: detect error → read source → edit fix
                // → verify. Measured from inter-test-invocation intervals
                // across sessions: ~4 minutes.
                // lambda = ln(2)/240 ≈ 0.0029
                lambda: 0.0029,
                // Units: weighted error count (1.0 per is_error, 0.5 per
                // parsed failure pattern). 6 active errors within the
                // 4-min decay window = sustained failure loop.
                critical: 6.0,
            },
            ChannelConfig {
                name: "progress".into(),
                // Derived from steady-state constraint: pure stalling at
                // base_rate should approach critical. lambda = base_rate / critical
                // = 0.1 / 55 ≈ 0.0018. Half-life ~6.3 min.
                lambda: 0.0018,
                // Pure stall steady-state: 0.1/0.0018 = 55.6. An agent that
                // produces NOTHING eventually saturates at critical. Healthy
                // sessions ~0-5% (edits dominate), v3 peaks at 82%.
                critical: 55.0,
            },
            ChannelConfig {
                name: "repetition".into(),
                // Healthy working set revisit interval: N files × T seconds
                // per file. With ~7 files and ~40s per file, the healthy
                // cycle is ~280s. Re-reads faster than this are thrashing.
                // lambda = ln(2)/280 ≈ 0.0025
                lambda: 0.0025,
                // Units: weighted re-access count (1.0 per Read revisit,
                // 0.5 per Edit revisit). 15 active re-accesses in a 4.6-min
                // window = reading the same files every few seconds.
                critical: 15.0,
            },
        ];

        // Weights redistributed proportionally after removing cost (was 0.10).
        let stress_weights = vec![
            0.28, // context — high weight, non-renewable resource
            0.11, // latency — informational
            0.33, // error — highest weight, errors drive cascades
            0.17, // progress — stalls waste tokens
            0.11, // repetition — indicator of thrashing
        ];

        Self {
            system: SystemConfig {
                channels,
                stress_weights,
            },
            thresholds: ChannelThresholds::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_has_five_channels() {
        let cfg = FeedbackConfig::from_defaults();
        assert_eq!(cfg.system.dim(), Channel::DIM);
    }

    #[test]
    fn channel_indices_are_contiguous() {
        for (i, ch) in Channel::ALL.iter().enumerate() {
            assert_eq!(ch.index(), i);
        }
    }

    #[test]
    fn stress_weights_sum_to_one() {
        let cfg = FeedbackConfig::from_defaults();
        let sum: f64 = cfg.system.stress_weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }
}
