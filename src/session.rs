//! Session logging and post-hoc analysis.
//!
//! Hands layer: IO-dependent. Accumulates governor verdicts over a
//! session's lifetime for post-session analysis and model fitting.

use std::io;
use std::path::Path;

use crate::channels::Channel;
use crate::decision::{Decision, GovernorVerdict};

/// Accumulated log of governor verdicts across a session.
#[derive(Debug, Clone)]
pub struct SessionLog {
    verdicts: Vec<GovernorVerdict>,
}

/// Summary statistics from a session.
#[derive(Debug, Clone)]
pub struct SessionSummary {
    /// Total number of tool calls evaluated.
    pub total_evaluations: usize,
    /// Number of tool calls denied.
    pub total_denials: usize,
    /// Peak pressure value per channel.
    pub peak_pressure: Vec<f64>,
    /// Channel that triggered the most denials.
    pub most_denied_channel: Option<String>,
    /// Session duration in seconds.
    pub duration_seconds: f64,
}

impl SessionLog {
    pub fn new() -> Self {
        Self {
            verdicts: Vec::new(),
        }
    }

    pub fn append(&mut self, verdict: GovernorVerdict) {
        self.verdicts.push(verdict);
    }

    pub fn verdicts(&self) -> &[GovernorVerdict] {
        &self.verdicts
    }

    /// Compute summary statistics from the session log.
    pub fn summary(&self) -> SessionSummary {
        let total = self.verdicts.len();
        let denials = self.verdicts
            .iter()
            .filter(|v| matches!(v.decision, Decision::Deny { .. }))
            .count();

        let mut peak = vec![0.0_f64; Channel::DIM];
        for v in &self.verdicts {
            for i in 0..Channel::DIM.min(v.snapshot.values.len()) {
                peak[i] = peak[i].max(v.snapshot.values[i]);
            }
        }

        let duration = self.verdicts
            .last()
            .map(|v| v.timestamp)
            .unwrap_or(0.0);

        // Find which channel caused the most denials by parsing reasons
        // (simplified — in production you'd tag denials with the channel)
        let most_denied = None; // deferred to v0.2

        SessionSummary {
            total_evaluations: total,
            total_denials: denials,
            peak_pressure: peak,
            most_denied_channel: most_denied,
            duration_seconds: duration,
        }
    }

    /// Write the session log to a JSON file for post-hoc analysis.
    pub fn write_to_file(&self, path: &Path) -> io::Result<()> {
        let json = serde_json::to_string_pretty(&self.verdicts)
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }
}

impl Default for SessionLog {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::PressureSnapshot;

    fn mock_verdict(decision: Decision, timestamp: f64) -> GovernorVerdict {
        GovernorVerdict {
            decision,
            snapshot: PressureSnapshot {
                values: vec![0.0; Channel::DIM],
                prev_values: None,
                composite_stress: 0.0,
                spectral_gap: 1.0,
                is_stable: true,
            },
            timestamp,
            tool_name: "Read".into(),
        }
    }

    #[test]
    fn empty_session_summary() {
        let log = SessionLog::new();
        let summary = log.summary();
        assert_eq!(summary.total_evaluations, 0);
        assert_eq!(summary.total_denials, 0);
    }

    #[test]
    fn summary_counts_denials() {
        let mut log = SessionLog::new();
        log.append(mock_verdict(Decision::Allow, 1.0));
        log.append(mock_verdict(
            Decision::Deny { reason: "test".into() },
            2.0,
        ));
        log.append(mock_verdict(Decision::Allow, 3.0));

        let summary = log.summary();
        assert_eq!(summary.total_evaluations, 3);
        assert_eq!(summary.total_denials, 1);
        assert!((summary.duration_seconds - 3.0).abs() < 1e-10);
    }
}
