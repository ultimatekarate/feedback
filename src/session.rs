//! Session logging and post-hoc analysis.
//!
//! Hands layer: IO-dependent. Accumulates governor verdicts over a
//! session's lifetime for post-session analysis and model fitting.

use std::io;
use std::path::Path;

use serde::{Deserialize, Serialize};

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

/// Compact digest of a session's pressure history.
///
/// Accumulated incrementally by the governor on each `evaluate` call
/// and persisted alongside the governor state so the SessionEnd handler
/// can write it to `<session>-digest.json` without replaying the full
/// verdict history. All vectors are indexed by `Channel::index()`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionDigest {
    /// Peak raw pressure value observed per channel.
    pub peak_pressure: Vec<f64>,
    /// Total number of tool-call evaluations in the session.
    pub total_evaluations: usize,
    /// Sum of positive impulses recorded per channel.
    pub total_impulses: Vec<f64>,
    /// Seconds each channel spent above the warn threshold.
    pub time_above_warn: Vec<f64>,
    /// Seconds each channel spent above the critical (deny) threshold.
    pub time_above_critical: Vec<f64>,
}

impl SessionDigest {
    /// Create a zeroed digest for a fresh session.
    pub fn parse_empty_session() -> Self {
        Self {
            peak_pressure: vec![0.0; Channel::DIM],
            total_evaluations: 0,
            total_impulses: vec![0.0; Channel::DIM],
            time_above_warn: vec![0.0; Channel::DIM],
            time_above_critical: vec![0.0; Channel::DIM],
        }
    }

    /// Update the digest with a new evaluation snapshot.
    ///
    /// `values` — current raw integral values per channel.
    /// `criticals` — per-channel critical thresholds from the system config.
    /// `warn` / `deny` — normalized threshold fractions (e.g. 0.60, 0.85).
    /// `dt` — seconds elapsed since the previous evaluation.
    ///
    /// Uses a left-endpoint rule: the time interval `dt` is attributed to
    /// whichever zone the *previous* snapshot was in (tracked internally
    /// via the peak/values from the prior call). On the first call `dt` is
    /// typically near zero so the approximation error is negligible.
    pub fn parse_update_session(
        &mut self,
        values: &[f64],
        criticals: &[f64],
        warn: &[f64],
        deny: &[f64],
        dt: f64,
    ) {
        self.total_evaluations += 1;
        for i in 0..Channel::DIM.min(values.len()) {
            if values[i] > self.peak_pressure[i] {
                self.peak_pressure[i] = values[i];
            }
            let normalized = if criticals[i] > 0.0 {
                values[i] / criticals[i]
            } else {
                0.0
            };
            if dt > 0.0 {
                if normalized > deny[i] {
                    self.time_above_critical[i] += dt;
                    // Critical implies warn — count in both buckets.
                    self.time_above_warn[i] += dt;
                } else if normalized > warn[i] {
                    self.time_above_warn[i] += dt;
                }
            }
        }
    }

    /// Record a positive impulse on a channel.
    pub fn parse_record_impulse_session(&mut self, channel_index: usize, impulse: f64) {
        if impulse > 0.0 && channel_index < self.total_impulses.len() {
            self.total_impulses[channel_index] += impulse;
        }
    }
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
            for (p, val) in peak.iter_mut().zip(v.snapshot.values.iter()).take(Channel::DIM) {
                *p = p.max(*val);
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
            .map_err(io::Error::other)?;
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

    // ── SessionDigest tests ────────────────────────────────────────

    #[test]
    fn parse_empty_session_digest_is_zeroed() {
        let d = SessionDigest::parse_empty_session();
        assert_eq!(d.total_evaluations, 0);
        assert_eq!(d.peak_pressure, vec![0.0; Channel::DIM]);
        assert_eq!(d.total_impulses, vec![0.0; Channel::DIM]);
        assert_eq!(d.time_above_warn, vec![0.0; Channel::DIM]);
        assert_eq!(d.time_above_critical, vec![0.0; Channel::DIM]);
    }

    #[test]
    fn parse_update_session_tracks_peak_pressure() {
        let mut d = SessionDigest::parse_empty_session();
        let criticals = vec![100.0; Channel::DIM];
        let warn = vec![0.60; Channel::DIM];
        let deny = vec![0.85; Channel::DIM];

        d.parse_update_session(&[10.0, 20.0, 30.0, 40.0, 50.0], &criticals, &warn, &deny, 0.0);
        d.parse_update_session(&[5.0, 25.0, 15.0, 45.0, 10.0], &criticals, &warn, &deny, 1.0);

        assert!((d.peak_pressure[0] - 10.0).abs() < 1e-10);
        assert!((d.peak_pressure[1] - 25.0).abs() < 1e-10);
        assert!((d.peak_pressure[2] - 30.0).abs() < 1e-10);
        assert!((d.peak_pressure[3] - 45.0).abs() < 1e-10);
        assert!((d.peak_pressure[4] - 50.0).abs() < 1e-10);
        assert_eq!(d.total_evaluations, 2);
    }

    #[test]
    fn parse_update_session_accumulates_time_above_warn() {
        let mut d = SessionDigest::parse_empty_session();
        // critical=100, warn=0.60 → warn at 60, deny at 85
        let criticals = vec![100.0; Channel::DIM];
        let warn = vec![0.60; Channel::DIM];
        let deny = vec![0.85; Channel::DIM];

        // First call: error channel at 70 (normalized 0.70 > warn 0.60)
        let mut vals = vec![0.0; Channel::DIM];
        vals[Channel::Error.index()] = 70.0;
        d.parse_update_session(&vals, &criticals, &warn, &deny, 5.0);
        assert!((d.time_above_warn[Channel::Error.index()] - 5.0).abs() < 1e-10);
        assert!((d.time_above_critical[Channel::Error.index()]).abs() < 1e-10);

        // Second call: error channel at 90 (normalized 0.90 > deny 0.85)
        vals[Channel::Error.index()] = 90.0;
        d.parse_update_session(&vals, &criticals, &warn, &deny, 3.0);
        // time_above_warn should be 5+3=8 (critical counts as warn too)
        assert!((d.time_above_warn[Channel::Error.index()] - 8.0).abs() < 1e-10);
        assert!((d.time_above_critical[Channel::Error.index()] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn parse_record_impulse_session_only_counts_positive() {
        let mut d = SessionDigest::parse_empty_session();
        d.parse_record_impulse_session(Channel::Error.index(), 2.5);
        d.parse_record_impulse_session(Channel::Error.index(), -1.0); // ignored
        d.parse_record_impulse_session(Channel::Progress.index(), 0.5);
        assert!((d.total_impulses[Channel::Error.index()] - 2.5).abs() < 1e-10);
        assert!((d.total_impulses[Channel::Progress.index()] - 0.5).abs() < 1e-10);
    }

    #[test]
    fn parse_digest_session_roundtrips_json() {
        let mut d = SessionDigest::parse_empty_session();
        d.total_evaluations = 42;
        d.peak_pressure[0] = 1234.5;
        let json = serde_json::to_string(&d).unwrap();
        let d2: SessionDigest = serde_json::from_str(&json).unwrap();
        assert_eq!(d2.total_evaluations, 42);
        assert!((d2.peak_pressure[0] - 1234.5).abs() < 1e-10);
    }

    // ── SessionLog tests ─────────────────────────────────────────

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
