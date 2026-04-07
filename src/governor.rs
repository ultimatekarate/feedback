//! Real-time governor: maintains live pressure state and produces verdicts.
//!
//! Hands layer: owns the mutable IntegralBank and the clock. This is the
//! stateful runtime object that the hook integration calls into.

use std::collections::HashMap;
use std::time::Instant;

use volterra_stability::config::ImpulseRates;
use volterra_stability::integral::IntegralBank;

use crate::channels::{Channel, FeedbackConfig};
use crate::decision::GovernorVerdict;
use crate::monitor;
use crate::pressure_model::AgentCouplingModel;

/// The live stability governor.
///
/// Maintains pressure state via an IntegralBank, computes spectral
/// stability on demand, and produces allow/deny decisions.
pub struct Governor {
    model: AgentCouplingModel,
    bank: IntegralBank,
    config: FeedbackConfig,
    rates: ImpulseRates,
    epoch: Instant,
    /// Per-file access count for repetition tracking.
    /// Key is normalized file path. Value >= 2 means "revisit."
    file_access_counts: HashMap<String, usize>,
    /// Timestamp of the last tool call (seconds since epoch).
    /// Used for time-proportional progress accumulation.
    last_call_t: f64,
}

impl Governor {
    /// Create a new governor with default configuration.
    pub fn new(config: FeedbackConfig, rates: ImpulseRates) -> Self {
        let lambdas = config.system.lambdas();
        let now = 0.0; // IntegralBank uses caller-provided f64 timestamps
        let bank = IntegralBank::from_lambdas(&lambdas, now);
        let model = AgentCouplingModel::new(config.clone());

        Self {
            model,
            bank,
            config,
            rates,
            epoch: Instant::now(),
            file_access_counts: HashMap::new(),
            last_call_t: 0.0,
        }
    }

    /// Create a governor with default parameters.
    pub fn from_defaults() -> Self {
        let config = FeedbackConfig::from_defaults();
        // Measured impulse rates from Phase 0 JSONL data.
        // Rates are in channel-units per second at ~1 call every 8s.
        let rates = ImpulseRates::from_slice(&[
            50.0,  // context: ~400 tokens/call / 8s (measured)
            0.001, // cost: 0.005 per call / 8s + overhead
            0.38,  // latency: ~3s per Bash, ~1 Bash per 8 calls
            0.01,  // error: ~2 errors per 300s session
            0.1,   // progress: 0.1/s base stall rate
            0.12,  // repetition: ~1.0 per call / 8s
        ]);
        Self::new(config, rates)
    }

    /// Current time in seconds since governor creation.
    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64()
    }

    /// Record a pressure impulse on a specific channel.
    ///
    /// Negative values are permitted — a burst of productive actions
    /// can drive progress below zero. The exponential decay naturally
    /// pulls the value back toward zero asymptotically, so the
    /// "productive credit" fades over time rather than being hard-clamped.
    pub fn record_impulse(&mut self, channel: Channel, impulse: f64) {
        let now = self.now();
        self.bank.record(channel.index(), impulse, now);
    }

    /// Record a file access and return the total access count.
    /// Returns 1 on first visit, 2+ on revisits. Normalizes path
    /// separators so Windows and Unix paths map to the same key.
    pub fn record_file_access(&mut self, file_path: &str) -> usize {
        let normalized = file_path.replace('\\', "/");
        let count = self.file_access_counts.entry(normalized).or_insert(0);
        *count += 1;
        *count
    }

    /// Accumulate time-proportional progress pressure since the last
    /// tool call. Called at the start of each hook invocation. Returns
    /// the elapsed seconds since the previous call.
    pub fn accumulate_progress_stall(&mut self) -> f64 {
        let now = self.now();
        let dt = now - self.last_call_t;
        if dt > 0.0 {
            // 0.1 progress-units per second of elapsed time.
            // This is the base stall rate — productive actions
            // (Edit, Write) give negative impulses to counteract.
            self.bank.record(Channel::Progress.index(), dt * 0.1, now);
        }
        self.last_call_t = now;
        dt
    }

    /// Read the current pressure values across all channels.
    pub fn current_values(&self) -> Vec<f64> {
        let now = self.now();
        self.bank.current_values(now)
    }

    /// Evaluate the current pressure state and produce a verdict.
    pub fn evaluate(&self, tool_name: &str) -> GovernorVerdict {
        let now = self.now();
        let values = self.bank.current_values(now);

        let snapshot = monitor::check_stability(&self.model, &values, &self.rates);
        let decision = monitor::evaluate_decision(
            &snapshot,
            &self.config.thresholds,
            tool_name,
        );

        GovernorVerdict {
            decision,
            snapshot,
            timestamp: now,
            tool_name: tool_name.to_string(),
        }
    }

    /// Record an impulse and immediately evaluate. Convenience method
    /// for the common hook pattern: observe tool call → update state → decide.
    pub fn record_and_evaluate(
        &mut self,
        channel: Channel,
        impulse: f64,
        tool_name: &str,
    ) -> GovernorVerdict {
        self.record_impulse(channel, impulse);
        self.evaluate(tool_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision::Decision;

    #[test]
    fn new_governor_starts_stable() {
        let gov = Governor::from_defaults();
        let verdict = gov.evaluate("Read");
        assert!(
            matches!(verdict.decision, Decision::Allow),
            "fresh governor should allow: {:?}",
            verdict.decision
        );
        assert!(verdict.snapshot.is_stable);
    }

    #[test]
    fn governor_records_impulses() {
        let mut gov = Governor::from_defaults();
        gov.record_impulse(Channel::Error, 3.0);
        let values = gov.current_values();
        assert!(
            values[Channel::Error.index()] > 0.0,
            "error channel should have pressure after impulse"
        );
    }
}
