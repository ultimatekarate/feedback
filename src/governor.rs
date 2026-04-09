//! Real-time governor: maintains live pressure state and produces verdicts.
//!
//! Hands layer: owns the mutable IntegralBank and the clock. This is the
//! stateful runtime object that the hook integration calls into.

use std::collections::HashMap;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use volterra_stability::config::ImpulseRates;
use volterra_stability::integral::IntegralBank;

use crate::channels::{Channel, FeedbackConfig};
use crate::decision::GovernorVerdict;
use crate::monitor;
use crate::pressure_model::AgentCouplingModel;

/// Serializable snapshot of the parts of `Governor` that have to
/// survive across process boundaries. Used by `feedback-hook` to
/// persist state to disk between PreToolUse invocations — each hook
/// invocation is a separate process, so without this the governor
/// would start fresh every time and never accumulate pressure.
///
/// `model`, `config`, and `rates` are NOT included: they are
/// reconstructed from `FeedbackConfig::from_defaults()` on resume.
/// If those defaults ever diverge between save and load (e.g. a
/// new feedback build with retuned thresholds), the resumed
/// governor uses the *new* defaults — bank values still apply
/// against the new thresholds, which is the desired behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PersistedState {
    pub bank: IntegralBank,
    pub file_access_counts: HashMap<String, usize>,
    pub last_call_t: f64,
    pub cumulative_cost: f64,
}

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
    /// Time offset added to `epoch.elapsed()` when computing `now()`.
    /// Zero for fresh governors. For governors resumed from a
    /// `PersistedState`, equals the saved `last_call_t` so that the
    /// new process's clock picks up exactly where the old one stopped
    /// — keeping the IntegralBank's per-channel `last_update`
    /// timestamps in the same time domain across the persistence
    /// boundary.
    time_offset: f64,
    /// Per-file access count for repetition tracking.
    /// Key is normalized file path. Value >= 2 means "revisit."
    file_access_counts: HashMap<String, usize>,
    /// Timestamp of the last tool call (seconds since epoch).
    /// Used for time-proportional progress accumulation.
    last_call_t: f64,
    /// Cumulative session cost in dollars. Standalone metric — not a
    /// pressure channel. Tracked for display and budget reporting only.
    cumulative_cost: f64,
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
            time_offset: 0.0,
            file_access_counts: HashMap::new(),
            last_call_t: 0.0,
            cumulative_cost: 0.0,
        }
    }

    /// Resume a governor from a persisted state. The new process's
    /// clock continues from where the previous one stopped — see the
    /// docstring on `time_offset` for the time-domain trick. The
    /// model, config, and rates are reconstructed from defaults
    /// rather than persisted; if you need a non-default tuning,
    /// resume manually via the field-by-field constructor.
    pub fn resume_from_defaults(state: PersistedState) -> Self {
        let config = FeedbackConfig::from_defaults();
        let rates = ImpulseRates::from_slice(&[50.0, 0.38, 0.01, 0.1, 0.12]);
        let model = AgentCouplingModel::new(config.clone());
        Self {
            model,
            bank: state.bank,
            config,
            rates,
            epoch: Instant::now(),
            time_offset: state.last_call_t.max(0.0),
            file_access_counts: state.file_access_counts,
            last_call_t: state.last_call_t,
            cumulative_cost: state.cumulative_cost,
        }
    }

    /// Take a serializable snapshot of the parts of `self` that need
    /// to survive across process boundaries. Cheap (clones a small
    /// `IntegralBank` and a `HashMap<String, usize>`).
    pub fn snapshot(&self) -> PersistedState {
        PersistedState {
            bank: self.bank.clone(),
            file_access_counts: self.file_access_counts.clone(),
            last_call_t: self.last_call_t,
            cumulative_cost: self.cumulative_cost,
        }
    }

    /// Create a governor with default parameters.
    pub fn from_defaults() -> Self {
        let config = FeedbackConfig::from_defaults();
        // Measured impulse rates from Phase 0 JSONL data.
        // Rates are in channel-units per second at ~1 call every 8s.
        let rates = ImpulseRates::from_slice(&[
            50.0,  // context: ~400 tokens/call / 8s (measured)
            0.38,  // latency: ~3s per Bash, ~1 Bash per 8 calls
            0.01,  // error: ~2 errors per 300s session
            0.1,   // progress: 0.1/s base stall rate
            0.12,  // repetition: ~1.0 per call / 8s
        ]);
        Self::new(config, rates)
    }

    /// Current time in seconds in the governor's time domain. For a
    /// fresh governor this equals seconds since construction. For a
    /// resumed governor it equals saved-`last_call_t` plus seconds
    /// since this process started — keeping the time domain stable
    /// across the persistence boundary.
    fn now(&self) -> f64 {
        self.epoch.elapsed().as_secs_f64() + self.time_offset
    }

    /// Record a cost increment. Cost is a standalone metric (cumulative
    /// dollar spend), not a pressure channel. It does not participate in
    /// the Jacobian, spectral analysis, or governor decisions.
    pub fn record_cost(&mut self, amount: f64) {
        self.cumulative_cost += amount;
    }

    /// Current cumulative session cost in dollars.
    pub fn cumulative_cost(&self) -> f64 {
        self.cumulative_cost
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
        /// Gaps longer than 10 minutes are session suspensions
        /// (overnight, lunch, context switch), not stalls.
        const MAX_STALL_GAP: f64 = 600.0;

        let now = self.now();
        let dt = now - self.last_call_t;
        let stall_dt = if dt > MAX_STALL_GAP { MAX_STALL_GAP } else { dt };
        if stall_dt > 0.0 {
            // 0.1 progress-units per second of elapsed time.
            // This is the base stall rate — productive actions
            // (Edit, Write) give negative impulses to counteract.
            self.bank.record(Channel::Progress.index(), stall_dt * 0.1, now);
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
    ///
    /// This is the input-agnostic entry point used by callers that
    /// only know which tool is about to run (e.g., the Coalition MCP
    /// server's `feedback_evaluate` tool). Without an input value,
    /// warn-zone Modify verdicts can only carry guidance, not a
    /// patched input. For the hook-side path that DOES have the
    /// tool's arguments, prefer `evaluate_with_input`.
    pub fn evaluate(&self, tool_name: &str) -> GovernorVerdict {
        self.evaluate_with_input(tool_name, None)
    }

    /// Evaluate the current pressure state with the tool's input
    /// arguments in hand. Lets the monitor produce concrete patched
    /// `updated_input` values for warn-zone Modify verdicts (e.g.,
    /// capping a Read's `limit` or a Bash's `timeout`).
    pub fn evaluate_with_input(
        &self,
        tool_name: &str,
        tool_input: Option<&serde_json::Value>,
    ) -> GovernorVerdict {
        let now = self.now();
        let values = self.bank.current_values(now);

        let snapshot = monitor::check_stability(&self.model, &values, &self.rates);
        let decision = monitor::evaluate_decision(
            &snapshot,
            &self.config.thresholds,
            tool_name,
            tool_input,
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

    #[test]
    fn snapshot_roundtrip_preserves_pressure() {
        // Save state from one governor, resume into another, and
        // verify the pressure carries over. This is the core
        // invariant the feedback-hook binary depends on.
        let mut gov1 = Governor::from_defaults();
        gov1.record_impulse(Channel::Error, 3.5);
        gov1.record_impulse(Channel::Context, 1500.0);
        gov1.record_file_access("/some/path.rs");
        gov1.record_cost(0.05);
        let values_before = gov1.current_values();
        let cost_before = gov1.cumulative_cost();

        // Snapshot → JSON → snapshot, the trip the binary actually makes.
        let snap = gov1.snapshot();
        let json = serde_json::to_string(&snap).expect("serialize");
        let snap2: PersistedState = serde_json::from_str(&json).expect("deserialize");

        let gov2 = Governor::resume_from_defaults(snap2);
        let values_after = gov2.current_values();
        let cost_after = gov2.cumulative_cost();

        // Cost is exact — it's just an f64.
        assert_eq!(cost_before, cost_after, "cost must roundtrip exactly");

        // Bank values may differ by a tiny amount due to whatever
        // wall-clock time elapsed between the two `current_values`
        // calls — both are decayed at read time. Tolerance covers
        // a millisecond's worth of decay at the slowest channel.
        for i in 0..values_before.len() {
            let diff = (values_before[i] - values_after[i]).abs();
            assert!(
                diff < 0.01,
                "channel {} drifted too much across snapshot/resume: {} -> {}",
                i,
                values_before[i],
                values_after[i]
            );
        }

        // File access count survived.
        let n = gov2
            .file_access_counts
            .get("/some/path.rs")
            .copied()
            .unwrap_or(0);
        assert_eq!(n, 1, "file access count must roundtrip");
    }
}
