//! Real-time governor: maintains live pressure state and produces verdicts.
//!
//! Hands layer: owns the mutable IntegralBank and the clock. This is the
//! stateful runtime object that the hook integration calls into.

use std::collections::{HashMap, VecDeque};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use volterra_stability::config::ImpulseRates;
use volterra_stability::integral::IntegralBank;

use crate::channels::{Channel, FeedbackConfig};
use crate::decision::GovernorVerdict;
use crate::monitor;
use crate::pressure_model::AgentCouplingModel;
use crate::session::SessionDigest;

/// Gaps longer than this (seconds) are treated as session suspensions
/// — overnight, lunch, context switch — rather than stalls or decay.
///
/// Used in two places that both care about "how much elapsed time do we
/// let flow into the pressure model across a persistence boundary":
///
/// 1. `accumulate_progress_stall` caps the progress impulse so an
///    overnight break does not dump 8 hours of stall into the channel.
/// 2. `resume_from_defaults` caps the wall-clock gap injected into the
///    bank's decay clock so the same overnight break does not collapse
///    every pressure channel to zero on the first impulse after resume.
///
/// A single shared constant keeps the two code paths consistent.
pub const MAX_STALL_GAP: f64 = 600.0;

/// Current wall-clock time in seconds since the UNIX epoch.
///
/// Used as the cross-process time reference at the persistence boundary
/// — `Instant` is per-process and cannot observe the gap between two
/// hook invocations, so the governor's clock would otherwise miss all
/// of the real-world time elapsed between tool calls.
fn wall_now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

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
    /// Wall-clock timestamps (in the governor's time domain) of
    /// recent PreCompact events. The hook binary trims entries outside
    /// its watchdog window before persisting. `Governor` itself does
    /// not touch this field — it flows through `snapshot` / `resume`
    /// untouched so the binary can observe compaction thrashing
    /// across per-process boundaries.
    ///
    /// `#[serde(default)]` so states written by older feedback builds
    /// (which lack this field) still deserialize cleanly. Same for
    /// `last_guidance_t`.
    #[serde(default)]
    pub recent_compactions: VecDeque<f64>,
    /// Timestamp of the last time the compaction watchdog injected
    /// guidance into a verdict. Used by the binary to enforce a
    /// cooldown so the agent does not get repeatedly nagged.
    #[serde(default)]
    pub last_guidance_t: f64,
    /// Markdown summary of Idiom deviations that the most recent
    /// PostToolUse observed in the file just written. The next
    /// PreToolUse drains this field (sets it back to `None`) and
    /// surfaces it as `additionalContext` so the agent sees the
    /// specific naming-convention deviations alongside the Error-
    /// channel pressure they generated, instead of a faceless
    /// numeric increment. Like the watchdog fields, `Governor`
    /// itself does not touch this — it round-trips through
    /// `snapshot` / `resume` untouched, owned by the hook binary.
    #[serde(default)]
    pub pending_idiom_context: Option<String>,
    /// Markdown summary of Basis architectural violations the
    /// most recent PostToolUse observed in (or near) the file
    /// just written. Mirror of `pending_idiom_context` for the
    /// other side of the dogfooding loop: naming drift on one
    /// channel, placement / values / completeness / purity drift
    /// on the same one. The next PreToolUse drains this field
    /// the same way it drains the Idiom one. Owned by the hook
    /// binary, not by `Governor`.
    #[serde(default)]
    pub pending_basis_context: Option<String>,
    /// Wall-clock time (seconds since UNIX_EPOCH) when this state
    /// was written to disk. Used on resume to advance the governor's
    /// time domain by the real-world gap between the previous hook
    /// invocation and the current one — without it, each hook
    /// process has a fresh `Instant::now()` epoch and the bank's
    /// decay clock effectively stops between tool calls.
    ///
    /// A value of `0.0` means "unknown" — either a freshly constructed
    /// state that was never persisted, or a state written by an older
    /// feedback build that predates this field. `resume_from_defaults`
    /// treats `0.0` as "no wall-clock gap to inject", so upgrading
    /// from an older build does not collapse the bank on first resume.
    #[serde(default)]
    pub saved_wall_secs: f64,
    /// Pressure values from the last `evaluate` call. Persisted so
    /// the next hook invocation can compute trajectory (rising vs
    /// falling) and include it in deny/warn messages.
    #[serde(default)]
    pub prev_values: Vec<f64>,
    /// Incrementally accumulated session digest. Updated on every
    /// `evaluate` call and persisted so SessionEnd can write it to
    /// `<session>-digest.json` without replaying the verdict history.
    #[serde(default = "SessionDigest::parse_empty_session")]
    pub digest: SessionDigest,
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
    /// Pressure values from the previous `evaluate` call. Populated
    /// after each evaluation and persisted across hook invocations so
    /// the deny/warn messages can report trajectory (rising/falling).
    prev_values: Vec<f64>,
    /// Incrementally accumulated session digest.
    digest: SessionDigest,
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
            prev_values: Vec::new(),
            digest: SessionDigest::parse_empty_session(),
        }
    }

    /// Resume a governor from a persisted state. The new process's
    /// clock is *caught up* to include the real-world time elapsed
    /// while no hook was running, and the progress channel is
    /// credited with the corresponding stall impulse inline.
    ///
    /// ## Why catch up here instead of in `accumulate_progress_stall`
    ///
    /// Each feedback-hook invocation is a separate OS process, and
    /// `Instant::now()` is per-process. Without an explicit
    /// cross-process time source, the governor would observe
    /// essentially zero elapsed time between tool calls, the
    /// `IntegralBank` would stop decaying, and pressure would
    /// accumulate monotonically until the session was bricked.
    ///
    /// `snapshot()` writes the current wall-clock time into
    /// `saved_wall_secs`. On resume we use that stamp as the
    /// cross-process reference, compute the real-world gap, cap it
    /// at `MAX_STALL_GAP` so an overnight break cannot collapse the
    /// bank, and advance the governor's time domain atomically:
    ///
    /// 1. `time_offset` is bumped by `wall_gap` so the bank's lazy
    ///    decay will observe a correct `dt` on its next read.
    /// 2. `last_call_t` is bumped by `wall_gap` so that successive
    ///    resumes without intervening `accumulate_progress_stall`
    ///    calls still persist the accumulated elapsed time — without
    ///    this, every resume would re-zero `time_offset` because
    ///    only `last_call_t` survives a persistence round trip.
    /// 3. The progress channel receives `wall_gap * 0.1` of stall
    ///    impulse directly, exactly what `accumulate_progress_stall`
    ///    would have recorded if it could see the gap.
    ///
    /// Subsequent `accumulate_progress_stall` calls will see
    /// `dt ≈ 0` on the first call after resume, which is correct:
    /// the catch-up already happened here atomically. This keeps
    /// the stall contribution a pure function of elapsed wall time,
    /// applied exactly once regardless of how the time was split
    /// across resume boundaries.
    ///
    /// The model, config, and rates are reconstructed from defaults
    /// rather than persisted; if you need a non-default tuning,
    /// resume manually via the field-by-field constructor.
    pub fn resume_from_defaults(state: PersistedState) -> Self {
        let config = FeedbackConfig::from_defaults();
        let rates = ImpulseRates::from_slice(&[50.0, 0.38, 0.01, 0.1, 0.12]);
        let model = AgentCouplingModel::new(config.clone());

        // Real-world seconds that elapsed while no hook process was
        // running. `saved_wall_secs == 0.0` means "unknown" — either a
        // default-constructed state or a state written by a feedback
        // build that predates this field — so we inject zero gap
        // rather than jumping the clock forward by 55 years of
        // UNIX_EPOCH drift.
        let wall_gap = if state.saved_wall_secs > 0.0 {
            (wall_now_secs() - state.saved_wall_secs)
                .max(0.0)
                .min(MAX_STALL_GAP)
        } else {
            0.0
        };

        let caught_up_t = state.last_call_t.max(0.0) + wall_gap;

        // The bank's decay clock is caught up by advancing
        // `time_offset` and `last_call_t` to `caught_up_t`. Each
        // channel's lazy decay will observe the correct `dt` on its
        // next `record` or `current_value` call.
        //
        // We intentionally do NOT inject a progress-stall impulse
        // here. A gap between sessions is an absence, not a stall —
        // the agent wasn't failing to make progress, it wasn't
        // running. Injecting `wall_gap * 0.1` of stall at resume
        // would push the progress channel above critical after any
        // gap longer than ~9 minutes (critical=55, 550s * 0.1 = 55),
        // bricking the governor on session start. The progress
        // channel accumulates stall only from within-session elapsed
        // time, via `accumulate_progress_stall`.
        let bank = state.bank;

        Self {
            model,
            bank,
            config,
            rates,
            epoch: Instant::now(),
            time_offset: caught_up_t,
            file_access_counts: state.file_access_counts,
            last_call_t: caught_up_t,
            cumulative_cost: state.cumulative_cost,
            prev_values: state.prev_values,
            digest: state.digest,
        }
    }

    /// Take a serializable snapshot of the parts of `self` that need
    /// to survive across process boundaries. Cheap (clones a small
    /// `IntegralBank` and a `HashMap<String, usize>`).
    ///
    /// The compaction-watchdog fields (`recent_compactions`,
    /// `last_guidance_t`) are returned as defaults here — `Governor`
    /// does not own them. The hook binary must merge in the values it
    /// loaded from disk (or just-updated during a PreCompact) before
    /// writing the state back. If the binary forgets, the watchdog
    /// silently resets each call; the tests cover the merge path.
    pub fn snapshot(&self) -> PersistedState {
        PersistedState {
            bank: self.bank.clone(),
            file_access_counts: self.file_access_counts.clone(),
            last_call_t: self.last_call_t,
            cumulative_cost: self.cumulative_cost,
            recent_compactions: VecDeque::new(),
            last_guidance_t: 0.0,
            pending_idiom_context: None,
            pending_basis_context: None,
            // Wall-clock stamp at save time. The counterpart to the
            // logic in `resume_from_defaults` — together they make
            // the bank's decay clock track real elapsed time across
            // the per-process boundary of each hook invocation.
            saved_wall_secs: wall_now_secs(),
            prev_values: self.prev_values.clone(),
            digest: self.digest.clone(),
        }
    }

    /// Return a reference to the current session digest.
    pub fn digest(&self) -> &SessionDigest {
        &self.digest
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
        self.digest.parse_record_impulse_session(channel.index(), impulse);
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
    ///
    /// Uses the module-level `MAX_STALL_GAP` to cap the stall
    /// contribution from session suspensions. `resume_from_defaults`
    /// uses the same cap on the wall-clock gap it injects, so a 9 AM
    /// resume after an overnight stop contributes MAX_STALL_GAP
    /// seconds of stall here (once), not 12 hours.
    pub fn accumulate_progress_stall(&mut self) -> f64 {
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
    pub fn evaluate(&mut self, tool_name: &str) -> GovernorVerdict {
        self.evaluate_with_input(tool_name, None)
    }

    /// Evaluate the current pressure state with the tool's input
    /// arguments in hand. Lets the monitor produce concrete patched
    /// `updated_input` values for warn-zone Modify verdicts (e.g.,
    /// capping a Read's `limit` or a Bash's `timeout`).
    pub fn evaluate_with_input(
        &mut self,
        tool_name: &str,
        tool_input: Option<&serde_json::Value>,
    ) -> GovernorVerdict {
        let now = self.now();
        let values = self.bank.current_values(now);

        let mut snapshot = monitor::check_stability(&self.model, &values, &self.rates);
        snapshot.prev_values = if self.prev_values.is_empty() {
            None
        } else {
            Some(self.prev_values.clone())
        };
        let decision = monitor::evaluate_decision(
            &snapshot,
            &self.config.thresholds,
            tool_name,
            tool_input,
        );

        // Update the session digest with this evaluation's snapshot.
        let criticals: Vec<f64> = self.config.system.channels.iter().map(|c| c.critical).collect();
        let dt = now - self.last_call_t;
        self.digest.parse_update_session(
            &values,
            &criticals,
            &self.config.thresholds.warn,
            &self.config.thresholds.deny,
            dt.max(0.0),
        );

        // Save current values for trajectory computation on next call.
        self.prev_values = values;

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
        let mut gov = Governor::from_defaults();
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

    // ── Wall-clock decay across resume (the bug that motivated
    //    `saved_wall_secs`). Each of these tests simulates a
    //    persistence gap by mutating `saved_wall_secs` on the
    //    snapshot before resuming — we cannot literally sleep for
    //    30 s in unit tests, and `SystemTime::now()` is not
    //    mockable from the outside. Mutating the saved stamp
    //    backward is equivalent to "30 s of wall-clock time passed
    //    while no hook was running," which is the scenario the
    //    fix targets.

    /// Core invariant: pressure that was recorded before a
    /// persistence gap must visibly decay after resume. Before
    /// the fix this test would have failed — the bank's decay
    /// clock was frozen at `last_call_t` across the resume
    /// boundary, so `current_values` on the resumed governor
    /// would return essentially the same pressure as `gov1` saw
    /// at save time.
    #[test]
    fn decay_advances_across_resume_with_wall_gap() {
        let mut gov1 = Governor::from_defaults();
        // Large impulse on the error channel (lambda ≈ 0.0231,
        // half-life ≈ 30 s). Big enough that a 60 s gap produces
        // an obviously detectable drop.
        gov1.record_impulse(Channel::Error, 10.0);
        let before = gov1.current_values()[Channel::Error.index()];

        // Simulate a 60-second persistence gap by backdating the
        // saved wall-clock stamp.
        let mut snap = gov1.snapshot();
        snap.saved_wall_secs -= 60.0;

        let gov2 = Governor::resume_from_defaults(snap);
        let after = gov2.current_values()[Channel::Error.index()];

        // At error-channel lambda, 60 s of decay should bring a
        // 10.0 impulse down by at least 20 % (1 - exp(-0.0231*60)
        // ≈ 0.75, i.e. the surviving value is ~2.5, a 7.5-unit
        // drop). We use a very loose bound — any real decay at
        // all proves the wall-clock gap is flowing through.
        assert!(
            after < before - 1.0,
            "error channel should decay across the persistence gap: \
             before={before}, after={after}"
        );
        assert!(
            after > 0.0,
            "error channel should not collapse to zero from a 60 s gap: \
             after={after}"
        );
    }

    /// The gap is capped at `MAX_STALL_GAP` so an overnight
    /// break cannot lobotomize the bank on first resume. Before
    /// the fix this was already broken for a different reason
    /// (the gap was always zero); the cap protects the fixed
    /// behavior.
    #[test]
    fn resume_bounds_wall_gap_to_max_stall() {
        let mut gov1 = Governor::from_defaults();
        gov1.record_impulse(Channel::Error, 10.0);

        // Backdate by 24 hours. Without the cap this would decay
        // the error channel essentially to zero
        // (exp(-0.0231*86400) = exp(-1996) ≈ 0).
        let mut snap = gov1.snapshot();
        snap.saved_wall_secs -= 86_400.0;

        let gov2 = Governor::resume_from_defaults(snap);
        let after = gov2.current_values()[Channel::Error.index()];

        // With the 600 s cap, the surviving value is
        // 10.0 * exp(-0.0231*600) ≈ 10.0 * 9.6e-7 ≈ 9.6e-6.
        // That IS almost zero — so the cap is saving us from
        // "astronomically smaller" rather than from full
        // annihilation. The real assertion is: the resume did
        // not panic, did not underflow to NaN, and produced a
        // finite non-negative value. Combined with the matching
        // cap in `accumulate_progress_stall`, this keeps the two
        // code paths consistent under the same assumption.
        assert!(after.is_finite(), "error channel must stay finite: {after}");
        assert!(after >= 0.0, "error channel must stay non-negative: {after}");
        // And the cap must be tight enough that a 60 s gap and a
        // 24 h gap produce meaningfully different surviving values.
        // (The 24 h gap is capped at 600 s; the 60 s gap is not.)
        let mut snap_short = gov1.snapshot();
        snap_short.saved_wall_secs -= 60.0;
        let short = Governor::resume_from_defaults(snap_short)
            .current_values()[Channel::Error.index()];
        assert!(
            short > after,
            "60 s gap should leave more pressure than a capped 24 h gap: \
             short={short}, capped={after}"
        );
    }

    /// Legacy states written by older feedback builds have
    /// `saved_wall_secs = 0.0` (the serde default). Resume must
    /// NOT interpret that as "55 years of UNIX_EPOCH drift"
    /// and jump the clock forward by MAX_STALL_GAP on every
    /// upgrade. Instead, it treats `0.0` as "unknown" and
    /// injects zero gap — the governor picks up where it left
    /// off, just as it did before the fix.
    #[test]
    fn legacy_state_without_wall_secs_resumes_cleanly() {
        let mut gov1 = Governor::from_defaults();
        gov1.record_impulse(Channel::Error, 5.0);
        let before = gov1.current_values()[Channel::Error.index()];

        let mut snap = gov1.snapshot();
        snap.saved_wall_secs = 0.0; // legacy / missing field

        let gov2 = Governor::resume_from_defaults(snap);
        let after = gov2.current_values()[Channel::Error.index()];

        // Zero gap → values should match almost exactly (the only
        // drift is a few microseconds of within-process time).
        let diff = (before - after).abs();
        assert!(
            diff < 0.01,
            "legacy state must not inject a spurious gap: \
             before={before}, after={after}"
        );
    }

    /// Resume must NOT inject progress stall from the wall-clock
    /// gap. A gap between sessions is an absence, not a stall —
    /// the agent wasn't failing to make progress, it wasn't
    /// running. Stall accumulates only from within-session elapsed
    /// time via `accumulate_progress_stall`.
    ///
    /// This test verifies that a long gap does NOT pre-load the
    /// progress channel. Without this invariant, any session gap
    /// longer than ~9 minutes would start the governor above
    /// critical on progress, bricking the agent on the first call.
    #[test]
    fn resume_does_not_inject_progress_stall() {
        let gov1 = Governor::from_defaults();

        // Simulate a 120-second persistence gap.
        let mut snap = gov1.snapshot();
        snap.saved_wall_secs -= 120.0;

        let gov2 = Governor::resume_from_defaults(snap);
        let progress = gov2.current_values()[Channel::Progress.index()];

        // Progress should be near zero — the gap is an absence,
        // not a stall. Any value above 1.0 means the resume is
        // leaking stall impulse into the progress channel.
        assert!(
            progress < 1.0,
            "resume must not inject progress stall from the \
             inter-session gap: progress={progress}"
        );
    }

    /// Denied-call flow: the hook records the impulse, saves
    /// state, and returns deny. On the next call the wall clock
    /// must still be advancing, so pressure can actually decay
    /// even while the agent is retrying against a deny wall.
    /// Before the fix, this was the self-reinforcing block —
    /// each denied call added pressure and zero decay happened
    /// between attempts.
    #[test]
    fn repeated_resume_still_decays_between_calls() {
        let mut gov = Governor::from_defaults();
        gov.record_impulse(Channel::Error, 10.0);

        // Call 1 → snap → backdate → resume (simulates one tool
        // call worth of wall time passing).
        let mut snap1 = gov.snapshot();
        snap1.saved_wall_secs -= 30.0;
        let gov = Governor::resume_from_defaults(snap1);
        let after_one = gov.current_values()[Channel::Error.index()];

        // Call 2 → snap → backdate another 30 s → resume.
        let mut snap2 = gov.snapshot();
        snap2.saved_wall_secs -= 30.0;
        let gov = Governor::resume_from_defaults(snap2);
        let after_two = gov.current_values()[Channel::Error.index()];

        // Pressure must monotonically decay across both resume
        // cycles, proving decay is cumulative and not reset by
        // the per-process `Instant::now()` epoch on each hop.
        assert!(
            after_one < 10.0,
            "first resume gap should decay pressure: {after_one}"
        );
        assert!(
            after_two < after_one,
            "second resume gap should decay further: one={after_one}, two={after_two}"
        );
    }
}
