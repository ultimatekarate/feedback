//! `feedback-hook` — Claude Code hook binary.
//!
//! Handles two Claude Code hook events from a single binary, chosen
//! via the `hook_event_name` field on the stdin payload:
//!
//! - **PreToolUse** (the hot path): reads a `HookInput` JSON
//!   document from stdin, loads the per-session governor state from
//!   disk, runs `handle_pre_tool_use`, persists the new state, and
//!   writes a `HookOutput` JSON document to stdout.
//! - **PreCompact**: deletes the on-disk session state file and
//!   returns an empty `{}`. When Claude Code compacts a session,
//!   the LLM context is replaced with a summary — the old context
//!   no longer exists, so the pressure signal that was measuring
//!   it must also be reset. Without this handler, a post-compact
//!   session inherits its pre-compact pressure and stays blocked
//!   on the very load that compacting was meant to relieve.
//!
//! # State persistence
//!
//! Each PreToolUse invocation is a separate process. The Governor
//! lives only for the duration of one call, so without persistence
//! pressure would never accumulate across calls and the loop would
//! be useless. State is keyed by `session_id` and stored at
//! `${FEEDBACK_STATE_DIR:-$HOME/.claude/feedback-state}/<id>.json`.
//!
//! # Failure mode: default-allow
//!
//! This binary runs on EVERY tool call in EVERY Claude Code session
//! once wired into `settings.json`. A bug here that returns non-zero
//! or invalid JSON would brick all tool calls. The contract is:
//! ANY error path — bad input, missing dirs, IO failure,
//! deserialization failure — produces an empty `{}` on stdout and
//! exits 0. Empty JSON is interpreted by Claude Code as no opinion,
//! which means "allow." Failure of the governor must never block
//! the agent; it must only fail to govern.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::{SystemTime, UNIX_EPOCH};

use feedback::channels::Channel;
use feedback::decision::{Decision, GovernorVerdict};
use feedback::governor::{Governor, PersistedState};
use feedback::hook_protocol::HookInput;
use feedback::hooks::{evaluate_pre_tool_use, verdict_to_output};
use feedback::post_tool;

/// Filename used for the append-only block-event log inside the
/// feedback state directory. Public so the Coalition adapter can
/// reference the same constant when reading.
const BLOCKS_LOG_FILENAME: &str = "blocks.jsonl";

/// Filename suffix for the per-session impulse log. The full path is
/// `<state>/<sanitized_session_id>.impulses.jsonl`. One JSONL line
/// per non-zero governance impulse — written from the PostToolUse
/// branch and read at SessionEnd to compute a session-level pressure
/// retrospective. Per-session (rather than a single global file) so
/// that concurrent hook processes for different sessions never
/// contend for append, the file shares the same lifecycle as the
/// per-session state file, and SessionEnd can read its own log
/// without filtering by session_id.
const IMPULSE_LOG_SUFFIX: &str = ".impulses.jsonl";

/// Resolve the absolute path of the per-session impulse log file
/// inside `dir` for `session_id`. Sanitizes the session id the same
/// way the per-session state file does, so the two are paired by name.
fn impulse_log_path(dir: &Path, session_id: &str) -> PathBuf {
    dir.join(format!("{}{}", sanitize_session_id(session_id), IMPULSE_LOG_SUFFIX))
}

/// Append a single per-source pressure entry to the per-session
/// impulse log. Mirrors `append_block_event`: best-effort, every IO
/// failure is silently swallowed because the hot path must never
/// block on telemetry. The log format is JSONL (one JSON object per
/// line) so SessionEnd can fold it into a retrospective without a
/// structured reader.
///
/// One line per non-zero governance source on a single PostToolUse.
/// When both Idiom and Basis fire on the same Write the caller emits
/// two lines, not one combined line — that keeps the digest math
/// simple ("X basis impulses, Y idiom impulses") without losing the
/// per-axis breakdown that only makes sense for Basis.
///
/// `impulse` here is the raw `deviations_to_impulse(count)` value
/// for *this source alone*, not the (potentially capped) combined
/// impulse the governor actually saw. Per-source attribution is more
/// useful for the retrospective; the cap is a property of the live
/// governor, not of the historical record.
fn append_impulse_log_entry(
    dir: &Path,
    session_id: &str,
    t: f64,
    source: &str,
    count: u64,
    impulse: f64,
    axis_breakdown: &std::collections::BTreeMap<String, u64>,
) {
    let entry = serde_json::json!({
        "t": t,
        "channel": "Error",
        "source": source,
        "count": count,
        "impulse": impulse,
        "axis_breakdown": axis_breakdown,
    });
    let Ok(line) = serde_json::to_string(&entry) else {
        return;
    };
    let log_path = impulse_log_path(dir, session_id);
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    else {
        return;
    };
    let _ = writeln!(f, "{line}");
}

// ── Compaction watchdog tuning ────────────────────────────────────
//
// Claude Code's own auto-compactor owns the happy path for context
// pressure: when the window fills it clears old tool outputs and
// eventually summarises the transcript. The watchdog only fires in
// the pathological case where a single tool output or a runaway
// loop refills the window faster than compaction can drain it —
// Claude Code stops auto-compacting after a few back-to-back
// attempts and surfaces an error instead. The watchdog catches
// that state BEFORE the error by looking at PreCompact frequency
// and injecting a plan-for-chunked-work nudge through
// `additionalContext` on the next PreToolUse.

/// Sliding window (in seconds) used to decide whether PreCompact
/// events are "recent." Two compactions within this window trip
/// the watchdog.
const COMPACTION_WINDOW_SECS: f64 = 60.0;

/// Number of PreCompact events inside `COMPACTION_WINDOW_SECS`
/// required before the watchdog considers the session to be
/// thrashing the auto-compactor.
const COMPACTION_THRESHOLD: usize = 2;

/// Minimum seconds between successive watchdog guidance injections.
/// Prevents the agent from being nagged on every single call once
/// the threshold is crossed — one nudge per five-minute window is
/// plenty to get the "break the work into chunks" point across.
const GUIDANCE_COOLDOWN_SECS: f64 = 300.0;

/// The guidance text injected as `additionalContext` when the
/// watchdog fires. Phrased as advice, not a command — the agent
/// must remain free to decide the compactions were expected (e.g.,
/// after a deliberately huge Read) and carry on.
const WATCHDOG_GUIDANCE: &str =
    "Claude Code has auto-compacted twice within the last minute. That usually means \
     a single file or tool output is refilling the context window as fast as it drains. \
     Before the next Read/Bash/Grep, consider pausing to write a plan that breaks the \
     remaining work into smaller chunks — each chunk small enough that its output will \
     not immediately require another compaction.";

/// Seconds since the UNIX epoch, used as the time source for the
/// watchdog's sliding window. We deliberately use wall-clock time
/// rather than the governor's internal clock because compaction is
/// a cross-process event — the governor resets itself to a fresh
/// clock domain on each PreCompact, so its `now()` would not be
/// comparable across compactions.
fn wall_clock_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Trim `recent_compactions` to entries newer than `now - window`.
/// Mutates in place — VecDeque makes this cheap from the front.
fn prune_old_compactions(
    recent: &mut std::collections::VecDeque<f64>,
    now: f64,
    window: f64,
) {
    let cutoff = now - window;
    while let Some(&front) = recent.front() {
        if front < cutoff {
            recent.pop_front();
        } else {
            break;
        }
    }
}

/// Combine up to three governance signals into a single
/// `additionalContext` payload, ordered by severity:
///
/// 1. **Watchdog** (compaction thrashing) — highest, because it
///    means the session is about to fall over;
/// 2. **Basis** (architectural violations) — middle, because B001–
///    B004 are hard errors the agent must fix to ship;
/// 3. **Idiom** (naming convention drift) — lowest, because it is
///    style guidance and the project may legitimately want to pin
///    the new pattern.
///
/// Sections are separated by blank lines so the agent sees them as
/// independent notes. Returns `None` only when all three are
/// absent — in that case the hook output omits `additionalContext`
/// entirely rather than emitting an empty string.
///
/// Pure function — extracted so the merge order is unit-testable
/// without spinning up a full hook event.
fn combine_extra_context(
    watchdog: Option<String>,
    idiom: Option<String>,
    basis: Option<String>,
) -> Option<String> {
    let mut sections: Vec<String> = Vec::new();
    if let Some(w) = watchdog {
        sections.push(w);
    }
    if let Some(b) = basis {
        sections.push(b);
    }
    if let Some(i) = idiom {
        sections.push(i);
    }
    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

// ── Session-boundary helpers ───────────────────────────────────────
//
// SessionStart and SessionEnd share a single on-disk artifact: a
// markdown digest of the *previous* session's blocked tool calls.
// SessionEnd writes the digest, SessionStart reads it (then deletes
// it so the same digest never echoes twice). The file lives in the
// same per-session state dir, so it shares the FEEDBACK_STATE_DIR
// override that the test suite already sets.

/// Filename for the cross-session block digest. One file overwrites
/// the previous on every SessionEnd — the digest is "what just
/// happened in the prior session," not a per-session log.
const LAST_SESSION_DIGEST_FILENAME: &str = "last-session-digest.md";

/// Soft cap on bytes of `basis.yaml` to inline into the SessionStart
/// `additionalContext`. Anything larger gets truncated with a
/// "(truncated — see <path> for the rest)" tail. The cap exists to
/// avoid blowing the agent's context window on a giant spec.
const BASIS_INLINE_CAP_BYTES: usize = 16 * 1024;

/// Maximum number of recent block events to enumerate in the
/// SessionEnd digest before collapsing the rest into a count summary.
const DIGEST_MAX_BLOCKS: usize = 10;

/// Build the SessionStart `additionalContext` payload by gathering
/// (1) the prior-session digest if any, (2) the inlined `basis.yaml`
/// if it exists in `cwd`, and (3) the markdown output of
/// `idiom-cli context <cwd>` if Idiom is on PATH. Each section is
/// labelled with a markdown header so the agent can find them.
/// Returns `None` if every source came up empty — in that case the
/// caller should emit `{}` rather than a hookSpecificOutput envelope
/// with an empty additionalContext.
///
/// Side effect: when the prior-session digest is consumed it is
/// deleted from disk so subsequent SessionStart calls do not echo
/// the same content. This is the read-and-clear half of the
/// SessionEnd → SessionStart handoff.
fn build_session_start_context(cwd: &Path, dir: &Path) -> Option<String> {
    let mut sections: Vec<String> = Vec::new();

    // 1. Prior session digest (read-and-delete).
    let digest_path = dir.join(LAST_SESSION_DIGEST_FILENAME);
    if let Ok(digest) = std::fs::read_to_string(&digest_path) {
        let trimmed = digest.trim();
        if !trimmed.is_empty() {
            sections.push(format!(
                "## Prior session — blocked / modified tool calls\n\n{trimmed}"
            ));
        }
        // Best-effort delete: if it fails, the same digest will appear
        // on the next SessionStart, which is harmless given it's just
        // advisory text.
        let _ = std::fs::remove_file(&digest_path);
    }

    // 2. basis.yaml inline (if any). Only the canonical
    //    `<cwd>/basis.yaml` location is checked; nested specs are out
    //    of scope for the SessionStart hook.
    let basis_path = cwd.join("basis.yaml");
    if let Ok(spec) = std::fs::read_to_string(&basis_path) {
        let body = if spec.len() > BASIS_INLINE_CAP_BYTES {
            // Truncate at a line boundary so the YAML stays parseable
            // by eye. Find the last newline before the cap, fall back
            // to a hard cut if none exists in range.
            let cut = spec[..BASIS_INLINE_CAP_BYTES]
                .rfind('\n')
                .unwrap_or(BASIS_INLINE_CAP_BYTES);
            format!(
                "{}\n# (truncated — full spec at {})\n",
                &spec[..cut],
                basis_path.display()
            )
        } else {
            spec
        };
        sections.push(format!(
            "## Architectural spec — `basis.yaml`\n\n```yaml\n{body}\n```"
        ));
    }

    // 3. Idiom context — `idiom-cli context <cwd>` produces markdown
    //    designed for direct injection. Honors FEEDBACK_IDIOM_BIN for
    //    test isolation, same as PostToolUse.
    let bin = std::env::var(post_tool::IDIOM_BIN_ENV).unwrap_or_else(|_| "idiom-cli".to_string());
    if let Ok(output) = std::process::Command::new(&bin)
        .args(["context"])
        .arg(cwd)
        .output()
    {
        // idiom-cli's `context` subcommand exits 0 on success and
        // prints markdown to stdout. Anything else → skip silently.
        if output.status.success() {
            if let Ok(text) = std::str::from_utf8(&output.stdout) {
                let trimmed = text.trim();
                if !trimmed.is_empty() {
                    sections.push(format!(
                        "## Local naming conventions — `idiom context`\n\n{trimmed}"
                    ));
                }
            }
        }
    }

    if sections.is_empty() {
        None
    } else {
        Some(sections.join("\n\n"))
    }
}

/// Render the SessionStart hook response as a JSON string. Wraps
/// `additional_context` in the `hookSpecificOutput` envelope Claude
/// Code expects for the SessionStart event. Built by hand rather
/// than reusing `HookOutput` because SessionStart has no
/// `permissionDecision` field — that DTO is PreToolUse-shaped.
fn session_start_response(additional_context: &str) -> String {
    serde_json::json!({
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": additional_context,
        }
    })
    .to_string()
}

/// Build the SessionEnd retrospective digest. Composes two sources:
///
/// 1. **Block events** from `blocks.jsonl`, filtered to `session_id`
///    (the file is global so cross-session entries must be excluded).
/// 2. **Pressure summary** from the per-session impulse log at
///    `impulse_log` (already keyed by session id, so no filtering).
///    Folds basis + idiom impulses into per-source totals plus a
///    Basis axis breakdown so the next session sees the *shape* of
///    the prior session's drift, not just the count.
///
/// Returns `None` only when BOTH sources are empty — there is
/// nothing to surface and the digest file should not be written.
/// When only one source has data, the digest contains just that
/// section. The two sections are blank-line separated so they read
/// as independent notes.
fn build_session_end_digest(
    blocks_log: &Path,
    impulse_log: &Path,
    session_id: &str,
) -> Option<String> {
    // ── Source 1: block events (filtered by session id) ──────────
    let mut events: Vec<serde_json::Value> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(blocks_log) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if v.get("session_id").and_then(|s| s.as_str()) == Some(session_id) {
                events.push(v);
            }
        }
    }

    // ── Source 2: per-session impulse log (no filtering needed) ──
    let mut impulse_entries: Vec<serde_json::Value> = Vec::new();
    if let Ok(text) = std::fs::read_to_string(impulse_log) {
        for line in text.lines() {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(line) {
                impulse_entries.push(v);
            }
        }
    }

    if events.is_empty() && impulse_entries.is_empty() {
        return None;
    }

    let mut sections: Vec<String> = Vec::new();

    // ── Section 1: blocked / modified tool calls ─────────────────
    if !events.is_empty() {
        let total = events.len();
        let recent: Vec<&serde_json::Value> =
            events.iter().rev().take(DIGEST_MAX_BLOCKS).collect();
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!(
            "Session {session_id} ended with {total} blocked / modified tool call(s). Recent:"
        ));
        for ev in recent.iter().rev() {
            let tool = ev.get("tool_name").and_then(|s| s.as_str()).unwrap_or("?");
            let decision = ev.get("decision").and_then(|s| s.as_str()).unwrap_or("?");
            let reason = ev.get("reason").and_then(|s| s.as_str()).unwrap_or("");
            lines.push(format!("- **{decision}** `{tool}` — {reason}"));
        }
        if total > DIGEST_MAX_BLOCKS {
            let elided = total - DIGEST_MAX_BLOCKS;
            lines.push(format!("- …and {elided} earlier event(s) elided"));
        }
        sections.push(lines.join("\n"));
    }

    // ── Section 2: pressure summary from the impulse log ─────────
    if !impulse_entries.is_empty() {
        sections.push(render_pressure_summary(&impulse_entries));
    }

    Some(sections.join("\n\n"))
}

/// Fold a list of impulse-log entries into a markdown summary block.
/// Pure function — split out from `build_session_end_digest` so the
/// aggregation logic can be unit-tested without planted files. Each
/// entry is a JSON value loaded from the per-session JSONL log.
///
/// The summary reports total impulses, total deviations, peak per-
/// call impulse, per-source breakdown, and a Basis axis breakdown
/// when any axis_breakdown fields were populated. An empty axis
/// breakdown (the Idiom case) is silently skipped — there are no
/// Idiom "axes" to report.
fn render_pressure_summary(entries: &[serde_json::Value]) -> String {
    let mut total_impulses = 0usize;
    let mut total_count = 0u64;
    let mut peak_impulse: f64 = 0.0;
    // (impulse_count, deviation_total, peak_impulse) per source.
    let mut by_source: std::collections::BTreeMap<String, (usize, u64, f64)> =
        std::collections::BTreeMap::new();
    let mut axis_totals: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();

    for entry in entries {
        total_impulses += 1;
        let count = entry.get("count").and_then(|c| c.as_u64()).unwrap_or(0);
        let imp = entry.get("impulse").and_then(|i| i.as_f64()).unwrap_or(0.0);
        let src = entry
            .get("source")
            .and_then(|s| s.as_str())
            .unwrap_or("?")
            .to_string();
        total_count += count;
        if imp > peak_impulse {
            peak_impulse = imp;
        }
        let bucket = by_source.entry(src).or_insert((0, 0, 0.0));
        bucket.0 += 1;
        bucket.1 += count;
        if imp > bucket.2 {
            bucket.2 = imp;
        }
        if let Some(axis_obj) = entry.get("axis_breakdown").and_then(|a| a.as_object()) {
            for (axis, n) in axis_obj {
                let nval = n.as_u64().unwrap_or(0);
                if nval > 0 {
                    *axis_totals.entry(axis.clone()).or_insert(0) += nval;
                }
            }
        }
    }

    let mut lines: Vec<String> = Vec::new();
    lines.push(format!(
        "Pressure summary: {total_impulses} impulse(s) over {total_count} deviation(s), peak {peak_impulse:.2}."
    ));
    for (src, (n, c, peak)) in &by_source {
        lines.push(format!(
            "- **{src}**: {n} impulse(s), {c} deviation(s), peak {peak:.2}"
        ));
    }
    if !axis_totals.is_empty() {
        let parts: Vec<String> = axis_totals
            .iter()
            .map(|(axis, n)| format!("{n} {axis}"))
            .collect();
        lines.push(format!("- Basis axes: {}", parts.join(", ")));
    }
    lines.join("\n")
}

/// Resolve the directory where per-session state files live.
/// Honors `FEEDBACK_STATE_DIR` for tests and tooling, otherwise
/// defaults to `<home>/.claude/feedback-state`.
fn state_dir() -> PathBuf {
    if let Ok(custom) = std::env::var("FEEDBACK_STATE_DIR") {
        return PathBuf::from(custom);
    }
    let home = std::env::var("USERPROFILE")
        .or_else(|_| std::env::var("HOME"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".claude").join("feedback-state")
}

/// Make a session id safe to use as a filename. Maps anything that
/// isn't `[A-Za-z0-9_-]` to `_`. Prevents path traversal via a
/// hostile or malformed session id field on the hook input.
fn sanitize_session_id(id: &str) -> String {
    if id.is_empty() {
        return "default".to_string();
    }
    id.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Tool names whose calls must never be blocked by the governor
/// or recorded as pressure events. These are Coalition's read-only
/// introspection tools — locking the agent out of them would mean
/// a saturated session could not ask what is wrong.
const INTROSPECTION_TOOLS: &[&str] = &[
    "mcp__coalition__feedback_status",
    "mcp__coalition__feedback_evaluate",
];

/// Append a single block-event line to `blocks.jsonl` in `dir`.
/// Best-effort: any IO failure is silently swallowed because the
/// hot path must never be blocked on telemetry. The log format is
/// JSONL (one JSON object per line) so it can be tailed and tail-
/// parsed without a structured reader.
fn append_block_event(
    dir: &Path,
    session_id: &str,
    verdict: &GovernorVerdict,
    cumulative_cost: f64,
) {
    // Only Modify and Deny are recorded — Allow events would
    // dominate the log and bury the signal we actually care about.
    let (kind, reason) = match &verdict.decision {
        Decision::Allow => return,
        Decision::Deny { reason } => ("Deny", reason.as_str()),
        Decision::Modify { reason, .. } => ("Modify", reason.as_str()),
    };

    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0);

    let event = serde_json::json!({
        "ts": ts,
        "session_id": session_id,
        "tool_name": verdict.tool_name,
        "decision": kind,
        "reason": reason,
        "channel_values": verdict.snapshot.values,
        "cumulative_cost": cumulative_cost,
    });

    let Ok(line) = serde_json::to_string(&event) else {
        return;
    };

    // Open in append mode. For small (<4KB) writes on Windows and
    // POSIX, append mode gives effectively-atomic line-level
    // appends — concurrent hook processes will not interleave
    // bytes inside a single record.
    let log_path = dir.join(BLOCKS_LOG_FILENAME);
    let Ok(mut f) = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
    else {
        return;
    };
    let _ = writeln!(f, "{line}");
}

/// Bundle of pending governance summaries that PostToolUse stashes
/// for the next PreToolUse to drain. Both fields are independent
/// drift signals from different sub-tools (Idiom = naming
/// conventions, Basis = the four architectural axes). Bundling them
/// into one struct keeps the persistence helper's signature stable
/// as more sources are added — without it, every new context source
/// would mean a new positional argument and ambiguous call sites.
#[derive(Debug, Default, Clone)]
struct PendingContext {
    idiom: Option<String>,
    basis: Option<String>,
}

impl PendingContext {
    /// True iff at least one source has something to surface. Used
    /// by `record_impulse_to_disk` to short-circuit the IO when the
    /// impulse is also zero — there is nothing to write.
    fn is_empty(&self) -> bool {
        self.idiom.is_none() && self.basis.is_none()
    }
}

/// Load per-session state, apply a single impulse to one channel,
/// optionally stash one or more pending governance summaries, and
/// persist atomically. Used by the PostToolUse branch — the helper
/// exists so the persistence guarantees can be unit-tested without
/// needing real `idiom-cli` or `basis-cli` binaries on PATH.
///
/// `pending` carries the markdown summaries the *next* PreToolUse
/// will drain and surface as `additionalContext`. An empty
/// `PendingContext` is fine; the impulse path still runs.
/// Passing populated fields for a session that already has pending
/// content will overwrite the old summaries — that is intentional,
/// the most recent set of deviations is the most relevant.
///
/// Default-allow contract: every IO error is silently swallowed.
/// Returns `true` if the state file ended up updated, `false`
/// otherwise. A non-positive `impulse` combined with an empty
/// `pending` is a no-op (returns `false`); a non-positive impulse
/// with at least one pending summary still writes the fields
/// through.
fn record_impulse_to_disk(
    dir: &Path,
    session_id: &str,
    channel: Channel,
    impulse: f64,
    pending: PendingContext,
) -> bool {
    // Skip the IO entirely when there is nothing to record at all.
    if impulse <= 0.0 && pending.is_empty() {
        return false;
    }
    if std::fs::create_dir_all(dir).is_err() {
        return false;
    }
    let session_file = format!("{}.json", sanitize_session_id(session_id));
    let state_path = dir.join(&session_file);
    // Load the prior state once. We need BOTH the governor-owned
    // fields (resumed into a Governor) AND the watchdog fields
    // (stashed separately so we can re-attach them to the snapshot
    // before persisting — `Governor::snapshot` returns defaults for
    // all of them). Without this merge, any PostToolUse impulse
    // would wipe `recent_compactions` and disarm the compaction
    // watchdog. The pending fields ARE overwritten on purpose:
    // PostToolUse always has the freshest information about the
    // file just edited.
    let prior = if state_path.exists() {
        std::fs::read(&state_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PersistedState>(&bytes).ok())
    } else {
        None
    };
    let (mut governor, recent_compactions, last_guidance_t) = match prior {
        Some(state) => {
            let recent = state.recent_compactions.clone();
            let last = state.last_guidance_t;
            (Governor::resume_from_defaults(state), recent, last)
        }
        None => (
            Governor::from_defaults(),
            std::collections::VecDeque::new(),
            0.0,
        ),
    };
    if impulse > 0.0 {
        governor.record_impulse(channel, impulse);
    }
    let mut snap = governor.snapshot();
    snap.recent_compactions = recent_compactions;
    snap.last_guidance_t = last_guidance_t;
    snap.pending_idiom_context = pending.idiom;
    snap.pending_basis_context = pending.basis;
    let Ok(snap_bytes) = serde_json::to_vec_pretty(&snap) else {
        return false;
    };
    let tmp_path = state_path.with_extension("json.tmp");
    if std::fs::write(&tmp_path, &snap_bytes).is_err() {
        return false;
    }
    std::fs::rename(&tmp_path, &state_path).is_ok()
}

/// Pure dispatch logic, separated from stdin/state-dir resolution
/// so it can be unit-tested with constructed inputs and a tempdir.
/// `dir` is the per-session state directory the binary would
/// otherwise resolve via `state_dir()`.
fn handle_event(input_str: &str, dir: &Path) -> Result<String, Box<dyn std::error::Error>> {
    // ── Event dispatch ─────────────────────────────────────────
    // Peek at `hook_event_name` via a raw Value lookup so we can
    // route PreCompact payloads (which lack `tool_name`/`tool_input`
    // and would fail strict HookInput deserialization) before
    // committing to the full PreToolUse schema.
    let raw: serde_json::Value = serde_json::from_str(input_str)?;
    let event = raw
        .get("hook_event_name")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if event == "PreCompact" {
        // Compaction replaces the LLM context with a summary. The
        // old context pressure profile no longer applies, so we
        // reset the governor to defaults — but we preserve the
        // compaction-watchdog fields (`recent_compactions` and
        // `last_guidance_t`) so the PreToolUse path can spot a
        // session that is thrashing the auto-compactor.
        //
        // We record THIS compaction's timestamp, prune anything
        // older than the sliding window, and write the fresh
        // default state back with the updated watchdog metadata.
        // Any IO error here silently falls through to the empty
        // no-opinion response — default-allow contract.
        let session_id = raw
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let session_file = format!("{}.json", sanitize_session_id(session_id));
        let state_path = dir.join(&session_file);

        // Load whatever watchdog history the prior state had, if any.
        let (mut recent_compactions, last_guidance_t) = match std::fs::read(&state_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PersistedState>(&bytes).ok())
        {
            Some(prior) => (prior.recent_compactions, prior.last_guidance_t),
            None => (std::collections::VecDeque::new(), 0.0),
        };

        let now = wall_clock_secs();
        recent_compactions.push_back(now);
        prune_old_compactions(&mut recent_compactions, now, COMPACTION_WINDOW_SECS);

        // Build a fresh-governor snapshot and merge the watchdog
        // fields in before persisting. Ignore IO errors — the hook
        // must not surface them to Claude Code.
        let gov = Governor::from_defaults();
        let mut snap = gov.snapshot();
        snap.recent_compactions = recent_compactions;
        snap.last_guidance_t = last_guidance_t;

        if std::fs::create_dir_all(dir).is_ok() {
            if let Ok(bytes) = serde_json::to_vec_pretty(&snap) {
                let tmp_path = state_path.with_extension("json.tmp");
                if std::fs::write(&tmp_path, &bytes).is_ok() {
                    let _ = std::fs::rename(&tmp_path, &state_path);
                }
            }
        }
        return Ok("{}".to_string());
    }

    // ── SessionStart: inject Basis spec + Idiom context + prior digest ──
    // Fired exactly once per session (startup, resume, clear, or
    // compact). Builds an `additionalContext` payload from three
    // sources, all optional. If every source is empty (no basis.yaml,
    // no idiom-cli, no prior digest), return `{}` rather than an
    // empty envelope. SessionStart payloads carry `cwd`; we need it
    // because the hook process's own cwd is wherever Claude Code was
    // launched from, not the project being edited.
    if event == "SessionStart" {
        let cwd_str = raw.get("cwd").and_then(|v| v.as_str()).unwrap_or("");
        if cwd_str.is_empty() {
            return Ok("{}".to_string());
        }
        let cwd = Path::new(cwd_str);
        return Ok(match build_session_start_context(cwd, dir) {
            Some(ctx) => session_start_response(&ctx),
            None => "{}".to_string(),
        });
    }

    // ── SessionEnd: write block-event digest for the next SessionStart ──
    // SessionEnd is write-only — the agent has already terminated and
    // any response body is discarded by Claude Code. We use it as the
    // place to compress the session's `blocks.jsonl` entries into a
    // markdown digest that the *next* SessionStart will pick up and
    // surface to the new agent. Default-allow contract still applies:
    // every IO error path returns `{}` and never panics.
    if event == "SessionEnd" {
        let session_id = raw
            .get("session_id")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if session_id.is_empty() {
            return Ok("{}".to_string());
        }
        let blocks_log = dir.join(BLOCKS_LOG_FILENAME);
        let imp_log = impulse_log_path(dir, session_id);
        if let Some(digest) = build_session_end_digest(&blocks_log, &imp_log, session_id) {
            // Write the digest to a fixed path. Best-effort: failure
            // is invisible to the agent.
            if std::fs::create_dir_all(dir).is_ok() {
                let digest_path = dir.join(LAST_SESSION_DIGEST_FILENAME);
                let tmp_path = digest_path.with_extension("md.tmp");
                if std::fs::write(&tmp_path, &digest).is_ok() {
                    let _ = std::fs::rename(&tmp_path, &digest_path);
                }
            }
        }
        // Read-and-clear semantics for the per-session impulse log:
        // SessionEnd is the last consumer of this file, so deleting
        // it now keeps the state directory tidy and prevents a
        // hypothetical second SessionEnd on the same id from
        // double-counting. Best-effort — a leftover file is
        // harmless, just slightly stale.
        let _ = std::fs::remove_file(&imp_log);
        return Ok("{}".to_string());
    }

    // Parse the strict HookInput. Anything that fails here falls
    // back to default-allow via `main`.
    let input: HookInput = serde_json::from_str(input_str)?;

    // ── Introspection exemption ────────────────────────────────
    // Bypass state load AND `apply_impulses`: the agent querying
    // governor state must not itself add pressure, and must work
    // even if the state file is corrupt. See `INTROSPECTION_TOOLS`.
    if INTROSPECTION_TOOLS.contains(&input.tool_name.as_str()) {
        return Ok("{}".to_string());
    }

    // ── PostToolUse: idiom-cli check → Error channel impulse ──
    // Pure side-channel. Claude Code ignores the body of a
    // PostToolUse reply, so we mutate persisted state for the
    // *next* PreToolUse to feel and return an empty no-opinion
    // object. Only Write/Edit/MultiEdit are inspected — Read,
    // Bash, etc. did not change any source file, so there is
    // nothing for Idiom to check.
    if event == "PostToolUse" {
        let tool = input.tool_name.as_str();
        if tool != "Write" && tool != "Edit" && tool != "MultiEdit" {
            return Ok("{}".to_string());
        }
        let Some(file_path) = input
            .tool_input
            .get("file_path")
            .and_then(|v| v.as_str())
        else {
            return Ok("{}".to_string());
        };

        // Two parallel governance checks against the file just
        // written. Either, both, or neither may produce a report:
        //
        // - Idiom checks naming conventions, returns None on a non-
        //   source file or missing binary;
        // - Basis checks the four architectural axes (placement,
        //   values, completeness, purity), returns None on a non-
        //   source file, missing binary, or file outside any
        //   governed crate.
        //
        // We sum the two counts for the Error-channel impulse
        // (per-deviation cost is currently uniform across both
        // sources) and stash both summaries in the PendingContext
        // bundle so the next PreToolUse can drain and surface them
        // independently with their original framing.
        let idiom_report = post_tool::idiom_check_report(file_path);
        let basis_report = post_tool::basis_check_report(file_path);
        if idiom_report.is_none() && basis_report.is_none() {
            return Ok("{}".to_string());
        }
        let idiom_count = idiom_report.as_ref().map(|r| r.count).unwrap_or(0);
        let basis_count = basis_report.as_ref().map(|r| r.count).unwrap_or(0);
        let impulse = post_tool::deviations_to_impulse(idiom_count + basis_count);

        // Append per-source impulse log entries before consuming the
        // reports into the pending bundle. SessionEnd will fold these
        // into a pressure retrospective the next SessionStart picks
        // up. Per-session log file means no session_id filtering at
        // digest time. Per-source (rather than one combined line)
        // means the digest can attribute pressure cleanly without
        // re-running the governance subprocesses.
        if idiom_count > 0 || basis_count > 0 {
            let now = wall_clock_secs();
            if idiom_count > 0 {
                append_impulse_log_entry(
                    dir,
                    &input.session_id,
                    now,
                    "idiom",
                    idiom_count,
                    post_tool::deviations_to_impulse(idiom_count),
                    &std::collections::BTreeMap::new(),
                );
            }
            if let Some(report) = basis_report.as_ref() {
                if report.count > 0 {
                    append_impulse_log_entry(
                        dir,
                        &input.session_id,
                        now,
                        "basis",
                        report.count,
                        post_tool::deviations_to_impulse(report.count),
                        &report.axis_breakdown,
                    );
                }
            }
        }

        // Stash the summaries even when impulse is zero — a single
        // sub-impulse deviation should still surface its explanation
        // to the agent. Drain happens on the next PreToolUse.
        let pending = PendingContext {
            idiom: idiom_report
                .filter(|r| !r.summary.is_empty())
                .map(|r| r.summary),
            basis: basis_report
                .filter(|r| !r.summary.is_empty())
                .map(|r| r.summary),
        };
        record_impulse_to_disk(dir, &input.session_id, Channel::Error, impulse, pending);
        return Ok("{}".to_string());
    }

    let session_file = format!("{}.json", sanitize_session_id(&input.session_id));
    std::fs::create_dir_all(dir)?;
    let state_path = dir.join(&session_file);

    // Resume from disk if a snapshot exists. A corrupt or unreadable
    // file is treated as a fresh session — better to lose history
    // than to brick the agent on every call until someone manually
    // clears the file.
    //
    // The watchdog metadata lives on `PersistedState` but is
    // intentionally NOT owned by the Governor (it is cross-process,
    // wall-clock indexed, and has nothing to do with the pressure
    // model). We stash it before `resume_from_defaults` swallows
    // the state, then attach it back to the snapshot before writing.
    let (
        mut governor,
        mut recent_compactions,
        mut last_guidance_t,
        pending_idiom_context,
        pending_basis_context,
    ) = if state_path.exists() {
        match std::fs::read(&state_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PersistedState>(&bytes).ok())
        {
            Some(state) => {
                let recent = state.recent_compactions.clone();
                let last = state.last_guidance_t;
                let pending_i = state.pending_idiom_context.clone();
                let pending_b = state.pending_basis_context.clone();
                (
                    Governor::resume_from_defaults(state),
                    recent,
                    last,
                    pending_i,
                    pending_b,
                )
            }
            None => (
                Governor::from_defaults(),
                std::collections::VecDeque::new(),
                0.0,
                None,
                None,
            ),
        }
    } else {
        (
            Governor::from_defaults(),
            std::collections::VecDeque::new(),
            0.0,
            None,
            None,
        )
    };

    // Two-step: get the verdict so we can log non-Allow events to
    // the blocks log before serializing it back to Claude Code.
    let verdict = evaluate_pre_tool_use(input_str, &mut governor)?;
    append_block_event(dir, &input.session_id, &verdict, governor.cumulative_cost());

    // Watchdog: after the verdict is known, check whether the
    // sliding-window compaction count crosses the threshold. If so,
    // and we are outside the cooldown, attach the chunking guidance
    // to the outgoing response and record the guidance timestamp so
    // we don't nag again immediately. Pruning happens before the
    // count check so stale entries don't keep the watchdog armed
    // forever.
    let now_wall = wall_clock_secs();
    prune_old_compactions(&mut recent_compactions, now_wall, COMPACTION_WINDOW_SECS);
    let should_guide = recent_compactions.len() >= COMPACTION_THRESHOLD
        && (now_wall - last_guidance_t) >= GUIDANCE_COOLDOWN_SECS;
    let watchdog_guidance = if should_guide {
        last_guidance_t = now_wall;
        Some(WATCHDOG_GUIDANCE.to_string())
    } else {
        None
    };

    // Combine the watchdog guidance with any pending governance
    // deviation summaries. Three independent signals — compaction
    // thrashing, naming drift, and architectural drift — any
    // subset of which may be present. The drain semantics live
    // here: the pending fields are consumed by being read out of
    // the load-side variables, and the persistence step below
    // writes `None` back to both `pending_*_context` fields on
    // the snapshot so the next PreToolUse starts clean.
    let extra_context = combine_extra_context(
        watchdog_guidance,
        pending_idiom_context,
        pending_basis_context,
    );

    let output_json = verdict_to_output(&verdict, extra_context)?;

    // Persist atomically: write to a sibling .tmp then rename. The
    // rename is atomic on the same filesystem, so a concurrent
    // reader either sees the old file or the new file, never a
    // partial write. Failures here are non-fatal — the hook output
    // is still returned to Claude Code. Watchdog metadata is merged
    // into the snapshot before writing. `pending_idiom_context` is
    // explicitly NOT re-attached — draining it is the whole point.
    let mut snap = governor.snapshot();
    snap.recent_compactions = recent_compactions;
    snap.last_guidance_t = last_guidance_t;
    if let Ok(snap_bytes) = serde_json::to_vec_pretty(&snap) {
        let tmp_path = state_path.with_extension("json.tmp");
        if std::fs::write(&tmp_path, &snap_bytes).is_ok() {
            let _ = std::fs::rename(&tmp_path, &state_path);
        }
    }

    Ok(output_json)
}

/// The fallible inner body. Reads stdin and resolves the state
/// directory from the environment, then delegates to `handle_event`.
/// Anything that returns an error here is caught in `main` and
/// converted to an empty allow.
fn run() -> Result<String, Box<dyn std::error::Error>> {
    let mut input_str = String::new();
    std::io::stdin().read_to_string(&mut input_str)?;
    let dir = state_dir();
    handle_event(&input_str, &dir)
}

fn main() -> ExitCode {
    match run() {
        Ok(json) => {
            println!("{json}");
            ExitCode::SUCCESS
        }
        Err(e) => {
            // Default-allow on every error. Empty JSON object means
            // "no opinion" → allow. Stderr surfaces under
            // `claude --debug` if the user investigates.
            eprintln!("feedback-hook: {e}");
            println!("{{}}");
            ExitCode::SUCCESS
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use feedback::channels::Channel;

    /// Plant a per-session state file in `dir` reflecting the
    /// pressure profile produced by `gov.snapshot()`. Mirrors what
    /// the binary does after handling a real PreToolUse call.
    fn plant_state_file(dir: &Path, session_id: &str, gov: &Governor) {
        let snap = gov.snapshot();
        let bytes = serde_json::to_vec_pretty(&snap).expect("serialize");
        let session_file = format!("{}.json", sanitize_session_id(session_id));
        std::fs::write(dir.join(session_file), bytes).expect("write");
    }

    fn pre_tool_use_payload(session_id: &str, tool: &str, file_path: &str) -> String {
        serde_json::json!({
            "hook_event_name": "PreToolUse",
            "session_id": session_id,
            "tool_name": tool,
            "tool_input": {"file_path": file_path},
        })
        .to_string()
    }

    fn pre_compact_payload(session_id: &str) -> String {
        serde_json::json!({
            "hook_event_name": "PreCompact",
            "session_id": session_id,
        })
        .to_string()
    }

    fn post_tool_use_payload(session_id: &str, tool: &str, file_path: &str) -> String {
        serde_json::json!({
            "hook_event_name": "PostToolUse",
            "session_id": session_id,
            "tool_name": tool,
            "tool_input": {"file_path": file_path},
            "tool_response": {},
        })
        .to_string()
    }

    #[test]
    fn precompact_resets_pressure_but_preserves_watchdog_metadata() {
        // PreCompact clears the per-channel pressure (since the LLM
        // context is now a summary and the old values no longer
        // describe it) but MUST preserve compaction-watchdog fields
        // so the PreToolUse path can spot a thrashing session that
        // compacts twice in 60s. The new contract: write a fresh
        // default state with a new compaction timestamp appended.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut gov = Governor::from_defaults();
        gov.record_impulse(Channel::Context, 5000.0);
        gov.record_impulse(Channel::Error, 4.0);
        plant_state_file(dir.path(), "compact-session", &gov);
        let state_path = dir.path().join("compact-session.json");

        let out = handle_event(&pre_compact_payload("compact-session"), dir.path())
            .expect("handle_event should not error");
        assert_eq!(out, "{}", "PreCompact must return empty no-opinion object");

        assert!(
            state_path.exists(),
            "PreCompact should rewrite, not delete, the state file"
        );
        let bytes = std::fs::read(&state_path).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();

        // Pressure was reset: Error should be zero (or near-zero
        // decay from an empty bank).
        let err = state
            .bank
            .current_value(Channel::Error.index(), state.last_call_t);
        assert!(
            err.abs() < 0.01,
            "Error channel must be reset on PreCompact, got {err}"
        );

        // Watchdog metadata: exactly one compaction recorded (this one).
        assert_eq!(state.recent_compactions.len(), 1);
        // last_guidance_t untouched (watchdog did not fire here).
        assert_eq!(state.last_guidance_t, 0.0);
    }

    #[test]
    fn precompact_accumulates_compaction_timestamps_inside_window() {
        // Two PreCompacts back-to-back should leave TWO entries in
        // recent_compactions — the watchdog will pick that up on the
        // next PreToolUse call.
        let dir = tempfile::tempdir().expect("tempdir");
        let _ = handle_event(&pre_compact_payload("dual"), dir.path()).unwrap();
        let _ = handle_event(&pre_compact_payload("dual"), dir.path()).unwrap();

        let bytes = std::fs::read(dir.path().join("dual.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            state.recent_compactions.len(),
            2,
            "two compactions in the window should both be recorded"
        );
    }

    #[test]
    fn precompact_also_cleans_half_written_tmp_sibling() {
        // Atomic-rename writers leave a `.json.tmp` sibling around
        // mid-flight. The new PreCompact path writes a FRESH tmp and
        // rename-installs it, so any stale `.json.tmp` left over from
        // a previous crash is clobbered by that rename.
        let dir = tempfile::tempdir().expect("tempdir");
        let session = "with-tmp";
        std::fs::write(dir.path().join("with-tmp.json"), b"{}").unwrap();
        std::fs::write(dir.path().join("with-tmp.json.tmp"), b"junk").unwrap();

        handle_event(&pre_compact_payload(session), dir.path()).expect("ok");
        // The main state file now contains the freshly-reset state
        // with watchdog metadata — not the stub `{}` we planted.
        let bytes = std::fs::read(dir.path().join("with-tmp.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(state.recent_compactions.len(), 1);
        // The atomic rename consumed the tmp sibling.
        assert!(!dir.path().join("with-tmp.json.tmp").exists());
    }

    #[test]
    fn introspection_tools_do_not_touch_state_or_blocks_log() {
        // Regression: a saturated session must always be able to
        // ask `feedback_status`/`feedback_evaluate` what is wrong.
        // The hook short-circuits before state load, so even a
        // corrupt state file cannot brick introspection.
        let dir = tempfile::tempdir().expect("tempdir");
        // Plant a deliberately corrupt state file.
        std::fs::write(
            dir.path().join("introspect.json"),
            b"{ this is not valid json",
        )
        .unwrap();

        for tool in [
            "mcp__coalition__feedback_status",
            "mcp__coalition__feedback_evaluate",
        ] {
            let payload = pre_tool_use_payload("introspect", tool, "");
            let out = handle_event(&payload, dir.path())
                .unwrap_or_else(|_| panic!("introspection on {tool} must not error"));
            assert_eq!(
                out, "{}",
                "introspection tool {tool} must return empty no-opinion object"
            );
        }
        // The corrupt state file is untouched (not rewritten).
        let bytes = std::fs::read(dir.path().join("introspect.json")).unwrap();
        assert_eq!(bytes, b"{ this is not valid json");
        // Blocks log is not created — introspection is not a block event.
        assert!(!dir.path().join(BLOCKS_LOG_FILENAME).exists());
    }

    #[test]
    fn allow_decisions_do_not_write_to_blocks_log() {
        // A baseline PreToolUse on a fresh session produces an
        // Allow verdict. The blocks log records non-Allow events
        // only — Allows would dominate the log and bury signal.
        let dir = tempfile::tempdir().expect("tempdir");
        let payload = pre_tool_use_payload("fresh", "Read", "/foo.rs");
        let _ = handle_event(&payload, dir.path()).expect("ok");
        assert!(
            !dir.path().join(BLOCKS_LOG_FILENAME).exists(),
            "Allow events must not create blocks.jsonl"
        );
        // The state file IS created — that's the per-call persistence.
        assert!(dir.path().join("fresh.json").exists());
    }

    #[test]
    fn high_pressure_state_produces_block_event_in_log() {
        // Plant pressure that is solidly past critical so the verdict
        // is non-Allow no matter the calibration. Verify the block
        // event lands in blocks.jsonl with the right shape.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut gov = Governor::from_defaults();
        // Error critical is 6.0; 100.0 is unambiguously past.
        gov.record_impulse(Channel::Error, 100.0);
        plant_state_file(dir.path(), "saturated", &gov);

        let payload = pre_tool_use_payload("saturated", "Read", "/foo.rs");
        let _ = handle_event(&payload, dir.path()).expect("ok");

        let log_path = dir.path().join(BLOCKS_LOG_FILENAME);
        assert!(log_path.exists(), "blocks.jsonl should be created on first block");
        let log = std::fs::read_to_string(&log_path).unwrap();
        let line = log.lines().next().expect("at least one block event");
        let event: serde_json::Value = serde_json::from_str(line).expect("valid JSON line");
        assert_eq!(event["session_id"], "saturated");
        assert_eq!(event["tool_name"], "Read");
        // Decision must be one of the two non-Allow tags.
        let decision = event["decision"].as_str().unwrap();
        assert!(
            decision == "Modify" || decision == "Deny",
            "decision should be Modify or Deny, got {decision}"
        );
        // Channel values array is the full snapshot (5 channels).
        let values = event["channel_values"].as_array().unwrap();
        assert_eq!(values.len(), 5);
        // ts is a positive number (seconds since UNIX epoch).
        assert!(event["ts"].as_f64().unwrap() > 0.0);
        // reason is non-empty.
        assert!(!event["reason"].as_str().unwrap().is_empty());
    }

    // ── PostToolUse path ───────────────────────────────────────

    /// Read the per-session state file back and return the IntegralBank's
    /// current `Channel::Error` value at the timestamp of the last
    /// recorded impulse. None if the file doesn't exist. Reading at
    /// `last_call_t` means the test sees the impulse value with
    /// effectively zero decay.
    fn read_error_pressure(dir: &Path, session_id: &str) -> Option<f64> {
        let session_file = format!("{}.json", sanitize_session_id(session_id));
        let bytes = std::fs::read(dir.join(session_file)).ok()?;
        let state: PersistedState = serde_json::from_slice(&bytes).ok()?;
        Some(
            state
                .bank
                .current_value(Channel::Error.index(), state.last_call_t),
        )
    }

    #[test]
    fn record_impulse_to_disk_creates_state_file_with_error_pressure() {
        // Helper-level test: bypasses idiom-cli entirely so we can
        // prove the persist path lands the impulse in the right
        // channel slot. The PostToolUse dispatch reuses this helper.
        let dir = tempfile::tempdir().expect("tempdir");
        let updated = record_impulse_to_disk(
            dir.path(),
            "rec1",
            Channel::Error,
            1.5,
            PendingContext::default(),
        );
        assert!(updated, "non-zero impulse should land");
        let pressure =
            read_error_pressure(dir.path(), "rec1").expect("state file should exist after record");
        assert!(
            pressure > 0.0,
            "Error channel should be non-zero after a 1.5 impulse, got {pressure}"
        );
    }

    #[test]
    fn record_impulse_to_disk_zero_impulse_is_noop() {
        // Skipping the disk write for zero-impulse calls keeps the
        // hook hot path silent on the common "Idiom found nothing"
        // case — there is no point churning the state file just to
        // record a no-op. Same applies when no pending context is
        // supplied; if either is present the write proceeds.
        let dir = tempfile::tempdir().expect("tempdir");
        let updated = record_impulse_to_disk(
            dir.path(),
            "rec0",
            Channel::Error,
            0.0,
            PendingContext::default(),
        );
        assert!(!updated, "zero impulse with no pending context must not touch disk");
        assert!(
            !dir.path().join("rec0.json").exists(),
            "no state file should be created for a zero-impulse call"
        );
    }

    #[test]
    fn record_impulse_to_disk_accumulates_across_calls() {
        // Two writes in a row should compound: the second call must
        // load the state the first call wrote, not start from zero.
        let dir = tempfile::tempdir().expect("tempdir");
        record_impulse_to_disk(
            dir.path(),
            "compound",
            Channel::Error,
            1.0,
            PendingContext::default(),
        );
        let after_one = read_error_pressure(dir.path(), "compound").unwrap();
        record_impulse_to_disk(
            dir.path(),
            "compound",
            Channel::Error,
            1.0,
            PendingContext::default(),
        );
        let after_two = read_error_pressure(dir.path(), "compound").unwrap();
        assert!(
            after_two > after_one,
            "second impulse must build on the first: {after_one} → {after_two}"
        );
    }

    #[test]
    fn record_impulse_to_disk_writes_pending_idiom_context() {
        // PostToolUse stashes the deviation summary; the next
        // PreToolUse must be able to read it back from disk. Verify
        // the round-trip without going through `idiom-cli`.
        let dir = tempfile::tempdir().expect("tempdir");
        let summary = "Idiom flagged 2 deviations:\n- [I001] line 4: x".to_string();
        let updated = record_impulse_to_disk(
            dir.path(),
            "pend",
            Channel::Error,
            1.0,
            PendingContext {
                idiom: Some(summary.clone()),
                basis: None,
            },
        );
        assert!(updated);
        let bytes = std::fs::read(dir.path().join("pend.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(state.pending_idiom_context.as_deref(), Some(summary.as_str()));
    }

    #[test]
    fn record_impulse_to_disk_writes_pending_even_with_zero_impulse() {
        // A clean file with zero count produces no impulse, but a
        // single deviation below the impulse cap (count=1 → 0.5
        // impulse) should always carry the summary through. Verify
        // the contract by passing impulse=0 + Some(summary): the
        // helper must still write, otherwise the agent would see no
        // explanation when the deviation was bracketed inside the
        // PostToolUse extension filter.
        let dir = tempfile::tempdir().expect("tempdir");
        let updated = record_impulse_to_disk(
            dir.path(),
            "pend0",
            Channel::Error,
            0.0,
            PendingContext {
                idiom: Some("explain".to_string()),
                basis: None,
            },
        );
        assert!(updated, "zero impulse + Some(context) must still persist");
        let state_path = dir.path().join("pend0.json");
        assert!(state_path.exists());
    }

    #[test]
    fn pre_tool_use_drains_pending_idiom_context_into_additional_context() {
        // The end-to-end contract: when a prior PostToolUse stashed a
        // pending Idiom summary, the next PreToolUse must surface it
        // via `additionalContext` AND clear the field on disk so a
        // *second* PreToolUse no longer sees it. Without the drain,
        // the same summary would echo on every call until the next
        // file edit.
        let dir = tempfile::tempdir().expect("tempdir");
        record_impulse_to_disk(
            dir.path(),
            "drainme",
            Channel::Error,
            0.5,
            PendingContext {
                idiom: Some("Idiom flagged 1 deviation:\n- [I001] line 7: bad name".to_string()),
                basis: None,
            },
        );

        // First PreToolUse — should see the context.
        let payload = pre_tool_use_payload("drainme", "Read", "/tmp/x.rs");
        let out = handle_event(&payload, dir.path()).expect("ok");
        assert!(
            out.contains("Idiom flagged 1 deviation"),
            "first PreToolUse must surface the pending summary, got: {out}"
        );
        assert!(
            out.contains("additionalContext"),
            "summary must be wrapped in additionalContext, got: {out}"
        );

        // State on disk must now have a cleared pending field.
        let bytes = std::fs::read(dir.path().join("drainme.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert!(
            state.pending_idiom_context.is_none(),
            "drain must clear the pending field on disk"
        );

        // Second PreToolUse — must NOT echo the same summary.
        let out2 = handle_event(&payload, dir.path()).expect("ok");
        assert!(
            !out2.contains("Idiom flagged 1 deviation"),
            "second PreToolUse must not re-echo the drained summary, got: {out2}"
        );
    }

    #[test]
    fn record_impulse_to_disk_writes_pending_basis_context() {
        // Mirror of the idiom-side test for the basis pending field.
        // PostToolUse stashes the basis report; the next PreToolUse
        // must be able to read it back from disk.
        let dir = tempfile::tempdir().expect("tempdir");
        let summary = "Basis flagged 1 architectural violation:\n- [B001 placement] line 4: x"
            .to_string();
        let updated = record_impulse_to_disk(
            dir.path(),
            "pendb",
            Channel::Error,
            1.0,
            PendingContext {
                idiom: None,
                basis: Some(summary.clone()),
            },
        );
        assert!(updated);
        let bytes = std::fs::read(dir.path().join("pendb.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(state.pending_basis_context.as_deref(), Some(summary.as_str()));
        assert!(
            state.pending_idiom_context.is_none(),
            "basis-only stash must not populate the idiom field"
        );
    }

    #[test]
    fn record_impulse_to_disk_can_stash_both_pending_fields_in_one_call() {
        // Realistic PostToolUse case: a single edited file flunks
        // both naming and architectural rules. The two summaries must
        // round-trip independently — no overwrite, no merge.
        let dir = tempfile::tempdir().expect("tempdir");
        let updated = record_impulse_to_disk(
            dir.path(),
            "both",
            Channel::Error,
            1.0,
            PendingContext {
                idiom: Some("IDIOM-SUMMARY".to_string()),
                basis: Some("BASIS-SUMMARY".to_string()),
            },
        );
        assert!(updated);
        let bytes = std::fs::read(dir.path().join("both.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(state.pending_idiom_context.as_deref(), Some("IDIOM-SUMMARY"));
        assert_eq!(state.pending_basis_context.as_deref(), Some("BASIS-SUMMARY"));
    }

    #[test]
    fn pre_tool_use_drains_pending_basis_context_into_additional_context() {
        // End-to-end contract for the basis side: stashed summary
        // surfaces on the first PreToolUse and is gone on the second.
        // Same drain semantics as the idiom test; failing this would
        // mean the agent gets nagged with the same architectural
        // violation on every subsequent tool call.
        let dir = tempfile::tempdir().expect("tempdir");
        let summary =
            "Basis flagged 1 architectural violation:\n- [B001 placement] line 7: cross-layer";
        record_impulse_to_disk(
            dir.path(),
            "drainb",
            Channel::Error,
            0.5,
            PendingContext {
                idiom: None,
                basis: Some(summary.to_string()),
            },
        );

        // First PreToolUse — should see the basis summary.
        let payload = pre_tool_use_payload("drainb", "Read", "/tmp/x.rs");
        let out = handle_event(&payload, dir.path()).expect("ok");
        assert!(
            out.contains("Basis flagged 1 architectural violation"),
            "first PreToolUse must surface the basis summary, got: {out}"
        );
        assert!(
            out.contains("additionalContext"),
            "summary must be wrapped in additionalContext, got: {out}"
        );

        // State on disk must now have a cleared basis pending field.
        let bytes = std::fs::read(dir.path().join("drainb.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert!(
            state.pending_basis_context.is_none(),
            "drain must clear the basis pending field on disk"
        );

        // Second PreToolUse — must NOT echo the same summary.
        let out2 = handle_event(&payload, dir.path()).expect("ok");
        assert!(
            !out2.contains("Basis flagged"),
            "second PreToolUse must not re-echo the drained summary, got: {out2}"
        );
    }

    #[test]
    fn pre_tool_use_surfaces_both_basis_and_idiom_when_both_pending() {
        // When both governance signals are present, the PreToolUse
        // response must include both — basis before idiom (severity
        // order) — and the drain must clear both fields. Anything
        // less means the agent silently misses one of the two reports.
        let dir = tempfile::tempdir().expect("tempdir");
        record_impulse_to_disk(
            dir.path(),
            "drainboth",
            Channel::Error,
            0.5,
            PendingContext {
                idiom: Some("IDIOM-NOTE".to_string()),
                basis: Some("BASIS-NOTE".to_string()),
            },
        );

        let payload = pre_tool_use_payload("drainboth", "Read", "/tmp/x.rs");
        let out = handle_event(&payload, dir.path()).expect("ok");
        assert!(out.contains("BASIS-NOTE"), "basis must surface, got: {out}");
        assert!(out.contains("IDIOM-NOTE"), "idiom must surface, got: {out}");
        // Severity ordering: basis must appear before idiom in the
        // serialized JSON. The strings are escaped but `find` on raw
        // substrings still gives the correct relative order.
        let b_pos = out.find("BASIS-NOTE").unwrap();
        let i_pos = out.find("IDIOM-NOTE").unwrap();
        assert!(b_pos < i_pos, "basis must precede idiom: {out}");

        // Both fields drained.
        let bytes = std::fs::read(dir.path().join("drainboth.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert!(state.pending_basis_context.is_none());
        assert!(state.pending_idiom_context.is_none());
    }

    #[test]
    fn combine_extra_context_orders_by_severity() {
        // All three signals: watchdog (compaction thrash) > basis
        // (architectural error) > idiom (style drift). Blank-line
        // separators so the agent sees them as independent notes.
        let combined = combine_extra_context(
            Some("WATCHDOG".to_string()),
            Some("IDIOM".to_string()),
            Some("BASIS".to_string()),
        )
        .expect("all Some → Some");
        // Watchdog first.
        assert!(combined.starts_with("WATCHDOG"), "got: {combined}");
        // Idiom last.
        assert!(combined.ends_with("IDIOM"), "got: {combined}");
        // Basis sandwiched.
        let w_end = combined.find("WATCHDOG").unwrap() + "WATCHDOG".len();
        let b_pos = combined.find("BASIS").expect("basis present");
        let i_pos = combined.find("IDIOM").expect("idiom present");
        assert!(w_end < b_pos, "basis must follow watchdog");
        assert!(b_pos < i_pos, "idiom must follow basis");
        assert!(combined.contains("\n\n"));
    }

    #[test]
    fn combine_extra_context_passes_singletons_through_unchanged() {
        assert_eq!(
            combine_extra_context(Some("only".to_string()), None, None).as_deref(),
            Some("only")
        );
        assert_eq!(
            combine_extra_context(None, Some("only".to_string()), None).as_deref(),
            Some("only")
        );
        assert_eq!(
            combine_extra_context(None, None, Some("only".to_string())).as_deref(),
            Some("only")
        );
        assert!(combine_extra_context(None, None, None).is_none());
    }

    // ── SessionStart / SessionEnd path ─────────────────────────

    fn session_start_payload(cwd: &Path, source: &str) -> String {
        serde_json::json!({
            "hook_event_name": "SessionStart",
            "session_id": "ss-test",
            "cwd": cwd.to_string_lossy(),
            "source": source,
        })
        .to_string()
    }

    fn session_end_payload(session_id: &str) -> String {
        serde_json::json!({
            "hook_event_name": "SessionEnd",
            "session_id": session_id,
            "cwd": ".",
            "source": "clear",
        })
        .to_string()
    }

    #[test]
    fn session_start_with_no_sources_returns_empty() {
        // Empty cwd, no basis.yaml, no idiom-cli, no prior digest →
        // the handler must return `{}` rather than an empty
        // hookSpecificOutput envelope (which would surface a blank
        // additionalContext to the agent on every cold start).
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd = tempfile::tempdir().expect("cwd tempdir");
        // Force the idiom subprocess to fail by pointing at a name
        // that cannot exist on PATH — same trick as the post_tool tests.
        let prior = std::env::var(post_tool::IDIOM_BIN_ENV).ok();
        unsafe {
            std::env::set_var(post_tool::IDIOM_BIN_ENV, "definitely_not_a_real_idiom_xyz");
        }
        let out = handle_event(&session_start_payload(cwd.path(), "startup"), dir.path())
            .expect("ok");
        unsafe {
            match prior {
                Some(v) => std::env::set_var(post_tool::IDIOM_BIN_ENV, v),
                None => std::env::remove_var(post_tool::IDIOM_BIN_ENV),
            }
        }
        assert_eq!(out, "{}");
    }

    #[test]
    fn session_start_inlines_basis_yaml_when_present() {
        // A `basis.yaml` in the cwd must be inlined into
        // additionalContext under a labelled section so the agent can
        // see the layer/axis configuration before its first action.
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd = tempfile::tempdir().expect("cwd tempdir");
        let yaml = "governance:\n  version: \"1.0\"\nlayers:\n  hands:\n    role: \"x\"\n";
        std::fs::write(cwd.path().join("basis.yaml"), yaml).expect("write yaml");
        // Suppress idiom subprocess for determinism.
        let prior = std::env::var(post_tool::IDIOM_BIN_ENV).ok();
        unsafe {
            std::env::set_var(post_tool::IDIOM_BIN_ENV, "definitely_not_a_real_idiom_xyz");
        }
        let out = handle_event(&session_start_payload(cwd.path(), "startup"), dir.path())
            .expect("ok");
        unsafe {
            match prior {
                Some(v) => std::env::set_var(post_tool::IDIOM_BIN_ENV, v),
                None => std::env::remove_var(post_tool::IDIOM_BIN_ENV),
            }
        }
        // Should be a hookSpecificOutput envelope, not `{}`.
        assert!(
            out.contains("hookSpecificOutput"),
            "must wrap in envelope when context exists, got: {out}"
        );
        assert!(out.contains("hookEventName"));
        assert!(out.contains("SessionStart"));
        assert!(
            out.contains("Architectural spec"),
            "must label the basis section, got: {out}"
        );
        assert!(out.contains("layers"), "must inline yaml content, got: {out}");
    }

    #[test]
    fn session_start_drains_prior_digest_and_deletes_it() {
        // The prior-session digest is read and the file deleted in
        // the same SessionStart call. A second call must NOT see the
        // same digest content again — same drain semantics as the
        // pending Idiom context, just at a different cadence.
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd = tempfile::tempdir().expect("cwd tempdir");
        let digest_path = dir.path().join(LAST_SESSION_DIGEST_FILENAME);
        std::fs::write(&digest_path, "Session foo ended with 3 blocked tool call(s).")
            .expect("write digest");
        let prior_env = std::env::var(post_tool::IDIOM_BIN_ENV).ok();
        unsafe {
            std::env::set_var(post_tool::IDIOM_BIN_ENV, "definitely_not_a_real_idiom_xyz");
        }
        let out =
            handle_event(&session_start_payload(cwd.path(), "startup"), dir.path()).expect("ok");
        let out2 =
            handle_event(&session_start_payload(cwd.path(), "startup"), dir.path()).expect("ok");
        unsafe {
            match prior_env {
                Some(v) => std::env::set_var(post_tool::IDIOM_BIN_ENV, v),
                None => std::env::remove_var(post_tool::IDIOM_BIN_ENV),
            }
        }
        assert!(out.contains("Prior session"));
        assert!(out.contains("3 blocked tool call"));
        assert!(
            !digest_path.exists(),
            "digest file must be deleted after consumption"
        );
        assert_eq!(out2, "{}", "second SessionStart must not echo the digest");
    }

    #[test]
    fn session_end_writes_digest_only_when_blocks_log_has_entries() {
        // No blocks.jsonl → no digest. SessionEnd must still return
        // `{}` cleanly (default-allow), and must not create an empty
        // digest file that the next SessionStart would mistakenly
        // surface as "prior session blocks: (empty)".
        let dir = tempfile::tempdir().expect("tempdir");
        let out = handle_event(&session_end_payload("nothing"), dir.path()).expect("ok");
        assert_eq!(out, "{}");
        assert!(!dir.path().join(LAST_SESSION_DIGEST_FILENAME).exists());
    }

    #[test]
    fn session_end_writes_digest_for_session_with_block_events() {
        // Plant a blocks.jsonl with a couple of entries for one
        // session and an unrelated entry for a different session.
        // The digest must include only the matching session and must
        // be readable as plain text.
        let dir = tempfile::tempdir().expect("tempdir");
        let lines = vec![
            r#"{"ts":1.0,"session_id":"alpha","tool_name":"Read","decision":"Modify","reason":"context","channel_values":[],"cumulative_cost":0.0}"#,
            r#"{"ts":2.0,"session_id":"beta","tool_name":"Bash","decision":"Deny","reason":"latency","channel_values":[],"cumulative_cost":0.0}"#,
            r#"{"ts":3.0,"session_id":"alpha","tool_name":"Bash","decision":"Deny","reason":"error","channel_values":[],"cumulative_cost":0.0}"#,
        ];
        std::fs::write(dir.path().join(BLOCKS_LOG_FILENAME), lines.join("\n"))
            .expect("write blocks log");
        let out = handle_event(&session_end_payload("alpha"), dir.path()).expect("ok");
        assert_eq!(out, "{}");
        let digest_path = dir.path().join(LAST_SESSION_DIGEST_FILENAME);
        assert!(digest_path.exists(), "SessionEnd must write the digest");
        let body = std::fs::read_to_string(&digest_path).expect("read digest");
        assert!(body.contains("Session alpha ended with 2"));
        assert!(body.contains("**Modify** `Read`"));
        assert!(body.contains("**Deny** `Bash`"));
        // The beta session must NOT bleed into alpha's digest.
        assert!(!body.contains("latency"));
    }

    #[test]
    fn session_end_then_session_start_chain_works_end_to_end() {
        // Full handoff loop: SessionEnd("a") writes a digest, the
        // next SessionStart picks it up, surfaces it via
        // additionalContext, and clears the file. Proves the two
        // hooks compose against the same state directory.
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd = tempfile::tempdir().expect("cwd tempdir");
        let line = r#"{"ts":1.0,"session_id":"a","tool_name":"Bash","decision":"Deny","reason":"saturated","channel_values":[],"cumulative_cost":0.0}"#;
        std::fs::write(dir.path().join(BLOCKS_LOG_FILENAME), line).expect("write blocks log");

        let _ = handle_event(&session_end_payload("a"), dir.path()).expect("ok");
        // Suppress idiom subprocess so the only context source is
        // the digest the SessionEnd just wrote.
        let prior = std::env::var(post_tool::IDIOM_BIN_ENV).ok();
        unsafe {
            std::env::set_var(post_tool::IDIOM_BIN_ENV, "definitely_not_a_real_idiom_xyz");
        }
        let out =
            handle_event(&session_start_payload(cwd.path(), "startup"), dir.path()).expect("ok");
        unsafe {
            match prior {
                Some(v) => std::env::set_var(post_tool::IDIOM_BIN_ENV, v),
                None => std::env::remove_var(post_tool::IDIOM_BIN_ENV),
            }
        }
        assert!(out.contains("Prior session"));
        assert!(out.contains("Session a ended with 1"));
        assert!(out.contains("**Deny** `Bash`"));
    }

    #[test]
    fn posttool_on_read_does_not_touch_state() {
        // Read is not a write — Idiom has nothing to check, the
        // dispatch must short-circuit before any subprocess or
        // disk write happens.
        let dir = tempfile::tempdir().expect("tempdir");
        let payload = post_tool_use_payload("readsess", "Read", "/foo.rs");
        let out = handle_event(&payload, dir.path()).expect("ok");
        assert_eq!(out, "{}");
        assert!(
            !dir.path().join("readsess.json").exists(),
            "Read PostToolUse must not create a state file"
        );
    }

    #[test]
    fn posttool_on_bash_does_not_touch_state() {
        // Same shape as Read: Bash never writes a source file.
        let dir = tempfile::tempdir().expect("tempdir");
        let payload = post_tool_use_payload("bashsess", "Bash", "");
        let out = handle_event(&payload, dir.path()).expect("ok");
        assert_eq!(out, "{}");
        assert!(!dir.path().join("bashsess.json").exists());
    }

    #[test]
    fn posttool_on_write_to_non_source_file_does_not_touch_state() {
        // Markdown / config edits are real Write tool calls but
        // Idiom has no language for them — the extension pre-filter
        // in `post_tool::idiom_deviation_count` returns None, so the
        // hook should produce no impulse and no state file.
        let dir = tempfile::tempdir().expect("tempdir");
        let payload = post_tool_use_payload("docsess", "Write", "C:/proj/CLAUDE.md");
        let out = handle_event(&payload, dir.path()).expect("ok");
        assert_eq!(out, "{}");
        assert!(!dir.path().join("docsess.json").exists());
    }

    #[test]
    fn posttool_with_missing_idiom_binary_does_not_touch_state() {
        // Default-allow contract: a missing/broken idiom-cli must
        // never cause the hook to error or to spuriously add
        // pressure. Point FEEDBACK_IDIOM_BIN at a name that cannot
        // exist on PATH and verify the state file is not created.
        let dir = tempfile::tempdir().expect("tempdir");
        let prior = std::env::var(post_tool::IDIOM_BIN_ENV).ok();
        // SAFETY: tests in this module run in a single process; the
        // env var is restored before any other test could observe it.
        unsafe {
            std::env::set_var(
                post_tool::IDIOM_BIN_ENV,
                "definitely_not_a_real_binary_xyz_99",
            );
        }
        let payload = post_tool_use_payload("missing", "Write", "/proj/foo.rs");
        let result = handle_event(&payload, dir.path());
        unsafe {
            match prior {
                Some(v) => std::env::set_var(post_tool::IDIOM_BIN_ENV, v),
                None => std::env::remove_var(post_tool::IDIOM_BIN_ENV),
            }
        }
        let out = result.expect("missing binary must not error");
        assert_eq!(out, "{}");
        assert!(
            !dir.path().join("missing.json").exists(),
            "missing idiom binary must not create a state file"
        );
    }

    #[test]
    fn posttool_payload_without_file_path_returns_empty() {
        // A malformed Write payload (no file_path field) must not
        // crash the hook — the let-else short-circuit returns "{}".
        let dir = tempfile::tempdir().expect("tempdir");
        let payload = serde_json::json!({
            "hook_event_name": "PostToolUse",
            "session_id": "nofp",
            "tool_name": "Write",
            "tool_input": {},
        })
        .to_string();
        let out = handle_event(&payload, dir.path()).expect("ok");
        assert_eq!(out, "{}");
        assert!(!dir.path().join("nofp.json").exists());
    }

    // ── Compaction watchdog ────────────────────────────────────

    /// Build a `PersistedState` with the given watchdog metadata
    /// already populated and plant it at `dir/<session>.json`. The
    /// governor's live pressure fields are defaulted — the watchdog
    /// path only cares about `recent_compactions` and
    /// `last_guidance_t`, not the bank contents.
    fn plant_state_with_compactions(
        dir: &Path,
        session_id: &str,
        recent: Vec<f64>,
        last_guidance_t: f64,
    ) {
        let gov = Governor::from_defaults();
        let mut snap = gov.snapshot();
        snap.recent_compactions = recent.into_iter().collect();
        snap.last_guidance_t = last_guidance_t;
        let bytes = serde_json::to_vec_pretty(&snap).expect("serialize");
        let session_file = format!("{}.json", sanitize_session_id(session_id));
        std::fs::write(dir.join(session_file), bytes).expect("write");
    }

    #[test]
    fn watchdog_fires_when_two_recent_compactions_trigger_pre_tool_use() {
        // Plant a session with two compactions inside the 60s window
        // and no prior guidance. The next PreToolUse call must
        // inject the chunking advice via additionalContext.
        let dir = tempfile::tempdir().expect("tempdir");
        let now = wall_clock_secs();
        plant_state_with_compactions(dir.path(), "thrashing", vec![now - 10.0, now - 5.0], 0.0);

        let out = handle_event(
            &pre_tool_use_payload("thrashing", "Read", "/foo.rs"),
            dir.path(),
        )
        .expect("ok");
        assert!(
            out.contains("additionalContext"),
            "watchdog should attach additionalContext: {out}"
        );
        assert!(
            out.contains("smaller chunks"),
            "guidance text should reach the agent: {out}"
        );

        // Cooldown: last_guidance_t must have been updated so the
        // watchdog does not fire again on the very next call.
        let bytes = std::fs::read(dir.path().join("thrashing.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert!(
            state.last_guidance_t > 0.0,
            "last_guidance_t must be stamped after firing"
        );
    }

    #[test]
    fn watchdog_is_silent_when_only_one_recent_compaction() {
        // One compaction is normal behavior — Claude Code's
        // auto-compactor runs constantly. The watchdog must only
        // fire on the "twice within the window" case.
        let dir = tempfile::tempdir().expect("tempdir");
        let now = wall_clock_secs();
        plant_state_with_compactions(dir.path(), "normal", vec![now - 20.0], 0.0);

        let out = handle_event(
            &pre_tool_use_payload("normal", "Read", "/foo.rs"),
            dir.path(),
        )
        .expect("ok");
        assert!(
            !out.contains("additionalContext"),
            "single compaction must not trip the watchdog: {out}"
        );
    }

    #[test]
    fn watchdog_respects_cooldown_between_injections() {
        // Even with the threshold met, a recent guidance timestamp
        // must suppress re-injection for the cooldown duration.
        // This prevents the agent from being nagged on every single
        // call of a thrashing session.
        let dir = tempfile::tempdir().expect("tempdir");
        let now = wall_clock_secs();
        // Threshold met + guidance fired 10s ago (<< 300s cooldown).
        plant_state_with_compactions(
            dir.path(),
            "cooldown",
            vec![now - 30.0, now - 10.0],
            now - 10.0,
        );

        let out = handle_event(
            &pre_tool_use_payload("cooldown", "Read", "/foo.rs"),
            dir.path(),
        )
        .expect("ok");
        assert!(
            !out.contains("additionalContext"),
            "cooldown must suppress the watchdog: {out}"
        );
    }

    #[test]
    fn watchdog_prunes_stale_compactions_before_counting() {
        // Two compactions, but the first is outside the 60s window.
        // The effective count is 1, so the watchdog should NOT fire.
        // This also proves the persisted `recent_compactions` list
        // gets trimmed, so we don't carry stale entries forever.
        let dir = tempfile::tempdir().expect("tempdir");
        let now = wall_clock_secs();
        plant_state_with_compactions(
            dir.path(),
            "stale",
            vec![now - 500.0, now - 5.0],
            0.0,
        );

        let out = handle_event(
            &pre_tool_use_payload("stale", "Read", "/foo.rs"),
            dir.path(),
        )
        .expect("ok");
        assert!(
            !out.contains("additionalContext"),
            "stale compactions must not trip the watchdog: {out}"
        );

        // The stale entry was pruned from the persisted list.
        let bytes = std::fs::read(dir.path().join("stale.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            state.recent_compactions.len(),
            1,
            "stale entry should be pruned on write"
        );
    }

    #[test]
    fn post_tool_use_preserves_watchdog_metadata() {
        // A PostToolUse impulse (from idiom-cli, etc.) must not
        // clobber the watchdog's persisted state. Plant a session
        // with two compactions + a guidance stamp, fire a PostToolUse
        // impulse via the helper, and verify the fields survive.
        let dir = tempfile::tempdir().expect("tempdir");
        let now = wall_clock_secs();
        plant_state_with_compactions(
            dir.path(),
            "post",
            vec![now - 30.0, now - 10.0],
            now - 20.0,
        );

        let updated = record_impulse_to_disk(
            dir.path(),
            "post",
            Channel::Error,
            1.5,
            PendingContext::default(),
        );
        assert!(updated);

        let bytes = std::fs::read(dir.path().join("post.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            state.recent_compactions.len(),
            2,
            "PostToolUse must not erase recent_compactions"
        );
        assert!(
            (state.last_guidance_t - (now - 20.0)).abs() < 0.01,
            "PostToolUse must preserve last_guidance_t"
        );
    }

    #[test]
    fn pre_compact_preserves_prior_guidance_timestamp() {
        // A PreCompact after guidance already fired must not reset
        // last_guidance_t — the cooldown has to survive across
        // compactions so the agent is not re-nagged moments after
        // compacting.
        let dir = tempfile::tempdir().expect("tempdir");
        let now = wall_clock_secs();
        plant_state_with_compactions(dir.path(), "post-guide", vec![now - 5.0], now - 30.0);

        let _ = handle_event(&pre_compact_payload("post-guide"), dir.path()).unwrap();

        let bytes = std::fs::read(dir.path().join("post-guide.json")).unwrap();
        let state: PersistedState = serde_json::from_slice(&bytes).unwrap();
        assert!(
            (state.last_guidance_t - (now - 30.0)).abs() < 0.01,
            "PreCompact must preserve last_guidance_t across the reset"
        );
    }

    // ── Per-session impulse log ────────────────────────────────

    #[test]
    fn append_impulse_log_creates_jsonl_with_expected_fields() {
        // Single non-zero call from the basis side. Verifies the
        // file lands at the per-session path, contains exactly one
        // valid JSON line, and that every field round-trips.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut axes = std::collections::BTreeMap::new();
        axes.insert("placement".to_string(), 2u64);
        axes.insert("values".to_string(), 1u64);
        append_impulse_log_entry(dir.path(), "logsess", 1234.5, "basis", 3, 1.5, &axes);

        let path = impulse_log_path(dir.path(), "logsess");
        assert!(path.exists(), "impulse log should be created at the per-session path");
        let log = std::fs::read_to_string(&path).expect("read log");
        let line = log.lines().next().expect("at least one line");
        let entry: serde_json::Value = serde_json::from_str(line).expect("valid JSON line");
        assert_eq!(entry["channel"], "Error");
        assert_eq!(entry["source"], "basis");
        assert_eq!(entry["count"], 3);
        assert!((entry["impulse"].as_f64().unwrap() - 1.5).abs() < 1e-9);
        assert!((entry["t"].as_f64().unwrap() - 1234.5).abs() < 1e-9);
        assert_eq!(entry["axis_breakdown"]["placement"], 2);
        assert_eq!(entry["axis_breakdown"]["values"], 1);
    }

    #[test]
    fn append_impulse_log_appends_across_calls() {
        // Append-only contract: three writes → three lines, each
        // independently parseable. Mirrors the blocks-log append test.
        let dir = tempfile::tempdir().expect("tempdir");
        for i in 0..3 {
            append_impulse_log_entry(
                dir.path(),
                "appendsess",
                i as f64,
                "idiom",
                1,
                0.5,
                &std::collections::BTreeMap::new(),
            );
        }
        let log = std::fs::read_to_string(impulse_log_path(dir.path(), "appendsess")).unwrap();
        let lines: Vec<&str> = log.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 3, "three calls should produce three lines");
        for line in &lines {
            let _: serde_json::Value =
                serde_json::from_str(line).expect("each line is valid JSON");
        }
    }

    #[test]
    fn append_impulse_log_isolates_sessions_into_separate_files() {
        // The whole point of per-session logs is no concurrent-append
        // contention between different Claude sessions. Verify two
        // session ids land in two distinct files.
        let dir = tempfile::tempdir().expect("tempdir");
        append_impulse_log_entry(
            dir.path(),
            "sessA",
            1.0,
            "basis",
            1,
            0.5,
            &std::collections::BTreeMap::new(),
        );
        append_impulse_log_entry(
            dir.path(),
            "sessB",
            2.0,
            "idiom",
            2,
            1.0,
            &std::collections::BTreeMap::new(),
        );
        let a = std::fs::read_to_string(impulse_log_path(dir.path(), "sessA")).unwrap();
        let b = std::fs::read_to_string(impulse_log_path(dir.path(), "sessB")).unwrap();
        assert!(a.contains("\"basis\"") && !a.contains("\"idiom\""));
        assert!(b.contains("\"idiom\"") && !b.contains("\"basis\""));
    }

    #[test]
    fn render_pressure_summary_includes_totals_sources_and_axes() {
        // Two basis impulses + one idiom impulse + axis spread. The
        // summary should report aggregate totals, per-source bullets,
        // and a Basis axis line with summed counts.
        let entries: Vec<serde_json::Value> = vec![
            serde_json::json!({
                "t": 1.0, "channel": "Error", "source": "basis",
                "count": 3, "impulse": 1.5,
                "axis_breakdown": {"placement": 2, "values": 1}
            }),
            serde_json::json!({
                "t": 2.0, "channel": "Error", "source": "basis",
                "count": 2, "impulse": 1.0,
                "axis_breakdown": {"purity": 2}
            }),
            serde_json::json!({
                "t": 3.0, "channel": "Error", "source": "idiom",
                "count": 4, "impulse": 2.0,
                "axis_breakdown": {}
            }),
        ];
        let summary = render_pressure_summary(&entries);
        assert!(summary.contains("Pressure summary"), "got: {summary}");
        assert!(summary.contains("3 impulse(s)"), "total impulses: {summary}");
        assert!(summary.contains("9 deviation(s)"), "total deviations: {summary}");
        assert!(summary.contains("peak 2.00"), "peak: {summary}");
        assert!(summary.contains("**basis**"), "basis bucket: {summary}");
        assert!(summary.contains("**idiom**"), "idiom bucket: {summary}");
        // Basis axis line: aggregated across both basis entries.
        assert!(summary.contains("Basis axes"), "axis line: {summary}");
        assert!(summary.contains("2 placement"));
        assert!(summary.contains("1 values"));
        assert!(summary.contains("2 purity"));
    }

    #[test]
    fn render_pressure_summary_omits_axis_line_when_only_idiom_present() {
        // Idiom entries have empty axis_breakdown. The Basis axes
        // line must be skipped — no Idiom "axes" exist to report.
        let entries: Vec<serde_json::Value> = vec![serde_json::json!({
            "t": 1.0, "channel": "Error", "source": "idiom",
            "count": 2, "impulse": 1.0, "axis_breakdown": {}
        })];
        let summary = render_pressure_summary(&entries);
        assert!(summary.contains("Pressure summary"));
        assert!(summary.contains("**idiom**"));
        assert!(
            !summary.contains("Basis axes"),
            "no axis line when no axes recorded: {summary}"
        );
    }

    #[test]
    fn build_session_end_digest_returns_some_when_only_impulses_present() {
        // No blocks.jsonl, but a per-session impulse log exists.
        // The digest must still be produced — pressure-only sessions
        // are real (governance subprocess found drift but the
        // governor never escalated to a block).
        let dir = tempfile::tempdir().expect("tempdir");
        let imp_path = impulse_log_path(dir.path(), "imponly");
        std::fs::write(
            &imp_path,
            r#"{"t":1.0,"channel":"Error","source":"basis","count":2,"impulse":1.0,"axis_breakdown":{"purity":2}}"#,
        )
        .unwrap();
        let blocks_log = dir.path().join(BLOCKS_LOG_FILENAME);
        let digest = build_session_end_digest(&blocks_log, &imp_path, "imponly")
            .expect("impulse-only session should still produce a digest");
        assert!(digest.contains("Pressure summary"), "got: {digest}");
        assert!(digest.contains("**basis**"));
        assert!(digest.contains("purity"));
        // No block-events section: the "blocked / modified" header
        // must be absent when blocks.jsonl is empty.
        assert!(
            !digest.contains("blocked / modified tool call"),
            "no block-events section without blocks log: {digest}"
        );
    }

    #[test]
    fn build_session_end_digest_combines_blocks_and_pressure_when_both_present() {
        // The realistic case: a session both produced block events
        // AND accumulated pressure from PostToolUse impulses. Both
        // sections must appear, blank-line separated.
        let dir = tempfile::tempdir().expect("tempdir");
        let blocks_log = dir.path().join(BLOCKS_LOG_FILENAME);
        std::fs::write(
            &blocks_log,
            r#"{"ts":1.0,"session_id":"both","tool_name":"Bash","decision":"Deny","reason":"saturated","channel_values":[],"cumulative_cost":0.0}"#,
        )
        .unwrap();
        let imp_path = impulse_log_path(dir.path(), "both");
        std::fs::write(
            &imp_path,
            r#"{"t":1.0,"channel":"Error","source":"basis","count":1,"impulse":0.5,"axis_breakdown":{"placement":1}}"#,
        )
        .unwrap();
        let digest = build_session_end_digest(&blocks_log, &imp_path, "both")
            .expect("digest should exist when both sources are populated");
        assert!(
            digest.contains("Session both ended with 1 blocked / modified"),
            "blocks header: {digest}"
        );
        assert!(digest.contains("**Deny** `Bash`"), "block bullet: {digest}");
        assert!(digest.contains("Pressure summary"), "pressure section: {digest}");
        assert!(digest.contains("placement"));
        assert!(digest.contains("\n\n"), "sections must be blank-line separated: {digest}");
    }

    #[test]
    fn session_end_writes_digest_for_impulse_only_session_via_handle_event() {
        // End-to-end through the dispatch: a per-session impulse log
        // exists, no blocks.jsonl, SessionEnd handler must persist
        // the digest AND clean up the impulse log file (read-and-clear).
        let dir = tempfile::tempdir().expect("tempdir");
        let imp_path = impulse_log_path(dir.path(), "imponly2");
        std::fs::write(
            &imp_path,
            r#"{"t":1.0,"channel":"Error","source":"idiom","count":2,"impulse":1.0,"axis_breakdown":{}}"#,
        )
        .unwrap();
        let out = handle_event(&session_end_payload("imponly2"), dir.path()).expect("ok");
        assert_eq!(out, "{}");
        let digest_path = dir.path().join(LAST_SESSION_DIGEST_FILENAME);
        assert!(digest_path.exists(), "digest should be written");
        let body = std::fs::read_to_string(&digest_path).unwrap();
        assert!(body.contains("Pressure summary"), "body: {body}");
        assert!(body.contains("**idiom**"));
        // Read-and-clear: the impulse log is removed by SessionEnd.
        assert!(
            !imp_path.exists(),
            "impulse log should be cleaned up after SessionEnd consumes it"
        );
    }

    #[test]
    fn session_end_then_session_start_chain_includes_pressure_summary() {
        // Full handoff: impulse log + blocks log → SessionEnd writes
        // a combined digest → next SessionStart drains it into
        // additionalContext. Proves the three-step chain (PostToolUse
        // → SessionEnd → SessionStart) hands the pressure shape
        // through to the next session without code in SessionStart
        // needing to know the digest format changed.
        let dir = tempfile::tempdir().expect("tempdir");
        let cwd = tempfile::tempdir().expect("cwd tempdir");
        std::fs::write(
            dir.path().join(BLOCKS_LOG_FILENAME),
            r#"{"ts":1.0,"session_id":"chain","tool_name":"Bash","decision":"Deny","reason":"saturated","channel_values":[],"cumulative_cost":0.0}"#,
        )
        .unwrap();
        std::fs::write(
            impulse_log_path(dir.path(), "chain"),
            r#"{"t":1.0,"channel":"Error","source":"basis","count":4,"impulse":2.0,"axis_breakdown":{"placement":4}}"#,
        )
        .unwrap();

        let _ = handle_event(&session_end_payload("chain"), dir.path()).expect("ok");

        // Suppress idiom subprocess so the only context source is
        // the digest the SessionEnd just wrote.
        let prior = std::env::var(post_tool::IDIOM_BIN_ENV).ok();
        unsafe {
            std::env::set_var(post_tool::IDIOM_BIN_ENV, "definitely_not_a_real_idiom_xyz");
        }
        let out =
            handle_event(&session_start_payload(cwd.path(), "startup"), dir.path()).expect("ok");
        unsafe {
            match prior {
                Some(v) => std::env::set_var(post_tool::IDIOM_BIN_ENV, v),
                None => std::env::remove_var(post_tool::IDIOM_BIN_ENV),
            }
        }
        assert!(out.contains("Prior session"), "got: {out}");
        assert!(out.contains("Session chain ended with 1"));
        assert!(out.contains("Pressure summary"), "pressure must reach SessionStart: {out}");
        assert!(out.contains("**basis**"));
        assert!(out.contains("placement"));
    }

    #[test]
    fn block_events_append_across_calls() {
        // The log is append-only — every block on the same session
        // adds another line, so the operator can scrub history later
        // to compute false-positive rates.
        let dir = tempfile::tempdir().expect("tempdir");
        let mut gov = Governor::from_defaults();
        gov.record_impulse(Channel::Error, 100.0);
        plant_state_file(dir.path(), "appender", &gov);

        for _ in 0..3 {
            let payload = pre_tool_use_payload("appender", "Read", "/foo.rs");
            let _ = handle_event(&payload, dir.path()).expect("ok");
        }

        let log = std::fs::read_to_string(dir.path().join(BLOCKS_LOG_FILENAME)).unwrap();
        let lines: Vec<&str> = log.lines().filter(|l| !l.is_empty()).collect();
        assert_eq!(lines.len(), 3, "three calls → three log lines");
        for line in &lines {
            let _: serde_json::Value =
                serde_json::from_str(line).expect("each line is valid JSON");
        }
    }
}
