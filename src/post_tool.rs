//! PostToolUse adapter — turns Idiom deviation counts into Error
//! channel impulses.
//!
//! Hands layer: IO-dependent. This module shells out to `idiom-cli`
//! after a Write/Edit lands, parses the `{"total": N}` JSON, and maps
//! the count to a bounded magnitude on the Error channel. The hook
//! binary then records that impulse against the per-session governor
//! state, so the agent feels naming-convention drift as accumulating
//! pressure rather than as a one-off advisory it can ignore.
//!
//! # Why this exists
//!
//! Idiom is wired into Coalition as MCP tools (`idiom_check`,
//! `idiom_infer`, `idiom_context`), but in practice the agent rarely
//! invokes them — they are advisory and cost a tool call. By piping
//! the deviation count back through the Feedback governor on every
//! Write, we make non-conforming code cost something automatically
//! without blocking the edit that produced it. The PreToolUse hook
//! still gates the *next* action, so writing one bad file is fine but
//! a streak of them will eventually push the Error channel past its
//! warn / critical thresholds.
//!
//! # Default-allow contract
//!
//! Like the rest of the hook binary, every error path here is
//! silent. A missing `idiom-cli`, a non-source file, a parse failure,
//! or a busy filesystem all return `None` from `idiom_deviation_count`
//! — the caller will see "no opinion" and leave the governor state
//! untouched. This module must never block, panic, or hang the agent.

use std::path::Path;
use std::process::Command;

/// Environment variable used to override the path to the `idiom-cli`
/// binary in tests. Outside tests we just trust `$PATH`.
pub const IDIOM_BIN_ENV: &str = "FEEDBACK_IDIOM_BIN";

/// Environment variable used to override the path to the `basis-cli`
/// binary in tests. Same role as `IDIOM_BIN_ENV` for the Idiom side.
pub const BASIS_BIN_ENV: &str = "FEEDBACK_BASIS_BIN";

/// Default binary name. Resolved through `$PATH` at spawn time.
const DEFAULT_IDIOM_BIN: &str = "idiom-cli";

/// Default binary name for Basis. Resolved through `$PATH`.
const DEFAULT_BASIS_BIN: &str = "basis-cli";

/// Hard ceiling on the impulse a single Write can deposit on the
/// Error channel. Error critical is 6.0, so 3.0 means a single
/// catastrophic file accounts for half the budget — enough to be
/// noticed, not enough to one-shot the channel into block territory.
const MAX_IMPULSE: f64 = 3.0;

/// Linear coefficient: each deviation contributes this much to the
/// raw impulse before clamping. Calibrated so that 6 deviations
/// saturates the cap (6 * 0.5 = 3.0 = MAX_IMPULSE).
const PER_DEVIATION: f64 = 0.5;

/// Source file extensions Idiom can analyse. Anything else (markdown,
/// json, lockfiles, screenshots) skips the subprocess entirely — both
/// to avoid wasted work and because Idiom would either no-op or
/// return an unhelpful error on a non-source file.
const SOURCE_EXTENSIONS: &[&str] = &[
    "rs", "py", "ts", "tsx", "js", "jsx", "go", "java", "c", "cc", "cpp", "h", "hpp", "rb",
];

/// Maximum number of individual deviation lines to include in the
/// human-readable summary handed to the next PreToolUse. Three is
/// enough to give the agent a concrete sample without flooding the
/// context window — the count comes back separately, so the agent
/// always knows when there are more than were shown.
const MAX_DEVIATIONS_IN_SUMMARY: usize = 3;

/// Result of running Idiom against one file. Carries both the bare
/// count (used to compute the Error-channel impulse) and a short
/// human-readable summary of the first few deviations (used as the
/// `additionalContext` payload on the *next* PreToolUse so the agent
/// can see *what* it tripped instead of just feeling generic Error
/// pressure). Both fields are derived from the same `idiom-cli`
/// invocation, so a single subprocess call covers both needs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IdiomReport {
    pub count: u64,
    /// Multi-line markdown-flavoured summary, suitable for direct
    /// embedding in `additionalContext`. Empty when `count == 0`.
    pub summary: String,
}

/// Result of running Basis against the file's containing crate,
/// filtered to violations whose `file` field matches the edited path.
/// Mirror of `IdiomReport`, plus an `axis_breakdown` so the agent
/// can see whether the deviations are placement, values,
/// completeness, or purity — that information lets the agent decide
/// which fix is cheapest before its next action.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BasisReport {
    pub count: u64,
    /// Multi-line markdown summary, suitable for direct embedding
    /// in `additionalContext`. Empty when `count == 0`.
    pub summary: String,
    /// Per-axis violation counts. Useful for axis-aware impulse
    /// weighting in the future; for v1 the only consumer is the
    /// human-readable summary header.
    pub axis_breakdown: std::collections::BTreeMap<String, u64>,
}

fn resolve_bin() -> String {
    std::env::var(IDIOM_BIN_ENV).unwrap_or_else(|_| DEFAULT_IDIOM_BIN.to_string())
}

/// Map a deviation count to an Error-channel impulse.
///
/// Pure function — extracted from `idiom_deviation_count` so the
/// calibration can be unit-tested without spawning a subprocess.
/// Zero deviations → zero impulse (no point recording a no-op).
pub fn deviations_to_impulse(count: u64) -> f64 {
    if count == 0 {
        return 0.0;
    }
    let raw = PER_DEVIATION * count as f64;
    if raw > MAX_IMPULSE {
        MAX_IMPULSE
    } else {
        raw
    }
}

/// Impulse deposited on the Error channel for a Bash tool failure
/// (nonzero exit code or `is_error` flag from Claude Code).
///
/// A single build failure is worth 1.5 error-units — half the channel's
/// critical budget (6.0). With the 24-second half-life, two consecutive
/// failures 30 seconds apart accumulate to `1.5 + 1.5*e^(-0.029*30) ≈ 2.5`
/// (below warn), and three at that pace reach `≈ 3.2` (near warn at 3.6).
/// This means the channel distinguishes a one-off failure (pressure decays
/// before warn) from a sustained spiral (3+ failures within a minute cross
/// warn and start generating Modify verdicts).
///
/// Set slightly higher than the per-deviation impulse (0.5) because a
/// build failure is more disruptive — the agent cannot make progress
/// until it's resolved, and each failed attempt wastes context tokens.
pub const BASH_ERROR_IMPULSE: f64 = 1.5;

/// Decide whether `path` is something Idiom should look at. Pure
/// extension test — Idiom's own language detection is the source of
/// truth, this is just a cheap pre-filter so we don't spawn `idiom-cli`
/// for every CLAUDE.md edit.
fn looks_like_source_file(path: &str) -> bool {
    Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|ext| SOURCE_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

/// Run `idiom-cli check <file> --format json` and return both the
/// `total` deviation count and a short human-readable summary of the
/// first few deviations. Returns `None` on any failure — missing
/// binary, non-zero exit beyond the documented `1` (which is
/// "deviations found, output still valid"), parse failure, or
/// non-source file.
///
/// The default-allow contract means a `None` here is indistinguishable
/// from "Idiom approved this file" at the call site, which is the
/// correct behavior: when in doubt, do not penalize.
pub fn idiom_check_report(file_path: &str) -> Option<IdiomReport> {
    if !looks_like_source_file(file_path) {
        return None;
    }
    let bin = resolve_bin();
    let output = Command::new(&bin)
        .args(["check", file_path, "--format", "json"])
        .output()
        .ok()?;
    // idiom-cli exits 0 on a clean file and 1 when deviations are
    // found — both are valid outcomes that produce parseable JSON on
    // stdout. Anything else means the binary failed to run the check
    // (missing file, language not supported, internal error).
    let code = output.status.code().unwrap_or(-1);
    if code != 0 && code != 1 {
        return None;
    }
    let stdout = std::str::from_utf8(&output.stdout).ok()?;
    parse_idiom_report(stdout)
}

/// Pure JSON-to-`IdiomReport` parser. Split out from
/// `idiom_check_report` so the summarization shape can be unit-tested
/// without spawning a subprocess. Returns `None` on any structural
/// failure — missing `total`, malformed `deviations`, etc.
pub fn parse_idiom_report(json_str: &str) -> Option<IdiomReport> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let count = parsed.get("total").and_then(|v| v.as_u64())?;
    if count == 0 {
        return Some(IdiomReport {
            count: 0,
            summary: String::new(),
        });
    }
    let deviations = parsed
        .get("deviations")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();
    let mut lines: Vec<String> = Vec::with_capacity(MAX_DEVIATIONS_IN_SUMMARY);
    for dev in deviations.iter().take(MAX_DEVIATIONS_IN_SUMMARY) {
        let code = dev.get("code").and_then(|v| v.as_str()).unwrap_or("I???");
        let line = dev.get("line").and_then(|v| v.as_u64()).unwrap_or(0);
        let message = dev
            .get("message")
            .and_then(|v| v.as_str())
            .unwrap_or("(no message)");
        lines.push(format!("- [{code}] line {line}: {message}"));
    }
    let header = if count == 1 {
        "Idiom flagged 1 naming convention deviation in the file you just wrote:".to_string()
    } else {
        format!(
            "Idiom flagged {count} naming convention deviations in the file you just wrote:"
        )
    };
    let mut summary = header;
    summary.push('\n');
    summary.push_str(&lines.join("\n"));
    if (count as usize) > lines.len() {
        let extra = count as usize - lines.len();
        summary.push_str(&format!("\n  …and {extra} more"));
    }
    summary.push_str(
        "\n\nThese are local-convention deviations, not Basis errors. Either rename to match the \
         project's prevailing pattern, or pin the new pattern in the spec if it is intentional.",
    );
    Some(IdiomReport { count, summary })
}

/// Backwards-compat shim. The hook binary used to call this directly;
/// new code should call `idiom_check_report` and read the `count`
/// field. Kept around so the existing tests for the
/// missing-binary / non-source-file paths continue to apply without
/// duplication, and so any out-of-tree caller does not break.
pub fn idiom_deviation_count(file_path: &str) -> Option<u64> {
    idiom_check_report(file_path).map(|r| r.count)
}

// ── Basis-side: full architectural check, scoped to one file ──────

fn resolve_basis_bin() -> String {
    std::env::var(BASIS_BIN_ENV).unwrap_or_else(|_| DEFAULT_BASIS_BIN.to_string())
}

/// Walk up the directory tree from `file_path` looking for the
/// nearest ancestor that contains BOTH a `Cargo.toml` and a
/// `basis.yaml`. Returns that ancestor path on success.
///
/// We require both files because:
/// - Cargo.toml alone identifies a Rust crate, but Basis is
///   language-agnostic and a non-Rust project may not have one;
/// - basis.yaml alone identifies the spec root, but the spec root
///   for a multi-crate workspace may live above the crate Basis
///   actually wants to check.
///
/// In practice, the project layouts we care about (Coalition,
/// Basis itself, Feedback, Idiom) all keep `basis.yaml` at the
/// crate root next to `Cargo.toml`, so the conjunction is the
/// right disambiguator. Returns `None` for files that live
/// outside any governed crate, which is the signal to silently
/// skip the Basis subprocess (default-allow contract).
pub fn find_crate_root(file_path: &str) -> Option<std::path::PathBuf> {
    let mut current = Path::new(file_path);
    if current.is_file() {
        current = current.parent()?;
    }
    loop {
        if current.join("Cargo.toml").exists() && current.join("basis.yaml").exists() {
            return Some(current.to_path_buf());
        }
        current = current.parent()?;
    }
}

/// Run `basis-cli check --spec <root>/basis.yaml --path <root>
/// --format json` and parse the result, filtering violations to
/// those whose `file` field matches the edited path. Returns
/// `None` on any failure — missing binary, missing crate root,
/// non-JSON output, etc.
///
/// The whole-crate check is fast enough (~150ms on the feedback
/// crate) that we don't bother caching. If that changes, the
/// natural cache key is `(crate_root, max_mtime_under_root)`.
///
/// Default-allow contract: every error returns `None`. The hook
/// must never panic, hang, or block on this code path.
pub fn basis_check_report(file_path: &str) -> Option<BasisReport> {
    if !looks_like_source_file(file_path) {
        return None;
    }
    let crate_root = find_crate_root(file_path)?;
    let spec_path = crate_root.join("basis.yaml");
    let bin = resolve_basis_bin();
    let output = Command::new(&bin)
        .arg("check")
        .arg("--spec")
        .arg(&spec_path)
        .arg("--format")
        .arg("json")
        .arg(&crate_root)
        .output()
        .ok()?;
    // basis-cli exits 0 on a clean repo and 1 when violations are
    // present (same convention as idiom-cli). Anything else means
    // the binary failed to run — bad spec, missing path, internal
    // error — and we silently default-allow.
    let code = output.status.code().unwrap_or(-1);
    if code != 0 && code != 1 {
        return None;
    }
    let stdout = std::str::from_utf8(&output.stdout).ok()?;
    parse_basis_report(stdout, file_path)
}

/// Pure JSON-to-`BasisReport` parser. Filters violations to those
/// whose `file` field matches `edited_file` (so the agent only
/// sees the consequences of the edit it just made, not the
/// codebase's pre-existing debt). Split out from
/// `basis_check_report` so the filtering and rendering can be
/// unit-tested without spawning a subprocess.
///
/// `edited_file` is matched against `violation.file` by suffix
/// after path normalization (replace `\\` with `/`), because
/// basis-cli emits paths relative to its `--path` argument while
/// the hook receives absolute paths from Claude Code. A suffix
/// match is loose enough to bridge that gap and tight enough that
/// `src/foo.rs` does not collide with `src/bar/foo.rs`.
pub fn parse_basis_report(json_str: &str, edited_file: &str) -> Option<BasisReport> {
    let parsed: serde_json::Value = serde_json::from_str(json_str).ok()?;
    let violations = parsed.get("violations").and_then(|v| v.as_array())?;

    let edited_norm = edited_file.replace('\\', "/");
    let edited_tail = std::path::Path::new(&edited_norm)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(&edited_norm);

    let matching: Vec<&serde_json::Value> = violations
        .iter()
        .filter(|v| {
            let Some(vfile) = v.get("file").and_then(|f| f.as_str()) else {
                return false;
            };
            let vfile_norm = vfile.replace('\\', "/");
            // Two-pronged match: full-suffix for absolute paths, or
            // basename match for the common case where basis-cli's
            // path is relative-to-crate-root and the edited file
            // path is absolute. The basename check would risk
            // false positives if two distinct files share a name,
            // but that is rare enough in well-organized crates and
            // the cost (a spurious deviation message) is low.
            edited_norm.ends_with(&vfile_norm)
                || vfile_norm.ends_with(&edited_norm)
                || std::path::Path::new(&vfile_norm)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|name| name == edited_tail && vfile_norm == edited_tail)
                    .unwrap_or(false)
        })
        .collect();

    let count = matching.len() as u64;
    let mut axis_breakdown: std::collections::BTreeMap<String, u64> =
        std::collections::BTreeMap::new();
    for v in &matching {
        if let Some(axis) = v.get("axis").and_then(|a| a.as_str()) {
            *axis_breakdown.entry(axis.to_string()).or_insert(0) += 1;
        }
    }

    if count == 0 {
        return Some(BasisReport {
            count: 0,
            summary: String::new(),
            axis_breakdown,
        });
    }

    let mut lines: Vec<String> = Vec::with_capacity(MAX_DEVIATIONS_IN_SUMMARY);
    for v in matching.iter().take(MAX_DEVIATIONS_IN_SUMMARY) {
        let code = v.get("code").and_then(|s| s.as_str()).unwrap_or("B???");
        let axis = v.get("axis").and_then(|s| s.as_str()).unwrap_or("?");
        let line = v.get("line").and_then(|s| s.as_u64()).unwrap_or(0);
        let message = v
            .get("message")
            .and_then(|s| s.as_str())
            .unwrap_or("(no message)");
        // The `help` field is the actionable fix line — same text a
        // human reading `basis check` would see on the help: row.
        // Defaults to empty for older basis-cli builds that pre-date
        // the field; in that case we omit the indented sub-line so
        // the bullet doesn't dangle a "fix:" with nothing after it.
        let help = v.get("help").and_then(|s| s.as_str()).unwrap_or("");
        if help.is_empty() {
            lines.push(format!("- [{code} {axis}] line {line}: {message}"));
        } else {
            lines.push(format!(
                "- [{code} {axis}] line {line}: {message}\n  fix: {help}"
            ));
        }
    }

    // Render the axis breakdown as "2 placement, 1 values" so the
    // agent sees the shape of the failure at a glance, even when
    // the count is large enough to elide most lines.
    let breakdown_str = if axis_breakdown.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = axis_breakdown
            .iter()
            .map(|(axis, n)| format!("{n} {axis}"))
            .collect();
        format!(" ({})", parts.join(", "))
    };

    let header = if count == 1 {
        format!(
            "Basis flagged 1 architectural violation in the file you just wrote{breakdown_str}:"
        )
    } else {
        format!(
            "Basis flagged {count} architectural violations in the file you just wrote{breakdown_str}:"
        )
    };

    let mut summary = header;
    summary.push('\n');
    summary.push_str(&lines.join("\n"));
    if (count as usize) > lines.len() {
        let extra = count as usize - lines.len();
        summary.push_str(&format!("\n  …and {extra} more"));
    }
    summary.push_str(
        "\n\nThese are Basis errors, not advisory. Apply the fix shown under each violation, \
         or run `basis check` to see the full list with full diagnostics.",
    );
    Some(BasisReport {
        count,
        summary,
        axis_breakdown,
    })
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deviations_to_impulse_zero_is_zero() {
        // Recording a zero impulse would be a no-op, so the function
        // short-circuits — verifies the contract callers depend on
        // when deciding whether to even touch the governor.
        assert_eq!(deviations_to_impulse(0), 0.0);
    }

    #[test]
    fn deviations_to_impulse_single_is_per_deviation() {
        assert!((deviations_to_impulse(1) - PER_DEVIATION).abs() < 1e-9);
    }

    #[test]
    fn deviations_to_impulse_is_linear_below_cap() {
        // Three deviations → 1.5, halfway between zero and the cap.
        assert!((deviations_to_impulse(3) - 1.5).abs() < 1e-9);
    }

    #[test]
    fn deviations_to_impulse_clamps_at_max() {
        // Six deviations exactly hits the cap; ten must not exceed it.
        assert!((deviations_to_impulse(6) - MAX_IMPULSE).abs() < 1e-9);
        assert!((deviations_to_impulse(10) - MAX_IMPULSE).abs() < 1e-9);
        assert!((deviations_to_impulse(1000) - MAX_IMPULSE).abs() < 1e-9);
    }

    #[test]
    fn looks_like_source_file_accepts_common_extensions() {
        for path in [
            "/foo/bar.rs",
            "src/main.py",
            "C:\\proj\\app.ts",
            "Component.tsx",
            "main.go",
            "Main.java",
            "x.cpp",
        ] {
            assert!(looks_like_source_file(path), "should accept {path}");
        }
    }

    #[test]
    fn looks_like_source_file_is_case_insensitive() {
        // Some Windows tools shout the extension; we should not let
        // capitalisation flip a real source file into the skip path.
        assert!(looks_like_source_file("README.RS"));
        assert!(looks_like_source_file("Foo.Py"));
    }

    #[test]
    fn looks_like_source_file_rejects_non_source() {
        for path in [
            "/foo/CLAUDE.md",
            "Cargo.lock",
            "config.json",
            "image.png",
            "no_extension",
            "",
        ] {
            assert!(!looks_like_source_file(path), "should reject {path}");
        }
    }

    #[test]
    fn idiom_deviation_count_skips_non_source_without_spawning() {
        // Even if FEEDBACK_IDIOM_BIN is unset, this must return None
        // without ever attempting the subprocess — the extension
        // pre-filter runs first. We can't mechanically prove "no
        // process was spawned" from inside the test, but if the
        // pre-filter were broken this would hang or error on Windows
        // CI where idiom-cli isn't installed.
        assert_eq!(idiom_deviation_count("/foo/CLAUDE.md"), None);
        assert_eq!(idiom_deviation_count("Cargo.toml"), None);
    }

    #[test]
    fn parse_idiom_report_clean_file_has_empty_summary() {
        // Zero-deviation files round-trip to a `count: 0` report with
        // an empty summary string. The hook binary uses the empty
        // summary as a signal that nothing needs to be persisted as
        // pending context.
        let json = r#"{"deviations": [], "total": 0}"#;
        let report = parse_idiom_report(json).expect("clean file should parse");
        assert_eq!(report.count, 0);
        assert!(report.summary.is_empty());
    }

    #[test]
    fn parse_idiom_report_single_deviation_uses_singular_header() {
        // The header swaps between "1 deviation" and "N deviations"
        // depending on count. Verify the singular path so a future
        // i18n-style refactor doesn't quietly regress to "1 deviations".
        let json = r#"{
            "deviations": [
                {"code": "I001", "file": "x.rs", "line": 12, "name": "do_thing",
                 "role": "Function", "expected": "handle_*",
                 "message": "function name 'do_thing' does not match local convention: prefix 'handle_'"}
            ],
            "total": 1
        }"#;
        let report = parse_idiom_report(json).expect("should parse");
        assert_eq!(report.count, 1);
        assert!(report.summary.starts_with("Idiom flagged 1 naming convention deviation "));
        assert!(report.summary.contains("[I001] line 12"));
        assert!(report.summary.contains("'handle_'"));
        assert!(!report.summary.contains("…and"));
    }

    #[test]
    fn parse_idiom_report_truncates_long_lists() {
        // When more deviations exist than the summary cap, the
        // summary must list the first MAX_DEVIATIONS_IN_SUMMARY and
        // append a "…and N more" tail so the agent knows there is
        // hidden tail it can dig into via the `idiom_check` MCP tool.
        let mut entries = String::new();
        for i in 0..7 {
            if i > 0 {
                entries.push(',');
            }
            entries.push_str(&format!(
                r#"{{"code":"I001","file":"x.rs","line":{i},"name":"n{i}","role":"Function","expected":"e","message":"m{i}"}}"#
            ));
        }
        let json = format!(r#"{{"deviations":[{entries}],"total":7}}"#);
        let report = parse_idiom_report(&json).expect("should parse");
        assert_eq!(report.count, 7);
        // Three entries kept + tail line.
        let bullet_count = report.summary.matches("- [I001]").count();
        assert_eq!(bullet_count, MAX_DEVIATIONS_IN_SUMMARY);
        assert!(report.summary.contains("…and 4 more"));
    }

    #[test]
    fn parse_idiom_report_returns_none_on_malformed_json() {
        // Default-allow contract: garbage in → None out, never panic.
        assert!(parse_idiom_report("not json").is_none());
        assert!(parse_idiom_report("{}").is_none()); // missing `total`
    }

    // ── Basis-side parser tests ───────────────────────────────

    fn basis_violation(code: &str, axis: &str, file: &str, line: u64, message: &str) -> String {
        format!(
            r#"{{"code":"{code}","axis":"{axis}","file":"{file}","line":{line},"message":"{message}","identity":"x","details":{{"type":"Placement","from_layer":"a","to_layer":"b","module":"m"}}}}"#
        )
    }

    /// Like `basis_violation` but includes the `help` field. Used by
    /// the test that proves the actionable fix line surfaces in the
    /// rendered summary. The help field on `UnifiedViolation` is the
    /// per-violation prescription that real basis-cli builds emit;
    /// the unadorned helper above tests the backward-compatible path
    /// for older JSON that lacks the field.
    fn basis_violation_with_help(
        code: &str,
        axis: &str,
        file: &str,
        line: u64,
        message: &str,
        help: &str,
    ) -> String {
        format!(
            r#"{{"code":"{code}","axis":"{axis}","file":"{file}","line":{line},"message":"{message}","identity":"x","help":"{help}","details":{{"type":"Placement","from_layer":"a","to_layer":"b","module":"m"}}}}"#
        )
    }

    fn basis_check_json(violations: &[String]) -> String {
        format!(
            r#"{{"version":"1.0","timestamp":"t","spec_path":"basis.yaml","target_path":".","axes_checked":["placement"],"summary":{{"total":{},"by_axis":{{}}}},"violations":[{}]}}"#,
            violations.len(),
            violations.join(",")
        )
    }

    #[test]
    fn parse_basis_report_clean_repo_has_empty_summary() {
        // Zero violations everywhere → zero count, empty summary,
        // empty axis breakdown. The hook uses the empty summary to
        // decide not to write a pending field.
        let json = basis_check_json(&[]);
        let report = parse_basis_report(&json, "/whatever/foo.rs").expect("clean parses");
        assert_eq!(report.count, 0);
        assert!(report.summary.is_empty());
        assert!(report.axis_breakdown.is_empty());
    }

    #[test]
    fn parse_basis_report_filters_to_edited_file() {
        // Two violations exist in the crate, but only one is in
        // the file the agent just edited. The report must surface
        // only that one — pre-existing debt in unrelated files is
        // not the agent's fault for THIS edit.
        let v1 = basis_violation("B001", "placement", "src/edited.rs", 12, "bad import");
        let v2 = basis_violation("B002", "values", "src/other.rs", 3, "raw primitive");
        let json = basis_check_json(&[v1, v2]);
        let report = parse_basis_report(&json, "/abs/path/src/edited.rs")
            .expect("should parse");
        assert_eq!(report.count, 1);
        assert!(report.summary.contains("[B001 placement] line 12"));
        assert!(!report.summary.contains("B002"));
        assert_eq!(report.axis_breakdown.get("placement"), Some(&1));
        assert_eq!(report.axis_breakdown.get("values"), None);
    }

    #[test]
    fn parse_basis_report_surfaces_help_text_under_each_violation() {
        // The whole point of plumbing `help` through the JSON: when a
        // basis-cli build emits the per-violation fix line, the
        // rendered summary must include it as a sub-line under each
        // bullet. Without this assertion the agent only ever sees
        // *what* broke, never *how to fix it* — even though basis-cli
        // already did the thinking.
        let v = basis_violation_with_help(
            "B001",
            "placement",
            "src/edited.rs",
            12,
            "import crosses boundary",
            "move this code into the 'logic' layer, or add 'logic' to depends_on in basis.yaml",
        );
        let json = basis_check_json(&[v]);
        let report = parse_basis_report(&json, "src/edited.rs").expect("should parse");
        assert_eq!(report.count, 1);
        assert!(
            report.summary.contains("[B001 placement] line 12: import crosses boundary"),
            "violation line missing, got: {}",
            report.summary
        );
        assert!(
            report.summary.contains(
                "  fix: move this code into the 'logic' layer, or add 'logic' to depends_on"
            ),
            "fix sub-line must appear indented under the bullet, got: {}",
            report.summary
        );
    }

    #[test]
    fn parse_basis_report_omits_fix_subline_when_help_field_missing() {
        // Backward compat: violations without a `help` field (older
        // basis-cli builds, hand-written test JSON) must NOT render
        // a dangling "fix:" with no content. The rendered bullet
        // looks identical to the pre-help-field shape, which is what
        // the existing render tests assert.
        let v = basis_violation("B001", "placement", "src/x.rs", 5, "msg");
        let json = basis_check_json(&[v]);
        let report = parse_basis_report(&json, "src/x.rs").expect("should parse");
        assert!(
            !report.summary.contains("fix:"),
            "no fix sub-line should appear when help is absent, got: {}",
            report.summary
        );
        assert!(report.summary.contains("[B001 placement] line 5: msg"));
    }

    #[test]
    fn parse_basis_report_axis_breakdown_appears_in_header() {
        // Mixed-axis report: header must show "(2 placement, 1 values)"
        // so the agent sees the shape before reading individual lines.
        let v1 = basis_violation("B001", "placement", "src/x.rs", 1, "p1");
        let v2 = basis_violation("B001", "placement", "src/x.rs", 2, "p2");
        let v3 = basis_violation("B002", "values", "src/x.rs", 3, "v1");
        let json = basis_check_json(&[v1, v2, v3]);
        let report = parse_basis_report(&json, "src/x.rs").expect("should parse");
        assert_eq!(report.count, 3);
        assert!(
            report.summary.contains("(2 placement, 1 values)"),
            "header must include axis breakdown, got: {}",
            report.summary
        );
    }

    #[test]
    fn parse_basis_report_truncates_long_lists_with_tail_count() {
        // More violations than the per-summary cap → first N lines
        // shown, "…and M more" tail, full count in the header.
        let mut vs: Vec<String> = Vec::new();
        for i in 0..7 {
            vs.push(basis_violation(
                "B001",
                "placement",
                "src/x.rs",
                i,
                "violation",
            ));
        }
        let json = basis_check_json(&vs);
        let report = parse_basis_report(&json, "src/x.rs").expect("should parse");
        assert_eq!(report.count, 7);
        let bullet_count = report.summary.matches("- [B001").count();
        assert_eq!(bullet_count, MAX_DEVIATIONS_IN_SUMMARY);
        assert!(report.summary.contains("…and 4 more"));
    }

    #[test]
    fn parse_basis_report_handles_windows_path_separators() {
        // basis-cli running on Windows emits paths with backslashes
        // mixed in. The hook receives a forward-slash absolute path
        // from Claude Code. The matcher must bridge that without
        // false negatives, otherwise the whole Basis loop is dead
        // on Windows.
        let v = r#"{"code":"B001","axis":"placement","file":"src\\bin\\feedback_hook.rs","line":42,"message":"x","identity":"y","details":{"type":"Placement","from_layer":"a","to_layer":"b","module":"m"}}"#;
        let json = format!(
            r#"{{"version":"1.0","timestamp":"t","spec_path":"basis.yaml","target_path":".","axes_checked":["placement"],"summary":{{"total":1,"by_axis":{{}}}},"violations":[{v}]}}"#
        );
        let report =
            parse_basis_report(&json, "C:/Users/joevo/git-repo/feedback/src/bin/feedback_hook.rs")
                .expect("should parse");
        assert_eq!(
            report.count, 1,
            "Windows-style violation paths must still match the absolute hook input: {}",
            report.summary
        );
    }

    #[test]
    fn parse_basis_report_returns_none_on_malformed_json() {
        assert!(parse_basis_report("not json", "x.rs").is_none());
        // Missing `violations` field — defensive default-allow.
        assert!(parse_basis_report(r#"{"summary":{"total":0,"by_axis":{}}}"#, "x.rs").is_none());
    }

    #[test]
    fn find_crate_root_walks_up_to_cargo_and_basis_yaml() {
        // Plant a fake crate layout: tempdir/Cargo.toml, basis.yaml,
        // and src/lib.rs nested under it. Verify find_crate_root
        // walks up from the leaf and stops at the conjunction.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"").unwrap();
        std::fs::write(root.join("basis.yaml"), "governance:\n  version: \"1.0\"").unwrap();
        let src = root.join("src");
        std::fs::create_dir(&src).unwrap();
        let leaf = src.join("lib.rs");
        std::fs::write(&leaf, "// empty").unwrap();
        let found = find_crate_root(leaf.to_str().unwrap()).expect("should find");
        assert_eq!(
            found.canonicalize().unwrap(),
            root.canonicalize().unwrap(),
            "must find the conjunction root"
        );
    }

    #[test]
    fn find_crate_root_returns_none_when_no_basis_yaml_above() {
        // Cargo.toml present but no basis.yaml anywhere up the tree
        // → silent skip, the file is in an ungoverned crate.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"").unwrap();
        // No basis.yaml.
        let leaf = root.join("foo.rs");
        std::fs::write(&leaf, "// empty").unwrap();
        assert!(find_crate_root(leaf.to_str().unwrap()).is_none());
    }

    #[test]
    fn basis_check_report_returns_none_when_binary_missing() {
        // Missing binary → default-allow None, no panic. Same
        // contract as `idiom_deviation_count_returns_none_when_binary_missing`.
        // Plant a fake crate root so find_crate_root succeeds and
        // we actually reach the subprocess attempt.
        let dir = tempfile::tempdir().expect("tempdir");
        let root = dir.path();
        std::fs::write(root.join("Cargo.toml"), "[package]\nname = \"x\"").unwrap();
        std::fs::write(root.join("basis.yaml"), "governance:\n  version: \"1.0\"").unwrap();
        let leaf = root.join("foo.rs");
        std::fs::write(&leaf, "// empty").unwrap();

        let prior = std::env::var(BASIS_BIN_ENV).ok();
        unsafe {
            std::env::set_var(BASIS_BIN_ENV, "definitely_not_a_real_basis_binary_xyz");
        }
        let result = basis_check_report(leaf.to_str().unwrap());
        unsafe {
            match prior {
                Some(v) => std::env::set_var(BASIS_BIN_ENV, v),
                None => std::env::remove_var(BASIS_BIN_ENV),
            }
        }
        assert!(
            result.is_none(),
            "missing basis binary must default-allow, not panic"
        );
    }

    #[test]
    fn basis_check_report_skips_non_source_files() {
        // Same extension pre-filter as the Idiom side: don't even
        // attempt the subprocess for markdown/json/lockfiles.
        assert!(basis_check_report("/foo/CLAUDE.md").is_none());
        assert!(basis_check_report("/foo/Cargo.toml").is_none());
    }

    #[test]
    fn idiom_deviation_count_returns_none_when_binary_missing() {
        // Force a binary name that cannot exist on PATH. The default-
        // allow contract: missing binary → None, never panic.
        // SAFETY: tests run single-threaded for env var mutation here.
        let prior = std::env::var(IDIOM_BIN_ENV).ok();
        // SAFETY: setting an env var is unsafe in multi-threaded code,
        // but cargo test serializes per-test by default and this test
        // restores the prior value before returning.
        unsafe {
            std::env::set_var(
                IDIOM_BIN_ENV,
                "definitely_not_a_real_binary_name_xyz_abc_123",
            );
        }
        let result = idiom_deviation_count("/tmp/foo.rs");
        // Restore env var so other tests are not affected.
        unsafe {
            match prior {
                Some(v) => std::env::set_var(IDIOM_BIN_ENV, v),
                None => std::env::remove_var(IDIOM_BIN_ENV),
            }
        }
        assert_eq!(
            result, None,
            "missing binary must default-allow, not panic or block"
        );
    }
}
