//! `feedback-hook` — Claude Code PreToolUse hook binary.
//!
//! Reads a `HookInput` JSON document from stdin, loads the
//! per-session governor state from disk, runs `handle_pre_tool_use`,
//! persists the new state, and writes a `HookOutput` JSON document
//! to stdout.
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

use std::io::Read;
use std::path::PathBuf;
use std::process::ExitCode;

use feedback::governor::{Governor, PersistedState};
use feedback::hook_protocol::HookInput;
use feedback::hooks::handle_pre_tool_use;

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

/// The fallible inner body. Anything that returns an error here is
/// caught in `main` and converted to an empty allow.
fn run() -> Result<String, Box<dyn std::error::Error>> {
    let mut input_str = String::new();
    std::io::stdin().read_to_string(&mut input_str)?;

    // Parse once for the session_id; the full input is reparsed
    // inside `handle_pre_tool_use`. The double-parse cost is
    // negligible (~1 KB JSON) and lets us keep the library API
    // unchanged.
    let input: HookInput = serde_json::from_str(&input_str)?;
    let session_file = format!("{}.json", sanitize_session_id(&input.session_id));
    let dir = state_dir();
    std::fs::create_dir_all(&dir)?;
    let state_path = dir.join(&session_file);

    // Resume from disk if a snapshot exists. A corrupt or unreadable
    // file is treated as a fresh session — better to lose history
    // than to brick the agent on every call until someone manually
    // clears the file.
    let mut governor = if state_path.exists() {
        match std::fs::read(&state_path)
            .ok()
            .and_then(|bytes| serde_json::from_slice::<PersistedState>(&bytes).ok())
        {
            Some(state) => Governor::resume_from_defaults(state),
            None => Governor::from_defaults(),
        }
    } else {
        Governor::from_defaults()
    };

    let output_json = handle_pre_tool_use(&input_str, &mut governor)?;

    // Persist atomically: write to a sibling .tmp then rename. The
    // rename is atomic on the same filesystem, so a concurrent
    // reader either sees the old file or the new file, never a
    // partial write. Failures here are non-fatal — the hook output
    // is still returned to Claude Code.
    let snap = governor.snapshot();
    if let Ok(snap_bytes) = serde_json::to_vec_pretty(&snap) {
        let tmp_path = state_path.with_extension("json.tmp");
        if std::fs::write(&tmp_path, &snap_bytes).is_ok() {
            let _ = std::fs::rename(&tmp_path, &state_path);
        }
    }

    Ok(output_json)
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
