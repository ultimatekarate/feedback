[![CI](https://github.com/ultimatekarate/feedback/actions/workflows/ci.yml/badge.svg)](https://github.com/ultimatekarate/feedback/actions/workflows/ci.yml)


# Feedback

A real-time stability governor for agentic coding workflows. Feedback monitors AI agent sessions using Volterra integral pressure analysis to detect when an agent is spiraling — filling context, looping on errors, thrashing files — and intervenes before the session degrades.

## How it works

Feedback tracks five "orthogonal" (In anticipation of friendly fire: I am fully aware that the channels aren't actually orthogonal. Do you see a definition of an inner product here?) pressure channels, each measuring a distinct failure mode:

| Channel | What it measures | Half-life |
|---------|-----------------|-----------|
| **Context** | Token I/O payload accumulation | 6.1 min |
| **Latency** | Wall-clock time in Bash commands | 130s |
| **Error** | Tool failures and error patterns | 4.0 min |
| **Progress** | Time-proportional stall vs productive actions | 6.3 min |
| **Repetition** | Repeated access to the same file paths | 4.6 min |

Each channel accumulates impulses from tool calls and decays exponentially over time. When pressure crosses calibrated thresholds, the governor issues graduated interventions:

1. **Allow** — all channels below 60% of critical
2. **Warn** — any channel between 60–85% of critical (logged, not blocked)
3. **Deny** — any channel above 85% of critical, or system is spectrally unstable

A 5×5 Jacobian encodes cross-channel coupling (e.g., errors trigger read cascades that fill context). Spectral stability analysis on the Jacobian's eigenvalues detects emerging instability before any single channel saturates.

## Architecture

The codebase follows a three-layer **linguistic code model**:

- **Dictionary** (`channels.rs`, `decision.rs`, `hook_protocol.rs`) — Configuration, types, and protocol definitions. Pure data, no logic.
- **Laboratory** (`pressure_model.rs`, `dynamics.rs`, `monitor.rs`) — Coupling model, nonlinear dynamics, and spectral analysis. Pure functions, no IO.
- **Hands** (`governor.rs`, `hooks.rs`, `session.rs`) — Runtime governor, Claude Code hook integration, and session logging.

## Integration

Feedback integrates with [Claude Code](https://docs.anthropic.com/en/docs/claude-code) via `PreToolUse` hooks. When Claude Code invokes a tool:

1. The hook receives the tool name and input
2. `apply_impulses()` maps the tool call to channel-specific impulses
3. The governor evaluates system state and spectral stability
4. A verdict (allow/warn/deny) is returned to Claude Code

## Building

```bash
cargo build
cargo test
```

Requires the `volterra-stability` crate (local path dependency).

## Offline analysis

Python scripts for replaying and analyzing recorded sessions:

```bash
python analyze_session.py     # Replay a JSONL session through the 5-channel model
python measure.py             # Extract raw observables from session data
python extract_session.py     # Extract tool sequences for debugging
```

## Calibration

Channel parameters (decay rates, critical thresholds, impulse magnitudes) were calibrated from Phase 0 honeypot sessions — controlled coding tasks with known difficulty profiles. See `PHASE0.md` for baseline data and `FEEDBACK-MODEL.md` for the full model specification.

## Documentation

- **[FEEDBACK-MODEL.md](FEEDBACK-MODEL.md)** — Complete model specification: channels, thresholds, coupling, and calibration data
- **[theory.md](theory.md)** — Mathematical foundation: Volterra integral equations and decay dynamics
- **[PHASE0.md](PHASE0.md)** — Phase 0 calibration results from honeypot sessions
