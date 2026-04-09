# Feedback — Agentic Workflow Stability Governor

Feedback is a real-time stability governor for agentic coding workflows. It uses Volterra integral pressure analysis to detect when an AI coding agent is spiraling — filling context, looping on errors, thrashing files — and intervenes before the session degrades.

## Model Architecture

Five pressure channels, each backed by a **DecayingIntegral**:

```
x(t+dt) = x(t) * exp(-lambda * dt) + impulse
```

Each channel has a unique impulse source. No single event fires on all channels simultaneously.

| Channel | Source | Lambda | Half-life | Critical | What it detects |
|---|---|---|---|---|---|
| **Context** | Token load from tool I/O | 0.0019 | 6.1 min | 50,000 tokens | Context window filling up |
| **Latency** | Wall-clock Bash elapsed time | 0.0053 | 130s | 350s | Catastrophic build/test times |
| **Error** | Tool failures + parsed patterns | 0.0029 | 4.0 min | 6.0 | Sustained failure loops |
| **Progress** | Time-proportional stall - productive actions | 0.0018 | 6.3 min | 55.0 | Agent producing nothing |
| **Repetition** | 2nd+ access to same file path | 0.0025 | 4.6 min | 15.0 | Thrashing same files |

**Cost** is tracked as a standalone dollar counter (`cumulative_cost: f64`), not a pressure channel. See [Why Cost Was Removed](#why-cost-was-removed).

## Impulse Mapping

Impulses are **measured physical quantities** — token counts extracted from real session JONLs via `measure.py`, not invented weights.

### Context impulses (tokens)

| Tool | Median tokens | Source |
|---|---|---|
| Read | 1,175 | Cross-session median |
| Edit | 200 | Cross-session median |
| Write | 2,400 | Cross-session median |
| Bash | 565 | Cross-session median |
| Grep/Glob | 100 | Cross-session median |
| Agent | 4,400 | Measured (n=1) |
| Other | 200 | Conservative default |

### Latency impulses (seconds)

Actual elapsed time from Bash `tool_use` timestamp to `tool_result` timestamp. No estimation.

### Error impulses (count)

- `is_error=True` on tool result: **1.0** (except ExitPlanMode — user steering, not failure)
- Parsed failure pattern in Bash output: **0.5** (`FAILED`, `error[E`, `panicked`, `Traceback`, etc.)

### Progress impulses

- Base stall: **+0.1 per second** between tool calls (time-proportional)
- Stall gap cap: **600s** (10 min). Overnight gaps don't inflate progress.
- Write: **-5.0** (large productive action)
- Edit: **-3.0** (productive action)
- Bash with test/build command: **-1.0**
- TodoWrite: **-0.5**
- ExitPlanMode (approved plan): **-5.0**
- ExitPlanMode (review round): **-2.0**

### Repetition impulses

- Read revisit (2nd+ access to same file): **1.0**
- Edit revisit: **0.5**
- Files under `.claude/plans/` are **suppressed** — plan docs are designed for iterative review.

### Cost metric

- Per-call tax: **$0.005**
- Agent surcharge: **$0.05**
- Tracked as cumulative counter, not a DecayingIntegral.

## Criticals Derivation

Criticals are set so healthy sessions peak at ~30% and struggling sessions cross 60% WARN.

| Channel | Critical | Derivation |
|---|---|---|
| Context | 50,000 | 4x healthy peak (~12K tokens). Rapid context churn. |
| Latency | 350s | Sustained 2-min Bash calls. Catastrophic builds. |
| Error | 6.0 | 6 active errors in decay window = sustained failure loop. |
| Progress | 55.0 | Pure stall steady-state: 0.1/0.0018 = 55.6. Agent produces nothing. |
| Repetition | 15.0 | 15 active re-accesses = thrashing same files every few seconds. |

## Governor Decision Logic

Three levels of intervention:

1. **Allow** — all channels below 60% of critical, system spectrally stable.
2. **Warn** — any channel above 60% of critical. Logged, not blocked.
3. **Deny** — any channel above 85% of critical, OR system spectrally unstable. Tool call blocked with channel-specific guidance.

Spectral stability is assessed via the Jacobian eigenvalues at the current operating point. If the spectral gap drops below 1e-6, the system is at the instability boundary.

## Jacobian and Coupling Model

The 5x5 Jacobian encodes how channels interact:

**Diagonal entries**: Self-decay minus self-coupling.
```
J[i,i] = -lambda_i + u_i * d_sigma/dI_i
```

**Cross-coupling** (the hypothesis — to be fitted from empirical data):

| Coupling | Direction | Coefficient | Mechanism |
|---|---|---|---|
| Error -> Context | Destabilizing | 0.02 * lambda_ctx | Errors trigger read cascades |
| Error -> Progress | Destabilizing | 0.01 * lambda_prg | Errors stall forward progress |
| Error -> Repetition | Destabilizing | 0.02 * lambda_rep | Errors cause re-reading files |
| Repetition -> Context | Destabilizing | 0.01 * lambda_ctx | Re-reading fills context |
| Context -> Error | Destabilizing | 0.005 * lambda_err | High context degrades error fixing |
| Latency -> Repetition | Weak | 0.005 * lambda_rep | High latency may cause retries |

The Context -> Error coupling is the critical one. If positive and strong enough, it creates the death spiral: errors fill context, full context causes more errors.

## Stress Weights

Composite stress for overall health assessment:

| Channel | Weight | Rationale |
|---|---|---|
| Context | 0.28 | Non-renewable resource |
| Latency | 0.11 | Informational |
| Error | 0.33 | Highest — errors drive cascades |
| Progress | 0.17 | Stalls waste tokens |
| Repetition | 0.11 | Indicator of thrashing |

## Calibration Data

### Phase 0: Three honeypot sessions

| Session | Calls | Duration | Outcome | Peak channels |
|---|---|---|---|---|
| v1 bug-fix | 38 | 5 min | Healthy | context 22%, repetition 57% |
| v2 bug-fix | 45 | 6 min | Healthy | context 25%, repetition 51% |
| v3 feature-on-bugs | 47 | 24 min | Struggling | latency 88%, progress 81% |

**Calibration invariant**: v1/v2 peak below WARN (60%). v3 crosses WARN on exactly the channels that distinguish it (latency, progress). v3 triggers 1 denial (latency=88% at 15:27).

### Phalanx sessions

| Session | Calls | Duration | Denials | Key signal |
|---|---|---|---|---|
| Crypto audit (cfb60f66) | 134 | 44 min | 9 | Error cascade drove context+repetition |
| Community feature (f756d750) | 265 | 20.5h | Varies by phase | Multi-day with overnight gap |
| Phase 6 (f00b1e95) | 144 | 44 min | 4 | Repetition on corroboration.rs (18 edits) |

## Why Cost Was Removed

Cost was originally the sixth pressure channel (lambda=0.00003, half-life 6.4 hours). It was removed for three reasons:

1. **Redundancy**. Every tool call fires cost AND at least one other channel. Cost going up is implied by any other signal going up. It has zero independent diagnostic value.

2. **Spectral distortion**. Cost's near-zero lambda produces a near-zero eigenvalue in the Jacobian. This distorts the spectral gap analysis — the gap between the largest eigenvalue and the next one becomes dominated by cost's artificial near-zero value rather than reflecting genuine coupling dynamics.

3. **Monotonic saturation**. Once cost crosses 85%, it fires a denial on every subsequent snapshot forever (6.4-hour half-life means it never decays within a session). In Phase 5 of the community feature, 34 of 43 denials were cost — drowning out all other signals.

Cost is now a standalone `f64` accumulator in the Governor, visible in reports as a dollar amount.

## Refinements History

Each refinement was driven by empirical analysis of real sessions:

| Issue | Root cause | Fix |
|---|---|---|
| Progress 10,447% from overnight gap | 19-hour gap deposited ~5,700 stall units | MAX_STALL_GAP = 600s cap |
| 34/43 denials from cost | Cost monotonic, never decays | Made display-only, then removed from Jacobian |
| Plan doc inflating repetition | Plan doc accessed 22 times during 4 review rounds | Suppress repetition for `.claude/plans/` paths |
| Planning treated as waste by progress | Progress accumulated stall during planning rounds | Planning credits: ExitPlanMode = -2.0/-5.0, TodoWrite = -0.5 |
| ExitPlanMode rejections counted as errors | Claude Code marks user plan rejections as `is_error=True` | Exclude ExitPlanMode from error channel |
| Equilibrium test took 100K steps | Cost's 6.4-hour half-life required ~27 hours simulation | Removed cost; reduced to 5K steps |

## Open Questions

- **Coupling coefficients are hypotheses.** The cross-coupling values (0.02, 0.01, 0.005) are initial estimates. They need to be fitted from a larger corpus of session data.
- **Progress credits are hand-tuned.** The -2.0/-5.0 planning credits and -3.0/-5.0 edit/write credits need empirical validation.
- **PostToolUse hook not live.** Error and latency channels currently only fire in offline replay (`analyze_session.py`). The live governor (`hooks.rs`) only handles PreToolUse — it can't see tool results yet.
- **Repetition threshold for focused work.** 18 edits to a single file during a feature build may be intensive focused work, not pathological thrashing. The repetition channel may need a higher critical or a decay-based "burstiness" filter.

## File Map

| File | Layer | Role |
|---|---|---|
| `src/channels.rs` | Dictionary | Channel enum, DIM=5, thresholds, config |
| `src/pressure_model.rs` | Laboratory | Jacobian coupling model (5x5) |
| `src/dynamics.rs` | Laboratory | Full nonlinear RHS for simulation |
| `src/monitor.rs` | Laboratory | Spectral analysis + decision logic |
| `src/governor.rs` | Hands | Live governor with integral bank + cost counter |
| `src/hooks.rs` | Hands | Claude Code hook integration, impulse mapping |
| `src/session.rs` | Hands | Session logging and post-hoc analysis |
| `analyze_session.py` | Offline | JSONL replay through the 5-channel model |
| `measure.py` | Offline | Extract raw observables from session JONLs |
| `extract_session.py` | Offline | Extract tool sequences for debugging |
