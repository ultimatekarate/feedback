Here's the full model.

---

## The Continuous System

The state vector is **x** ∈ ℝ⁶, where each component is a decaying integral — one per pressure channel:

```
x = [Context, Cost, Latency, Error, Progress, Repetition]
```

Each integral evolves according to a Volterra second-kind equation:

```
dxᵢ/dt = fᵢ(x) − λᵢ xᵢ
```

The term **−λᵢ xᵢ** is exponential decay. It pulls the integral toward zero with a characteristic timescale of **ln(2)/λᵢ** (the half-life). This is the "forgetting" term — old pressure bleeds off.

The term **fᵢ(x)** is the effective impulse rate. It's where the coupling lives. If fᵢ depended only on external inputs and not on x, the system would be six independent decaying integrals — no interaction, no instability possible. The coupling comes from fᵢ being a function of the *other* channels.

## The Decay Parameters (λ)

Each λ encodes how far back the integral looks — its memory horizon:

| Channel | λ | Half-life | Physical meaning |
|---------|---|-----------|-----------------|
| Context | 0.00231 | 5 min | Tokens consumed stay relevant for minutes |
| Cost | 0.00116 | 10 min | Spend fades as a sunk cost over the session |
| Latency | 0.0077 | 90s | Recent slow responses matter for a few cycles |
| Error | 0.00578 | 2 min | An error persists through its fix cycle |
| Progress | 0.00231 | 5 min | Stalls build slowly and persist |
| Repetition | 0.00578 | 2 min | Thrashing pattern spans a fix cycle |

## The Scaler Function σ

Before fᵢ can couple channels together, each channel's raw integral value is mapped to a dimensionless [0, 1] signal through a **scaler**:

```
σ(xᵢ) = max(0, 1 − xᵢ/cᵢ)
```

where cᵢ is the **critical threshold** for channel i. This is a linear ramp: σ = 1 when the channel is idle (xᵢ = 0), σ = 0 when the channel is saturated (xᵢ ≥ cᵢ). The derivative is:

```
dσ/dxᵢ = −1/cᵢ  (when xᵢ < cᵢ)
        = 0       (when xᵢ ≥ cᵢ)
```

The saturation is important: once a channel hits critical, it stops feeding the coupling. This prevents the system from diverging to infinity — it creates a natural basin of attraction.

## The Coupling (fᵢ)

This is the behavioral hypothesis encoded in `dynamics.rs`. The effective impulse rate for each channel depends on the state of other channels:

**Context** (index 0):
```
f_ctx = u_ctx · σ(x_ctx)           base rate, self-attenuating
      + u_ctx · 0.5 · ε(x_err)     errors trigger read cascades
      + u_ctx · 0.3 · ρ(x_rep)     repetition drives re-reads
```

where ε(x) = min(x/c_err, 1) is the normalized error stress and ρ(x) = min(x/c_rep, 1) is the normalized repetition stress. When errors are high, the agent reads more files trying to diagnose. When repetition is high, those reads pile into context.

**Cost** (index 1):
```
f_cst = u_cst + u_cst · 0.1 · min(x_ctx/c_ctx, 1)
```

Cost accumulates at a base rate, weakly amplified by context pressure. High context means more tool calls, which means more spend.

**Latency** (index 2):
```
f_lat = u_lat
```

Latency is uncoupled — it only responds to external Bash call timing. This is deliberate: latency is an observable, not a behavioral consequence.

**Error** (index 3):
```
f_err = u_err · σ(x_err)           base rate, self-attenuating
      + u_err · 0.1 · min(x_ctx/c_ctx, 1)   context degrades error resolution
```

This is the **critical coupling**: `Context → Error`. The hypothesis is that as the context window fills, the agent's ability to fix errors degrades. It can't hold enough state in working memory. Errors accumulate. This is positive (destabilizing) — it creates a feedback loop.

**Progress** (index 4):
```
f_prg = u_prg · (1 + 0.3 · ε(x_err))
```

Stall pressure grows at a base rate, amplified by errors. When there are active errors, the agent is less likely to make forward progress.

**Repetition** (index 5):
```
f_rep = u_rep + u_rep · 0.4 · ε(x_err)
```

Re-reading frequency increases when errors are active — the agent keeps going back to the same files trying to find the bug.

## The Jacobian (Linearized Coupling)

The Jacobian `J` linearizes the system around an operating point. Its entries determine stability:

```
J[i,i] = −λᵢ + uᵢ · dσ/dxᵢ        (diagonal: self-decay + self-coupling)
J[i,j] = coupling coefficient       (off-diagonal: cross-channel coupling)
```

The diagonal is always negative at idle (−λᵢ + negative ≈ strongly negative). This means each channel, in isolation, is stable — perturbations decay.

The off-diagonal entries encode the coupling hypothesis:

```
J[ctx, err] = +0.02 · λ_ctx    Error → Context (destabilizing)
J[prg, err] = +0.01 · λ_prg    Error → Progress (destabilizing)
J[rep, err] = +0.02 · λ_rep    Error → Repetition (destabilizing)
J[ctx, rep] = +0.01 · λ_ctx    Repetition → Context (destabilizing)
J[err, ctx] = +0.005 · λ_err   Context → Error (THE critical coupling)
J[rep, lat] = +0.005 · λ_rep   Latency → Repetition (weak)
```

All positive. All destabilizing. Each one says: "when channel j is elevated, channel i gets worse."

## The Death Spiral

The coupling creates a feedback loop:

```
Error ↑  →  Context ↑  (agent reads more, trying to diagnose)
Context ↑  →  Error ↑  (context window fills, agent can't fix errors)
Error ↑  →  Repetition ↑  (agent re-reads same files)
Repetition ↑  →  Context ↑  (re-reads fill context faster)
```

This is a positive feedback cycle. Whether it's stable or unstable depends on whether the **diagonal decay** (negative, stabilizing) is stronger than the **off-diagonal coupling** (positive, destabilizing).

## Spectral Stability Analysis

The eigenvalues of J determine stability. If every eigenvalue has a negative real part, the system is stable — perturbations decay. The **spectral gap** γ₁ is the real part of the least-negative eigenvalue. It measures how much margin the system has before instability:

- **γ₁ ≪ 0**: strongly stable. Perturbations decay fast. Healthy session.
- **γ₁ → 0**: the stability margin is narrowing. The system is approaching the bifurcation point.
- **γ₁ > 0**: unstable. One eigenvalue has crossed into the right half-plane. Perturbations grow. Death spiral.

The governor monitors γ₁ in real time. When it narrows below a threshold, the governor intervenes — even if no single channel has crossed its individual threshold. This is the spectral signal: the channels look individually fine, but the coupling between them is about to go critical.

## The Discrete Hook Implementation

The continuous model describes the dynamics. The actual system is event-driven: a hook fires on each tool call, records impulses, and evaluates.

Each tool call maps to impulses on specific channels (the orthogonal mapping):

```
Read   →  Context=500, Repetition=1.0 (revisit only)
Edit   →  Context=50,  Progress=-3.0, Repetition=0.5 (revisit only)
Write  →  Context=100, Progress=-5.0
Bash   →  Context=200, Latency=2.0-5.0, Progress=-1.0 (test only)
Agent  →  Context=2000, Cost=0.05
Every  →  Cost=0.005
Time   →  Progress += 0.1 · dt (between calls)
Error  →  Error=1.0 (is_error), Error=0.5 (parsed failure patterns)
```

Between hook calls, the `DecayingIntegral` applies exponential decay:

```
x(t + dt) = x(t) · exp(−λ · dt) + impulse
```

This is the exact solution to `dx/dt = −λx` between impulse events. The continuous dynamics (coupling) are captured separately in the Jacobian evaluation. The hook-level integrals track the raw pressure; the spectral analysis evaluates whether the coupling is driving the system toward instability.

## What the Governor Decides

At each tool call:

1. **Record impulses** — update the integral bank
2. **Build the Jacobian** at the current operating point
3. **Compute eigenvalues** — check spectral stability
4. **Check thresholds** — per-channel warn (60%) and deny (85%)
5. **Decide**: Allow, Warn, or Deny

A denial happens when either:
- Any single channel exceeds 85% of its critical threshold
- The spectral gap γ₁ approaches zero (system approaching instability)
- The system is spectrally unstable (γ₁ > 0)

The denial includes channel-specific guidance: "you've re-read the same files repeatedly — try a different approach" or "no forward progress — break the task into smaller steps."