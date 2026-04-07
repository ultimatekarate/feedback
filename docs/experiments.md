| | v1 Bug-fix | v2 Bug-fix | v3 Feature-on-bugs |
|---|---|---|---|
| **Task** | Fix 16 failing tests | Fix 16 failing tests (3x code) | Add pause/resume feature (bugs hidden) |
| **Duration** | 5 min | 6 min | 24 min |
| **Tool calls** | 38 | 45 | 47 |
| **Tool mix** | Edit=24, Read=7, Bash=6 | Edit=23, Read=10 | Bash=16, Read=10, Edit=7 |
| **Errors detected** | 5 | 2 | 8 |
| | | | |
| **Context** | 22.1% | 29.5% | 25.4% |
| **Cost** | 25.4% | 27.4% | 20.6% |
| **Latency** | 38.0% | 21.3% | **68.0% ⚠ WARN** |
| **Error** | 34.8% | 20.3% | 38.9% |
| **Progress** | 0.0% | 12.7% | **69.0% ⚠ WARN** |
| **Repetition** | 38.9% | 36.9% | 22.9% |
| | | | |
| **Governor warnings** | 0 | 0 | 5 |
| **Governor denials** | 0 | 0 | 0 |

The healthy sessions are uniformly flat across all six channels at 20–39%. No channel dominates. Progress sits at 0% for v1 because 24 edits in 5 minutes outpaces the stall accumulation — the integral goes negative and the display floors at zero.

The struggling session has a completely different shape. Context, cost, and repetition stay in the healthy band (20–25%). The pathology shows up exclusively in progress and latency — the two channels that measure "time passing without productive output" and "repeated Bash execution." Both cross the 60% WARN threshold. Error is elevated at 39% but doesn't reach WARN because there were only 8 detected failures.

The v3 timeline shows the cascade developing: progress crosses WARN at minute 16, oscillates, then converges with latency at minute 24 when both hit 68–69% simultaneously with error spiking to 39%. Three channels rising together in the final minute — that's the intervention signal.