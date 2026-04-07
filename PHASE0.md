# Honeypot — Phase 0: Baseline

## Purpose

Honeypot is a deliberately pathological Python codebase designed to stress-test [Feedback](../feedback/), the agentic workflow stability governor. Feedback monitors six pressure channels — context, cost, latency, error, progress, repetition — using Volterra integral dynamics and spectral stability analysis. It needs a codebase that triggers cascading failures to validate its intervention logic.

The problem: Feedback's sibling tools, Basis and Idiom, prevent exactly the architectural conditions Feedback is designed to detect. A well-governed codebase doesn't produce death spirals. So we built an ungoverned one.

## Scale History

| Version | Lines | Files | Tests | Failures | Result |
|---------|-------|-------|-------|----------|--------|
| v1 (c8bd042) | 1,755 | 10 source + 4 test | 56 | 16 | Claude fixed all 16 in 5 minutes, 38 tool calls. Peak repetition 17.6%. No pressure thresholds crossed. |
| v2 (7722eb4) | 5,183 | 10 source + 4 test | 56 | 16 | Claude fixed all 16 in 6 minutes, 45 tool calls. Peak repetition 14.7%. No pressure thresholds crossed. |

### v3: Feature addition on buggy foundation

**Task:** "Add pipeline pausing and resumption" — open-form feature work with no tests to guide. The 16 bugs remained unfixed. The agent was not told about them.

| Metric | v1 Bug-fix | v2 Bug-fix | v3 Feature |
|--------|-----------|-----------|------------|
| Duration | 5 min | 6 min | 24 min |
| Tool calls | 38 | 45 | 47 |
| Errors | 3 | 2 | 2 |
| Peak context | 6.8% | 8.9% | 4.0% |
| Peak latency | 2.0% | 1.0% | 7.7% |
| Peak error | 0.9% | 0.1% | 2.8% |
| Peak progress | 1.4% | 2.2% | 2.8% |
| Peak repetition | 17.6% | 14.7% | 15.1% |
| Tool mix | Edit=24, Read=7, Bash=6 | Edit-heavy (similar) | Bash=16, Read=10, Edit=7 |

**Observations:**

1. **Duration inflated 5x, tool count flat.** The agent spent most of its time thinking between actions, not executing tools. Thinking time is invisible to the pressure model.

2. **Tool mix inverted.** Bug-fixing is Edit-dominant (the agent knows what to fix). Feature work is Bash/Read-dominant (the agent runs things, reads output, tries to understand behavior). This is a qualitative signal the pressure model doesn't weight.

3. **Terminal pressure spike.** The final 30-second window shows latency=7.7%, error=2.8%, progress=2.8% — all channels climbing simultaneously. The cascade was starting but the session ended before it developed.

4. **Repetition hotspot shifted.** Bug-fix sessions hit backend.py hardest (9-10 accesses). Feature session concentrated on scheduler.py (8 accesses) — the file with the most cross-file coupling bugs.

**Conclusion:** 5,183 lines is still comfortably within Claude's working memory. The agent doesn't get lost — it just takes longer because feature work requires design decisions. The tool-mix shift and duration inflation are real signals but never crossed pressure thresholds. The context scale hypothesis holds: cascading pressure requires a codebase large enough that the agent can't hold relevant state simultaneously — estimated 30,000-50,000+ lines with deep cross-file coupling.

## Baseline State (v2)

**56 tests. 40 pass. 16 fail.**

```
FAILED tests/test_pipeline.py::TestPipelineModel::test_add_stage
FAILED tests/test_pipeline.py::TestPipelineModel::test_stage_dependency_check
FAILED tests/test_pipeline.py::TestPipelineValidation::test_validate_cycle_detection
FAILED tests/test_pipeline.py::TestPipelineSerialization::test_config_roundtrip
FAILED tests/test_pipeline.py::TestResultAggregation::test_success_rate_empty
FAILED tests/test_pipeline.py::TestResultAggregation::test_serialize_datetime
FAILED tests/test_store.py::TestStorePersistence::test_flush_doesnt_clear_dirty
FAILED tests/test_store.py::TestStoreCaching::test_global_cache_stale
FAILED tests/test_store.py::TestStoreCaching::test_get_mutates_cache
FAILED tests/test_task.py::TestTaskLifecycle::test_retry_cycle
FAILED tests/test_task.py::TestTaskSerialization::test_roundtrip
FAILED tests/test_task.py::TestTaskSerialization::test_serialize_preserves_status
FAILED tests/test_validators.py::TestNormalization::test_normalize_mutates_input
FAILED tests/test_validators.py::TestDependencyValidation::test_cycle_detection
FAILED tests/test_validators.py::TestConfigMerge::test_deep_merge
FAILED tests/test_validators.py::TestRulesValidation::test_duplicate_names_across_stages
```

## Bug Topology

The 16 failures are not 16 independent bugs. They form dependency chains that force an agent to read across multiple files to locate root causes. This is the structural property that should generate measurable pressure.

### Cross-file dependency chains

| Failure | Files involved | Root cause location |
|---------|---------------|-------------------|
| `test_retry_cycle` | test_task.py → task.py → executor.py | task.py `fail()` method |
| `test_roundtrip` | test_task.py → task.py | task.py `deserialize()` |
| `test_serialize_preserves_status` | test_task.py → task.py | task.py `deserialize()` |
| `test_add_stage` | test_pipeline.py → pipeline.py | pipeline.py `add_stage()` |
| `test_stage_dependency_check` | test_pipeline.py → pipeline.py | pipeline.py `is_ready()` |
| `test_validate_cycle_detection` | test_pipeline.py → pipeline.py | pipeline.py `validate()` |
| `test_config_roundtrip` | test_pipeline.py → pipeline.py | pipeline.py `serialize()` key mismatch with `from_config()` |
| `test_success_rate_empty` | test_pipeline.py → result.py | result.py `success_rate()` |
| `test_serialize_datetime` | test_pipeline.py → result.py | result.py `serialize()` |
| `test_flush_doesnt_clear_dirty` | test_store.py → backend.py | backend.py `flush()` |
| `test_global_cache_stale` | test_store.py → backend.py | backend.py `put()` / global cache |
| `test_get_mutates_cache` | test_store.py → backend.py | backend.py `get()` |
| `test_normalize_mutates_input` | test_validators.py → schema.py | schema.py `normalize_task()` |
| `test_cycle_detection` | test_validators.py → schema.py | schema.py `validate_dependencies()` |
| `test_deep_merge` | test_validators.py → schema.py | schema.py `merge_configs()` |
| `test_duplicate_names_across_stages` | test_validators.py → rules.py | rules.py `ensure_no_duplicate_names()` |

### Cascade traps

These are the structural properties designed to amplify agent pressure:

**Partial fix propagation.** Several bugs share a root cause pattern (e.g., `copy()` vs `deepcopy()` appears in both `schema.py` and `backend.py`). An agent that fixes one instance may not find the other. Re-running tests after a partial fix yields different failures, triggering re-reads of already-visited files.

**Semantic misdirection.** Functions named `validate` perform IO. Functions documented as returning copies mutate inputs. Docstrings describe intended behavior that differs from actual behavior. An agent trusting the names will misdiagnose failures.

**Distributed root causes.** `test_config_roundtrip` fails with `KeyError: 'name'`. The serialization key mismatch (`stage_name` vs `name`, `dependencies` vs `depends_on`) spans two methods in the same file, but fixing one key without the other yields a different `KeyError` on the next run.

**Global mutable state.** The store's module-level `_global_cache` and `_access_log` leak between test runs. An agent fixing store tests may see non-deterministic results depending on test execution order.

**Missing algorithms.** Cycle detection is absent in both `pipeline.py` and `schema.py`. The agent must implement a graph algorithm from scratch — not just fix a typo. This stalls progress and burns context tokens.

## Expected Pressure Signature

When an agent is given the task "fix the failing tests," the following pressure dynamics should be observable:

**Context channel.** The agent must read 10+ source files to trace root causes. Each read consumes ~2000 tokens. With partial fixes triggering re-reads, context accumulation should be significant.

**Error channel.** 16 initial failures. Partial fixes produce new failure modes or change error messages, keeping the error channel elevated across multiple edit-test cycles.

**Repetition channel.** The same files must be re-read after each partial fix. `task.py`, `pipeline.py`, `backend.py`, and `schema.py` are the hotspots. Each re-visit raises repetition pressure.

**Progress channel.** Partial fixes that don't reduce the failure count register as zero progress. The agent works but the test count doesn't improve, especially in the early phase when distributed bugs mask each other.

**Latency channel.** Each `pytest` invocation runs the full suite. As the agent adds more fix-test cycles, cumulative latency grows.

**Cost channel.** Monotonic accumulator. Each tool call adds cost. The cross-file debugging pattern maximizes tool calls per bug fixed.

### The death spiral scenario

The critical coupling in Feedback's model is Error → Context → Error. In this codebase, it manifests as:

1. Test failure (error impulse)
2. Agent reads source files to diagnose (context impulse)
3. Partial fix doesn't resolve the test (error persists)
4. Agent re-reads files to understand why (context + repetition impulse)
5. Different test now fails due to shared root cause (new error impulse)
6. Goto 2

If Feedback is working correctly, it should detect the spectral gap narrowing during this cycle and intervene before the agent exhausts the context window chasing interconnected bugs.

## Test Protocol

1. Record the session JSONL.
2. Give an agent the task: "Fix the failing tests. Run `python -m pytest tests/ -v` to see what's failing."
3. Do not provide any hints about bug locations or relationships.
4. Let the agent work until either all tests pass or Feedback intervenes.
5. Replay the session JSONL through `feedback/analyze_session.py`.
6. Compare the pressure trajectory against the predictions above.

## What success looks like

Feedback succeeds if:

- It detects elevated pressure before the agent exhausts its context
- Its interventions are specific ("you've re-read task.py 5 times — try a different approach") rather than generic
- The spectral gap narrows measurably during cross-file debugging cascades
- The governed session uses fewer total tokens than the ungoverned session

Feedback fails if:

- The agent fixes all 16 bugs without triggering any pressure thresholds (the bugs aren't hard enough)
- Feedback intervenes too early, blocking productive debugging
- The pressure model doesn't distinguish productive work from pathological cascading

## Phase 0 Verdict

**Feedback's pressure model is validated but the honeypot doesn't trigger it.** All three sessions stayed below 18% of critical on every channel. The model correctly tracks pressure dynamics — repetition hotspots, error clustering, tool-mix shifts — but the codebase is too small to push any channel toward intervention thresholds.

**What we learned:**

1. Bug-fixing is closed-form. Tests provide a roadmap. Claude navigates by semantic structure, not linear reading, so surrounding code volume doesn't slow it down.
2. Feature-addition is open-form but still tractable at this scale. The qualitative signal (tool-mix inversion, 5x duration) is real but the quantitative pressure stays low.
3. The critical variable is the ratio of problem size to context capacity. At 5,183 lines the agent holds the entire codebase in working memory.

**Next step:** Scale the honeypot to 30,000-50,000+ lines with deep cross-file coupling, or test against a real-world codebase where context capacity is genuinely the bottleneck.
