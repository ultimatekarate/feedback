#!/usr/bin/env python3
"""Feedback pressure analysis -- replay a Claude Code session JSONL through the 5-channel model.

Orthogonal channel design: each channel has a unique impulse source.
  Context    -- token load from tool I/O payload size
  Latency    -- wall-clock time waiting for Bash commands
  Error      -- tool failures + parsed test/build failure patterns
  Progress   -- time-proportional stall minus productive actions
  Repetition -- 2nd+ access to the same file path

Cost is tracked as a standalone metric (cumulative dollar spend), not
a pressure channel. It's redundant with every other channel and its
near-zero decay rate distorts spectral analysis.

Impulse mapping synced with feedback/src/hooks.rs.
Channel parameters synced with feedback/src/channels.rs.
"""

import json, math, sys
from collections import Counter, defaultdict
from datetime import datetime

JSONL_PATH = r'C:\Users\joevo\.claude\projects\C--Users-joevo-git-repo-phalanx\7cf5283e-0a63-477d-94db-9c7a03654f87.jsonl'

# -- Decaying integral (mirrors feedback/src/governor.rs) ----------

class DecayingIntegral:
    def __init__(self, lam):
        self.lam = lam
        self.value = 0.0
        self.last_t = 0.0

    def update(self, t):
        dt = t - self.last_t
        if dt > 0:
            self.value *= math.exp(-self.lam * dt)
            self.last_t = t

    def record(self, impulse, t):
        self.update(t)
        self.value += impulse


# -- Channel parameters (synced with feedback/src/channels.rs) ----

LAMBDAS = {
    'context':    0.0019,     # half-life 6.1 min (working set inter-reference time)
    'latency':    0.0053,     # half-life 130s    (EWMA over ~15 tool calls)
    'error':      0.0029,     # half-life 4.0 min (fix cycle duration)
    'progress':   0.0018,     # half-life 6.3 min (base_rate / critical steady-state)
    'repetition': 0.0025,     # half-life 4.6 min (healthy working set revisit interval)
}

# Criticals: calibrated from Phase 0 empirical data (orthogonal model).
# Healthy sessions (v1/v2 bug-fix) peak at ~25-39%.
# Struggling session (v3 feature-on-bugs) crosses 60% WARN on progress+latency.
CRITICALS = {
    # Derived from measured data + physical thresholds.
    # Impulses are physical quantities (tokens, seconds, counts).
    # Criticals are the integral value where sustained behavior is genuinely bad.
    'context':    50_000.0,   # tokens: 4x healthy peak (~12K). Rapid context churn.
    'latency':    350.0,      # seconds: sustained 2-min Bash calls. Catastrophic builds.
    'error':      6.0,        # 6 active errors in decay window = sustained failure loop.
    'progress':   55.0,       # pure stall steady-state = 0.1/0.0018 = 55.6. Agent produces nothing.
    'repetition': 15.0,       # 15 active re-accesses = thrashing same files every few seconds.
}

# Cost: standalone metric, not a pressure channel.
COST_PER_CALL = 0.005    # base per-call tax
COST_AGENT_SURCHARGE = 0.05  # additional cost for Agent tool calls

PROGRESS_BASE_RATE = 0.1  # stall accumulation: 0.1 per second between calls

# Failure patterns to detect in Bash output (Error channel).
FAILURE_PATTERNS = ['FAILED', 'error[E', 'panicked', 'Traceback', 'Error:',
                    'ERRORS', 'failures=', 'CalledProcessError']


def parse_events(path):
    events = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    return events


def build_timeline(events):
    timeline = []
    for e in events:
        ts_str = e.get('timestamp', '')
        if not ts_str:
            continue
        dt = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
        msg = e.get('message', {})
        if not isinstance(msg, dict):
            continue
        content = msg.get('content', [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get('type') == 'tool_use':
                inp = block.get('input', {})
                fpath = inp.get('file_path', '')
                sep = chr(92)
                fpath_normalized = fpath.replace(sep, '/')
                fname = fpath_normalized.split('/')[-1] if fpath else ''
                timeline.append({
                    'dt': dt, 'kind': 'call',
                    'tool': block.get('name', ''),
                    'id': block.get('id', ''),
                    'file': fname,
                    'fpath': fpath_normalized,
                    'command': inp.get('command', '')[:200] if 'command' in inp else '',
                })
            elif block.get('type') == 'tool_result':
                # Extract result text for failure-pattern parsing.
                result_content = block.get('content', '')
                content_text = ''
                if isinstance(result_content, list):
                    content_text = ' '.join(
                        str(b.get('text', ''))
                        for b in result_content
                        if isinstance(b, dict)
                    )
                elif isinstance(result_content, str):
                    content_text = result_content
                timeline.append({
                    'dt': dt, 'kind': 'result',
                    'id': block.get('tool_use_id', ''),
                    'is_error': block.get('is_error', False),
                    'content': content_text[:3000],
                })
    timeline.sort(key=lambda x: x['dt'])
    return timeline


def simulate(timeline):
    channels = {name: DecayingIntegral(lam) for name, lam in LAMBDAS.items()}
    cumulative_cost = 0.0  # standalone metric, not a pressure channel
    file_access_count = defaultdict(int)
    t0 = timeline[0]['dt']
    error_events = []
    snapshots = []
    last_snapshot = 0
    last_call_t = 0.0

    # Map tool_use IDs to (tool_name, timestamp, command) for result correlation.
    call_info = {}  # id -> {'tool': str, 't': float, 'cmd': str}

    def secs(dt):
        return (dt - t0).total_seconds()

    for ev in timeline:
        t = secs(ev['dt'])

        if ev['kind'] == 'call':
            tool = ev['tool']
            fname = ev.get('file', '')
            cmd = ev.get('command', '')
            fpath_full = ev.get('fpath', '')
            call_info[ev.get('id', '')] = {'tool': tool, 't': t, 'cmd': cmd, 'fpath': fpath_full}

            # -- Progress: time-proportional stall accumulation --
            # Cap inter-call gap at 10 minutes. Gaps longer than that are
            # session suspensions (overnight, lunch), not stalls. Without
            # the cap, a 19-hour overnight gap deposits ~5700 stall-units
            # and progress hits 10,000%+ — pure noise.
            MAX_STALL_GAP = 600.0  # 10 minutes
            dt_since_last = t - last_call_t
            stall_dt = min(dt_since_last, MAX_STALL_GAP)
            if stall_dt > 0:
                channels['progress'].record(stall_dt * PROGRESS_BASE_RATE, t)
            last_call_t = t

            # -- Orthogonal impulse mapping ---
            # Context impulses: measured token counts (cross-session medians).
            # Source: measure.py Phase 0 extraction.
            if tool == 'Read':
                channels['context'].record(1175.0, t)   # measured median: 1175 tokens
                if fname:
                    file_access_count[fname] += 1
                    # Plan docs are designed for iterative review.
                    # Suppress repetition for .claude/plans/ paths.
                    is_plan = '.claude/plans' in fpath_full
                    if file_access_count[fname] > 1 and not is_plan:
                        channels['repetition'].record(1.0, t)

            elif tool in ('Glob', 'Grep'):
                channels['context'].record(100.0, t)     # measured median: 94-108 tokens

            elif tool == 'Write':
                channels['context'].record(2400.0, t)    # measured median: 2404 tokens
                channels['progress'].record(-5.0, t)

            elif tool == 'Edit':
                channels['context'].record(200.0, t)     # measured median: 208 tokens
                channels['progress'].record(-3.0, t)
                if fname:
                    file_access_count[fname] += 1
                    is_plan = '.claude/plans' in fpath_full
                    if file_access_count[fname] > 1 and not is_plan:
                        channels['repetition'].record(0.5, t)

            elif tool == 'Bash':
                channels['context'].record(565.0, t)     # measured median: 565 tokens
                # Latency impulse moved to result handler (actual elapsed).
                is_test = any(k in cmd for k in (
                    'cargo test', 'cargo check', 'cargo build',
                    'pytest', 'python -m pytest',
                ))
                if is_test:
                    channels['progress'].record(-1.0, t)

            elif tool == 'Agent':
                channels['context'].record(4400.0, t)    # measured: 4384 tokens
                cumulative_cost += COST_AGENT_SURCHARGE

            elif tool == 'TodoWrite':
                # Task tracking is lightweight forward motion.
                channels['progress'].record(-0.5, t)

            elif tool == 'ExitPlanMode':
                # Planning credit applied in result handler (approval vs review).
                pass

            else:
                channels['context'].record(200.0, t)     # conservative default

            # Cost metric: per-call spend tax.
            cumulative_cost += COST_PER_CALL

        elif ev['kind'] == 'result':
            call_id = ev.get('id', '')
            correlated = call_info.get(call_id, {})
            correlated_tool = correlated.get('tool', '')
            call_t = correlated.get('t', 0.0)

            # Latency channel: actual elapsed seconds for Bash commands.
            # Measured from tool_use timestamp to tool_result timestamp.
            if correlated_tool == 'Bash' and call_t > 0:
                elapsed = t - call_t
                if elapsed > 0:
                    channels['latency'].record(elapsed, t)

            # Progress credit: planning rounds with user feedback.
            # ExitPlanMode with user direction ("red team", "check for",
            # "does it adhere") is productive work, not stalling.
            # Approved plans get a larger credit (forward motion achieved).
            if correlated_tool == 'ExitPlanMode':
                result_text_plan = ev.get('content', '')
                if 'approved' in result_text_plan.lower():
                    channels['progress'].record(-5.0, t)  # plan landed
                elif not ev.get('is_error'):
                    channels['progress'].record(-2.0, t)  # review round

            # Error channel: tool errors.
            # ExitPlanMode rejections are user steering, not failures.
            if ev.get('is_error') and correlated_tool != 'ExitPlanMode':
                channels['error'].record(1.0, t)
                error_events.append({'t': t, 'dt': ev['dt']})

            # Error channel: parsed failure patterns from Bash output.
            result_text = ev.get('content', '')
            if correlated_tool == 'Bash' and result_text:
                if any(p in result_text for p in FAILURE_PATTERNS):
                    channels['error'].record(0.5, t)
                    if not ev.get('is_error'):
                        error_events.append({'t': t, 'dt': ev['dt']})

        if t - last_snapshot >= 30:
            for ch in channels.values():
                ch.update(t)
            snap = {'t': t, 'dt': ev['dt'], 'cost': cumulative_cost}
            for name, ch in channels.items():
                snap[name] = ch.value
                snap[name + '_pct'] = max(0.0, (ch.value / CRITICALS[name]) * 100)
            snapshots.append(snap)
            last_snapshot = t

    return channels, cumulative_cost, file_access_count, error_events, snapshots


def report(timeline, channels, cumulative_cost, file_access_count, error_events, snapshots):
    t0 = timeline[0]['dt']
    t_end = timeline[-1]['dt']
    duration_s = (t_end - t0).total_seconds()
    total_calls = sum(1 for e in timeline if e['kind'] == 'call')
    total_errors = len(error_events)

    path = sys.argv[1] if len(sys.argv) > 1 else JSONL_PATH
    sid = path.replace('\\', '/').split('/')[-1].replace('.jsonl', '')[:8]

    print("=" * 80)
    print(f"FEEDBACK PRESSURE ANALYSIS - Session {sid}")
    print(f"Duration: {duration_s:.0f}s ({duration_s/3600:.1f}h)")
    print(f"Total tool calls: {total_calls}")
    print(f"Total errors: {total_errors}")
    print("=" * 80)

    # Peak pressure
    print()
    print("-- PEAK PRESSURE (% of critical) --")
    for name in LAMBDAS:
        vals = [s[name + '_pct'] for s in snapshots]
        pk = max(vals) if vals else 0
        peak_dt = None
        for s in snapshots:
            if s[name + '_pct'] == pk:
                peak_dt = s['dt']
                break
        bar = '#' * min(int(pk / 2), 40)
        warn = ' ** WARN' if pk > 60 else (' * elevated' if pk > 40 else '')
        print(f"  {name:12s}: {pk:5.1f}%  {bar}{warn}")

    # Pressure timeline
    print()
    print("-- PRESSURE TIMELINE (30s windows, top 3 channels) --")
    for s in snapshots:
        t = s['t']
        mins = int(t // 60)
        secs_val = int(t % 60)
        chans = sorted(
            [(name, s[name + '_pct']) for name in LAMBDAS],
            key=lambda x: -x[1]
        )[:3]
        top = '  '.join(f'{n}={v:.1f}%' for n, v in chans if v > 0.1)
        if top:
            print(f"  {mins:3d}:{secs_val:02d}  {top}")

    # Error clusters
    print()
    print("-- ERROR CLUSTERS --")
    if error_events:
        clusters = []
        current_cluster = [error_events[0]]
        for err in error_events[1:]:
            if err['t'] - current_cluster[-1]['t'] < 120:
                current_cluster.append(err)
            else:
                clusters.append(current_cluster)
                current_cluster = [err]
        clusters.append(current_cluster)
        for i, cluster in enumerate(clusters):
            start = cluster[0]['dt']
            end = cluster[-1]['dt']
            dur = cluster[-1]['t'] - cluster[0]['t']
            print(f"  Cluster {i+1}: {len(cluster)} errors over {dur:.0f}s  [{str(start)[:19]} -> {str(end)[:19]}]")
    else:
        print("  No errors detected.")

    # Repetition hotspots
    print()
    print("-- REPETITION HOTSPOTS --")
    for fname, count in sorted(file_access_count.items(), key=lambda x: -x[1])[:10]:
        bar = '#' * count
        print(f"  {fname:40s}: {count:2d} accesses  {bar}")

    # Phase analysis
    print()
    print("-- PHASE ANALYSIS --")
    calls_by_time = [ev for ev in timeline if ev['kind'] == 'call']
    phases = []
    if calls_by_time:
        phase_calls = [calls_by_time[0]]
        for ev in calls_by_time[1:]:
            gap = (ev['dt'] - phase_calls[-1]['dt']).total_seconds()
            if gap > 600:
                phases.append(phase_calls)
                phase_calls = [ev]
            else:
                phase_calls.append(ev)
        phases.append(phase_calls)

    for i, phase in enumerate(phases):
        dur = (phase[-1]['dt'] - phase[0]['dt']).total_seconds()
        tools = Counter(ev['tool'] for ev in phase)
        errs = sum(1 for ev in error_events
                   if phase[0]['dt'] <= ev['dt'] <= phase[-1]['dt'])
        print(f"  Phase {i+1}: {str(phase[0]['dt'])[:16]} -> {str(phase[-1]['dt'])[:16]}  ({dur/60:.0f}min, {len(phase)} calls, {errs} errors)")
        top3 = ', '.join(f'{n}={c}' for n, c in tools.most_common(3))
        print(f"           [{top3}]")

    # Final state
    print()
    print("-- FINAL PRESSURE STATE --")
    t_final = (t_end - t0).total_seconds()
    for name, ch in channels.items():
        ch.update(t_final)
        pct = max(0.0, (ch.value / CRITICALS[name]) * 100)
        warn = ' ** WARN' if pct > 60 else (' * elevated' if pct > 40 else '')
        print(f"  {name:12s}: {ch.value:8.2f} / {CRITICALS[name]:8.2f}  ({pct:.1f}%){warn}")
    print(f"  {'cost':12s}: ${cumulative_cost:.3f}  (metric only, not a pressure channel)")

    # Stability assessment with intervention breakdown
    print()
    print("-- STABILITY ASSESSMENT --")
    warn_count = 0
    deny_count = 0
    # Track which channel triggered each deny, and when.
    deny_triggers = []  # (time_str, channel, pct)
    deny_by_channel = Counter()
    warn_by_channel = Counter()

    for s in snapshots:
        for name in LAMBDAS:
            pct = s[name + '_pct']
            if pct > 85:
                deny_count += 1
                deny_by_channel[name] += 1
                t_min = s['t'] / 60
                deny_triggers.append((t_min, name, pct))
            elif pct > 60:
                warn_count += 1
                warn_by_channel[name] += 1

    print(f"  Snapshots where a channel exceeded WARN (60%): {warn_count}")
    print(f"  Snapshots where a channel exceeded DENY (85%): {deny_count}")

    if deny_count == 0 and warn_count == 0:
        print("  -> Session remained well within stable operating envelope.")
    elif deny_count == 0:
        print(f"  -> Governor would have warned {warn_count} times. No denials.")
    else:
        print(f"  -> Governor would have intervened {deny_count} times.")

    if deny_by_channel:
        print()
        print("-- INTERVENTION BREAKDOWN --")
        print("  Denials by channel:")
        for name, count in deny_by_channel.most_common():
            print(f"    {name:12s}: {count:2d}")
        if warn_by_channel:
            print("  Warnings by channel:")
            for name, count in warn_by_channel.most_common():
                print(f"    {name:12s}: {count:2d}")

        # Cluster denials by time to show when interventions fired.
        # Group denials that fire within the same 30s snapshot.
        print()
        print("  Intervention timeline:")
        i = 0
        while i < len(deny_triggers):
            t_min, ch, pct = deny_triggers[i]
            # Gather all denials at the same timestamp (same snapshot).
            batch = [(ch, pct)]
            while i + 1 < len(deny_triggers) and abs(deny_triggers[i + 1][0] - t_min) < 0.1:
                i += 1
                batch.append((deny_triggers[i][1], deny_triggers[i][2]))
            channels_str = ', '.join(f'{c}={p:.0f}%' for c, p in batch)
            print(f"    {int(t_min):3d}:{int((t_min % 1) * 60):02d}  DENY  {channels_str}")
            i += 1


if __name__ == '__main__':
    path = sys.argv[1] if len(sys.argv) > 1 else JSONL_PATH
    events = parse_events(path)
    timeline = build_timeline(events)
    if not timeline:
        print("No events found.")
        sys.exit(1)
    channels, cost, fac, errs, snaps = simulate(timeline)
    report(timeline, channels, cost, fac, errs, snaps)
