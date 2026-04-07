#!/usr/bin/env python3
"""Extract raw observables from honeypot session JONLs.

Measures actual token counts, elapsed times, and file access patterns
so impulse magnitudes can be set from data rather than invented.
"""

import json, sys, statistics
from collections import defaultdict
from datetime import datetime

SESSIONS = {
    'v1_bugfix': r'C:\Users\joevo\.claude\projects\C--Users-joevo-git-repo-honeypot\d7496929-940a-4070-92d9-df8da8ea2ffd.jsonl',
    'v2_bugfix': r'C:\Users\joevo\.claude\projects\C--Users-joevo-git-repo-honeypot\d31784e9-6616-4cad-a95c-ed34a6b392e8.jsonl',
    'v3_feature': r'C:\Users\joevo\.claude\projects\C--Users-joevo-git-repo-honeypot\c0ce33a0-4ffd-4ffd-983b-0795f5481f8a.jsonl',
}


def estimate_tokens(text):
    """Rough token estimate: ~4 chars per token for code."""
    if not text:
        return 0
    return len(str(text)) / 4


def parse_session(path):
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


def measure_session(events):
    """Extract raw observables from a session."""

    # Collect tool_use and tool_result blocks with timestamps
    tool_calls = {}   # id -> {tool, input_tokens, timestamp}
    tool_results = {} # id -> {output_tokens, is_error, elapsed_s, content, timestamp}

    call_sequence = []  # ordered list of (timestamp, tool, id, input_tokens)
    result_sequence = []

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
                tool_id = block.get('id', '')
                tool_name = block.get('name', '')
                inp = block.get('input', {})
                input_text = json.dumps(inp)
                input_tokens = estimate_tokens(input_text)

                tool_calls[tool_id] = {
                    'tool': tool_name,
                    'input_tokens': input_tokens,
                    'timestamp': dt,
                    'input': inp,
                }
                call_sequence.append((dt, tool_name, tool_id, input_tokens))

            elif block.get('type') == 'tool_result':
                tool_id = block.get('tool_use_id', '')
                result_content = block.get('content', '')
                if isinstance(result_content, list):
                    result_text = ' '.join(
                        str(b.get('text', ''))
                        for b in result_content
                        if isinstance(b, dict)
                    )
                elif isinstance(result_content, str):
                    result_text = result_content
                else:
                    result_text = str(result_content)

                output_tokens = estimate_tokens(result_text)

                # Calculate elapsed time if we have the call
                elapsed = 0
                if tool_id in tool_calls:
                    elapsed = (dt - tool_calls[tool_id]['timestamp']).total_seconds()

                tool_results[tool_id] = {
                    'output_tokens': output_tokens,
                    'is_error': block.get('is_error', False),
                    'elapsed_s': elapsed,
                    'content': result_text[:500],
                    'timestamp': dt,
                }
                result_sequence.append((dt, tool_id))

    # Now aggregate by tool type
    measurements = defaultdict(lambda: {
        'count': 0,
        'input_tokens': [],
        'output_tokens': [],
        'total_tokens': [],
        'elapsed_s': [],
    })

    for tool_id, call in tool_calls.items():
        tool = call['tool']
        m = measurements[tool]
        m['count'] += 1
        m['input_tokens'].append(call['input_tokens'])

        if tool_id in tool_results:
            result = tool_results[tool_id]
            m['output_tokens'].append(result['output_tokens'])
            m['total_tokens'].append(call['input_tokens'] + result['output_tokens'])
            m['elapsed_s'].append(result['elapsed_s'])

    # Inter-call timing
    inter_call_times = []
    for i in range(1, len(call_sequence)):
        dt = (call_sequence[i][0] - call_sequence[i-1][0]).total_seconds()
        inter_call_times.append(dt)

    return measurements, inter_call_times, call_sequence, tool_calls, tool_results


def fmt_stats(values):
    if not values:
        return "n/a"
    mn = min(values)
    mx = max(values)
    md = statistics.median(values)
    avg = statistics.mean(values)
    return f"mean={avg:.0f}  median={md:.0f}  min={mn:.0f}  max={mx:.0f}"


if __name__ == '__main__':
    all_measurements = {}

    for label, path in SESSIONS.items():
        events = parse_session(path)
        measurements, inter_call, call_seq, calls, results = measure_session(events)

        print(f"{'=' * 80}")
        print(f"SESSION: {label} ({len(call_seq)} tool calls)")
        print(f"{'=' * 80}")
        print()

        # Duration
        if call_seq:
            dur = (call_seq[-1][0] - call_seq[0][0]).total_seconds()
            print(f"  Duration: {dur:.0f}s ({dur/60:.1f} min)")
            print(f"  Inter-call timing: {fmt_stats(inter_call)}")
            print()

        # Per-tool measurements
        for tool in ['Read', 'Edit', 'Write', 'Bash', 'Grep', 'Glob', 'Agent', 'TodoWrite']:
            m = measurements.get(tool)
            if not m or m['count'] == 0:
                continue
            print(f"  {tool} (n={m['count']}):")
            print(f"    Input tokens:  {fmt_stats(m['input_tokens'])}")
            if m['output_tokens']:
                print(f"    Output tokens: {fmt_stats(m['output_tokens'])}")
            if m['total_tokens']:
                print(f"    Total tokens:  {fmt_stats(m['total_tokens'])}")
            if m['elapsed_s'] and any(s > 0 for s in m['elapsed_s']):
                print(f"    Elapsed (s):   {fmt_stats(m['elapsed_s'])}")
            print()

        # Bash command breakdown
        print("  Bash commands by type:")
        test_elapsed = []
        other_elapsed = []
        for tool_id, call in calls.items():
            if call['tool'] != 'Bash':
                continue
            cmd = call['input'].get('command', '')[:80]
            is_test = any(k in cmd for k in ('pytest', 'cargo test', 'cargo check'))
            if tool_id in results:
                elapsed = results[tool_id]['elapsed_s']
                if is_test:
                    test_elapsed.append(elapsed)
                else:
                    other_elapsed.append(elapsed)
        if test_elapsed:
            print(f"    Test/check:  n={len(test_elapsed)}  elapsed: {fmt_stats(test_elapsed)}")
        if other_elapsed:
            print(f"    Other:       n={len(other_elapsed)}  elapsed: {fmt_stats(other_elapsed)}")
        print()

        all_measurements[label] = measurements

    # Cross-session aggregation for impulse derivation
    print("=" * 80)
    print("CROSS-SESSION AGGREGATION (for impulse calibration)")
    print("=" * 80)
    print()

    for tool in ['Read', 'Edit', 'Write', 'Bash', 'Grep', 'Glob', 'Agent']:
        all_total = []
        all_input = []
        all_output = []
        all_elapsed = []
        total_count = 0
        for label, measurements in all_measurements.items():
            m = measurements.get(tool)
            if m:
                total_count += m['count']
                all_total.extend(m['total_tokens'])
                all_input.extend(m['input_tokens'])
                all_output.extend(m['output_tokens'])
                all_elapsed.extend(m['elapsed_s'])

        if total_count == 0:
            continue
        print(f"  {tool} (n={total_count} across all sessions):")
        if all_total:
            print(f"    Total tokens:  {fmt_stats(all_total)}")
            avg = statistics.mean(all_total)
            print(f"    --> Suggested context impulse: {avg:.0f} tokens")
        if all_elapsed and any(s > 0 for s in all_elapsed):
            nonzero = [s for s in all_elapsed if s > 0]
            if nonzero:
                print(f"    Elapsed (s):   {fmt_stats(nonzero)}")
                avg_e = statistics.mean(nonzero)
                print(f"    --> Suggested latency impulse: {avg_e:.1f} seconds")
        print()
