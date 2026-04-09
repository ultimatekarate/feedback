"""Extract tool call sequence and errors from a Claude Code session JSONL."""
import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else ""
events = []
with open(path, 'r', encoding='utf-8', errors='replace') as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except:
            pass

tool_calls = []
tool_results = {}
user_messages = []

for ev in events:
    msg = ev.get('message', {})
    if msg.get('role') == 'user' and isinstance(msg.get('content'), str):
        user_messages.append((ev.get('timestamp', ''), msg['content'][:200]))
    content = msg.get('content', [])
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict):
                if block.get('type') == 'tool_use':
                    tool_calls.append((ev.get('timestamp', ''), block))
                elif block.get('type') == 'tool_result':
                    tid = block.get('tool_use_id', '')
                    tool_results[tid] = (ev.get('timestamp', ''), block)

FAILURES = ['FAILED', 'error[E', 'panicked', 'Traceback', 'Error:',
            'ERRORS', 'failures=', 'CalledProcessError']
PREFIX = 'C:/Users/joevo/git-repo/phalanx/'
PREFIX2 = 'C:\\Users\\joevo\\git-repo\\phalanx\\'

print("=== USER MESSAGES ===")
for ts, msg in user_messages:
    print(f"  [{ts[11:19]}] {msg[:150]}")

print(f"\n=== FULL TOOL SEQUENCE ({len(tool_calls)} calls) ===")
errors = []
for i, (ts, tc) in enumerate(tool_calls):
    name = tc.get('name', '?')
    inp = tc.get('input', {})
    tid = tc.get('id', '')

    fp = inp.get('file_path', '')
    fp = fp.replace(PREFIX, '').replace(PREFIX2, '')
    cmd = inp.get('command', '')[:80]
    pat = inp.get('pattern', '')[:40]
    old = inp.get('old_string', '')[:50] if 'old_string' in inp else ''
    detail = fp[-55:] if fp else cmd if cmd else pat if pat else ''

    # Check result
    r_ts, r_block = tool_results.get(tid, ('', {}))
    is_err = r_block.get('is_error', False)
    r_content = ''
    rc = r_block.get('content', '')
    if isinstance(rc, list):
        for b in rc:
            if isinstance(b, dict) and b.get('text'):
                r_content += b['text']
    elif isinstance(rc, str):
        r_content = rc

    has_failure = any(p in r_content[:3000] for p in FAILURES)

    marker = ''
    if is_err:
        marker = ' *** ERROR'
        errors.append((i + 1, ts, name, detail, old, r_content[:400]))
    elif has_failure:
        marker = ' *** FAILURE'
        errors.append((i + 1, ts, name, detail, old, r_content[:400]))

    print(f"  {i + 1:3d}. [{ts[11:19]}] {name:8s} {detail}{marker}")

print(f"\n=== {len(errors)} ERRORS/FAILURES ===")
for num, ts, name, detail, old, content in errors:
    print(f"\n--- Call #{num} [{ts[11:19]}] {name} {detail} ---")
    if old:
        print(f"  old_string: {old}")
    snippet = content.replace('\n', ' | ')[:400]
    print(f"  {snippet}")
