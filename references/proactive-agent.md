# Proactive Agent Reference

## Architecture

```
proactive_agent.py (daemon)
    ↓ every 30 min
    ├── Check cron jobs (stalled?)
    ├── Scan ATSM outcomes (failures?)
    ├── Time-of-day pattern analysis
    ├── Git status (unpushed commits?)
    └── Skills count (new installs?)
    ↓
    If issue found:
    ├── Write to /tmp/proactive_tasks.json
    ├── Record in agent_memory
    └── Dispatch alert via agent_alerts.py
```

## Integration

```python
# Each cycle:
memory = agent_memory.load_memory()          # persistent state
stats = atsm.compute_stats(records)          # per-skill health
issues = []                                  # detected issues

# Check 1: ATSM failures
for skill, s in stats.items():
    if s['expected_success'] < 0.4:
        issues.append({'type': 'atsm_failure', 'skill': skill, ...})

# Check 2: Time-of-day
hour = datetime.now(timezone.utc).hour
if hour in bad_hours:  # learned from history
    issues.append({'type': 'time_warning', 'hour': hour, ...})

# Process issues
for issue in issues:
    agent_memory.record_issue(issue)
    agent_alerts.dispatch_alert(issue)

agent_memory.save_memory(memory)  # persist
```

## Daemon Management

```bash
python3 scripts/proactive_agent.py start     # daemonize
python3 scripts/proactive_agent.py stop      # SIGTERM
python3 scripts/proactive_agent.py status    # PID + last log
python3 scripts/proactive_agent.py once      # single cycle (for cron)
```

## Memory Schema

```json
{
  "agent_name": "proactive-agent",
  "instance_id": "ad0cdac3",
  "cycle_count": 5,
  "started_at": "2026-09-20T21:45:07",
  "last_cycle": "2026-09-20T21:45:11",
  "issue_history": [
    {
      "id": "abc123",
      "type": "atsm_failure",
      "skill": "gif-search",
      "success_rate": 0.0,
      "timestamp": "...",
      "resolved": false
    }
  ],
  "pending_tasks": [],
  "learned_lessons": [
    {
      "id": "38630fdb",
      "text": "arxiv works best with site:arxiv.org filter",
      "context": "research-workflow"
    }
  ]
}
```

## Alert Levels

| Level | Trigger | Action |
|-------|---------|--------|
| info | Minor deviation | Log only |
| warning | Success < 50% OR time risk | Desktop notification |
| critical | Success < 20% | Urgent notification + task creation |

## Pitfalls

- **Stateless by default**: Always load memory at cycle start, save at end — otherwise lessons are lost
- **Deduplication**: Use task content hash to avoid duplicate alerts (1h cooldown)
- **Memory bloat**: Periodically archive resolved issues to avoid unbounded growth
- **Cron interaction**: `proactive_agent.py once` is safe for cron — it handles its own locking
