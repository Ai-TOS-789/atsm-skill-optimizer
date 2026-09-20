#!/usr/bin/env python3
"""Agent Alert and Notification System.

Monitors proactive tasks/issues and dispatches alerts:
  - Desktop notification via notify-send (Linux)
  - Log to data/alerts.log
  - Critical issues flagged urgent (success rate < 20%)

Alert levels: info, warning, critical

Usage:
    python3 agent_alerts.py check
    python3 agent_alerts.py watch
    python3 agent_alerts.py history
    python3 agent_alerts.py test
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Paths ---
DATA_DIR = Path(__file__).parent.parent / "data"
ALERTS_LOG = DATA_DIR / "alerts.log"
TASKS_FILE = Path("/tmp/proactive_tasks.json")
STATE_FILE = DATA_DIR / "alerts_state.json"

# Thresholds
CRITICAL_SUCCESS_RATE = 0.20  # below this = critical alert


# --- Logging ---
def log_alert(level: str, message: str):
    """Write an alert entry to the alerts log."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level.upper():<8}] {message}"
    with open(ALERTS_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


# --- State tracking ---
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"alerted_ids": []}

def save_state(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# --- Severity classification ---
def classify_severity(issue: dict) -> str:
    """Determine alert severity from issue data."""
    t = issue.get("type", "")
    if t == "atsm_failure":
        rate = issue.get("success_rate", 1.0)
        if rate < CRITICAL_SUCCESS_RATE:
            return "critical"
        elif rate < 0.4:
            return "warning"
        return "info"
    elif t == "git_unpushed":
        return "warning"
    elif t == "new_skills":
        return "info"
    return "info"


# --- Notification dispatch ---
def notify_send(title: str, body: str, urgency: str = "normal") -> bool:
    """Send desktop notification via notify-send."""
    try:
        cmd = ["notify-send", "--urgency", urgency, title, body]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# --- Alert processing ---
def process_alerts(issues: list) -> list:
    """Process a list of issues, sending alerts for new ones.
    Returns list of alert records created."""
    state = load_state()
    alerted_ids = set(state.get("alerted_ids", []))
    alerts = []

    for issue in issues:
        desc = issue.get("description", str(issue))
        # Use description as dedup key
        if desc in alerted_ids:
            continue

        severity = classify_severity(issue)
        issue_type = issue.get("type", "unknown")

        # Log it
        line = log_alert(severity, f"[{issue_type}] {desc}")
        print(line)

        # Desktop notification
        title = f"Proactive Agent: {severity.upper()}"
        urgency_map = {"critical": "critical", "warning": "normal", "info": "low"}
        notify_send(title, desc, urgency=urgency_map.get(severity, "normal"))

        alerted_ids.add(desc)
        alerts.append({
            "severity": severity,
            "type": issue_type,
            "description": desc,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    # Save state (keep last 500 IDs to prevent unbounded growth)
    state["alerted_ids"] = list(alerted_ids)[-500:]
    save_state(state)
    return alerts


def check_now() -> list:
    """Check tasks file for issues and alert on new ones."""
    if not TASKS_FILE.exists():
        print("No tasks file found.")
        return []
    try:
        with open(TASKS_FILE, "r", encoding="utf-8") as f:
            tasks = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"Error reading tasks: {e}")
        return []
    return process_alerts(tasks)


# --- Watch loop ---
def watch_loop(interval: int = 60):
    """Continuously watch for new issues."""
    print(f"Watching for alerts (interval: {interval}s, Ctrl+C to stop)...")
    while True:
        try:
            check_now()
            time.sleep(interval)
        except KeyboardInterrupt:
            print("\nStopped.")
            break


# --- History ---
def show_history(lines: int = 30):
    if not ALERTS_LOG.exists():
        print("No alert history.")
        return
    with open(ALERTS_LOG, "r", encoding="utf-8") as f:
        all_lines = f.readlines()
    for line in all_lines[-lines:]:
        print(line.rstrip())


# --- CLI ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "check":
        alerts = check_now()
        if alerts:
            print(f"\n{len(alerts)} alert(s) dispatched.")
        else:
            print("No new alerts.")
    elif cmd == "watch":
        interval = 60
        if "--interval" in sys.argv:
            idx = sys.argv.index("--interval")
            if idx + 1 < len(sys.argv):
                interval = int(sys.argv[idx + 1])
        watch_loop(interval)
    elif cmd == "history":
        n = 30
        if "--lines" in sys.argv:
            idx = sys.argv.index("--lines")
            if idx + 1 < len(sys.argv):
                n = int(sys.argv[idx + 1])
        show_history(n)
    elif cmd == "test":
        test_issue = {
            "type": "test",
            "description": "Test alert from agent_alerts.py",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        alerts = process_alerts([test_issue])
        print(f"Test alert dispatched: {len(alerts)}")
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)


if __name__ == "__main__":
    main()
