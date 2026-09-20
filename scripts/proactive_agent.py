#!/usr/bin/env python3
"""Proactive Agent Loop Engine — periodic background checks.

Runs as a daemon, performing these checks every 30 minutes:
  • Stalled cron jobs
  • Failed ATSM outcomes needing retry
  • Time-of-day success patterns
  • Git repo: unpushed commits, new skills
  • Writes detected issues to /tmp/proactive_tasks.json
  • Reads completions from /tmp/proactive_completed.json

CLI:
  python3 proactive_agent.py start    # start daemon
  python3 proactive_agent.py stop     # stop daemon
  python3 proactive_agent.py status   # show status
  python3 proactive_agent.py once     # single check (for cron)
"""

import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Paths ---
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = Path(__file__).parent.parent / "data"
PID_FILE = DATA_DIR / "proactive_agent.pid"
LOG_FILE = DATA_DIR / "proactive.log"
TASKS_FILE = Path("/tmp/proactive_tasks.json")
COMPLETED_FILE = Path("/tmp/proactive_completed.json")
DB_FILE = DATA_DIR / "atsm_db.jsonl"

CHECK_INTERVAL = 1800  # 30 minutes


# --- Logging ---
def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Task I/O ---
def load_tasks() -> list:
    if TASKS_FILE.exists():
        try:
            with open(TASKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


def save_tasks(tasks: list):
    TASKS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(TASKS_FILE, "w", encoding="utf-8") as f:
        json.dump(tasks, f, indent=2, ensure_ascii=False)


def add_task(task: dict):
    tasks = load_tasks()
    # Deduplicate by description
    existing = {t.get("description") for t in tasks}
    if task["description"] not in existing:
        tasks.append(task)
        save_tasks(tasks)
        log(f"TASK ADDED: {task['description']}")
    else:
        log(f"TASK EXISTS: {task['description']}")


def load_completed() -> list:
    if COMPLETED_FILE.exists():
        try:
            with open(COMPLETED_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return []


# --- Checks ---
def check_cron_stalled():
    """Check for stalled cron jobs."""
    issues = []
    try:
        result = subprocess.run(
            ["crontab", "-l"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            lines = [l.strip() for l in result.stdout.splitlines() if l.strip() and not l.startswith("#")]
            if not lines:
                log("CRON: no active jobs")
            else:
                log(f"CRON: {len(lines)} job(s) configured")
        else:
            log("CRON: crontab not available or no crontab")
    except (subprocess.TimeoutExpired, FileNotFoundError) as e:
        log(f"CRON: check failed — {e}")

    # Check for known hermes cron issues via atsm data patterns
    return issues


def check_atsm_failures():
    """Find ATSM skills with recent failures (expected_success < 0.4)."""
    issues = []
    if not DB_FILE.exists():
        log("ATSM: no outcome database found")
        return issues

    try:
        records = []
        with open(DB_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError as e:
        log(f"ATSM: read error — {e}")
        return issues

    # Group by skill, find failures
    from collections import defaultdict
    skill_results = defaultdict(lambda: {"success": 0, "fail": 0})
    for rec in records:
        s = skill_results[rec["skill"]]
        if rec["success"]:
            s["success"] += 1
        else:
            s["fail"] += 1

    for skill, counts in skill_results.items():
        total = counts["success"] + counts["fail"]
        if total == 0:
            continue
        success_rate = counts["success"] / total
        if success_rate < 0.4 and total >= 2:
            issue = {
                "type": "atsm_failure",
                "description": f"Skill '{skill}' has low success rate ({success_rate:.0%}, {total} attempts) — review or retrain",
                "skill": skill,
                "success_rate": round(success_rate, 3),
                "total_attempts": total,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            issues.append(issue)
            log(f"ATSM FAILURE: {skill} ({success_rate:.0%} over {total} tries)")

    if not issues:
        log("ATSM: all skills within acceptable success thresholds")
    return issues


def check_time_patterns():
    """Analyze ATSM data for time-of-day success patterns."""
    if not DB_FILE.exists():
        log("PATTERN: no ATSM data for time analysis")
        return []

    try:
        records = []
        with open(DB_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        return []

    if not records:
        return []

    # Group by hour
    hour_success = {}
    for rec in records:
        try:
            dt = datetime.fromisoformat(rec["timestamp"])
            h = dt.hour
        except (ValueError, KeyError):
            continue
        if h not in hour_success:
            hour_success[h] = {"success": 0, "fail": 0}
        if rec["success"]:
            hour_success[h]["success"] += 1
        else:
            hour_success[h]["fail"] += 1

    now_hour = datetime.now(timezone.utc).hour
    current = hour_success.get(now_hour, {"success": 0, "fail": 0})
    total = current["success"] + current["fail"]
    if total > 0:
        rate = current["success"] / total
        log(f"PATTERN: hour {now_hour}:00 UTC — {rate:.0%} success ({total} records)")
    else:
        log(f"PATTERN: hour {now_hour}:00 UTC — no historical data")

    # Note: this is informational, doesn't generate tasks
    return []


def check_git_status():
    """Check for unpushed commits and new skills."""
    issues = []

    # Check skills repo git status
    skill_parent = SKILLS_ROOT.parent  # ~/.hermes/
    git_dir = skill_parent / ".git"

    if not git_dir.exists():
        # Try ~/.hermes directly or search upward
        for candidate in [skill_parent, Path.home()]:
            if (candidate / ".git").exists():
                git_dir = candidate / ".git"
                break

    if git_dir.exists():
        repo_root = git_dir.parent
        try:
            # Unpushed commits
            result = subprocess.run(
                ["git", "log", "--branches", "--not", "--remotes", "--oneline"],
                capture_output=True, text=True, timeout=10, cwd=repo_root
            )
            if result.returncode == 0 and result.stdout.strip():
                unpushed = result.stdout.strip().count("\n") + 1
                issue = {
                    "type": "git_unpushed",
                    "description": f"Git repo has {unpushed} unpushed commit(s) in {repo_root.name}",
                    "count": unpushed,
                    "path": str(repo_root),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
                issues.append(issue)
                log(f"GIT: {unpushed} unpushed commit(s)")
            else:
                log("GIT: all commits pushed")
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            log(f"GIT: check failed — {e}")

    # Check for new skills (compare count to a stored baseline)
    baseline_file = DATA_DIR / "skill_count_baseline.json"
    current_skills = []
    if SKILLS_ROOT.exists():
        for skill_dir in SKILLS_ROOT.rglob("SKILL.md"):
            current_skills.append(skill_dir.parent.name)

    current_count = len(current_skills)
    previous_count = 0
    if baseline_file.exists():
        try:
            with open(baseline_file, "r") as f:
                previous_count = json.load(f).get("count", 0)
        except (json.JSONDecodeError, OSError):
            pass

    if previous_count > 0 and current_count > previous_count:
        new_count = current_count - previous_count
        issue = {
            "type": "new_skills",
            "description": f"{new_count} new skill(s) installed (total: {current_count})",
            "new_count": new_count,
            "total": current_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        issues.append(issue)
        log(f"SKILLS: {new_count} new skill(s) detected")
    else:
        log(f"SKILLS: {current_count} skills (no change)")

    # Update baseline
    with open(baseline_file, "w") as f:
        json.dump({"count": current_count, "timestamp": datetime.now(timezone.utc).isoformat()}, f)

    return issues


def check_completed_tasks():
    """Process completed tasks — remove from active list."""
    completed = load_completed()
    if not completed:
        return

    active = load_tasks()
    completed_ids = {t.get("description") for t in completed}
    remaining = [t for t in active if t.get("description") not in completed_ids]

    if len(remaining) < len(active):
        removed = len(active) - len(remaining)
        log(f"COMPLETED: {removed} task(s) resolved")
        save_tasks(remaining)


# --- Main check cycle ---
def run_checks():
    """Run all proactive checks."""
    log("=" * 50)
    log("PROACTIVE CHECK CYCLE START")
    log("=" * 50)

    all_issues = []

    # 1. Cron status
    log("--- Checking cron jobs ---")
    all_issues.extend(check_cron_stalled())

    # 2. ATSM failures
    log("--- Checking ATSM outcomes ---")
    all_issues.extend(check_atsm_failures())

    # 3. Time-of-day patterns
    log("--- Analyzing time patterns ---")
    all_issues.extend(check_time_patterns())

    # 4. Git & skills
    log("--- Checking git/skills ---")
    all_issues.extend(check_git_status())

    # 5. Process completions
    log("--- Processing completions ---")
    check_completed_tasks()

    # Add all issues as tasks
    for issue in all_issues:
        add_task(issue)

    log(f"CYCLE COMPLETE: {len(all_issues)} issue(s) detected")
    return all_issues


# --- Daemon ---
def daemonize():
    """Fork into background daemon."""
    if os.fork() > 0:
        sys.exit(0)
    os.setsid()
    if os.fork() > 0:
        sys.exit(0)

    # Redirect stdio
    sys.stdout.flush()
    sys.stderr.flush()
    devnull = open(os.devnull, "r")
    os.dup2(devnull.fileno(), sys.stdin.fileno())

    # Write PID
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    log(f"DAEMON STARTED (pid {os.getpid()})")

    def handle_signal(signum, frame):
        log(f"DAEMON: received signal {signum}, shutting down")
        if PID_FILE.exists():
            PID_FILE.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Main loop
    while True:
        try:
            run_checks()
        except Exception as e:
            log(f"ERROR in check cycle: {e}")
        time.sleep(CHECK_INTERVAL)


def cmd_start():
    """Start the daemon."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # Check if process exists
            print(f"Daemon already running (pid {pid})")
            return
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink()

    print("Starting proactive agent daemon...")
    daemonize()


def cmd_stop():
    """Stop the daemon."""
    if not PID_FILE.exists():
        print("Daemon not running (no PID file)")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to daemon (pid {pid})")
        # Wait for PID file to be removed
        for _ in range(10):
            if not PID_FILE.exists():
                print("Daemon stopped")
                return
            time.sleep(0.5)
        print("Daemon may still be shutting down")
    except (ValueError, ProcessLookupError):
        print("Daemon not running (stale PID file)")
        PID_FILE.unlink()
    except PermissionError:
        print(f"Permission denied to signal pid {pid}")


def cmd_status():
    """Show daemon status."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Daemon running (pid {pid})")
        except (ValueError, ProcessLookupError, PermissionError):
            print("Daemon not running (stale PID file)")
    else:
        print("Daemon not running")

    # Show recent log
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().strip().splitlines()
        if lines:
            print(f"\n--- Last 10 log entries ---")
            for line in lines[-10:]:
                print(f"  {line}")

    # Show active tasks
    tasks = load_tasks()
    if tasks:
        print(f"\n--- Active tasks ({len(tasks)}) ---")
        for t in tasks:
            print(f"  [{t.get('type', '?')}] {t.get('description', '?')}")
    else:
        print("\nNo active tasks")


def cmd_once():
    """Run a single check cycle."""
    issues = run_checks()
    if issues:
        print(f"\n{len(issues)} issue(s) detected:")
        for issue in issues:
            print(f"  • [{issue['type']}] {issue['description']}")
    else:
        print("\nNo issues detected. System healthy.")
    return issues


# --- Entry ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "status":
        cmd_status()
    elif cmd == "once":
        cmd_once()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
