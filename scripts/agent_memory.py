#!/usr/bin/env python3
"""Persistent Agent Memory System — stores agent state across cycles.

Maintains a JSON-backed memory file that survives daemon restarts.
Tracks issues, skill performance, pending tasks, and learned lessons.

CLI:
  python3 agent_memory.py status
  python3 agent_memory.py lessons
  python3 agent_memory.py trend <skill_name>
  python3 agent_memory.py forget <skill_name>
  python3 agent_memory.py add-lesson "lesson text" [--context "ctx"]
  python3 agent_memory.py log-cycle [--agent NAME] [--instance ID]
"""

import argparse
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
MEMORY_FILE = DATA_DIR / "agent_memory.json"


# --- Memory Load/Save ---
def _ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def _default_memory() -> dict:
    return {
        "agent_name": "proactive-agent",
        "instance_id": str(uuid.uuid4())[:8],
        "cycle_count": 0,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "last_cycle": None,
        "issue_history": [],
        "skill_performance_log": {},
        "pending_tasks": [],
        "learned_lessons": [],
    }


def load_memory() -> dict:
    """Load agent memory from disk, returning default if missing/corrupt."""
    _ensure_data_dir()
    if MEMORY_FILE.exists():
        try:
            with open(MEMORY_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Ensure all keys exist
            default = _default_memory()
            for key, val in default.items():
                if key not in data:
                    data[key] = val
            return data
        except (json.JSONDecodeError, OSError):
            pass
    data = _default_memory()
    save_memory(data)
    return data


def save_memory(data: dict):
    """Persist agent memory to disk atomically."""
    _ensure_data_dir()
    tmp = MEMORY_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, MEMORY_FILE)


# --- Issue Tracking ---
def record_issue(issue: dict, data: dict = None) -> str:
    """Record a new issue. Returns the issue_id."""
    if data is None:
        data = load_memory()
    issue_id = str(uuid.uuid4())[:8]
    entry = {
        "id": issue_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "description": issue.get("description", ""),
        "source": issue.get("source", "unknown"),
        "severity": issue.get("severity", "info"),
        "resolution": None,
        "resolved_at": None,
    }
    data["issue_history"].append(entry)
    save_memory(data)
    return issue_id


def record_resolution(issue_id: str, resolution: str, data: dict = None) -> bool:
    """Resolve an issue by its ID."""
    if data is None:
        data = load_memory()
    for issue in data["issue_history"]:
        if issue["id"] == issue_id:
            issue["status"] = "resolved"
            issue["resolution"] = resolution
            issue["resolved_at"] = datetime.now(timezone.utc).isoformat()
            save_memory(data)
            return True
    return False


# --- Skill Performance ---
def _ensure_skill_entry(data: dict, skill_name: str):
    if skill_name not in data["skill_performance_log"]:
        data["skill_performance_log"][skill_name] = {
            "uses": 0,
            "successes": 0,
            "failures": 0,
            "last_used": None,
            "scores": [],  # rolling window of success/fail as 1/0
        }


def log_skill_use(skill_name: str, success: bool = True, data: dict = None):
    """Log a skill invocation outcome."""
    if data is None:
        data = load_memory()
    _ensure_skill_entry(data, skill_name)
    entry = data["skill_performance_log"][skill_name]
    entry["uses"] += 1
    if success:
        entry["successes"] += 1
        entry["scores"].append(1)
    else:
        entry["failures"] += 1
        entry["scores"].append(0)
    entry["last_used"] = datetime.now(timezone.utc).isoformat()
    # Keep last 20 scores for trend analysis
    if len(entry["scores"]) > 20:
        entry["scores"] = entry["scores"][-20:]
    save_memory(data)


def get_skill_trend(skill_name: str, data: dict = None) -> str:
    """Determine trend: improving / declining / stable / unknown."""
    if data is None:
        data = load_memory()
    if skill_name not in data["skill_performance_log"]:
        return "unknown"
    entry = data["skill_performance_log"][skill_name]
    scores = entry.get("scores", [])
    if len(scores) < 3:
        return "stable"  # not enough data
    # Compare first half vs second half
    mid = len(scores) // 2
    first_half = sum(scores[:mid]) / max(mid, 1)
    second_half = sum(scores[mid:]) / max(len(scores) - mid, 1)
    diff = second_half - first_half
    if diff > 0.15:
        return "improving"
    elif diff < -0.15:
        return "declining"
    else:
        return "stable"


def forget_skill(skill_name: str) -> bool:
    """Remove a skill from the performance log."""
    data = load_memory()
    if skill_name in data["skill_performance_log"]:
        del data["skill_performance_log"][skill_name]
        save_memory(data)
        return True
    return False


# --- Lessons ---
def add_lesson(lesson_text: str, context: str = "", data: dict = None) -> str:
    """Add a learned lesson. Returns lesson ID."""
    if data is None:
        data = load_memory()
    lesson_id = str(uuid.uuid4())[:8]
    entry = {
        "id": lesson_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": lesson_text,
        "context": context,
        "applied_count": 0,
    }
    data["learned_lessons"].append(entry)
    save_memory(data)
    return lesson_id


# --- Cycle Logging ---
def log_cycle(agent_name: str = None, instance_id: str = None, data: dict = None):
    """Increment cycle count and update timestamps."""
    if data is None:
        data = load_memory()
    data["cycle_count"] += 1
    data["last_cycle"] = datetime.now(timezone.utc).isoformat()
    if agent_name:
        data["agent_name"] = agent_name
    if instance_id:
        data["instance_id"] = instance_id
    save_memory(data)
    return data["cycle_count"]


# --- Pending Tasks ---
def add_pending_task(task: dict, data: dict = None) -> str:
    """Add a pending task. Returns task ID."""
    if data is None:
        data = load_memory()
    task_id = str(uuid.uuid4())[:8]
    entry = {
        "id": task_id,
        "added": datetime.now(timezone.utc).isoformat(),
        "description": task.get("description", ""),
        "status": "pending",
    }
    data["pending_tasks"].append(entry)
    save_memory(data)
    return task_id


def complete_task(task_id: str) -> bool:
    """Mark a pending task as completed."""
    data = load_memory()
    for t in data["pending_tasks"]:
        if t["id"] == task_id:
            t["status"] = "completed"
            t["completed_at"] = datetime.now(timezone.utc).isoformat()
            save_memory(data)
            return True
    return False


# --- CLI Commands ---
def cmd_status():
    """Show agent memory status."""
    data = load_memory()
    print("=" * 50)
    print(f"  Agent:       {data['agent_name']}")
    print(f"  Instance:    {data['instance_id']}")
    print(f"  Started:     {data['started_at']}")
    print(f"  Last Cycle:  {data['last_cycle'] or 'never'}")
    print(f"  Cycles:      {data['cycle_count']}")
    print("-" * 50)
    print(f"  Issues:      {len(data['issue_history'])} total")
    open_issues = sum(1 for i in data["issue_history"] if i["status"] == "open")
    resolved = sum(1 for i in data["issue_history"] if i["status"] == "resolved")
    print(f"    Open:      {open_issues}")
    print(f"    Resolved:  {resolved}")
    print(f"  Skills:      {len(data['skill_performance_log'])} tracked")
    print(f"  Lessons:     {len(data['learned_lessons'])} learned")
    print(f"  Pending:     {len(data['pending_tasks'])} tasks")
    print("=" * 50)


def cmd_lessons():
    """Display all learned lessons."""
    data = load_memory()
    lessons = data["learned_lessons"]
    if not lessons:
        print("No lessons learned yet.")
        return
    print(f"Learned Lessons ({len(lessons)}):")
    print("-" * 40)
    for lesson in lessons:
        print(f"  [{lesson['id']}] {lesson['text']}")
        if lesson.get("context"):
            print(f"           Context: {lesson['context']}")
        print(f"           Added: {lesson['timestamp'][:19]}")


def cmd_trend(skill_name: str):
    """Show skill performance trend."""
    data = load_memory()
    trend = get_skill_trend(skill_name, data)
    entry = data["skill_performance_log"].get(skill_name)
    if entry is None:
        print(f"No data for skill: {skill_name}")
        print("Tracked skills:")
        for s in data["skill_performance_log"]:
            print(f"  - {s}")
        return
    success_rate = (entry["successes"] / entry["uses"] * 100) if entry["uses"] > 0 else 0
    print(f"Skill:        {skill_name}")
    print(f"  Trend:      {trend}")
    print(f"  Uses:       {entry['uses']}")
    print(f"  Successes:  {entry['successes']}")
    print(f"  Failures:   {entry['failures']}")
    print(f"  Rate:       {success_rate:.1f}%")
    print(f"  Last used:  {entry['last_used'][:19] if entry['last_used'] else 'never'}")


def cmd_forget(skill_name: str):
    """Remove a skill from the performance log."""
    if forget_skill(skill_name):
        print(f"Forget skill: {skill_name}")
    else:
        print(f"Skill not found: {skill_name}")


def cmd_add_lesson(lesson_text: str, context: str = ""):
    """Add a new lesson."""
    lesson_id = add_lesson(lesson_text, context)
    print(f"Lesson added [{lesson_id}]: {lesson_text}")


def cmd_log_cycle(agent_name: str = None, instance_id: str = None):
    """Log a proactive agent cycle."""
    n = log_cycle(agent_name, instance_id)
    print(f"Cycle #{n} logged at {datetime.now(timezone.utc).isoformat()[:19]}")


# --- Main ---
def main():
    parser = argparse.ArgumentParser(description="Agent Memory System")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("status", help="Show agent status")

    sub.add_parser("lessons", help="List learned lessons")

    p_trend = sub.add_parser("trend", help="Show skill trend")
    p_trend.add_argument("skill_name")

    p_forget = sub.add_parser("forget", help="Forget a skill")
    p_forget.add_argument("skill_name")

    p_lesson = sub.add_parser("add-lesson", help="Add a lesson")
    p_lesson.add_argument("text")
    p_lesson.add_argument("--context", default="")

    p_cycle = sub.add_parser("log-cycle", help="Log a cycle")
    p_cycle.add_argument("--agent", default=None)
    p_cycle.add_argument("--instance", default=None)

    args = parser.parse_args()

    if args.command == "status":
        cmd_status()
    elif args.command == "lessons":
        cmd_lessons()
    elif args.command == "trend":
        cmd_trend(args.skill_name)
    elif args.command == "forget":
        cmd_forget(args.skill_name)
    elif args.command == "add-lesson":
        cmd_add_lesson(args.text, args.context)
    elif args.command == "log-cycle":
        cmd_log_cycle(args.agent, args.instance)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
