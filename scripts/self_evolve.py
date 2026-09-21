#!/usr/bin/env python3
"""Self-Evolving Autonomous Loop — daemon that sets goals, executes, and learns.

Runs as a background daemon. Every cycle (default 10 minutes):
  1. Load agent memory (lessons, preferences, issue history)
  2. Check proactive_tasks.json for pending tasks
  3. If no pending tasks → generate a new goal from:
     - User preferences (memory)
     - Time-of-day patterns
     - System state (disk, CPU, pending updates)
     - Skill gaps (low P(success) skills that need practice)
  4. Plan the goal via goal_planner.py
  5. Execute via execution_engine.py
  6. Record outcome in ATSM
  7. Save lesson to agent_memory

Self-modification:
  - If a skill fails 3+ times → auto-update its description/pitfalls
  - If new external tool detected → create a skill for it (auto_skill.py)
  - If pattern detected → create new prior in ATSM

CLI:
  python3 self_evolve.py start           # start daemon
  python3 self_evolve.py stop            # stop daemon
  python3 self_evolve.py status          # show status
  python3 self_evolve.py once            # single cycle (foreground)
  python3 self_evolve.py set-pref key=value  # set a user preference
  python3 self_evolve.py lessons         # show learned lessons
  python3 self-evolve.py self-modify     # run self-modification pass
"""

import json
import os
import random
import re
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPT_DIR.parent
DATA_DIR = SKILL_DIR / "data"
PREFS_FILE = DATA_DIR / "evolve_prefs.json"
PID_FILE = DATA_DIR / "self_evolve.pid"
LOG_FILE = DATA_DIR / "self_evolve.log"
CYCLE_STATE_FILE = DATA_DIR / "evolve_cycle_state.json"
GOALS_DB_FILE = DATA_DIR / "goals_db.json"
ATSM_DB_FILE = DATA_DIR / "atsm_db.jsonl"
PROACTIVE_TASKS_FILE = Path("/tmp/proactive_tasks.json")
AGENT_MEMORY_FILE = DATA_DIR / "agent_memory.json"

# --- Config ---
DEFAULT_CYCLE_SECONDS = 600  # 10 minutes
MIN_CYCLE_SECONDS = 60
MAX_CYCLE_SECONDS = 3600
GOAL_GENERATION_WEIGHTS = {
    "skill_gap": 0.35,
    "time_pattern": 0.25,
    "system_state": 0.25,
    "user_preference": 0.15,
}


# ============================================================
# Logging
# ============================================================

def log(msg: str, level: str = "INFO"):
    """Log a message to file and optionally stdout."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    return line


def log_print(msg: str, level: str = "INFO"):
    """Log and print."""
    line = log(msg, level)
    print(line, flush=True)


# ============================================================
# Preferences
# ============================================================

def load_preferences() -> dict:
    """Load user preferences for goal generation."""
    if PREFS_FILE.exists():
        try:
            with open(PREFS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    # Default preferences
    return {
        "focus_areas": ["productivity", "automation", "research"],
        "skill_practice_enabled": True,
        "max_concurrent_goals": 3,
        "auto_self_modify": True,
        "goal_history_depth": 10,
    }


def save_preferences(prefs: dict):
    """Save user preferences."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREFS_FILE, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)


def set_pref(key: str, value: str):
    """Set a single preference key=value."""
    prefs = load_preferences()
    # Try to parse value as JSON (for bools, numbers, lists)
    try:
        parsed = json.loads(value)
    except (json.JSONDecodeError, ValueError):
        parsed = value
    prefs[key] = parsed
    save_preferences(prefs)
    log_print(f"Preference set: {key} = {parsed}")


# ============================================================
# Agent Memory Integration
# ============================================================

def load_agent_memory() -> dict:
    """Load agent memory from disk."""
    if AGENT_MEMORY_FILE.exists():
        try:
            with open(AGENT_MEMORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return {}


def save_lesson(lesson_text: str, context: str = "self-evolve"):
    """Save a lesson to agent memory."""
    memory = load_agent_memory()
    lessons = memory.get("learned_lessons", [])
    lesson = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "text": lesson_text,
        "context": context,
        "applied_count": 0,
    }
    lessons.append(lesson)
    memory["learned_lessons"] = lessons
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
    log(f"Lesson saved: {lesson_text[:80]}")


def load_lessons() -> list:
    """Load all learned lessons."""
    memory = load_agent_memory()
    return memory.get("learned_lessons", [])


def record_issue(description: str, issue_type: str = "evolve_generated"):
    """Record an issue to agent memory."""
    memory = load_agent_memory()
    issues = memory.get("issue_history", [])
    issue = {
        "id": uuid.uuid4().hex[:8],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "open",
        "description": description,
        "source": "self-evolve",
        "severity": "info",
    }
    issues.append(issue)
    memory["issue_history"] = issues
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)


# ============================================================
# ATSM Integration
# ============================================================

def atsm_rank_skills(task: str, top_k: int = 5) -> list:
    """Rank skills using ATSM engine. Returns list of dicts."""
    try:
        sys.path.insert(0, str(SCRIPT_DIR))
        import atsm
        results, _ = atsm.rank_skills(task, top_k=top_k)
        sys.path.pop(0)
        return results
    except Exception as e:
        log(f"ATSM rank failed: {e}", "WARN")
        return []


def atsm_record_outcome(skill: str, success: int):
    """Record an outcome in ATSM."""
    try:
        cmd = [
            sys.executable, str(SCRIPT_DIR / "atsm.py"),
            "record", skill, str(success),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        log(f"ATSM record failed: {e}", "WARN")


def atsm_get_stats() -> dict:
    """Get ATSM statistics per skill."""
    if not ATSM_DB_FILE.exists():
        return {}
    skill_stats = {}
    try:
        with open(ATSM_DB_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    s = rec.get("skill", "unknown")
                    if s not in skill_stats:
                        skill_stats[s] = {"success": 0, "fail": 0}
                    if rec.get("success"):
                        skill_stats[s]["success"] += 1
                    else:
                        skill_stats[s]["fail"] += 1
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return skill_stats


# ============================================================
# Proactive Tasks
# ============================================================

def load_pending_tasks() -> list:
    """Load pending tasks from proactive_tasks.json."""
    if PROACTIVE_TASKS_FILE.exists():
        try:
            with open(PROACTIVE_TASKS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return []


# ============================================================
# System State
# ============================================================

def get_system_state() -> dict:
    """Get current system state: CPU, memory, disk, etc."""
    state = {"timestamp": datetime.now(timezone.utc).isoformat()}

    # CPU usage (quick read from /proc/stat)
    try:
        with open("/proc/stat", "r") as f:
            line1 = f.readline()
        time.sleep(0.3)
        with open("/proc/stat", "r") as f:
            line2 = f.readline()
        parts1 = line1.split()[1:]
        parts2 = line2.split()[1:]
        idle1 = int(parts1[3])
        idle2 = int(parts2[3])
        total1 = sum(int(x) for x in parts1)
        total2 = sum(int(x) for x in parts2)
        idle_delta = idle2 - idle1
        total_delta = total2 - total1
        cpu_pct = (1.0 - idle_delta / total_delta) * 100 if total_delta > 0 else 0
        state["cpu_percent"] = round(cpu_pct, 1)
    except Exception:
        state["cpu_percent"] = None

    # Memory usage (from /proc/meminfo)
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        mem_total = None
        mem_available = None
        for line in lines:
            if line.startswith("MemTotal:"):
                mem_total = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                mem_available = int(line.split()[1])
        if mem_total and mem_available is not None:
            mem_pct = (1 - mem_available / mem_total) * 100
            state["memory_percent"] = round(mem_pct, 1)
    except Exception:
        state["memory_percent"] = None

    # Disk usage (root /)
    try:
        result = subprocess.run(
            ["df", "-h", "/"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")
            if len(lines) >= 2:
                parts = lines[1].split()
                disk_pct = int(parts[4].rstrip("%"))
                state["disk_percent"] = disk_pct
    except Exception:
        state["disk_percent"] = None

    # Pending updates (apt)
    try:
        result = subprocess.run(
            ["apt", "list", "--upgradable"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            upgradable = [
                l for l in result.stdout.splitlines()
                if "upgradable" in l
            ]
            state["pending_updates"] = len(upgradable)
    except Exception:
        state["pending_updates"] = None

    return state


# ============================================================
# Goal Generation
# ============================================================

def detect_skill_gaps() -> list[str]:
    """Find skills with low P(success) that need practice."""
    stats = atsm_get_stats()
    gaps = []
    for skill, counts in stats.items():
        total = counts["success"] + counts["fail"]
        if total < 2:
            continue
        success_rate = counts["success"] / total
        if success_rate < 0.4:
            gaps.append(skill)
    return gaps


def generate_goal_from_skill_gap(skill: str) -> str:
    """Generate a goal to practice a failing skill."""
    templates = [
        f"Practice and improve the '{skill}' skill by running a test workflow",
        f"Review and retrain the '{skill}' skill — test with a simple task",
        f"Debug why '{skill}' is failing and verify it works end-to-end",
    ]
    return random.choice(templates)


def generate_goal_from_time_pattern() -> str:
    """Generate a goal based on time-of-day patterns."""
    hour = datetime.now(timezone.utc).hour
    if 0 <= hour < 6:
        return "Review and organize learned lessons from agent memory"
    elif 6 <= hour < 9:
        return "Check system health and generate a daily status report"
    elif 9 <= hour < 12:
        return "Research latest AI agent frameworks and summarize findings"
    elif 12 <= hour < 14:
        return "Optimize ATSM priors and clean up stale skill records"
    elif 14 <= hour < 17:
        return "Run proactive checks and verify all cron jobs are functional"
    elif 17 <= hour < 20:
        return "Review today's execution history and record outcomes"
    else:
        return "Plan tomorrow's goals and update agent preferences"


def generate_goal_from_system_state(state: dict) -> Optional[str]:
    """Generate a goal based on system state."""
    goals = []
    disk = state.get("disk_percent")
    if disk and disk > 80:
        goals.append("Clean up disk space — remove temp files and old logs")
    mem = state.get("memory_percent")
    if mem and mem > 85:
        goals.append("Investigate high memory usage and terminate idle processes")
    cpu = state.get("cpu_percent")
    if cpu and cpu > 85:
        goals.append("Investigate high CPU usage and identify resource-heavy processes")
    updates = state.get("pending_updates")
    if updates and updates > 20:
        goals.append(f"Review {updates} pending system updates and plan maintenance")
    if goals:
        return random.choice(goals)
    return None


def generate_goal_from_preferences(prefs: dict) -> Optional[str]:
    """Generate a goal from user preferences."""
    focus_areas = prefs.get("focus_areas", [])
    if not focus_areas:
        return None
    area = random.choice(focus_areas)
    templates = {
        "productivity": "Find and automate a repetitive task in the workflow",
        "automation": "Create a new skill for an undetected external tool",
        "research": "Search for recent papers on autonomous agent architectures",
        "testing": "Run a full integration test of the ATSM ranking pipeline",
        "optimization": "Profile the execution engine and identify bottlenecks",
    }
    return templates.get(area, f"Explore improvements in {area}")


def generate_goal() -> str:
    """Generate a new goal using weighted random selection from sources."""
    prefs = load_preferences()
    sources = []

    # Skill gaps (highest priority)
    if prefs.get("skill_practice_enabled", True):
        gaps = detect_skill_gaps()
        if gaps:
            skill = random.choice(gaps)
            sources.append(("skill_gap", generate_goal_from_skill_gap(skill)))

    # Time-of-day patterns
    sources.append(("time_pattern", generate_goal_from_time_pattern()))

    # System state
    state = get_system_state()
    sys_goal = generate_goal_from_system_state(state)
    if sys_goal:
        sources.append(("system_state", sys_goal))

    # User preferences
    pref_goal = generate_goal_from_preferences(prefs)
    if pref_goal:
        sources.append(("user_preference", pref_goal))

    if not sources:
        return "Run a general system health check and record results"

    # Weighted random selection
    weights = []
    for source_name, _ in sources:
        w = GOAL_GENERATION_WEIGHTS.get(source_name, 0.1)
        weights.append(w)

    # Normalize
    total_w = sum(weights)
    if total_w > 0:
        weights = [w / total_w for w in weights]
    else:
        weights = [1.0 / len(sources)] * len(sources)

    selected = random.choices(sources, weights=weights, k=1)[0]
    log(f"Goal generated from source '{selected[0]}': {selected[1]}")
    return selected[1]


# ============================================================
# Goal Planning & Execution
# ============================================================

def plan_goal(goal_text: str) -> Optional[dict]:
    """Plan a goal using goal_planner.py. Returns goal tree dict or None."""
    try:
        cmd = [
            sys.executable, str(SCRIPT_DIR / "goal_planner.py"),
            "plan", goal_text,
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60,
            cwd=str(SKILL_DIR),
        )
        if result.returncode == 0:
            # Parse the goal ID from output
            output = result.stdout
            match = re.search(r"Goal saved as:\s*(goal_\w+)", output)
            if match:
                goal_id = match.group(1)
                log(f"Goal planned: {goal_id}")
                return {"id": goal_id, "description": goal_text, "output": output}
            # Even if we can't parse ID, return success
            return {"id": None, "description": goal_text, "output": output}
        else:
            log(f"Goal planner failed: {result.stderr[:200]}", "WARN")
            return None
    except subprocess.TimeoutExpired:
        log("Goal planner timed out", "WARN")
        return None
    except Exception as e:
        log(f"Goal planner error: {e}", "WARN")
        return None


def execute_goal(goal_text: str) -> dict:
    """Execute a goal using execution_engine.py. Returns execution entry."""
    try:
        cmd = [
            sys.executable, str(SCRIPT_DIR / "execution_engine.py"),
            "run", goal_text,
            "--timeout", "120",
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=150,
            cwd=str(SKILL_DIR),
        )
        success = result.returncode == 0
        output = result.stdout.strip()
        stderr_out = result.stderr.strip()

        # Parse skill from output
        skill_match = re.search(r"Skill:\s*(\S+)", output)
        skill = skill_match.group(1) if skill_match else "unknown"

        entry = {
            "execution_id": uuid.uuid4().hex[:8],
            "task": goal_text,
            "skill": skill,
            "success": success,
            "return_code": result.returncode,
            "output": output[:1000],
            "stderr": stderr_out[:500],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        log(f"Execution {'SUCCESS' if success else 'FAILURE'}: {goal_text[:60]} (skill={skill})")
        return entry

    except subprocess.TimeoutExpired:
        log(f"Execution timed out: {goal_text[:60]}", "WARN")
        return {
            "execution_id": uuid.uuid4().hex[:8],
            "task": goal_text,
            "skill": "unknown",
            "success": False,
            "return_code": -1,
            "output": "TIMEOUT",
            "stderr": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
    except Exception as e:
        log(f"Execution error: {e}", "WARN")
        return {
            "execution_id": uuid.uuid4().hex[:8],
            "task": goal_text,
            "skill": "unknown",
            "success": False,
            "return_code": -1,
            "output": str(e),
            "stderr": "",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ============================================================
# Self-Modification
# ============================================================

def self_modify_skill(skill: str):
    """Auto-update a skill that has failed 3+ times."""
    log(f"SELF-MODIFY: Updating skill '{skill}' due to repeated failures")

    # Find the skill's SKILL.md
    skills_root = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
    skill_md = None
    for candidate in skills_root.rglob("SKILL.md"):
        if candidate.parent.name == skill:
            skill_md = candidate
            break

    if not skill_md or not skill_md.exists():
        log(f"SELF-MODIFY: SKILL.md not found for '{skill}'", "WARN")
        return

    try:
        content = skill_md.read_text(encoding="utf-8")
        # Add a "Pitfalls" section or append to existing one
        failure_note = f"\n## Auto-Generated Pitfall Note\n\nAdded {datetime.now(timezone.utc).strftime('%Y-%m-%d')}: This skill has been auto-flagged for review due to repeated failures. Consider:\n- Verifying external dependencies\n- Checking API rate limits\n- Reviewing input validation\n"

        if "## Pitfalls" in content:
            # Append to existing pitfalls
            content = content.replace("## Pitfalls", f"## Pitfalls{failure_note}")
        else:
            # Add pitfalls section before end
            content = content.rstrip() + "\n" + failure_note + "\n"

        skill_md.write_text(content, encoding="utf-8")
        log(f"SELF-MODIFY: Updated {skill_md}")
        save_lesson(f"Auto-updated skill '{skill}' — added failure review pitfalls", "self-modify")
    except Exception as e:
        log(f"SELF-MODIFY: Failed to update skill: {e}", "WARN")


def detect_new_tools() -> list[str]:
    """Detect new external tools that might need skills."""
    known_tools = set()
    skills_root = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
    if skills_root.exists():
        for skill_dir in skills_root.iterdir():
            if skill_dir.is_dir():
                known_tools.add(skill_dir.name.lower())

    # Check common tools
    common_tools = [
        "docker", "kubectl", "terraform", "ansible", "nginx",
        "postgresql", "redis", "mongodb", "elasticsearch",
        "prometheus", "grafana", "github-cli", "aws-cli",
    ]
    new_tools = []
    for tool in common_tools:
        if tool in known_tools:
            continue
        try:
            result = subprocess.run(
                ["which", tool.replace("-", "")],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                new_tools.append(tool)
        except Exception:
            pass
    return new_tools


def create_skill_for_tool(tool: str):
    """Create a new skill for a detected external tool."""
    log(f"SELF-MODIFY: Creating skill for detected tool '{tool}'")
    try:
        cmd = [
            sys.executable, str(SCRIPT_DIR / "auto_skill.py"),
            "--register", tool,
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=15, cwd=str(SKILL_DIR))
        save_lesson(f"Auto-created skill for detected tool: {tool}", "self-modify")
        log(f"SELF-MODIFY: Skill registered for '{tool}'")
    except Exception as e:
        log(f"SELF-MODIFY: Failed to create skill: {e}", "WARN")


def run_self_modification():
    """Run self-modification pass."""
    log("SELF-MODIFY: Starting self-modification pass")

    # 1. Check for skills with 3+ failures
    stats = atsm_get_stats()
    for skill, counts in stats.items():
        total = counts["success"] + counts["fail"]
        if total >= 3 and counts["success"] / total < 0.3:
            self_modify_skill(skill)

    # 2. Detect new external tools
    new_tools = detect_new_tools()
    for tool in new_tools[:2]:  # Limit to 2 per cycle
        create_skill_for_tool(tool)

    # 3. Create new priors for patterns
    # (This is handled implicitly by ATSM's Bayesian update)

    log("SELF-MODIFY: Self-modification pass complete")


# ============================================================
# Main Cycle
# ============================================================

def run_cycle() -> dict:
    """Run a single evolution cycle. Returns cycle result dict."""
    cycle_id = uuid.uuid4().hex[:8]
    log_print(f"=== CYCLE {cycle_id} START ===")
    start_time = time.time()

    result = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "goal": None,
        "planned": False,
        "executed": False,
        "success": False,
        "lesson": None,
    }

    # Step 1: Load agent memory
    memory = load_agent_memory()
    cycle_count = memory.get("cycle_count", 0) + 1
    memory["cycle_count"] = cycle_count
    memory["last_cycle"] = datetime.now(timezone.utc).isoformat()
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(AGENT_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
    log(f"Cycle #{cycle_count}")

    # Step 2: Check pending tasks
    pending = load_pending_tasks()
    if pending:
        log(f"Processing {len(pending)} pending proactive task(s)")
        # Pick the first pending task as the goal
        task = pending[0]
        goal_text = task.get("description", "Process pending task")
        result["goal"] = goal_text
        result["source"] = "proactive_task"

        # Execute the task
        entry = execute_goal(goal_text)
        result["executed"] = True
        result["success"] = entry.get("success", False)
        result["entry"] = entry

        # Record outcome
        if entry.get("skill"):
            atsm_record_outcome(entry["skill"], 1 if entry["success"] else 0)

        # Save lesson
        lesson = f"Processed proactive task: {goal_text[:60]} — {'success' if entry['success'] else 'failure'}"
        save_lesson(lesson, "proactive-task")
        result["lesson"] = lesson

    else:
        # Step 3: No pending tasks → generate a new goal
        log("No pending tasks — generating new goal")
        goal_text = generate_goal()
        result["goal"] = goal_text
        result["source"] = "generated"

        # Step 4: Plan the goal
        plan_result = plan_goal(goal_text)
        result["planned"] = plan_result is not None

        # Step 5: Execute
        entry = execute_goal(goal_text)
        result["executed"] = True
        result["success"] = entry.get("success", False)
        result["entry"] = entry

        # Step 6: Record outcome in ATSM
        if entry.get("skill"):
            atsm_record_outcome(entry["skill"], 1 if entry["success"] else 0)

        # Step 7: Save lesson
        lesson = f"Goal '{goal_text[:50]}' → {'success' if entry['success'] else 'failure'} (skill: {entry.get('skill', '?')})"
        save_lesson(lesson, "self-evolve")
        result["lesson"] = lesson

    # Step 8: Self-modification (every 5th cycle)
    prefs = load_preferences()
    if prefs.get("auto_self_modify", True) and cycle_count % 5 == 0:
        run_self_modification()

    duration = time.time() - start_time
    result["duration"] = round(duration, 2)
    log_print(f"=== CYCLE {cycle_id} END ({duration:.1f}s) ===")

    # Save cycle state
    save_cycle_state(result)
    return result


def save_cycle_state(result: dict):
    """Save the latest cycle state."""
    state = {}
    if CYCLE_STATE_FILE.exists():
        try:
            with open(CYCLE_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError):
            state = {}
    state["last_cycle"] = result
    state["history"] = state.get("history", [])
    state["history"].append(result)
    # Keep only last 50 cycles
    state["history"] = state["history"][-50:]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CYCLE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# ============================================================
# Daemon
# ============================================================

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

    log_print(f"DAEMON STARTED (pid {os.getpid()})")

    def handle_signal(signum, frame):
        log_print(f"DAEMON: received signal {signum}, shutting down")
        if PID_FILE.exists():
            PID_FILE.unlink()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Main loop
    cycle_seconds = DEFAULT_CYCLE_SECONDS
    while True:
        try:
            run_cycle()
        except Exception as e:
            log(f"ERROR in cycle: {e}", "ERROR")
        time.sleep(cycle_seconds)


def cmd_start(args=None):
    """Start the daemon."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Daemon already running (pid {pid})")
            return
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink()

    print("Starting self-evolve daemon...")
    daemonize()


def cmd_stop(args=None):
    """Stop the daemon."""
    if not PID_FILE.exists():
        print("Daemon not running (no PID file)")
        return

    try:
        pid = int(PID_FILE.read_text().strip())
        os.kill(pid, signal.SIGTERM)
        print(f"Sent SIGTERM to daemon (pid {pid})")
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


def cmd_status(args=None):
    """Show daemon status."""
    running = False
    pid = None
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            running = True
        except (ValueError, ProcessLookupError, PermissionError):
            pass

    if running:
        print(f"Daemon running (pid {pid})")
    else:
        print("Daemon not running")

    # Show cycle state
    if CYCLE_STATE_FILE.exists():
        try:
            with open(CYCLE_STATE_FILE, "r", encoding="utf-8") as f:
                state = json.load(f)
            last = state.get("last_cycle", {})
            history = state.get("history", [])
            print(f"\nCycles completed: {len(history)}")
            if last:
                print(f"Last cycle: {last.get('cycle_id', '?')}")
                print(f"  Goal: {last.get('goal', '?')[:60]}")
                print(f"  Source: {last.get('source', '?')}")
                print(f"  Success: {last.get('success', '?')}")
                print(f"  Duration: {last.get('duration', '?')}s")
                print(f"  Lesson: {last.get('lesson', '?')[:60]}")
        except (json.JSONDecodeError, IOError):
            pass

    # Show recent log
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().strip().splitlines()
        if lines:
            print(f"\n--- Last 10 log entries ---")
            for line in lines[-10:]:
                print(f"  {line}")

    # Show pending tasks
    pending = load_pending_tasks()
    if pending:
        print(f"\n--- Pending tasks ({len(pending)}) ---")
        for t in pending:
            print(f"  [{t.get('type', '?')}] {t.get('description', '?')[:60]}")


def cmd_once(args=None):
    """Run a single cycle in foreground."""
    result = run_cycle()
    print(f"\n{'='*60}")
    print(f"CYCLE RESULT")
    print(f"{'='*60}")
    print(f"  Cycle ID: {result.get('cycle_id')}")
    print(f"  Goal: {result.get('goal', '?')}")
    print(f"  Source: {result.get('source', '?')}")
    print(f"  Planned: {result.get('planned', '?')}")
    print(f"  Executed: {result.get('executed', '?')}")
    print(f"  Success: {result.get('success', '?')}")
    print(f"  Duration: {result.get('duration', '?')}s")
    print(f"  Lesson: {result.get('lesson', '?')}")
    if result.get("entry"):
        entry = result["entry"]
        print(f"  Skill: {entry.get('skill', '?')}")
        print(f"  Return code: {entry.get('return_code', '?')}")
    print(f"{'='*60}")
    return result


def cmd_set_pref(args):
    """Set a preference: key=value."""
    if not args:
        print("Usage: self_evolve.py set-pref key=value")
        print("Example: self_evolve.py set-pref focus_areas=[\"research\",\"automation\"]")
        sys.exit(1)

    arg = " ".join(args)
    if "=" not in arg:
        print("Error: format must be key=value")
        sys.exit(1)

    key, value = arg.split("=", 1)
    key = key.strip()
    value = value.strip()
    set_pref(key, value)


def cmd_lessons(args=None):
    """Show learned lessons."""
    lessons = load_lessons()
    if not lessons:
        print("No lessons learned yet.")
        return

    print(f"\n📚 Learned Lessons ({len(lessons)}):")
    print("─" * 60)
    for lesson in lessons[-20:]:
        ts = lesson.get("timestamp", "?")[:19]
        text = lesson.get("text", "?")
        ctx = lesson.get("context", "?")
        print(f"  [{ts}] ({ctx}) {text[:70]}")
    print()


def cmd_self_modify(args):
    """Run self-modification pass."""
    run_self_modification()
    print("Self-modification pass complete.")


def cmd_config(args):
    """Show or set cycle configuration."""
    prefs = load_preferences()
    if not args:
        print("\n📋 Self-Evolve Configuration:")
        print("─" * 40)
        for k, v in sorted(prefs.items()):
            print(f"  {k}: {v}")
        print(f"\n  Cycle interval: {DEFAULT_CYCLE_SECONDS}s ({DEFAULT_CYCLE_SECONDS//60} min)")
        print(f"  PID file: {PID_FILE}")
        print(f"  Log file: {LOG_FILE}")
        print(f"  Data dir: {DATA_DIR}")
    else:
        # Allow setting cycle_seconds via args
        for arg in args:
            if arg.startswith("cycle_seconds="):
                try:
                    val = int(arg.split("=")[1])
                    val = max(MIN_CYCLE_SECONDS, min(MAX_CYCLE_SECONDS, val))
                    prefs["cycle_seconds"] = val
                    save_preferences(prefs)
                    print(f"Cycle interval set to {val}s")
                except ValueError:
                    print("Invalid value for cycle_seconds")


# ============================================================
# Entry Point
# ============================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "once": cmd_once,
        "set-pref": cmd_set_pref,
        "lessons": cmd_lessons,
        "self-modify": cmd_self_modify,
        "config": cmd_config,
    }

    handler = commands.get(command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
