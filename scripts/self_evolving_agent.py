#!/usr/bin/env python3
"""Self-Evolving Agent — the autonomous loop that ties ALL ATSM modules together.

Every cycle:
  1. Check proactive tasks (proactive_agent)
  2. If idle → generate goal from memory patterns
  3. Plan goal decomposition (goal_planner)
  4. Execute each subgoal (execution_engine)
  5. Record outcomes (atsm record)
  6. Save lessons (agent_memory)
  7. If skill fails 3x → auto-update skill description
  8. If new tool detected → create skill (auto_skill)

Runs as a background daemon with full lifecycle management.

CLI:
  python3 self_evolving_agent.py start      # start daemon
  python3 self_evolving_agent.py stop       # stop daemon
  python3 self_evolving_agent.py status     # show daemon + cycle status
  python3 self_evolving_agent.py once       # single full cycle
  python3 self_evolving_agent.py set-pref key=value  # set preference
  python3 self_evolving_agent.py goals      # list active goals
  python3 self_evolving_agent.py lessons    # show learned lessons
"""

import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
SKILL_DIR = SCRIPT_DIR.parent
DATA_DIR = SKILL_DIR / "data"
PID_FILE = DATA_DIR / "self_evolving_agent.pid"
LOG_FILE = DATA_DIR / "self_evolving_agent.log"
CYCLE_STATE_FILE = DATA_DIR / "cycle_state.json"
PREF_FILE = DATA_DIR / "agent_preferences.json"

CHECK_INTERVAL = int(os.environ.get("ATSM_CYCLE_INTERVAL", "1800"))  # 30 min default
IDLE_CYCLES_BEFORE_GOAL = int(os.environ.get("ATSM_IDLE_THRESHOLD", "2"))
SKILL_FAIL_THRESHOLD = 3

# --- Logging ---
def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [{level}] {msg}"
    print(line, flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Preferences ---
def load_prefs() -> dict:
    if PREF_FILE.exists():
        try:
            with open(PREF_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"auto_goal_generation": True, "auto_skill_heal": True, "auto_skill_create": True, "verbose": False}

def save_prefs(prefs: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREF_FILE, "w", encoding="utf-8") as f:
        json.dump(prefs, f, indent=2)

def set_pref(key: str, value: str):
    prefs = load_prefs()
    # Type coercion
    if value.lower() in ("true", "1", "yes"):
        value = True
    elif value.lower() in ("false", "0", "no"):
        value = False
    elif value.isdigit():
        value = int(value)
    prefs[key] = value
    save_prefs(prefs)
    print(f"Set {key} = {value}")


# --- Cycle State ---
def load_cycle_state() -> dict:
    if CYCLE_STATE_FILE.exists():
        try:
            with open(CYCLE_STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {"total_cycles": 0, "idle_cycles": 0, "last_goal": None, "last_issues": [], "skill_fail_counts": {}, "goals_completed": 0}

def save_cycle_state(state: dict):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CYCLE_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


# --- Lazy Module Loaders ---
_proactive_mod = None
_goal_planner_mod = None
_execution_engine_mod = None
_agent_memory_mod = None
_auto_skill_mod = None
_atsm_mod = None
_self_heal_mod = None
_atsm_learn_mod = None

def _get_proactive():
    global _proactive_mod
    if _proactive_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import proactive_agent as _proactive_mod
        except ImportError:
            _proactive_mod = False
    return _proactive_mod if _proactive_mod else None

def _get_goal_planner():
    global _goal_planner_mod
    if _goal_planner_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import goal_planner as _goal_planner_mod
        except ImportError:
            _goal_planner_mod = False
    return _goal_planner_mod if _goal_planner_mod else None

def _get_execution_engine():
    global _execution_engine_mod
    if _execution_engine_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import execution_engine as _execution_engine_mod
        except ImportError:
            _execution_engine_mod = False
    return _execution_engine_mod if _execution_engine_mod else None

def _get_agent_memory():
    global _agent_memory_mod
    if _agent_memory_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import agent_memory as _agent_memory_mod
        except ImportError:
            _agent_memory_mod = False
    return _agent_memory_mod if _agent_memory_mod else None

def _get_auto_skill():
    global _auto_skill_mod
    if _auto_skill_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import auto_skill as _auto_skill_mod
        except ImportError:
            _auto_skill_mod = False
    return _auto_skill_mod if _auto_skill_mod else None

def _get_atsm():
    global _atsm_mod
    if _atsm_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import atsm as _atsm_mod
        except ImportError:
            _atsm_mod = False
    return _atsm_mod if _atsm_mod else None

def _get_self_heal():
    global _self_heal_mod
    if _self_heal_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import self_heal as _self_heal_mod
        except ImportError:
            _self_heal_mod = False
    return _self_heal_mod if _self_heal_mod else None

def _get_atsm_learn():
    global _atsm_learn_mod
    if _atsm_learn_mod is None:
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import atsm_learn as _atsm_learn_mod
        except ImportError:
            _atsm_learn_mod = False
    return _atsm_learn_mod if _atsm_learn_mod else None


# --- Goal Generation from Memory Patterns ---
def generate_goal_from_memory() -> Optional[str]:
    """Analyze memory patterns to generate a meaningful goal."""
    mem = _get_agent_memory()
    if not mem:
        return None

    try:
        data = mem.load_memory()
    except Exception as e:
        log(f"Memory load failed: {e}", "WARN")
        return None

    # Look at recent issues for patterns
    issues = data.get("issue_history", [])
    open_issues = [i for i in issues if i.get("status") == "open"]

    # Look at skill performance for declining skills
    skill_log = data.get("skill_performance_log", {})
    declining_skills = []
    for skill_name, entry in skill_log.items():
        trend = mem.get_skill_trend(skill_name, data)
        if trend == "declining":
            declining_skills.append(skill_name)

    # Generate goal based on patterns
    if declining_skills:
        return f"Review and improve underperforming skills: {', '.join(declining_skills[:3])}"
    elif open_issues:
        # Group issues by type
        from collections import Counter
        types = Counter(i.get("type", "unknown") for i in open_issues)
        top_type = types.most_common(1)[0][0]
        return f"Resolve recurring {top_type} issues ({len(open_issues)} open)"
    elif data.get("cycle_count", 0) > 10:
        return "Analyze system health and optimize ATSM performance"

    return None


# --- Skill Auto-Healing ---
def check_and_heal_skills(state: dict):
    """If a skill failed SKILL_FAIL_THRESHOLD times, auto-update its description."""
    prefs = load_prefs()
    if not prefs.get("auto_skill_heal", True):
        return

    mem = _get_agent_memory()
    if not mem:
        return

    try:
        data = mem.load_memory()
    except Exception:
        return

    skill_log = data.get("skill_performance_log", {})
    for skill_name, entry in skill_log.items():
        failures = entry.get("failures", 0)
        uses = entry.get("uses", 0)
        if uses < 3:
            continue
        fail_rate = failures / uses
        if fail_rate > 0.6 and failures >= SKILL_FAIL_THRESHOLD:
            # Record the failure count for this skill
            fail_key = f"consecutive_failures_{skill_name}"
            current = state.get("skill_fail_counts", {}).get(fail_key, 0)
            current += 1
            state.setdefault("skill_fail_counts", {})[fail_key] = current

            if current >= SKILL_FAIL_THRESHOLD:
                log(f"SKILL HEAL: '{skill_name}' failed {failures}/{uses} times — flagging for review", "WARN")
                lesson = f"Skill '{skill_name}' has high failure rate ({fail_rate:.0%}) — needs review or retraining"
                try:
                    mem.add_lesson(lesson, context="self_evolving_agent")
                    log(f"LESSON SAVED: {lesson}")
                except Exception as e:
                    log(f"Failed to save lesson: {e}", "WARN")
                # Reset counter after flagging
                state["skill_fail_counts"][fail_key] = 0


# --- Auto Skill Creation ---
def check_and_create_skill(goal: str):
    """If ATSM can't match a goal, auto-create a new skill."""
    prefs = load_prefs()
    if not prefs.get("auto_skill_create", True):
        return

    auto_skill = _get_auto_skill()
    atsm = _get_atsm()
    if not auto_skill or not atsm:
        return

    try:
        results, _ = atsm.rank_skills(goal, top_k=5)
        if not results or all(r["final_score"] < 0.05 for r in results):
            log(f"AUTO-SKILL: No match for '{goal[:60]}' — would create new skill", "INFO")
            # In non-interactive mode, we log the suggestion rather than prompting
            mem = _get_agent_memory()
            if mem:
                mem.add_lesson(
                    f"New skill needed for: {goal}",
                    context="auto_skill_suggestion"
                )
    except Exception as e:
        log(f"Auto-skill check failed: {e}", "WARN")


# --- Main Cycle ---
def run_cycle() -> dict:
    """Execute one full self-evolution cycle."""
    cycle_id = str(uuid.uuid4())[:8]
    state = load_cycle_state()
    state["total_cycles"] += 1
    cycle_num = state["total_cycles"]
    prefs = load_prefs()

    log("=" * 60)
    log(f"CYCLE #{cycle_id} (total: {cycle_num})")
    log("=" * 60)

    result = {
        "cycle_id": cycle_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "issues_found": 0,
        "goal_generated": None,
        "subgoals_executed": 0,
        "lessons_saved": 0,
        "skills_healed": 0,
    }

    # ── Step 1: Proactive Check ──
    log("[1/8] Running proactive checks...")
    proactive = _get_proactive()
    issues = []
    if proactive:
        try:
            issues = proactive.run_checks()
            result["issues_found"] = len(issues)
            log(f"  → {len(issues)} issue(s) detected")
        except Exception as e:
            log(f"  → Proactive check failed: {e}", "WARN")
    else:
        log("  → proactive_agent not available, skipping", "WARN")

    # ── Step 2: Goal Generation (if idle) ──
    goal_text = None
    if not issues:
        state["idle_cycles"] += 1
        log(f"[2/8] Idle ({state['idle_cycles']}/{IDLE_CYCLES_BEFORE_GOAL}) — checking for goal generation...")
        if state["idle_cycles"] >= IDLE_CYCLES_BEFORE_GOAL:
            if prefs.get("auto_goal_generation", True):
                goal_text = generate_goal_from_memory()
                if goal_text:
                    log(f"  → Generated goal: {goal_text}")
                    state["idle_cycles"] = 0
                else:
                    log("  → No goal pattern detected")
            else:
                log("  → Auto goal generation disabled")
        else:
            log(f"  → Not yet idle enough (need {IDLE_CYCLES_BEFORE_GOAL})")
    else:
        state["idle_cycles"] = 0
        log("[2/8] Issues found — skipping goal generation")

    result["goal_generated"] = goal_text

    # ── Step 3: Goal Planning ──
    goal_tree = None
    if goal_text:
        log("[3/8] Planning goal decomposition...")
        gp = _get_goal_planner()
        if gp:
            try:
                goal_tree = gp.create_goal_tree(goal_text)
                subgoal_count = len(goal_tree.root.subgoals) if goal_tree.root else 0
                log(f"  → Decomposed into {subgoal_count} subgoals")
                state["last_goal"] = goal_text
            except Exception as e:
                log(f"  → Goal planning failed: {e}", "WARN")
        else:
            log("  → goal_planner not available, skipping", "WARN")

    # ── Step 4: Execute Subgoals ──
    if goal_tree and goal_tree.root and goal_tree.root.subgoals:
        log("[4/8] Executing subgoals...")
        ee = _get_execution_engine()
        atsm = _get_atsm()
        for sg in goal_tree.root.subgoals:
            sg_desc = sg.description
            log(f"  → Executing: {sg_desc}")
            if ee:
                try:
                    entry = ee.execute_task(sg_desc, mode="sync", timeout=60)
                    success = entry.get("success", False)
                    result["subgoals_executed"] += 1
                    log(f"    → {'SUCCESS' if success else 'FAILURE'} (skill: {entry.get('skill', '?')})")

                    # Record outcome in ATSM
                    if atsm:
                        try:
                            skill_name = entry.get("skill", "unknown")
                            atsm.append_record(skill_name, 1 if success else 0)
                        except Exception as e:
                            log(f"    → ATSM record failed: {e}", "WARN")

                    # Update subgoal status
                    sg.status = "completed" if success else "failed"
                    sg.outcome = 1 if success else 0

                except Exception as e:
                    log(f"    → Execution failed: {e}", "WARN")
                    sg.status = "failed"
                    sg.outcome = 0
            else:
                log("    → execution_engine not available", "WARN")
                sg.status = "pending"

        # Save goal tree state
        try:
            goals = gp.load_goals_db()
            goals[goal_tree.id] = goal_tree
            gp.save_goals_db(goals)
        except Exception as e:
            log(f"  → Failed to save goal state: {e}", "WARN")

    # ── Step 5: Record Outcomes ──
    log("[5/8] Recording outcomes...")
    if goal_tree and goal_tree.root:
        try:
            all_success = all(sg.outcome == 1 for sg in goal_tree.root.subgoals if sg.outcome is not None)
            any_executed = any(sg.outcome is not None for sg in goal_tree.root.subgoals)
            if any_executed:
                goal_tree.status = "completed" if all_success else "failed"
                if all_success:
                    state["goals_completed"] += 1
                log(f"  → Goal outcome: {'SUCCESS' if all_success else 'PARTIAL/FAILURE'}")
        except Exception as e:
            log(f"  → Outcome recording failed: {e}", "WARN")

    # ── Step 6: Save Lessons ──
    log("[6/8] Saving lessons...")
    mem = _get_agent_memory()
    if mem:
        try:
            # Log the cycle
            mem.log_cycle(agent_name="self_evolving_agent", data=mem.load_memory())

            # Save lesson about this cycle
            lesson_parts = [f"Cycle #{cycle_id}: {result['issues_found']} issues"]
            if goal_text:
                lesson_parts.append(f"goal='{goal_text[:50]}'")
            if result["subgoals_executed"] > 0:
                lesson_parts.append(f"executed={result['subgoals_executed']}")

            lesson_text = " | ".join(lesson_parts)
            mem.add_lesson(lesson_text, context="cycle_summary")
            result["lessons_saved"] += 1
            log(f"  → Lesson saved: {lesson_text}")
        except Exception as e:
            log(f"  → Lesson save failed: {e}", "WARN")

    # ── Step 7: Skill Auto-Heal ──
    log("[7/8] Checking skill health...")
    check_and_heal_skills(state)

    # ── Step 8: Auto Skill Creation ──
    log("[8/8] Checking for new skill needs...")
    if goal_text:
        check_and_create_skill(goal_text)

    # Also run self-heal check periodically
    sh = _get_self_heal()
    if sh and cycle_num % 5 == 0:  # Every 5 cycles
        log("  → Running periodic self-heal check...")
        try:
            results = sh.run_checks()
            issues_count = sum(1 for v in results.values() if not v.get("ok", True))
            if issues_count > 0:
                log(f"  → Self-heal found {issues_count} issue(s)", "WARN")
        except Exception as e:
            log(f"  → Self-heal check failed: {e}", "WARN")

    # Save state
    save_cycle_state(state)

    log(f"CYCLE #{cycle_id} COMPLETE: {result['issues_found']} issues, {result['subgoals_executed']} subgoals, {result['lessons_saved']} lessons")
    return result


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
            run_cycle()
        except Exception as e:
            log(f"ERROR in cycle: {e}", "ERROR")
        log(f"Sleeping {CHECK_INTERVAL}s until next cycle...")
        time.sleep(CHECK_INTERVAL)


# --- CLI Commands ---
def cmd_start():
    """Start the daemon."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Daemon already running (pid {pid})")
            return
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink()

    print("Starting self-evolving agent daemon...")
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
    """Show daemon and cycle status."""
    # Daemon status
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Daemon: RUNNING (pid {pid})")
        except (ValueError, ProcessLookupError, PermissionError):
            print("Daemon: NOT RUNNING (stale PID file)")
    else:
        print("Daemon: NOT RUNNING")

    # Cycle state
    state = load_cycle_state()
    print(f"\nCycle State:")
    print(f"  Total cycles:      {state.get('total_cycles', 0)}")
    print(f"  Idle cycles:       {state.get('idle_cycles', 0)}/{IDLE_CYCLES_BEFORE_GOAL}")
    print(f"  Goals completed:   {state.get('goals_completed', 0)}")
    print(f"  Last goal:         {state.get('last_goal', 'none')}")

    # Preferences
    prefs = load_prefs()
    print(f"\nPreferences:")
    for k, v in prefs.items():
        print(f"  {k}: {v}")

    # Recent log
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().strip().splitlines()
        if lines:
            print(f"\n--- Last 15 log entries ---")
            for line in lines[-15:]:
                print(f"  {line}")

    # Active goals
    gp = _get_goal_planner()
    if gp:
        try:
            goals = gp.load_goals_db()
            active = [g for g in goals.values() if g.status != "completed"]
            if active:
                print(f"\n--- Active Goals ({len(active)}) ---")
                for g in active:
                    print(f"  [{g.status}] {g.description[:60]}")
        except Exception:
            pass


def cmd_once():
    """Run a single full cycle."""
    print("Running single self-evolution cycle...\n")
    result = run_cycle()
    print(f"\n{'='*60}")
    print(f"  Cycle Result: {result['cycle_id']}")
    print(f"  Issues found: {result['issues_found']}")
    print(f"  Goal: {result['goal_generated'] or 'none'}")
    print(f"  Subgoals executed: {result['subgoals_executed']}")
    print(f"  Lessons saved: {result['lessons_saved']}")
    print(f"{'='*60}")
    return result


def cmd_set_pref(args):
    """Set a preference key=value."""
    if not args or "=" not in args[0]:
        print("Usage: self_evolving_agent.py set-pref key=value")
        print("Example: self_evolving_agent.py set-pref auto_goal_generation=false")
        sys.exit(1)
    key, value = args[0].split("=", 1)
    set_pref(key.strip(), value.strip())


def cmd_goals():
    """List active goals."""
    gp = _get_goal_planner()
    if not gp:
        print("goal_planner module not available")
        return
    try:
        goals = gp.load_goals_db()
        if not goals:
            print("No goals yet.")
            return
        for gid, g in sorted(goals.items(), key=lambda x: x[1].created_at, reverse=True):
            status_icon = {"pending": "○", "in_progress": "◐", "completed": "●", "failed": "✗"}.get(g.status, "?")
            print(f"  {status_icon} {g.id} — {g.description[:60]}")
            if g.root:
                for sg in g.root.subgoals:
                    sg_icon = {"pending": "○", "in_progress": "◐", "completed": "●", "failed": "✗"}.get(sg.status, "?")
                    print(f"      {sg_icon} {sg.description[:50]}")
    except Exception as e:
        print(f"Failed to load goals: {e}")


def cmd_lessons():
    """Show learned lessons."""
    mem = _get_agent_memory()
    if not mem:
        print("agent_memory module not available")
        return
    try:
        data = mem.load_memory()
        lessons = data.get("learned_lessons", [])
        if not lessons:
            print("No lessons learned yet.")
            return
        print(f"Learned Lessons ({len(lessons)}):")
        for lesson in lessons[-20:]:
            print(f"  [{lesson.get('id', '?')}] {lesson.get('text', '?')[:70]}")
    except Exception as e:
        print(f"Failed to load lessons: {e}")


# --- Entry ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "start": cmd_start,
        "stop": cmd_stop,
        "status": cmd_status,
        "once": cmd_once,
        "set-pref": lambda: cmd_set_pref(args),
        "goals": cmd_goals,
        "lessons": cmd_lessons,
    }

    handler = commands.get(cmd)
    if handler:
        handler()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
