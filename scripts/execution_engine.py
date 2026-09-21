#!/usr/bin/env python3
"""Execution Engine: actually executes ranked skills via Hermes CLI.

Workflow:
  1. Takes a task description
  2. Runs ATSM rank to get top skill
  3. Generates and executes: hermes chat -q "<task>"
  4. Captures output, records outcome, logs to execution_log.jsonl

Usage:
  python3 execution_engine.py run "task description"
  python3 execution_engine.py batch "task1" "task2" "task3"
  python3 execution_engine.py history
  python3 execution_engine.py stats
  python3 execution_engine.py history --limit 5
"""

import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
DATA_DIR = Path(__file__).parent.parent / "data"
EXECUTION_LOG = DATA_DIR / "execution_log.jsonl"
ATSM_SCRIPT = Path(__file__).parent / "atsm.py"
HERMES_CLI = os.environ.get("HERMES_CLI", "/home/aorus/.local/bin/hermes")

# Default execution timeout per task (seconds)
DEFAULT_TIMEOUT = 120


# --- Logging ---
def log_execution(entry: dict):
    """Append execution entry to JSONL log."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(EXECUTION_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")


def load_history() -> list[dict]:
    """Load all execution history."""
    if not EXECUTION_LOG.exists():
        return []
    records = []
    with open(EXECUTION_LOG, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


# --- ATSM rank wrapper ---
def rank_skills(task: str, top_k: int = 1) -> list[dict]:
    """Run ATSM rank and return top results."""
    cmd = [
        sys.executable, str(ATSM_SCRIPT), "rank", task,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=30,
        )
        # Parse JSON-like output? The atsm.py outputs human-readable text.
        # We need a programmatic way. Let's call rank_skills via import.
        return _rank_via_import(task, top_k)
    except Exception:
        return []


def _rank_via_import(task: str, top_k: int) -> list[dict]:
    """Import atsm module and call rank_skills programmatically."""
    try:
        # Add scripts dir to path
        scripts_dir = str(Path(__file__).parent)
        if scripts_dir not in sys.path:
            sys.path.insert(0, scripts_dir)

        # Clear any cached module to force fresh import
        if "atsm" in sys.modules:
            # Reload to pick up fresh data
            import importlib
            atsm_mod = importlib.reload(sys.modules["atsm"])
        else:
            import atsm as atsm_mod

        results, elapsed = atsm_mod.rank_skills(task, top_k=top_k)
        return results
    except Exception as e:
        print(f"  ⚠ ATSM rank failed: {e}")
        return []


# --- Execution ---
def execute_task(task: str, mode: str = "sync", timeout: int = DEFAULT_TIMEOUT) -> dict:
    """Execute a task by ranking, building command, and running Hermes CLI.

    Args:
        task: Task description.
        mode: 'sync' or 'async'.
        timeout: Max seconds for sync execution.

    Returns:
        Execution entry dict.
    """
    execution_id = str(uuid.uuid4())[:8]
    print(f"\n{'='*60}")
    print(f"  Execution ID: {execution_id}")
    print(f"  Task: {task}")
    print(f"  Mode: {mode}")
    print(f"{'='*60}")

    # Step 1: Rank skills
    print(f"\n  [1/3] Ranking skills via ATSM...")
    start_time = time.time()
    ranked = _rank_via_import(task, top_k=3)

    if ranked:
        top = ranked[0]
        skill = top["name"]
        score = top["final_score"]
        relevance = top["relevance"]
        success_prob = top["success_prob"]
        print(f"  ✓ Top skill: {skill} (score={score}, rel={relevance}, P(success)={success_prob})")
        if len(ranked) > 1:
            others = ", ".join(f"{r['name']}({r['final_score']})" for r in ranked[1:])
            print(f"    Alternatives: {others}")
    else:
        skill = "unknown"
        score = 0.0
        relevance = 0.0
        success_prob = 0.5
        print(f"  ⚠ No ranked results — executing with generic prompt")

    rank_time = time.time() - start_time
    print(f"  ⏱ Rank took {rank_time*1000:.1f}ms")

    # Step 2: Build command
    print(f"\n  [2/3] Building Hermes command...")
    # hermes chat -q "<task>"
    cmd = [HERMES_CLI, "chat", "-q", task]
    print(f"  Command: {' '.join(cmd)}")

    # Step 3: Execute
    print(f"\n  [3/3] Executing ({mode} mode)...")
    exec_start = time.time()

    if mode == "sync":
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            duration = time.time() - exec_start
            output = result.stdout.strip()
            stderr_out = result.stderr.strip()
            success = result.returncode == 0

            print(f"  Return code: {result.returncode}")
            print(f"  Duration: {duration:.2f}s")
            if output:
                # Truncate long output for display
                display_out = output[:500] + "..." if len(output) > 500 else output
                print(f"  Output (first 500 chars):\n{display_out}")
            if stderr_out:
                display_err = stderr_out[:300] + "..." if len(stderr_out) > 300 else stderr_out
                print(f"  Stderr: {display_err}")

        except subprocess.TimeoutExpired:
            duration = time.time() - exec_start
            success = False
            output = f"TIMEOUT after {timeout}s"
            stderr_out = ""
            print(f"  ⚠ Timeout after {timeout}s")

        except FileNotFoundError:
            duration = time.time() - exec_start
            success = False
            output = f"CLI not found at {HERMES_CLI}"
            stderr_out = ""
            print(f"  ✗ Hermes CLI not found at {HERMES_CLI}")

    elif mode == "async":
        # Run in background, return session ID immediately
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        duration = time.time() - exec_start
        success = True  # Process started
        output = f"BACKGROUND pid={proc.pid}"
        stderr_out = ""
        print(f"  ✓ Background process started (pid={proc.pid})")

    else:
        duration = 0
        success = False
        output = f"Unknown mode: {mode}"
        stderr_out = ""

    # Build entry
    entry = {
        "execution_id": execution_id,
        "task": task,
        "skill": skill,
        "skill_score": score,
        "relevance": relevance,
        "success_prob": success_prob,
        "command": " ".join(cmd),
        "mode": mode,
        "output": output,
        "stderr": stderr_out,
        "success": success,
        "return_code": result.returncode if mode == "sync" and 'result' in dir() else (0 if success else -1),
        "duration": round(duration, 3),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Log execution
    log_execution(entry)
    print(f"\n  ✓ Execution logged to {EXECUTION_LOG.name}")

    # Record outcome in ATSM (only for sync)
    if mode == "sync":
        _record_outcome(skill, 1 if success else 0)

    return entry


def _record_outcome(skill: str, success: int):
    """Record outcome to ATSM."""
    try:
        cmd = [
            sys.executable, str(ATSM_SCRIPT), "record", skill, str(success),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        print(f"  ✓ Outcome recorded: {skill} → {'success' if success else 'failure'}")
    except Exception as e:
        print(f"  ⚠ Failed to record outcome: {e}")


def execute_batch(tasks: list[str], parallel: bool = True) -> list[dict]:
    """Execute multiple tasks in batch.

    Args:
        tasks: List of task descriptions.
        parallel: If True, run all in parallel (async); else sequential.

    Returns:
        List of execution entries.
    """
    print(f"\n{'='*60}")
    print(f"  BATCH: {len(tasks)} tasks")
    print(f"  Mode: {'parallel' if parallel else 'sequential'}")
    print(f"{'='*60}")

    entries = []

    if parallel:
        # Launch all processes in parallel
        processes = []
        for task in tasks:
            execution_id = str(uuid.uuid4())[:8]
            ranked = _rank_via_import(task, top_k=1)
            skill = ranked[0]["name"] if ranked else "unknown"
            cmd = [HERMES_CLI, "chat", "-q", task]

            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            processes.append((execution_id, task, skill, cmd, proc))
            print(f"  [{execution_id}] Launched: {task} (skill={skill}, pid={proc.pid})")

        # Wait for all to complete
        print(f"\n  Waiting for {len(processes)} processes...")
        batch_start = time.time()
        for execution_id, task, skill, cmd, proc in processes:
            try:
                stdout, stderr = proc.communicate(timeout=DEFAULT_TIMEOUT)
                duration = time.time() - batch_start
                success = proc.returncode == 0
                output = stdout.strip()
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                duration = DEFAULT_TIMEOUT
                success = False
                output = f"TIMEOUT after {DEFAULT_TIMEOUT}s"

            entry = {
                "execution_id": execution_id,
                "task": task,
                "skill": skill,
                "skill_score": 0,
                "relevance": 0,
                "success_prob": 0,
                "command": " ".join(cmd),
                "mode": "batch-parallel",
                "output": output[:1000],
                "stderr": stderr.strip() if stderr else "",
                "success": success,
                "return_code": proc.returncode,
                "duration": round(duration, 3),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            log_execution(entry)
            entries.append(entry)
            print(f"  [{execution_id}] Done: {task} ({'✓' if success else '✗'})")
    else:
        # Sequential
        for task in tasks:
            entry = execute_task(task, mode="sync")
            entries.append(entry)

    return entries


# --- History & Stats ---
def show_history(limit: int = 20):
    """Show execution history."""
    records = load_history()
    if not records:
        print("No execution history yet.")
        return

    print(f"\n{'='*80}")
    print(f"  Execution History (last {min(limit, len(records))} of {len(records)} total)")
    print(f"{'='*80}")

    for rec in records[-limit:]:
        status = "✓" if rec.get("success") else "✗"
        exec_id = rec.get("execution_id", "?")[:8]
        skill = rec.get("skill", "?")
        task = rec.get("task", "?")[:50]
        duration = rec.get("duration", 0)
        ts = rec.get("timestamp", "?")
        mode = rec.get("mode", "sync")

        print(f"\n  [{exec_id}] {status} {task}")
        print(f"    skill={skill} | {mode} | {duration}s | {ts[:19]}")

        # Show first 200 chars of output
        output = rec.get("output", "")
        if output:
            display = output[:200] + "..." if len(output) > 200 else output
            print(f"    output: {display}")


def show_stats():
    """Show aggregate execution statistics."""
    records = load_history()
    if not records:
        print("No execution history yet.")
        return

    total = len(records)
    successes = sum(1 for r in records if r.get("success"))
    failures = total - successes
    avg_duration = sum(r.get("duration", 0) for r in records) / total

    # Per-skill stats
    skill_stats = {}
    for r in records:
        s = r.get("skill", "unknown")
        if s not in skill_stats:
            skill_stats[s] = {"total": 0, "success": 0, "durations": []}
        skill_stats[s]["total"] += 1
        if r.get("success"):
            skill_stats[s]["success"] += 1
        skill_stats[s]["durations"].append(r.get("duration", 0))

    # Mode breakdown
    mode_stats = {}
    for r in records:
        m = r.get("mode", "sync")
        mode_stats[m] = mode_stats.get(m, 0) + 1

    print(f"\n{'='*60}")
    print(f"  Execution Engine Statistics")
    print(f"{'='*60}")
    print(f"\n  Total executions:  {total}")
    print(f"  Successes:         {successes} ({100*successes/total:.1f}%)")
    print(f"  Failures:          {failures} ({100*failures/total:.1f}%)")
    print(f"  Avg duration:      {avg_duration:.2f}s")
    print(f"  Total time:        {sum(r.get('duration', 0) for r in records):.2f}s")

    print(f"\n  --- Mode Breakdown ---")
    for mode, count in sorted(mode_stats.items()):
        print(f"    {mode}: {count}")

    print(f"\n  --- Per-Skill Breakdown ---")
    for skill, stats in sorted(skill_stats.items(), key=lambda x: x[1]["total"], reverse=True):
        rate = 100 * stats["success"] / stats["total"]
        avg_dur = sum(stats["durations"]) / len(stats["durations"])
        print(f"    {skill:<25} {stats['total']:>3} runs, {rate:.0f}% success, {avg_dur:.2f}s avg")


# --- CLI ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "run":
        if not args:
            print("Usage: execution_engine.py run \"task description\" [--timeout N] [--mode sync|async]")
            sys.exit(1)

        # Parse optional flags
        task = None
        timeout = DEFAULT_TIMEOUT
        mode = "sync"

        i = 0
        while i < len(args):
            if args[i] == "--timeout" and i + 1 < len(args):
                timeout = int(args[i + 1])
                i += 2
            elif args[i] == "--mode" and i + 1 < len(args):
                mode = args[i + 1]
                i += 2
            else:
                # Task description (could be multiple args)
                task = " ".join(args[i:])
                break

        if not task:
            print("Error: no task description provided")
            sys.exit(1)

        entry = execute_task(task, mode=mode, timeout=timeout)

        # Print final summary
        print(f"\n{'='*60}")
        print(f"  Summary: {'SUCCESS' if entry['success'] else 'FAILURE'}")
        print(f"  Execution ID: {entry['execution_id']}")
        print(f"  Skill: {entry['skill']}")
        print(f"  Duration: {entry['duration']}s")
        print(f"{'='*60}")

        sys.exit(0 if entry["success"] else 1)

    elif command == "batch":
        if not args:
            print("Usage: execution_engine.py batch \"task1\" \"task2\" ... [--parallel]")
            sys.exit(1)

        parallel = True
        if "--parallel" in args:
            parallel = True
            args.remove("--parallel")
        elif "--sequential" in args:
            parallel = False
            args.remove("--sequential")

        entries = execute_batch(args, parallel=parallel)

        # Summary
        succeeded = sum(1 for e in entries if e["success"])
        print(f"\n{'='*60}")
        print(f"  Batch Complete: {succeeded}/{len(entries)} succeeded")
        print(f"{'='*60}")

    elif command == "history":
        limit = 20
        if "--limit" in args:
            idx = args.index("--limit")
            if idx + 1 < len(args):
                limit = int(args[idx + 1])
        show_history(limit=limit)

    elif command == "stats":
        show_stats()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
