#!/usr/bin/env python3
"""
Agent Spawner - Parallel sub-agent execution manager.

Spawns real background processes for parallel task execution with
lifecycle management (spawn, status, collect, kill), concurrency
limits, timeouts, and auto-retry.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

# --- Configuration ---
WORKSPACE = Path.home() / ".hermes" / "cache" / "agent_spawner"
AGENTS_DIR = WORKSPACE / "agents"
LOGS_DIR = WORKSPACE / "logs"
MAX_AGENTS_DEFAULT = 4
TIMEOUT_DEFAULT = 300  # seconds
RETRY_DEFAULT = 0

# Ensure directories exist
AGENTS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now().isoformat()


def _agent_file(agent_id: str) -> Path:
    return AGENTS_DIR / f"{agent_id}.json"


def _log_file(agent_id: str) -> Path:
    return LOGS_DIR / f"{agent_id}.log"


def _load_agent(agent_id: str) -> Optional[dict]:
    f = _agent_file(agent_id)
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def _save_agent(agent_id: str, data: dict) -> None:
    _agent_file(agent_id).write_text(json.dumps(data, indent=2))


def _list_agents() -> list[str]:
    agents = []
    for f in AGENTS_DIR.glob("*.json"):
        agents.append(f.stem)
    return sorted(agents)


def _count_running() -> int:
    count = 0
    for aid in _list_agents():
        a = _load_agent(aid)
        if a and a.get("status") == "running":
            # Verify process is actually alive
            pid = a.get("pid")
            if pid and not _is_pid_alive(pid):
                a["status"] = "failed"
                a["error"] = "Process died unexpectedly"
                _save_agent(aid, a)
                continue
            count += 1
    return count


def _is_pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _generate_agent_id() -> str:
    return f"agent_{uuid.uuid4().hex[:8]}_{int(time.time())}"


# --- Agent Lifecycle ---

def spawn(
    task: str,
    max_agents: int = MAX_AGENTS_DEFAULT,
    timeout: int = TIMEOUT_DEFAULT,
    retry: int = RETRY_DEFAULT,
    agent_id: Optional[str] = None,
) -> str:
    """Spawn a new agent. Returns agent_id."""
    # Check concurrency limit
    running = _count_running()
    if running >= max_agents:
        raise RuntimeError(
            f"Max concurrent agents reached ({running}/{max_agents}). "
            f"Kill an agent or increase max_agents."
        )

    if agent_id is None:
        agent_id = _generate_agent_id()

    log_file = _log_file(agent_id)
    agent_script = _build_agent_script(task, agent_id)

    # Write the agent script to a temp file
    script_path = WORKSPACE / f"_agent_{agent_id}.py"
    script_path.write_text(agent_script)

    # Launch background process
    log_fh = open(log_file, "w")
    proc = subprocess.Popen(
        [sys.executable, str(script_path)],
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,  # Detach from parent
    )

    # Record agent state
    agent_data = {
        "agent_id": agent_id,
        "task": task,
        "status": "running",
        "pid": proc.pid,
        "created_at": _now(),
        "timeout": timeout,
        "retry": retry,
        "retry_count": 0,
        "max_retries": retry,
        "output_file": None,
        "error": None,
        "exit_code": None,
    }
    _save_agent(agent_id, agent_data)

    # Start timeout monitor in background
    if timeout > 0:
        _start_timeout_monitor(agent_id, timeout)

    return agent_id


def _build_agent_script(task: str, agent_id: str) -> str:
    """Build the Python script that the agent will execute."""
    # Escape the task for embedding in the script
    task_escaped = json.dumps(task)
    agent_id_escaped = json.dumps(agent_id)
    agent_file = json.dumps(str(_agent_file(agent_id)))

    return f'''#!/usr/bin/env python3
"""
Agent subprocess: executes a task and reports back.
"""
import json
import os
import sys
import time
from datetime import datetime

AGENT_ID = {agent_id_escaped}
AGENT_FILE = {agent_file}
TASK = {task_escaped}

def _now():
    return datetime.now().isoformat()

def update_agent(data):
    """Update the agent state file."""
    try:
        existing = {{}}
        if os.path.exists(AGENT_FILE):
            with open(AGENT_FILE, "r") as f:
                existing = json.load(f)
        existing.update(data)
        with open(AGENT_FILE, "w") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        print(f"[Agent {{AGENT_ID}}] Failed to update state: {{e}}", file=sys.stderr)

def main():
    print(f"[Agent {{AGENT_ID}}] Started at {{_now()}}")
    print(f"[Agent {{AGENT_ID}}] Task: {{TASK}}")
    update_agent({{"started_at": _now()}})

    try:
        # --- Execute the task ---
        result = execute_task(TASK)

        # Write output
        output = {{
            "result": result,
            "completed_at": _now(),
        }}
        output_path = os.path.join(os.path.dirname(AGENT_FILE), "..", "logs", f"{{AGENT_ID}}_output.json")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)

        update_agent({{
            "status": "done",
            "exit_code": 0,
            "completed_at": _now(),
            "output_file": output_path,
        }})
        print(f"[Agent {{AGENT_ID}}] Completed successfully at {{_now()}}")

    except Exception as e:
        update_agent({{
            "status": "failed",
            "exit_code": 1,
            "error": str(e),
            "completed_at": _now(),
        }})
        print(f"[Agent {{AGENT_ID}}] Failed: {{e}}", file=sys.stderr)
        sys.exit(1)

def execute_task(task: str) -> str:
    """
    Execute the given task. This is the core logic.
    Tasks are treated as shell commands by default.
    For more complex tasks, this can be extended.
    """
    import subprocess as sp

    # Run the task as a shell command
    result = sp.run(
        task,
        shell=True,
        capture_output=True,
        text=True,
        timeout=600,  # 10 min hard limit per task
    )

    output = result.stdout
    if result.stderr:
        output += "\\n--- stderr ---\\n" + result.stderr

    if result.returncode != 0:
        raise RuntimeError(
            f"Task exited with code {{result.returncode}}: {{output.strip()}}"
        )

    return output.strip()

if __name__ == "__main__":
    main()
'''


def _start_timeout_monitor(agent_id: str, timeout: int) -> None:
    """Start a background monitor that kills the agent after timeout."""
    monitor_script = f'''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path

agent_file = "{str(_agent_file(agent_id))}"
timeout = {timeout}
agent_id = "{agent_id}"

time.sleep(timeout)

if os.path.exists(agent_file):
    with open(agent_file) as f:
        data = json.load(f)
    if data.get("status") == "running":
        pid = data.get("pid")
        if pid:
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
        data["status"] = "failed"
        data["error"] = f"Timeout after {{timeout}}s"
        data["completed_at"] = datetime.now().isoformat() if 'datetime' in dir() else time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(agent_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"[Monitor] Agent {{agent_id}} killed after {{timeout}}s timeout")
'''
    # Fix the monitor script - add missing imports
    monitor_script = monitor_script.replace(
        'data["completed_at"] = datetime.now().isoformat() if \'datetime\' in dir() else time.strftime("%Y-%m-%dT%H:%M:%S")',
        'data["completed_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")'
    )
    # Add signal import
    monitor_script = 'import signal\n' + monitor_script

    monitor_path = WORKSPACE / f"_monitor_{agent_id}.py"
    monitor_path.write_text(monitor_script)

    subprocess.Popen(
        [sys.executable, str(monitor_path)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )


def status(agent_id: str) -> dict:
    """Get the status of an agent."""
    agent = _load_agent(agent_id)
    if agent is None:
        return {"error": f"Agent '{agent_id}' not found"}

    # Check if process is still alive
    if agent.get("status") == "running":
        pid = agent.get("pid")
        if pid and not _is_pid_alive(pid):
            agent["status"] = "failed"
            agent["error"] = "Process died unexpectedly"
            _save_agent(agent_id, agent)

    return {
        "agent_id": agent["agent_id"],
        "status": agent["status"],
        "task": agent.get("task", ""),
        "created_at": agent.get("created_at"),
        "completed_at": agent.get("completed_at"),
        "exit_code": agent.get("exit_code"),
        "error": agent.get("error"),
    }


def collect(agent_id: str) -> dict:
    """Collect the output of a completed agent."""
    agent = _load_agent(agent_id)
    if agent is None:
        return {"error": f"Agent '{agent_id}' not found"}

    result = status(agent_id)

    if agent["status"] == "running":
        result["output"] = None
        result["message"] = "Agent still running"
        return result

    # Try to read output file
    output_file = agent.get("output_file")
    if output_file and os.path.exists(output_file):
        try:
            output_data = json.loads(Path(output_file).read_text())
            result["output"] = output_data.get("result", "")
        except (json.JSONDecodeError, IOError) as e:
            result["output"] = None
            result["output_error"] = str(e)
    else:
        # Fall back to log file
        log_file = _log_file(agent_id)
        if log_file.exists():
            result["output"] = log_file.read_text()
        else:
            result["output"] = None

    return result


def kill(agent_id: str) -> dict:
    """Kill a running agent."""
    agent = _load_agent(agent_id)
    if agent is None:
        return {"error": f"Agent '{agent_id}' not found"}

    if agent.get("status") != "running":
        return {
            "agent_id": agent_id,
            "status": agent["status"],
            "message": "Agent is not running",
        }

    pid = agent.get("pid")
    if pid:
        try:
            os.kill(pid, signal.SIGTERM)
            # Give it a moment to die
            time.sleep(0.5)
            if _is_pid_alive(pid):
                os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, OSError):
            pass

    agent["status"] = "killed"
    agent["completed_at"] = _now()
    _save_agent(agent_id, agent)

    return {
        "agent_id": agent_id,
        "status": "killed",
        "message": f"Agent {agent_id} terminated",
    }


def list_agents(filter_status: Optional[str] = None) -> list[dict]:
    """List all agents, optionally filtered by status."""
    results = []
    for aid in _list_agents():
        s = status(aid)
        if filter_status is None or s.get("status") == filter_status:
            results.append(s)
    return results


def cleanup() -> int:
    """Remove completed/failed/killed agents older than 1 hour."""
    cutoff = datetime.now() - timedelta(hours=1)
    removed = 0
    for aid in _list_agents():
        agent = _load_agent(aid)
        if not agent:
            continue
        if agent.get("status") in ("done", "failed", "killed"):
            completed = agent.get("completed_at", "")
            try:
                if completed and datetime.fromisoformat(completed) < cutoff:
                    _agent_file(aid).unlink(missing_ok=True)
                    _log_file(aid).unlink(missing_ok=True)
                    output_file = agent.get("output_file")
                    if output_file:
                        Path(output_file).unlink(missing_ok=True)
                    removed += 1
            except (ValueError, OSError):
                pass
    return removed


# --- CLI ---

def main():
    parser = argparse.ArgumentParser(
        description="Agent Spawner - Parallel sub-agent execution manager"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # spawn
    spawn_parser = subparsers.add_parser("spawn", help="Spawn a new agent")
    spawn_parser.add_argument("task", help="Task description or command")
    spawn_parser.add_argument(
        "--max-agents", type=int, default=MAX_AGENTS_DEFAULT,
        help=f"Max concurrent agents (default: {MAX_AGENTS_DEFAULT})"
    )
    spawn_parser.add_argument(
        "--timeout", type=int, default=TIMEOUT_DEFAULT,
        help=f"Timeout in seconds (default: {TIMEOUT_DEFAULT})"
    )
    spawn_parser.add_argument(
        "--retry", type=int, default=RETRY_DEFAULT,
        help=f"Number of retries on failure (default: {RETRY_DEFAULT})"
    )

    # status
    status_parser = subparsers.add_parser("status", help="Show agent status")
    status_parser.add_argument("agent_id", nargs="?", help="Agent ID (omit for all)")

    # collect
    collect_parser = subparsers.add_parser("collect", help="Collect agent output")
    collect_parser.add_argument("agent_id", help="Agent ID")

    # list
    list_parser = subparsers.add_parser("list", help="List all agents")
    list_parser.add_argument(
        "--status", choices=["running", "done", "failed", "killed"],
        help="Filter by status"
    )

    # kill
    kill_parser = subparsers.add_parser("kill", help="Kill a running agent")
    kill_parser.add_argument("agent_id", help="Agent ID")

    # cleanup
    cleanup_parser = subparsers.add_parser("cleanup", help="Clean up old completed agents")

    args = parser.parse_args()

    if args.command == "spawn":
        try:
            agent_id = spawn(
                task=args.task,
                max_agents=args.max_agents,
                timeout=args.timeout,
                retry=args.retry,
            )
            print(json.dumps({"agent_id": agent_id, "status": "spawned"}, indent=2))
        except RuntimeError as e:
            print(json.dumps({"error": str(e)}, indent=2), file=sys.stderr)
            sys.exit(1)

    elif args.command == "status":
        if args.agent_id:
            result = status(args.agent_id)
            print(json.dumps(result, indent=2))
        else:
            agents = list_agents()
            print(json.dumps(agents, indent=2))

    elif args.command == "collect":
        result = collect(args.agent_id)
        print(json.dumps(result, indent=2))

    elif args.command == "list":
        agents = list_agents(filter_status=args.status)
        print(json.dumps(agents, indent=2))

    elif args.command == "kill":
        result = kill(args.agent_id)
        print(json.dumps(result, indent=2))

    elif args.command == "cleanup":
        removed = cleanup()
        print(json.dumps({"removed": removed, "message": f"Cleaned up {removed} old agents"}, indent=2))

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
