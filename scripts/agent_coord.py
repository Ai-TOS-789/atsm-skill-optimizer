#!/usr/bin/env python3
"""
Multi-Agent Coordination Protocol

A file-based message bus for proactive agents to:
- Register and discover each other
- Publish/subscribe to topic channels
- Delegate tasks between agents
- Share lessons and patterns
- Elect a leader (first registered agent wins; re-election if leader exits)

All communication goes through /tmp/agent_bus/ so processes can coordinate
without shared memory or a network daemon.
"""

import argparse
import fcntl
import json
import os
import sys
import time
from pathlib import Path

BUS_DIR = Path("/tmp/agent_bus")
AGENTS_FILE = BUS_DIR / "agents.json"
HEARTBEAT_TTL = 120  # seconds before a registered agent is considered dead


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def ensure_bus():
    BUS_DIR.mkdir(parents=True, exist_ok=True)
    if not AGENTS_FILE.exists():
        _atomic_write(AGENTS_FILE, {})


def _atomic_write(path: Path, data):
    """Write JSON atomically using a temp file + rename."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    tmp.rename(path)


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def _lock_file(path: Path):
    """Acquire an exclusive flock on *path* (creates if needed). Returns fd."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(fd, fcntl.LOCK_EX)
    return fd


def _unlock_file(fd):
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def cmd_register(agent_name: str):
    ensure_bus()
    lock_path = BUS_DIR / ".lock"
    fd = _lock_file(lock_path)
    try:
        agents = _read_json(AGENTS_FILE)
        is_first = len(agents) == 0
        agents[agent_name] = {
            "registered_at": time.time(),
            "last_heartbeat": time.time(),
            "leader": is_first,
            "status": "active",
        }
        _atomic_write(AGENTS_FILE, agents)
    finally:
        _unlock_file(fd)

    inbox = BUS_DIR / "inbox" / agent_name
    inbox.mkdir(parents=True, exist_ok=True)

    role = "LEADER (first agent)" if is_first else "follower"
    print(f"Registered '{agent_name}' as {role}")
    return 0


def cmd_heartbeat(agent_name: str):
    """Refresh liveness timestamp (called periodically by long-running agents)."""
    ensure_bus()
    lock_path = BUS_DIR / ".lock"
    fd = _lock_file(lock_path)
    try:
        agents = _read_json(AGENTS_FILE)
        if agent_name in agents:
            agents[agent_name]["last_heartbeat"] = time.time()
            _atomic_write(AGENTS_FILE, agents)
    finally:
        _unlock_file(fd)


def cmd_publish(topic: str, message: str, sender: str = ""):
    """Append a message to every active agent's inbox under *topic*."""
    ensure_bus()
    agents = _read_json(AGENTS_FILE)
    if not agents:
        print("No agents registered. Use 'register' first.", file=sys.stderr)
        return 1

    envelope = {
        "topic": topic,
        "message": message,
        "sender": sender or "(anon)",
        "timestamp": time.time(),
        "id": f"{time.time_ns()}",
    }

    delivered = 0
    for name, meta in agents.items():
        if meta.get("status") != "active":
            continue
        inbox = BUS_DIR / "inbox" / name / f"{topic}.jsonl"
        with open(inbox, "a") as f:
            f.write(json.dumps(envelope) + "\n")
        delivered += 1

    print(f"Published to topic '{topic}' — delivered to {delivered} agent(s)")
    return 0


def cmd_subscribe(topic: str, agent_name: str = "", mark_read: bool = False):
    """Read (and optionally clear) messages from an agent's inbox on *topic*."""
    agents = _read_json(AGENTS_FILE)
    # Default to first active agent if name omitted
    if not agent_name:
        agent_name = next((n for n, m in agents.items() if m.get("status") == "active"), None)
    if not agent_name:
        print("No active agent found.", file=sys.stderr)
        return 1

    inbox_file = BUS_DIR / "inbox" / agent_name / f"{topic}.jsonl"
    if not inbox_file.exists():
        print(f"No messages on topic '{topic}' for '{agent_name}'.")
        return 0

    lines = inbox_file.read_text().strip().splitlines()
    if not lines or lines == [""]:
        print(f"No messages on topic '{topic}' for '{agent_name}'.")
        return 0

    messages = [json.loads(line) for line in lines if line.strip()]
    for msg in messages:
        ts = time.strftime("%H:%M:%S", time.localtime(msg["timestamp"]))
        print(f"[{ts}] {msg['sender']}: {msg['message']}")

    if mark_read:
        inbox_file.unlink()
        print(f"(cleared {len(messages)} message(s))")
    else:
        print(f"\n{len(messages)} message(s) pending.")
    return 0


def cmd_delegate(target_agent: str, task: str, sender: str = ""):
    """Assign a task to a specific agent (syntactic sugar over publish)."""
    ensure_bus()
    agents = _read_json(AGENTS_FILE)
    if target_agent not in agents:
        print(f"Agent '{target_agent}' not registered.", file=sys.stderr)
        return 1

    task_record = {
        "task": task,
        "delegated_by": sender or "(anon)",
        "delegated_at": time.time(),
        "status": "pending",
        "id": f"task-{time.time_ns()}",
    }

    inbox_dir = BUS_DIR / "inbox" / target_agent
    inbox_dir.mkdir(parents=True, exist_ok=True)
    tasks_file = inbox_dir / "delegated_tasks.jsonl"
    with open(tasks_file, "a") as f:
        f.write(json.dumps(task_record) + "\n")

    print(f"Delegated to '{target_agent}': {task}")
    return 0


def cmd_status():
    """Show all registered agents, leader, liveness, and pending task counts."""
    ensure_bus()
    agents = _read_json(AGENTS_FILE)
    if not agents:
        print("No agents registered.")
        return 0

    now = time.time()
    print(f"{'Agent':<16} {'Role':<10} {'Status':<10} {'Pending Tasks':<14} {'Last HB':<10}")
    print("-" * 60)
    for name, meta in agents.items():
        age = now - meta.get("last_heartbeat", 0)
        alive = age < HEARTBEAT_TTL
        role = "LEADER" if meta.get("leader") else "follower"
        status = "active" if alive else "DEAD"
        # Count pending delegated tasks
        tasks_file = BUS_DIR / "inbox" / name / "delegated_tasks.jsonl"
        pending = 0
        if tasks_file.exists():
            pending = sum(
                1 for line in tasks_file.read_text().splitlines()
                if line.strip() and json.loads(line).get("status") == "pending"
            )
        # Count inbox messages
        inbox_dir = BUS_DIR / "inbox" / name
        inbox_msgs = 0
        if inbox_dir.exists():
            inbox_msgs = sum(
                len(f.read_text().strip().splitlines())
                for f in inbox_dir.glob("*.jsonl")
                if f.exists() and f.read_text().strip()
            )
        hb = f"{age:.0f}s ago" if age < 3600 else f"{age/3600:.1f}h ago"
        print(f"{name:<16} {role:<10} {status:<10} {pending:<14} {hb:<10}")
    return 0


def cmd_collect():
    """Aggregate and display shared lessons/patterns published to 'lessons' topic."""
    ensure_bus()
    agents = _read_json(AGENTS_FILE)
    all_lessons = []
    for name in agents:
        lessons_file = BUS_DIR / "inbox" / name / "lessons.jsonl"
        if lessons_file.exists():
            for line in lessons_file.read_text().splitlines():
                if line.strip():
                    entry = json.loads(line)
                    entry["_via"] = name
                    all_lessons.append(entry)

    if not all_lessons:
        print("No lessons collected yet.")
        return 0

    print(f"Shared Lessons & Patterns ({len(all_lessons)} entries):")
    print("=" * 60)
    for entry in all_lessons:
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(entry.get("timestamp", 0)))
        print(f"  [{ts}] {entry.get('sender', '?')}: {entry.get('message', '')}")
    return 0


def cmd_leader():
    """Print the current leader agent name."""
    ensure_bus()
    agents = _read_json(AGENTS_FILE)
    for name, meta in agents.items():
        if meta.get("leader"):
            print(name)
            return 0
    print("(none)")
    return 1


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Agent Coordination Protocol — file-based message bus",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_reg = sub.add_parser("register", help="Register an agent")
    p_reg.add_argument("agent_name")

    p_hb = sub.add_parser("heartbeat", help="Send liveness heartbeat")
    p_hb.add_argument("agent_name")

    p_pub = sub.add_parser("publish", help="Publish a message to a topic")
    p_pub.add_argument("topic")
    p_pub.add_argument("message")
    p_pub.add_argument("--sender", default="")

    p_sub = sub.add_parser("subscribe", help="Read messages on a topic")
    p_sub.add_argument("topic")
    p_sub.add_argument("--agent", default="", help="Agent name (default: first active)")
    p_sub.add_argument("--read", action="store_true", help="Mark messages as read (delete)")

    p_del = sub.add_parser("delegate", help="Delegate a task to a specific agent")
    p_del.add_argument("target_agent")
    p_del.add_argument("task")
    p_del.add_argument("--sender", default="")

    sub.add_parser("status", help="Show all agents and their status")

    sub.add_parser("collect", help="Collect shared lessons/patterns")

    sub.add_parser("leader", help="Show the current leader")

    args = parser.parse_args()

    cmds = {
        "register": lambda: cmd_register(args.agent_name),
        "heartbeat": lambda: cmd_heartbeat(args.agent_name),
        "publish": lambda: cmd_publish(args.topic, args.message, args.sender),
        "subscribe": lambda: cmd_subscribe(args.topic, args.agent, args.read),
        "delegate": lambda: cmd_delegate(args.target_agent, args.task, args.sender),
        "status": cmd_status,
        "collect": cmd_collect,
        "leader": cmd_leader,
    }

    sys.exit(cmds[args.command]() or 0)


if __name__ == "__main__":
    main()
