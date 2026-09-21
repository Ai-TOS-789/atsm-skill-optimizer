#!/usr/bin/env python3
"""ATSM Service Manager — unified daemon controller for all ATSM services.

Manages these services as a single daemon group:
  1. gateway          atsm_gateway.py   (REST API on port 8766)
  2. self_evolve      self_evolve.py     (autonomous evolution loop)
  3. proactive        proactive_agent.py (periodic checks every 30 min)
  4. sys_monitor      sys_monitor.py     (metrics collection loop)
  5. agent_alerts     agent_alerts.py    (alert watch loop)

CLI:
    python3 atsm_service.py start            # start all services
    python3 atsm_service.py stop             # stop all services
    python3 atsm_service.py restart [name]   # restart one (or all)
    python3 atsm_service.py status           # show status of all services
    python3 atsm_service.py health           # health check (exit 0 = all OK)
    python3 atsm_service.py logs [name]      # tail service log (default: manager)
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
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPT_DIR.parent
DATA_DIR = SKILL_DIR / "data"
LOG_DIR = DATA_DIR / "logs"
PID_FILE = DATA_DIR / "atsm_service.pid"
STATE_FILE = DATA_DIR / "atsm_service.state.json"

# --- Service definitions ---
SERVICES = {
    "gateway": {
        "script": SCRIPT_DIR / "atsm_gateway.py",
        "description": "ATSM REST Gateway (port 8766)",
        "foreground_cmd": ["start"],  # gateway.py start runs in foreground
        "port": 8766,
        "health_url": "http://localhost:8766/api/health",
    },
    "self_evolve": {
        "script": SCRIPT_DIR / "self_evolve.py",
        "description": "Self-Evolving Autonomous Agent",
        "foreground_cmd": ["start"],
    },
    "proactive": {
        "script": SCRIPT_DIR / "proactive_agent.py",
        "description": "Proactive Agent (every 30 min)",
        "foreground_cmd": ["start"],
    },
    "sys_monitor": {
        "script": SCRIPT_DIR / "sys_monitor.py",
        "description": "System Metrics Collector",
        "foreground_cmd": None,  # run in custom loop
        "loop_cmd": ["snapshot"],
        "interval": 300,  # 5 minutes
    },
    "agent_alerts": {
        "script": SCRIPT_DIR / "agent_alerts.py",
        "description": "Alert & Notification System",
        "foreground_cmd": None,  # run in custom loop
        "loop_cmd": ["watch"],
        "interval": 120,  # 2 minutes
    },
}


# --- Utilities ---
def ensure_dirs():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] [MANAGER] {msg}"
    print(line, flush=True)
    ensure_dirs()
    with open(LOG_DIR / "atsm_service.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_pid() -> int | None:
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)  # check alive
            return pid
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink(missing_ok=True)
    return None


def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {"services": {}}


def save_state(state: dict):
    ensure_dirs()
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, default=str)


def save_pid(pid: int):
    ensure_dirs()
    PID_FILE.write_text(str(pid))


def clear_pid():
    PID_FILE.unlink(missing_ok=True)


def is_port_in_use(port: int) -> bool:
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


# --- Process management ---
_procs: dict[str, subprocess.Popen | None] = {}


def start_service(name: str) -> bool:
    """Start a single service. Returns True if started successfully."""
    svc = SERVICES[name]
    script = svc["script"]
    if not script.exists():
        log(f"  ERROR: script not found: {script}")
        return False

    log_file = LOG_DIR / f"{name}.log"
    log_fh = open(log_file, "a", encoding="utf-8")

    if svc.get("foreground_cmd"):
        # Service has its own daemon loop (gateway, self_evolve, proactive)
        cmd = [sys.executable, str(script)] + svc["foreground_cmd"]
    else:
        # Service needs our wrapper loop (sys_monitor, agent_alerts)
        # We run a custom loop via the manager
        cmd = [sys.executable, "-c", _LOOP_WRAPPER.format(
            script=str(script),
            loop_cmd=json.dumps(svc["loop_cmd"]),
            interval=svc["interval"],
            log_file=str(log_file),
        )]

    try:
        proc = subprocess.Popen(
            cmd,
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            cwd=str(SCRIPT_DIR),
            # don't inherit signal handler; let manager handle
            preexec_fn=os.setsid,
        )
        _procs[name] = proc
        log(f"  Started {name} (pid {proc.pid})")
        return True
    except Exception as e:
        log(f"  ERROR starting {name}: {e}")
        log_fh.close()
        return False


def stop_service(name: str):
    """Stop a single service."""
    proc = _procs.get(name)
    if proc and proc.poll() is None:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                proc.wait(timeout=3)
            except Exception:
                pass
        log(f"  Stopped {name} (pid {proc.pid})")
    _procs[name] = None

    # Also clean up any external PID files the service may have created
    for svc_name, svc in SERVICES.items():
        pid_file = DATA_DIR / f"{svc_name}.pid"
        if pid_file.exists() and svc_name == name:
            try:
                pid = int(pid_file.read_text().strip())
                os.kill(pid, signal.SIGTERM)
            except (ValueError, ProcessLookupError, PermissionError):
                pass
            pid_file.unlink(missing_ok=True)


_LOOP_WRAPPER = """#!/usr/bin/env python3
import subprocess, sys, time, signal, json, os
script = "{script}"
loop_cmd = json.loads('{loop_cmd}')
interval = {interval}
log_file = "{log_file}"

def handler(sig, frame):
    sys.exit(0)

signal.signal(signal.SIGTERM, handler)
signal.signal(signal.SIGINT, handler)

while True:
    try:
        with open(log_file, "a") as f:
            f.write(f"Starting {{loop_cmd}}...\\n")
            f.flush()
            result = subprocess.run(
                [sys.executable, script] + loop_cmd,
                stdout=f, stderr=subprocess.STDOUT, timeout=interval
            )
            f.write(f"Finished (rc={{result.returncode}}), sleeping {{interval}}s...\\n")
    except subprocess.TimeoutExpired:
        pass
    except Exception as e:
        with open(log_file, "a") as f:
            f.write(f"Error: {{e}}\\n")
    time.sleep(interval)
"""


# --- CLI commands ---
def cmd_start():
    """Start the service manager and all services."""
    existing_pid = read_pid()
    if existing_pid:
        print(f"ATSM Service Manager already running (pid {existing_pid})")
        return

    ensure_dirs()
    log("=" * 60)
    log("Starting ATSM Service Manager...")

    # Write PID file before daemonizing so stop/status work immediately
    save_pid(os.getpid())

    # Daemonize the manager itself
    if os.fork() > 0:
        sys.exit(0)  # parent exits

    os.setsid()

    if os.fork() > 0:
        sys.exit(0)  # session leader exits

    # Redirect stdio to log
    sys.stdout = open(LOG_DIR / "atsm_service.log", "a", encoding="utf-8")
    sys.stderr = sys.stdout

    log("Manager daemonized")
    save_pid(os.getpid())

    # Start all services
    state = {"services": {}, "started_at": datetime.now(timezone.utc).isoformat()}
    for name in SERVICES:
        success = start_service(name)
        state["services"][name] = {
            "status": "running" if success else "failed",
            "started_at": datetime.now(timezone.utc).isoformat(),
            "pid": _procs.get(name).pid if _procs.get(name) else None,
        }

    save_state(state)

    # Monitor loop: restart crashed services
    while True:
        time.sleep(10)
        for name, proc in list(_procs.items()):
            if proc is not None and proc.poll() is not None:
                # Service exited — restart it
                log(f"  Service {name} exited (rc {proc.returncode}), restarting...")
                state = load_state()
                state["services"][name] = {
                    "status": "restarting",
                    "exited_at": datetime.now(timezone.utc).isoformat(),
                    "exit_code": proc.returncode,
                }
                save_state(state)
                start_service(name)
                state["services"][name]["status"] = "running"
                state["services"][name]["pid"] = _procs[name].pid
                state["services"][name]["restarted_at"] = datetime.now(timezone.utc).isoformat()
                save_state(state)


def cmd_stop():
    """Stop the service manager and all services."""
    pid = read_pid()
    if not pid:
        print("ATSM Service Manager not running (no PID file)")

    # Always try to stop all services from state file
    state = load_state()
    stopped_any = False
    for svc_name, svc_info in state.get("services", {}).items():
        svc_pid = svc_info.get("pid")
        if svc_pid:
            try:
                os.kill(svc_pid, signal.SIGTERM)
                log(f"  Stopped {svc_name} (pid {svc_pid})")
                stopped_any = True
            except (ProcessLookupError, PermissionError):
                pass
        # Also check for orphaned wrapper processes
        try:
            result = subprocess.run(
                ["pgrep", "-f", f"{SERVICES[svc_name]['script'].name}"],
                capture_output=True, text=True, timeout=5
            )
            for pid_str in result.stdout.strip().splitlines():
                if pid_str:
                    try:
                        os.kill(int(pid_str), signal.SIGTERM)
                        stopped_any = True
                    except (ProcessLookupError, ValueError, PermissionError):
                        pass
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Clean up PID files from individual services
    for svc_name in SERVICES:
        pid_file = DATA_DIR / f"{svc_name}.pid"
        if pid_file.exists():
            try:
                svc_pid = int(pid_file.read_text().strip())
                os.kill(svc_pid, signal.SIGTERM)
                stopped_any = True
            except (ValueError, ProcessLookupError, PermissionError):
                pass
            pid_file.unlink(missing_ok=True)

    if not pid:
        if stopped_any:
            print("Orphaned services stopped")
        clear_pid()
        return

    log(f"Stopping ATSM Service Manager (pid {pid})...")

    # Kill the manager daemon itself
    try:
        os.kill(pid, signal.SIGTERM)
        # Wait for PID file to be removed
        for _ in range(20):
            if not PID_FILE.exists():
                break
            time.sleep(0.3)
        PID_FILE.unlink(missing_ok=True)
        print("ATSM Service Manager stopped")
    except ProcessLookupError:
        print("Manager not running (stale PID file)")
        clear_pid()
    except PermissionError:
        print(f"Permission denied to signal manager pid {pid}")


def cmd_restart(args):
    """Restart one service or all services."""
    name = args[0] if args else None
    if name and name not in SERVICES:
        print(f"Unknown service: {name}")
        print(f"Available: {', '.join(SERVICES.keys())}")
        sys.exit(1)

    pid = read_pid()
    if pid:
        # Manager is running — signal it
        if name:
            # Restart specific service
            try:
                # Send SIGHUP + service name via state file to trigger restart
                state = load_state()
                state["services"][name] = {"status": "restart_requested"}
                save_state(state)
                os.kill(pid, signal.SIGHUP)
                print(f"Restart signal sent to {name}")
            except (ProcessLookupError, PermissionError):
                print(f"Cannot signal manager (pid {pid})")
        else:
            try:
                os.kill(pid, signal.SIGTERM)
                print("Manager stopping (will auto-restart via systemctl or similar)")
            except (ProcessLookupError, PermissionError):
                print(f"Cannot signal manager (pid {pid})")
    else:
        # Manager not running — use stop/start pattern
        print("Manager not running, starting fresh...")
        for svc_name in ([name] if name else list(SERVICES.keys())):
            stop_service(svc_name)
        if not name:
            cmd_start()
        else:
            start_service(name)
            state = load_state()
            state["services"][name] = {
                "status": "running",
                "pid": _procs[name].pid,
            }
            save_state(state)


def cmd_status():
    """Show status of the manager and all services."""
    pid = read_pid()
    if pid:
        print(f"ATSM Service Manager: running (pid {pid})")
    else:
        print("ATSM Service Manager: not running")

    print()
    print(f"{'Service':<16} {'Status':<12} {'PID':<8} {'Details'}")
    print("─" * 65)

    state = load_state()
    for name, svc in SERVICES.items():
        proc = _procs.get(name)
        entry = state.get("services", {}).get(name, {})

        # Check if process is alive
        is_running = False
        if proc and proc.poll() is None:
            is_running = True
        elif entry.get("pid"):
            try:
                os.kill(entry["pid"], 0)
                is_running = True
            except (ProcessLookupError, PermissionError):
                pass

        status = "running" if is_running else "stopped"
        if entry.get("status") == "restart_requested":
            status = "restarting"

        pid_str = str(entry.get("pid", proc.pid if proc else "─"))
        if not is_running:
            pid_str = "─"

        details = svc["description"]
        if name == "gateway" and svc.get("port"):
            if is_port_in_use(svc["port"]):
                details += " ✓"
            else:
                details += " ✗ (port closed)"

        print(f"{name:<16} {status:<12} {pid_str:<8} {details}")

    print()
    print(f"Log directory: {LOG_DIR}")
    print(f"PID file: {PID_FILE}")
    print(f"State file: {STATE_FILE}")


def cmd_health():
    """Health check — exit 0 if all services healthy, exit 1 otherwise."""
    pid = read_pid()
    if not pid:
        print("CRITICAL: ATSM Service Manager not running")
        sys.exit(1)

    state = load_state()
    all_ok = True

    for name, svc in SERVICES.items():
        entry = state.get("services", {}).get(name, {})
        proc = _procs.get(name)

        is_running = False
        if proc and proc.poll() is None:
            is_running = True
        elif entry.get("pid"):
            try:
                os.kill(entry["pid"], 0)
                is_running = True
            except (ProcessLookupError, PermissionError):
                pass

        if not is_running:
            all_ok = False

    if all_ok:
        print("OK: All services running")
        sys.exit(0)
    else:
        print("WARNING: Some services not running")
        sys.exit(1)


def cmd_logs(args):
    """Tail the service logs."""
    name = args[0] if args else "atsm_service"
    if name in SERVICES:
        log_file = LOG_DIR / f"{name}.log"
    else:
        log_file = LOG_DIR / f"{name}.log"

    if not log_file.exists():
        print(f"No log file: {log_file}")
        return

    # Print last 50 lines
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
        for line in lines[-50:]:
            print(line, end="")
    except IOError as e:
        print(f"Error reading log: {e}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    args = sys.argv[2:]

    if cmd == "start":
        cmd_start()
    elif cmd == "stop":
        cmd_stop()
    elif cmd == "restart":
        cmd_restart(args)
    elif cmd == "status":
        cmd_status()
    elif cmd == "health":
        cmd_health()
    elif cmd == "logs":
        cmd_logs(args)
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
