#!/usr/bin/env python3
"""
Self-Healing Module - monitors and auto-remediates Hermes system health.

Checks:
  - Hermes process running?
  - Chromium/CDP responding?
  - Whisper model loaded?
  - Disk space OK?

Auto-remediation:
  - Hermes down → restart with `hermes --tui`
  - Chromium down → start with `--remote-debugging-port=9223`
  - Whisper fails → fallback to TF-IDF ranking
  - Disk >90% → alert + suggest cleanup

CLI:
    python3 self_heal.py check    - Run health checks, show status
    python3 self_heal.py heal     - Run checks and auto-remediate
    python3 self_heal.py history  - Show remediation log
"""

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
HEALING_LOG = DATA_DIR / "healing_log.jsonl"

# --- Thresholds ---
DISK_THRESHOLD = 90  # percent
CDP_PORT = 9223
CDP_TIMEOUT = 5  # seconds

# --- Colors for terminal output ---
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"


def ensure_data_dir():
    """Ensure data directory exists."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def log_healing(action: str, target: str, status: str, detail: str = ""):
    """Append a remediation entry to the healing log."""
    ensure_data_dir()
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "target": target,
        "status": status,
        "detail": detail,
    }
    with open(HEALING_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")
    return entry


def read_history(limit=None):
    """Read healing log entries."""
    entries = []
    if HEALING_LOG.exists():
        with open(HEALING_LOG, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    if limit:
        entries = entries[-limit:]
    return entries


# --- Check Functions ---

def check_hermes_process():
    """Check if Hermes TUI process is running."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "hermes"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            pids = result.stdout.strip().split("\n")
            return True, f"running (PIDs: {', '.join(pids[:3])})"
        return False, "not running"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "check failed (pgrep unavailable)"


def check_chromium_cdp():
    """Check if Chromium is responding on CDP port 9223."""
    try:
        import urllib.request
        url = f"http://127.0.0.1:{CDP_PORT}/json/version"
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=CDP_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
            browser = data.get("Browser", "unknown")
            return True, f"CDP responding ({browser})"
    except ImportError:
        # Fallback to curl
        try:
            result = subprocess.run(
                ["curl", "-s", "--max-time", str(CDP_TIMEOUT),
                 f"http://127.0.0.1:{CDP_PORT}/json/version"],
                capture_output=True, text=True, timeout=CDP_TIMEOUT + 2
            )
            if result.returncode == 0 and result.stdout.strip():
                data = json.loads(result.stdout)
                browser = data.get("Browser", "unknown")
                return True, f"CDP responding ({browser})"
            return False, "CDP not responding"
        except (subprocess.TimeoutExpired, FileNotFoundError, json.JSONDecodeError):
            return False, "CDP check failed"
    except Exception as e:
        return False, f"CDP not responding ({type(e).__name__})"


def check_whisper_model():
    """Check if Whisper model is available/loaded."""
    # Check for faster-whisper or whisper
    try:
        result = subprocess.run(
            ["python3", "-c", "import whisper; print('ok')"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True, "whisper module available"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for faster-whisper
    try:
        result = subprocess.run(
            ["python3", "-c", "import faster_whisper; print('ok')"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return True, "faster_whisper module available"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for any whisper model files in common locations
    home = Path.home()
    model_paths = [
        home / ".cache" / "whisper",
        home / ".hermes" / "models" / "whisper",
    ]
    for p in model_paths:
        if p.exists() and any(p.iterdir()):
            return True, f"model files found at {p}"

    return False, "whisper not available (will use TF-IDF fallback)"


def check_disk_space():
    """Check disk space usage."""
    try:
        stat = shutil.disk_usage("/")
        total_gb = stat.total / (1024 ** 3)
        used_gb = stat.used / (1024 ** 3)
        free_gb = stat.free / (1024 ** 3)
        percent_used = (stat.used / stat.total) * 100

        status = {
            "total_gb": round(total_gb, 1),
            "used_gb": round(used_gb, 1),
            "free_gb": round(free_gb, 1),
            "percent_used": round(percent_used, 1),
        }

        if percent_used >= DISK_THRESHOLD:
            return False, f"{percent_used:.1f}% used ({free_gb:.1f}GB free) — ABOVE {DISK_THRESHOLD}% threshold", status
        return True, f"{percent_used:.1f}% used ({free_gb:.1f}GB free)", status
    except Exception as e:
        return False, f"check failed: {e}", {}


# --- Remediation Functions ---

def remediate_hermes():
    """Restart Hermes with --tui."""
    try:
        # Start hermes in background
        subprocess.Popen(
            ["hermes", "--tui"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        # Wait briefly and verify
        time.sleep(3)
        running, detail = check_hermes_process()
        if running:
            return True, "hermes restarted successfully"
        return False, "hermes started but not detected running"
    except FileNotFoundError:
        return False, "hermes command not found in PATH"
    except Exception as e:
        return False, f"restart failed: {e}"


def remediate_chromium():
    """Start Chromium with remote debugging."""
    try:
        subprocess.Popen(
            [
                "chromium-browser",
                "--headless",
                "--no-sandbox",
                f"--remote-debugging-port={CDP_PORT}",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--remote-allow-origins=*",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        time.sleep(3)
        running, detail = check_chromium_cdp()
        if running:
            return True, "chromium started with CDP"
        return False, "chromium started but CDP not responding"
    except FileNotFoundError:
        # Try google-chrome as fallback
        try:
            subprocess.Popen(
                [
                    "google-chrome",
                    "--headless",
                    "--no-sandbox",
                    f"--remote-debugging-port={CDP_PORT}",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    "--remote-allow-origins=*",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            time.sleep(3)
            running, detail = check_chromium_cdp()
            if running:
                return True, "google-chrome started with CDP"
            return False, "google-chrome started but CDP not responding"
        except FileNotFoundError:
            return False, "neither chromium-browser nor google-chrome found"
    except Exception as e:
        return False, f"chromium start failed: {e}"


def remediate_whisper():
    """Set fallback to TF-IDF ranking."""
    # This is a configuration fallback - set env var or create marker
    fallback_marker = DATA_DIR / ".whisper_fallback"
    try:
        fallback_marker.write_text(
            f"TF-IDF fallback enabled at {datetime.now(timezone.utc).isoformat()}\n"
        )
        return True, "TF-IDF fallback mode activated"
    except Exception as e:
        return False, f"fallback setup failed: {e}"


def remediate_disk():
    """Alert and suggest cleanup commands."""
    suggestions = [
        "Run: sudo apt autoremove && sudo apt clean",
        "Run: journalctl --vacuum-size=100M",
        "Run: docker system prune -f (if using Docker)",
        "Check large files: du -sh /* | sort -rh | head -20",
    ]
    return False, "Disk cleanup required — manual action needed", suggestions


# --- Main Logic ---

def run_checks():
    """Run all health checks and return results."""
    results = {}

    # Check Hermes
    ok, detail = check_hermes_process()
    results["hermes"] = {"ok": ok, "detail": detail}

    # Check Chromium/CDP
    ok, detail = check_chromium_cdp()
    results["chromium_cdp"] = {"ok": ok, "detail": detail}

    # Check Whisper
    ok, detail = check_whisper_model()
    results["whisper"] = {"ok": ok, "detail": detail}

    # Check Disk
    ok, detail, status = check_disk_space()
    results["disk"] = {"ok": ok, "detail": detail, "status": status}

    return results


def display_results(results, title="Health Check Results"):
    """Display check results with colors."""
    print(f"\n{BOLD}{'='*50}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{'='*50}{RESET}\n")

    issues = 0
    for name, info in results().items() if callable(results) else results.items():
        if name == "disk":
            # Disk has nested status
            ok = info.get("ok", False)
            detail = info.get("detail", "")
            status = info.get("status", {})
        else:
            ok = info.get("ok", False)
            detail = info.get("detail", "")

        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        label = name.replace("_", " ").title()

        if not ok:
            issues += 1
            print(f"  {icon} {BOLD}{label}{RESET}: {RED}{detail}{RESET}")
        else:
            print(f"  {icon} {BOLD}{label}{RESET}: {detail}")

    print(f"\n{BOLD}{'─'*50}{RESET}")
    if issues == 0:
        print(f"  {GREEN}{BOLD}All systems healthy ✓{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}{issues} issue(s) detected — run 'heal' to remediate{RESET}")
    print(f"{BOLD}{'─'*50}{RESET}\n")

    return issues


def run_heal():
    """Run checks and auto-remediate issues."""
    print(f"\n{BOLD}{CYAN}Running Self-Heal Sequence...{RESET}\n")

    results = run_checks()
    issues_found = 0

    # Hermes
    hermes_info = results["hermes"]
    if not hermes_info["ok"]:
        issues_found += 1
        print(f"  {YELLOW}→ Hermes down, restarting...{RESET}")
        ok, detail = remediate_hermes()
        status = "success" if ok else "failed"
        log_healing("restart", "hermes", status, detail)
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"    {icon} {detail}")
    else:
        print(f"  {GREEN}✓ Hermes OK{RESET}")

    # Chromium
    chromium_info = results["chromium_cdp"]
    if not chromium_info["ok"]:
        issues_found += 1
        print(f"  {YELLOW}→ Chromium CDP down, starting...{RESET}")
        ok, detail = remediate_chromium()
        status = "success" if ok else "failed"
        log_healing("start", "chromium", status, detail)
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"    {icon} {detail}")
    else:
        print(f"  {GREEN}✓ Chromium CDP OK{RESET}")

    # Whisper
    whisper_info = results["whisper"]
    if not whisper_info["ok"]:
        issues_found += 1
        print(f"  {YELLOW}→ Whisper unavailable, enabling TF-IDF fallback...{RESET}")
        ok, detail = remediate_whisper()
        status = "fallback" if ok else "failed"
        log_healing("fallback", "whisper", status, detail)
        icon = f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"
        print(f"    {icon} {detail}")
    else:
        print(f"  {GREEN}✓ Whisper OK{RESET}")

    # Disk
    disk_info = results["disk"]
    if not disk_info["ok"]:
        issues_found += 1
        print(f"  {YELLOW}→ Disk space critical!{RESET}")
        ok, detail, suggestions = remediate_disk()
        log_healing("alert", "disk", "warning", detail)
        print(f"    {RED}⚠ {detail}{RESET}")
        if suggestions:
            print(f"    {CYAN}Suggested cleanup:{RESET}")
            for s in suggestions:
                print(f"      • {s}")
    else:
        print(f"  {GREEN}✓ Disk OK ({disk_info['detail']}){RESET}")

    print(f"\n{BOLD}{'─'*50}{RESET}")
    if issues_found == 0:
        print(f"  {GREEN}{BOLD}No remediation needed — all healthy{RESET}")
    else:
        print(f"  {YELLOW}{BOLD}{issues_found} issue(s) addressed — see log for details{RESET}")
    print(f"{BOLD}{'─'*50}{RESET}\n")

    return issues_found


def show_history(limit=20):
    """Display remediation history."""
    entries = read_history(limit=limit)

    print(f"\n{BOLD}{'='*60}{RESET}")
    print(f"{BOLD}{CYAN}  Remediation History (last {min(limit, len(entries))} entries){RESET}")
    print(f"{BOLD}{'='*60}{RESET}\n")

    if not entries:
        print(f"  {YELLOW}No remediation events logged yet.{RESET}\n")
        return

    for entry in entries:
        ts = entry.get("timestamp", "unknown")
        action = entry.get("action", "?")
        target = entry.get("target", "?")
        status = entry.get("status", "?")
        detail = entry.get("detail", "")

        if status == "success":
            color = GREEN
            icon = "✓"
        elif status == "fallback":
            color = YELLOW
            icon = "↻"
        elif status == "warning":
            color = YELLOW
            icon = "⚠"
        else:
            color = RED
            icon = "✗"

        print(f"  {color}{icon}{RESET} [{ts}] {BOLD}{action}{RESET} {target}: {color}{status}{RESET}")
        if detail:
            print(f"    └─ {detail}")

    print()


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        print(f"Usage: {sys.argv[0]} <check|heal|history>")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "check":
        results = run_checks()
        display_results(results)

    elif command == "heal":
        run_heal()

    elif command == "history":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        show_history(limit)

    else:
        print(f"Unknown command: {command}")
        print(f"Usage: {sys.argv[0]} <check|heal|history>")
        sys.exit(1)


if __name__ == "__main__":
    main()
