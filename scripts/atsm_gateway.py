#!/usr/bin/env python3
"""ATSM Gateway — REST API server for the Adaptive Task-Skill Matching engine.

Exposes HTTP endpoints for ranking skills, recording outcomes, viewing stats,
planning skill chains, serving the dashboard UI, and proxying ATSM module CLIs.
Runs a background proactive agent every 5 minutes.

CLI:
    python3 atsm_gateway.py start    # start server (foreground)
    python3 atsm_gateway.py stop     # stop running server
    python3 atsm_gateway.py status   # show server status

API Endpoints:
    POST /api/rank     {"task": "...", "top_k": 10}
    POST /api/record   {"skill": "...", "success": 1}
    GET  /api/stats
    GET  /api/health
    GET  /api/lessons
    POST /api/chain    {"command": "..."}
    GET  /api/status
    GET  /ui            → serves the ATSM dashboard HTML
    GET  /api/dashboard → returns dashboard data as JSON
    GET  /api/flow      → returns visual flow DOT
    POST /api/proxy     {"module": "...", "args": [...]} → forwards to any ATSM module CLI
"""

import json
import mimetypes
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# --- Paths ---
SKILL_DIR = Path(__file__).parent
DATA_DIR = SKILL_DIR.parent / "data"
UI_DIR = SKILL_DIR.parent / "ui"
DB_FILE = DATA_DIR / "atsm_db.jsonl"
CHAINS_FILE = DATA_DIR / "chains_db.jsonl"
PROACTIVE_AGENT = SKILL_DIR / "proactive_agent.py"
PID_FILE = DATA_DIR / "atsm_gateway.pid"
LOG_FILE = DATA_DIR / "atsm_gateway.log"

PORT = 8766
PROACTIVE_INTERVAL = 300  # 5 minutes

# --- Import ATSM modules ---
sys.path.insert(0, str(SKILL_DIR))
import atsm
import skill_chain


# --- Logging ---
def log(msg: str):
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# --- Background proactive agent ---
class ProactiveScheduler:
    """Runs proactive_agent.py once every PROACTIVE_INTERVAL seconds."""

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        log("PROACTIVE scheduler started")

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                result = subprocess.run(
                    [sys.executable, str(PROACTIVE_AGENT), "once"],
                    capture_output=True, text=True, timeout=120
                )
                if result.returncode == 0:
                    log("PROACTIVE: cycle completed")
                else:
                    log(f"PROACTIVE: error — {result.stderr[:200]}")
            except subprocess.TimeoutExpired:
                log("PROACTIVE: cycle timed out")
            except Exception as e:
                log(f"PROACTIVE: exception — {e}")
            self._stop_event.wait(PROACTIVE_INTERVAL)


# --- Request Handler ---
class ATSMHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ATSM Gateway API."""

    def log_message(self, format, *args):
        """Override to use our logging."""
        log(f"HTTP: {args[0] if args else ''}")

    def _send_json(self, data: dict, status: int = 200):
        """Send JSON response with CORS headers."""
        body = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _send_text(self, text: str, content_type: str = "text/plain; charset=utf-8", status: int = 200):
        """Send text response with CORS headers."""
        body = text.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        """Read and parse JSON request body."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        raw = self.rfile.read(content_length)
        try:
            return json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            return {}

    def do_OPTIONS(self):
        """Handle CORS preflight."""
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def _serve_static(self, file_path: Path):
        """Serve a static file from the ui/ directory."""
        if not file_path.exists() or not file_path.is_file():
            self._send_json({"error": "Not found", "path": str(file_path)}, status=404)
            return

        # Security: prevent path traversal
        try:
            file_path.resolve().relative_to(UI_DIR.resolve())
        except ValueError:
            self._send_json({"error": "Forbidden"}, status=403)
            return

        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = "application/octet-stream"

        try:
            content = file_path.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(content)))
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(content)
        except Exception as e:
            self._send_json({"error": f"Failed to read file: {e}"}, status=500)

    def _generate_flow_dot(self) -> str:
        """Generate a DOT graph of ATSM skill flow."""
        records = atsm.load_db()
        stats = atsm.compute_stats(records)
        chains = skill_chain.load_chains()

        # Build co-occurrence from chains
        cooc = {}
        for chain in chains:
            steps = chain.get("steps", [])
            for i in range(len(steps) - 1):
                key = f"{steps[i]}|||{steps[i+1]}"
                cooc[key] = cooc.get(key, 0) + 1

        lines = ["digraph ATSM_Flow {", "  rankdir=LR;"]
        lines.append('  node [shape=box, style=rounded, fontname="sans-serif"];')
        lines.append('  edge [fontname="sans-serif", fontsize=10];')
        lines.append("")

        # Add skill nodes with success rate coloring
        all_skills = set()
        for name in stats:
            all_skills.add(name)
        for chain in chains:
            for step in chain.get("steps", []):
                all_skills.add(step)

        for skill in sorted(all_skills):
            s = stats.get(skill)
            if s:
                success_rate = s.get("expected_success", 0.5)
                # Color from red (0) to green (1)
                r = int(255 * (1 - success_rate))
                g = int(255 * success_rate)
                b = 100
                label = f"{skill}\\nP={success_rate:.2f}"
            else:
                r, g, b = 180, 180, 180
                label = skill
            lines.append(f'  "{skill}" [label="{label}", fillcolor="rgb({r},{g},{b})", style="filled,rounded"];')

        lines.append("")

        # Add edges from co-occurrence
        for key, weight in sorted(cooc.items(), key=lambda x: -x[1]):
            parts = key.split("|||")
            if len(parts) == 2:
                src, dst = parts
                penwidth = min(max(weight, 1), 5)
                lines.append(f'  "{src}" -> "{dst}" [label={weight}, penwidth={penwidth}];')

        lines.append("}")
        return "\n".join(lines)

    def _get_dashboard_data(self) -> dict:
        """Return comprehensive dashboard data as JSON."""
        records = atsm.load_db()
        stats = atsm.compute_stats(records)
        chains = skill_chain.load_chains()

        # Build skill summary
        skills_summary = []
        for name, s in sorted(stats.items(), key=lambda x: x[1].get("expected_success", 0), reverse=True):
            skills_summary.append({
                "name": name,
                "successes": s["raw_successes"],
                "failures": s["raw_failures"],
                "total": s["total"],
                "expected_success": round(s["expected_success"], 4),
                "confidence": round(s["confidence"], 4),
                "effective_total": round(s["effective_total"], 2),
            })

        # Build co-occurrence from chains
        cooc = {}
        for chain in chains:
            steps = chain.get("steps", [])
            for i in range(len(steps) - 1):
                key = f"{steps[i]}|||{steps[i+1]}"
                cooc[key] = cooc.get(key, 0) + 1

        # Flow data (nodes and edges)
        nodes = [{"id": name, "expected_success": round(s.get("expected_success", 0.5), 4)} for name, s in stats.items()]
        edges = []
        for key, weight in cooc.items():
            parts = key.split("|||")
            if len(parts) == 2:
                edges.append({"source": parts[0], "target": parts[1], "weight": weight})

        # Recent log entries
        recent_log = []
        if LOG_FILE.exists():
            try:
                log_lines = LOG_FILE.read_text().strip().splitlines()
                recent_log = log_lines[-20:]
            except:
                pass

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_records": len(records),
            "total_skills_tracked": len(stats),
            "total_chains_executed": len(chains),
            "skills_summary": skills_summary,
            "flow": {
                "nodes": nodes,
                "edges": edges,
            },
            "recent_log": recent_log,
            "lessons": _get_lessons(),
        }

    def do_GET(self):
        """Handle GET requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/ui":
            # Serve the ATSM dashboard HTML
            html_file = UI_DIR / "atsm-visualizer.html"
            if not html_file.exists():
                self._send_json({"error": "Dashboard HTML not found", "path": str(html_file)}, status=404)
                return
            self._serve_static(html_file)
            return

        # Static file serving for ui/ directory
        if path.startswith("/ui/"):
            relative = path[len("/ui/"):]
            file_path = UI_DIR / relative
            self._serve_static(file_path)
            return

        if path == "/api/health":
            self._send_json({
                "status": "ok",
                "service": "atsm-gateway",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "port": PORT,
            })

        elif path == "/api/stats":
            records = atsm.load_db()
            stats = atsm.compute_stats(records)
            # Convert to serializable format
            serializable = {}
            for name, s in stats.items():
                serializable[name] = {
                    "successes": s["raw_successes"],
                    "failures": s["raw_failures"],
                    "total": s["total"],
                    "expected_success": round(s["expected_success"], 4),
                    "confidence": round(s["confidence"], 4),
                    "effective_total": round(s["effective_total"], 2),
                }
            self._send_json({
                "skills": serializable,
                "total_records": len(records),
                "total_skills": len(serializable),
            })

        elif path == "/api/lessons":
            lessons = _get_lessons()
            self._send_json({"lessons": lessons})

        elif path == "/api/status":
            records = atsm.load_db()
            stats = atsm.compute_stats(records)
            chains = skill_chain.load_chains()
            # Compute overall metrics
            total_success = sum(s["raw_successes"] for s in stats.values())
            total_fail = sum(s["raw_failures"] for s in stats.values())
            total_obs = total_success + total_fail
            avg_confidence = (
                sum(s["confidence"] for s in stats.values()) / len(stats)
                if stats else 0.0
            )
            self._send_json({
                "status": "running",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_records": len(records),
                "total_skills_tracked": len(stats),
                "total_chains_executed": len(chains),
                "overall_success_rate": (
                    round(total_success / total_obs, 4) if total_obs > 0 else None
                ),
                "average_confidence": round(avg_confidence, 4),
                "proactive_interval_sec": PROACTIVE_INTERVAL,
            })

        elif path == "/api/dashboard":
            # Return comprehensive dashboard data as JSON
            dashboard_data = self._get_dashboard_data()
            self._send_json(dashboard_data)

        elif path == "/api/flow":
            # Return visual flow DOT
            dot = self._generate_flow_dot()
            self._send_text(dot, content_type="text/vnd.graphviz; charset=utf-8")

        else:
            self._send_json({"error": "Not found", "path": path}, status=404)

    def do_POST(self):
        """Handle POST requests."""
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")
        body = self._read_body()

        if path == "/api/rank":
            task = body.get("task", "")
            top_k = int(body.get("top_k", 10))
            if not task:
                self._send_json({"error": "Missing 'task' field"}, status=400)
                return
            try:
                results, elapsed = atsm.rank_skills(task, top_k=top_k)
                self._send_json({
                    "task": task,
                    "results": results,
                    "elapsed_ms": round(elapsed * 1000, 2),
                    "count": len(results),
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/record":
            skill = body.get("skill", "")
            success = body.get("success")
            if not skill or success is None:
                self._send_json(
                    {"error": "Missing 'skill' or 'success' field"}, status=400
                )
                return
            try:
                success = int(success)
                if success not in (0, 1):
                    raise ValueError("success must be 0 or 1")
                atsm.append_record(skill, success)
                self._send_json({
                    "status": "recorded",
                    "skill": skill,
                    "success": success,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=400)

        elif path == "/api/chain":
            command = body.get("command", "")
            if not command:
                self._send_json({"error": "Missing 'command' field"}, status=400)
                return
            try:
                chain_result = skill_chain.cmd_plan(command)
                # Also run the chain to get full execution record
                execution = skill_chain.cmd_run(command, simulate=True)
                self._send_json({
                    "command": command,
                    "planned_chain": [
                        {"name": c["name"], "description": c["description"],
                         "score": c["final_score"]}
                        for c in chain_result
                    ],
                    "execution": execution,
                })
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        elif path == "/api/proxy":
            # Forward request to any ATSM module CLI
            module = body.get("module", "")
            args = body.get("args", [])
            if not module:
                self._send_json({"error": "Missing 'module' field"}, status=400)
                return

            # Whitelist of allowed modules for security
            allowed_modules = {
                "atsm", "skill_chain", "proactive_agent", "embed_rank",
                "self_heal", "goal_planner", "execution_engine",
            }

            if module not in allowed_modules:
                self._send_json(
                    {"error": f"Module '{module}' not allowed", "allowed": sorted(allowed_modules)},
                    status=403,
                )
                return

            script_path = SKILL_DIR / f"{module}.py"
            if not script_path.exists():
                self._send_json(
                    {"error": f"Module script not found: {script_path}"}, status=404
                )
                return

            # Sanitize args
            clean_args = [str(a) for a in args]

            try:
                result = subprocess.run(
                    [sys.executable, str(script_path)] + clean_args,
                    capture_output=True, text=True, timeout=60
                )
                self._send_json({
                    "module": module,
                    "args": clean_args,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                })
            except subprocess.TimeoutExpired:
                self._send_json(
                    {"error": "Module execution timed out (60s limit)"},
                    status=504,
                )
            except Exception as e:
                self._send_json({"error": str(e)}, status=500)

        else:
            self._send_json({"error": "Not found", "path": path}, status=404)


def _get_lessons() -> list[dict]:
    """Extract learned lessons from chain history and ATSM stats."""
    lessons = []
    chains = skill_chain.load_chains()
    avoid_pairs = skill_chain.get_avoid_pairs()

    # Lessons from avoid pairs
    for prev, curr in avoid_pairs:
        lessons.append({
            "type": "avoid_pair",
            "lesson": f"Skill chain '{prev} → {curr}' has historically failed — avoid this sequence",
            "evidence": f"Co-failed 2+ times in chain history",
            "confidence": "high",
        })

    # Lessons from low-success skills
    records = atsm.load_db()
    stats = atsm.compute_stats(records)
    for name, s in stats.items():
        if s["total"] >= 3 and s["expected_success"] < 0.4:
            lessons.append({
                "type": "low_success_skill",
                "lesson": f"Skill '{name}' has low expected success ({s['expected_success']:.0%}) — review or retrain",
                "evidence": f"{s['raw_successes']}/{s['total']} successes",
                "confidence": "medium" if s["confidence"] < 0.7 else "high",
            })

    # Lessons from high-success skills
    for name, s in stats.items():
        if s["total"] >= 3 and s["expected_success"] > 0.8:
            lessons.append({
                "type": "high_success_skill",
                "lesson": f"Skill '{name}' is highly reliable ({s['expected_success']:.0%}) — prefer for relevant tasks",
                "evidence": f"{s['raw_successes']}/{s['total']} successes",
                "confidence": "high",
            })

    return lessons


# --- Server ---
_server = None
_proactive = None


def start_server():
    """Start the HTTP server and proactive scheduler."""
    global _server, _proactive

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already running
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Gateway already running (pid {pid})")
            return
        except (ValueError, ProcessLookupError, PermissionError):
            PID_FILE.unlink()

    # Write PID
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))

    def handle_signal(signum, frame):
        log(f"Received signal {signum}, shutting down")
        stop_server()
        sys.exit(0)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    # Start proactive scheduler
    _proactive = ProactiveScheduler()
    _proactive.start()

    # Start HTTP server
    _server = HTTPServer(("0.0.0.0", PORT), ATSMHandler)
    log(f"ATSM Gateway listening on port {PORT}")
    print(f"ATSM Gateway running on http://localhost:{PORT}")
    print(f"  Health:    GET  http://localhost:{PORT}/api/health")
    print(f"  Rank:      POST http://localhost:{PORT}/api/rank")
    print(f"  Record:    POST http://localhost:{PORT}/api/record")
    print(f"  Stats:     GET  http://localhost:{PORT}/api/stats")
    print(f"  Lessons:   GET  http://localhost:{PORT}/api/lessons")
    print(f"  Chain:     POST http://localhost:{PORT}/api/chain")
    print(f"  Status:    GET  http://localhost:{PORT}/api/status")
    print(f"  UI:        GET  http://localhost:{PORT}/ui")
    print(f"  Dashboard: GET  http://localhost:{PORT}/api/dashboard")
    print(f"  Flow:      GET  http://localhost:{PORT}/api/flow")
    print(f"  Proxy:     POST http://localhost:{PORT}/api/proxy")
    try:
        _server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        stop_server()


def stop_server():
    """Stop the server and proactive scheduler."""
    global _server, _proactive

    if _proactive:
        _proactive.stop()
        _proactive = None

    if _server:
        _server.shutdown()
        _server.server_close()
        _server = None

    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, signal.SIGTERM)
            # Wait for PID file removal
            for _ in range(10):
                if not PID_FILE.exists():
                    print("Gateway stopped")
                    return
                time.sleep(0.3)
            # Force remove if still exists
            PID_FILE.unlink(missing_ok=True)
            print("Gateway stopped (PID file cleaned)")
        except (ValueError, ProcessLookupError):
            PID_FILE.unlink(missing_ok=True)
            print("Gateway was not running (stale PID file removed)")
        except PermissionError:
            print(f"Permission denied to signal pid {pid}")


def show_status():
    """Show gateway status."""
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text().strip())
            os.kill(pid, 0)
            print(f"Gateway running (pid {pid})")
        except (ValueError, ProcessLookupError, PermissionError):
            print("Gateway not running (stale PID file)")
    else:
        print("Gateway not running")

    # Show recent log
    if LOG_FILE.exists():
        lines = LOG_FILE.read_text().strip().splitlines()
        if lines:
            print(f"\n--- Last 10 log entries ---")
            for line in lines[-10:]:
                print(f"  {line}")

    # Show quick stats
    records = atsm.load_db()
    stats = atsm.compute_stats(records)
    if stats:
        print(f"\n--- Quick Stats ---")
        print(f"  Total records: {len(records)}")
        print(f"  Skills tracked: {len(stats)}")
        for name, s in sorted(
            stats.items(), key=lambda x: x[1]["expected_success"], reverse=True
        )[:5]:
            print(f"    {name}: E={s['expected_success']:.2f} N={s['total']}")


# --- CLI ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]
    if cmd == "start":
        start_server()
    elif cmd == "stop":
        stop_server()
    elif cmd == "status":
        show_status()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
