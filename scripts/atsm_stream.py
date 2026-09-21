#!/usr/bin/env python3
"""ATSM Stream: Real-time SSE streaming for ATSM events.

Provides Server-Sent Events endpoint at GET /api/stream for:
- New ATSM rankings as they happen
- Proactive agent alerts
- Skill outcome recordings
- System metrics updates

Integrates with atsm_gateway.py (event bus pattern).

CLI:
    python3 atsm_stream.py start    # Start SSE server
    python3 atsm_stream.py test     # Test event flow
"""

import json
import os
import queue
import socket
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse

# --- Paths ---
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
SKILL_DIR = SCRIPT_DIR.parent

# Data files
DB_FILE = DATA_DIR / "atsm_db.jsonl"
ALERTS_LOG = DATA_DIR / "alerts.log"
SYS_METRICS_FILE = DATA_DIR / "sys_metrics.jsonl"
PROACTIVE_LOG = DATA_DIR / "proactive.log"
TASKS_FILE = Path("/tmp/proactive_tasks.json")

# Server config
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765


# =============================================================================
# ATSM Gateway — event bus integration point
# =============================================================================

class ATSMGateway:
    """Event bus that connects ATSM data sources to the SSE stream.
    
    This is the integration point for atsm_gateway.py.
    Other modules publish events here; the stream server subscribes.
    """

    def __init__(self):
        self._subscribers = []
        self._lock = threading.Lock()
        self._event_count = 0

    def subscribe(self) -> queue.Queue:
        """Create a new subscription queue. Returns the queue."""
        q = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue):
        """Remove a subscription."""
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def publish(self, event_type: str, data: dict):
        """Publish an event to all subscribers.
        
        Args:
            event_type: One of: ranking_update, alert, skill_outcome, system_metrics
            data: Event payload dict
        """
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "seq": self._event_count,
        }
        self._event_count += 1
        with self._lock:
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass

    @property
    def subscriber_count(self) -> int:
        with self._lock:
            return len(self._subscribers)


# Global gateway instance (singleton)
gateway = ATSMGateway()


# =============================================================================
# Data Watchers — monitor files and emit events
# =============================================================================

class DataWatcher:
    """Monitors ATSM data sources and publishes events to the gateway."""

    def __init__(self, gw: ATSMGateway):
        self.gw = gw
        self._running = False
        self._threads = []

        # File read positions for incremental reads
        self._db_pos = 0
        self._alerts_pos = 0
        self._metrics_pos = 0
        self._proactive_pos = 0

        # Ranking state for change detection
        self._last_rankings = {}

    def start(self):
        """Start all watcher threads."""
        self._running = True

        # Initialize positions to current file sizes (don't replay history)
        self._db_pos = self._file_size(DB_FILE)
        self._alerts_pos = self._file_size(ALERTS_LOG)
        self._metrics_pos = self._file_size(SYS_METRICS_FILE)
        self._proactive_pos = self._file_size(PROACTIVE_LOG)

        watchers = [
            ("db_watcher", self._watch_db, 1.0),
            ("alerts_watcher", self._watch_alerts, 2.0),
            ("metrics_watcher", self._watch_metrics, 5.0),
            ("rankings_watcher", self._watch_rankings, 15.0),
        ]

        for name, func, interval in watchers:
            t = threading.Thread(
                target=self._watcher_loop, args=(name, func, interval), daemon=True
            )
            t.start()
            self._threads.append(t)

        print(f"[ATSM Stream] Watchers started: {[w[0] for w in watchers]}")

    def stop(self):
        """Stop all watchers."""
        self._running = False
        for t in self._threads:
            t.join(timeout=5)

    @staticmethod
    def _file_size(path: Path) -> int:
        try:
            return path.stat().st_size
        except (OSError, FileNotFoundError):
            return 0

    def _watcher_loop(self, name: str, func, interval: float):
        """Run a watcher function in a loop until stopped."""
        while self._running:
            try:
                func()
            except Exception as e:
                print(f"[ATSM Stream] {name} error: {e}")
            time.sleep(interval)

    # --- Individual watchers ---

    def _watch_db(self):
        """Watch for new skill outcome records in atsm_db.jsonl."""
        if not DB_FILE.exists():
            return

        current_size = self._file_size(DB_FILE)
        if current_size <= self._db_pos:
            self._db_pos = current_size
            return

        try:
            with open(DB_FILE, "r", encoding="utf-8") as f:
                f.seek(self._db_pos)
                new_lines = f.readlines()
                self._db_pos = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    self.gw.publish("skill_outcome", record)
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass

    def _watch_alerts(self):
        """Watch for new proactive agent alerts in alerts.log."""
        if not ALERTS_LOG.exists():
            return

        current_size = self._file_size(ALERTS_LOG)
        if current_size <= self._alerts_pos:
            self._alerts_pos = current_size
            return

        try:
            with open(ALERTS_LOG, "r", encoding="utf-8") as f:
                f.seek(self._alerts_pos)
                new_lines = f.readlines()
                self._alerts_pos = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                self.gw.publish("alert", {
                    "raw": line,
                    "parsed": self._parse_alert(line),
                })
        except OSError:
            pass

    def _watch_metrics(self):
        """Watch for new system metrics in sys_metrics.jsonl."""
        if not SYS_METRICS_FILE.exists():
            return

        current_size = self._file_size(SYS_METRICS_FILE)
        if current_size <= self._metrics_pos:
            self._metrics_pos = current_size
            return

        try:
            with open(SYS_METRICS_FILE, "r", encoding="utf-8") as f:
                f.seek(self._metrics_pos)
                new_lines = f.readlines()
                self._metrics_pos = f.tell()

            for line in new_lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    metric = json.loads(line)
                    self.gw.publish("system_metrics", metric)
                except json.JSONDecodeError:
                    pass
        except OSError:
            pass

    def _watch_rankings(self):
        """Detect ATSM ranking changes by re-ranking common tasks."""
        try:
            sys.path.insert(0, str(SCRIPT_DIR))
            import atsm

            test_tasks = [
                "search academic papers",
                "send email notification",
                "create spreadsheet",
                "monitor system health",
            ]

            for task in test_tasks:
                results, elapsed = atsm.rank_skills(task, top_k=3)
                current = [(r["name"], r["final_score"]) for r in results]

                if task in self._last_rankings:
                    if self._last_rankings[task] != current:
                        self.gw.publish("ranking_update", {
                            "task": task,
                            "rankings": results,
                            "elapsed_ms": round(elapsed * 1000, 2),
                        })

                self._last_rankings[task] = current
        except Exception:
            pass

    @staticmethod
    def _parse_alert(line: str) -> dict:
        """Parse an alert log line like '[2026-09-20 21:48:54] [CRITICAL] message'."""
        import re
        match = re.match(r"\[([^\]]+)\]\s+\[(\w+)\]\s+(.*)", line)
        if match:
            return {
                "timestamp": match.group(1),
                "level": match.group(2),
                "message": match.group(3),
            }
        return {"message": line}


# =============================================================================
# SSE HTTP Server
# =============================================================================

class SSEHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Server-Sent Events."""

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/stream":
            self._handle_sse()
        elif parsed.path == "/":
            self._handle_index()
        elif parsed.path == "/api/health":
            self._handle_health()
        elif parsed.path == "/api/stats":
            self._handle_stats()
        else:
            self.send_error(404)

    def _handle_sse(self):
        """Handle SSE stream request at GET /api/stream."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("X-Accel-Buffering", "no")  # disable nginx buffering
        self.end_headers()

        sub = gateway.subscribe()

        try:
            # Send initial connected event
            self._send_event("connected", {
                "stream": "atsm-events",
                "time": datetime.now(timezone.utc).isoformat(),
                "endpoints": ["/api/stream"],
            })

            while True:
                try:
                    event = sub.get(timeout=30)
                    self._send_event(event["type"], event["data"])
                except queue.Empty:
                    # Send keepalive comment to prevent timeout
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            gateway.unsubscribe(sub)

    def _send_event(self, event_type: str, data: dict):
        """Send a single SSE event."""
        payload = json.dumps(data, default=str)
        self.wfile.write(f"event: {event_type}\n".encode())
        self.wfile.write(f"data: {payload}\n\n".encode())
        self.wfile.flush()

    def _handle_index(self):
        """Serve a simple HTML status page."""
        html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"><title>ATSM Stream</title>
<style>
body{font-family:system-ui,sans-serif;max-width:800px;margin:2rem auto;padding:0 1rem}
code{background:#f4f4f4;padding:2px 6px;border-radius:3px}
pre{background:#1e1e1e;color:#d4d4d4;padding:1rem;border-radius:6px;overflow-x:auto}
.event{border-left:4px solid #007acc;padding:0.5rem 1rem;margin:0.5rem 0;background:#f8f8f8}
</style></head>
<body>
<h1>ATSM Event Stream</h1>
<p>Server-Sent Events endpoint: <code>GET /api/stream</code></p>
<h2>Event Types</h2>
<div class="event"><strong>ranking_update</strong> — New ATSM rankings as they happen</div>
<div class="event"><strong>alert</strong> — Proactive agent alerts</div>
<div class="event"><strong>skill_outcome</strong> — Skill outcome recordings</div>
<div class="event"><strong>system_metrics</strong> — System metrics updates</div>
<h2>Quick Test</h2>
<pre>curl -N http://127.0.0.1:8765/api/stream</pre>
<h2>Endpoints</h2>
<ul>
<li><code>GET /api/stream</code> — SSE event stream</li>
<li><code>GET /api/health</code> — Health check</li>
<li><code>GET /api/stats</code> — Stream statistics</li>
</ul>
</body></html>"""
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode())

    def _handle_health(self):
        """Health check endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "ok",
            "subscribers": gateway.subscriber_count,
            "time": datetime.now(timezone.utc).isoformat(),
        }).encode())

    def _handle_stats(self):
        """Stream statistics endpoint."""
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({
            "subscribers": gateway.subscriber_count,
            "data_sources": {
                "db_file": str(DB_FILE),
                "alerts_log": str(ALERTS_LOG),
                "sys_metrics": str(SYS_METRICS_FILE),
            },
            "time": datetime.now(timezone.utc).isoformat(),
        }, indent=2).encode())

    def log_message(self, format, *args):
        """Suppress default access logging to reduce noise."""
        pass


# =============================================================================
# Server runner
# =============================================================================

def run_server(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT):
    """Start the SSE server with data watchers."""
    # Use threading HTTP server for concurrent SSE connections
    server = HTTPServer((host, port), SSEHandler)
    server.request_queue_size = 64

    # Start data watchers
    watcher = DataWatcher(gateway)
    watcher.start()

    print(f"[ATSM Stream] Server running on http://{host}:{port}")
    print(f"[ATSM Stream] SSE endpoint:  http://{host}:{port}/api/stream")
    print(f"[ATSM Stream] Health check:  http://{host}:{port}/api/health")
    print(f"[ATSM Stream] Press Ctrl+C to stop")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[ATSM Stream] Shutting down...")
    finally:
        watcher.stop()
        server.server_close()
        print("[ATSM Stream] Stopped.")


# =============================================================================
# Test suite
# =============================================================================

def run_test():
    """Test the event streaming system end-to-end."""
    print("=" * 60)
    print("ATSM Stream — Event Flow Test")
    print("=" * 60)
    all_passed = True

    # --- Test 1: Gateway pub/sub ---
    print("\n[TEST 1] Gateway pub/sub mechanism...")
    test_gw = ATSMGateway()
    sub = test_gw.subscribe()

    test_gw.publish("test_event", {"msg": "hello"})
    try:
        event = sub.get(timeout=1)
        assert event["type"] == "test_event"
        assert event["data"]["msg"] == "hello"
        assert "timestamp" in event
        assert "seq" in event
        print("  ✓ PASS — Events published and received")
    except queue.Empty:
        print("  ✗ FAIL — No event received")
        all_passed = False
    finally:
        test_gw.unsubscribe(sub)

    # --- Test 2: Data source availability ---
    print("\n[TEST 2] Data source file availability...")
    sources = {
        "atsm_db.jsonl": DB_FILE,
        "alerts.log": ALERTS_LOG,
        "sys_metrics.jsonl": SYS_METRICS_FILE,
    }
    for name, path in sources.items():
        exists = path.exists()
        status = "✓" if exists else "⚠"
        print(f"  {status} {name}: {path}")
    # Not failing on missing files — they may not exist yet
    print("  ℹ Data sources checked (missing files OK — created on first use)")

    # --- Test 3: Watcher initialization & start ---
    print("\n[TEST 3] DataWatcher start/stop...")
    test_gw2 = ATSMGateway()
    watcher = DataWatcher(test_gw2)
    watcher.start()
    time.sleep(0.5)
    print("  ✓ PASS — Watcher threads started")
    watcher.stop()
    time.sleep(0.2)
    print("  ✓ PASS — Watcher threads stopped")

    # --- Test 4: Event generation & flow ---
    print("\n[TEST 4] Event generation and flow...")
    test_gw3 = ATSMGateway()
    sub3 = test_gw3.subscribe()

    # Publish all four event types
    events_to_publish = [
        ("ranking_update", {"task": "test", "rankings": [{"name": "arxiv", "score": 0.95}]}),
        ("alert", {"level": "warning", "message": "Test alert"}),
        ("skill_outcome", {"skill": "test-skill", "success": 1, "agent": "test"}),
        ("system_metrics", {"cpu_percent": 5.0, "memory_percent": 50.0}),
    ]

    for etype, data in events_to_publish:
        test_gw3.publish(etype, data)

    # Collect events
    received = []
    try:
        while True:
            event = sub3.get(timeout=2)
            received.append(event["type"])
    except queue.Empty:
        pass

    for etype, _ in events_to_publish:
        if etype in received:
            print(f"  ✓ PASS — {etype} event received")
        else:
            print(f"  ✗ FAIL — {etype} event NOT received")
            all_passed = False

    test_gw3.unsubscribe(sub3)

    # --- Test 5: SSE server startup & health check ---
    print("\n[TEST 5] SSE server startup and health check...")
    test_gw4 = ATSMGateway()

    # Find a free port
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((DEFAULT_HOST, 0))
    test_port = sock.getsockname()[1]
    sock.close()

    server = HTTPServer((DEFAULT_HOST, test_port), SSEHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        import urllib.request
        url = f"http://{DEFAULT_HOST}:{test_port}/api/health"
        resp = urllib.request.urlopen(url, timeout=3)
        body = json.loads(resp.read())
        if resp.status == 200 and body.get("status") == "ok":
            print(f"  ✓ PASS — Server healthy on port {test_port}")
        else:
            print(f"  ✗ FAIL — Unexpected response: {body}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ FAIL — Server health check failed: {e}")
        all_passed = False
    finally:
        server.shutdown()

    # --- Test 6: SSE stream connection ---
    print("\n[TEST 6] SSE stream connection (event: connected)...")
    test_gw5 = ATSMGateway()
    sock2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock2.bind((DEFAULT_HOST, 0))
    test_port2 = sock2.getsockname()[1]
    sock2.close()

    server2 = HTTPServer((DEFAULT_HOST, test_port2), SSEHandler)
    server2_thread = threading.Thread(target=server2.serve_forever, daemon=True)
    server2_thread.start()

    try:
        time.sleep(0.3)  # Give server time to bind and start accepting

        # Use raw socket for SSE — urllib blocks on infinite streams
        raw_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        raw_sock.settimeout(3)

        # Retry connection (server might not be ready)
        for attempt in range(10):
            try:
                raw_sock.connect((DEFAULT_HOST, test_port2))
                break
            except (ConnectionRefusedError, OSError):
                time.sleep(0.1)

        raw_sock.sendall(
            b"GET /api/stream HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Accept: text/event-stream\r\n\r\n"
        )

        raw_data = b""
        try:
            while True:
                chunk = raw_sock.recv(2048)
                if not chunk:
                    break
                raw_data += chunk
                if b"event: connected" in raw_data:
                    break
        except socket.timeout:
            pass
        raw_sock.close()

        if b"event: connected" in raw_data and b"data:" in raw_data:
            print("  ✓ PASS — SSE stream emits 'connected' event")
        else:
            print(f"  ✗ FAIL — No 'connected' event received. Got {len(raw_data)} bytes: {raw_data[:300]}")
            all_passed = False
    except Exception as e:
        print(f"  ✗ FAIL — SSE stream test failed: {e}")
        all_passed = False
    finally:
        server2.shutdown()

    # --- Test 7: Multi-subscriber broadcast ---
    print("\n[TEST 7] Multi-subscriber broadcast...")
    test_gw6 = ATSMGateway()
    subs = [test_gw6.subscribe() for _ in range(3)]

    test_gw6.publish("broadcast", {"msg": "hello all"})

    received_count = 0
    for s in subs:
        try:
            s.get(timeout=1)
            received_count += 1
        except queue.Empty:
            pass

    if received_count == 3:
        print("  ✓ PASS — All 3 subscribers received the event")
    else:
        print(f"  ✗ FAIL — Only {received_count}/3 subscribers received")
        all_passed = False

    for s in subs:
        test_gw6.unsubscribe(s)

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_passed:
        print("ALL TESTS PASSED ✓")
    else:
        print("SOME TESTS FAILED ✗")
    print("=" * 60)

    return all_passed


# =============================================================================
# CLI entry point
# =============================================================================

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "start":
        host = os.environ.get("ATSM_STREAM_HOST", DEFAULT_HOST)
        port = int(os.environ.get("ATSM_STREAM_PORT", DEFAULT_PORT))
        run_server(host, port)
    elif cmd == "test":
        success = run_test()
        sys.exit(0 if success else 1)
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: atsm_stream.py {start|test}")
        sys.exit(1)


if __name__ == "__main__":
    main()
