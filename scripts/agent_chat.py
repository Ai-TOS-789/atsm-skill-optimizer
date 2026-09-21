#!/usr/bin/env python3
"""Agent Chat Channel — local HTTP chat interface between agent and user.

CLI:
    python3 agent_chat.py start          # Start the server (foreground)
    python3 agent_chat.py say "message"  # Send a message to the chat
    python3 agent_chat.py stop           # Stop the running server
"""

import argparse
import json
import os
import sys
import time
import threading
import http.server
import socketserver
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime

PORT = 8765
PID_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".agent_chat.pid")
LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agent_chat.log")

# ---------------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Agent Chat</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #1a1a2e; color: #eee; height: 100vh; display: flex;
         flex-direction: column; }
  #chat-log { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { margin-bottom: 12px; padding: 8px 12px; border-radius: 8px;
         max-width: 80%; word-wrap: break-word; }
  .msg.user { background: #16213e; margin-left: auto; }
  .msg.agent { background: #0f3460; margin-right: auto; }
  .msg .meta { font-size: 0.75em; color: #888; margin-bottom: 2px; }
  #input-area { display: flex; padding: 12px; border-top: 1px solid #333; }
  #input { flex: 1; padding: 10px; border: 1px solid #444; border-radius: 6px;
           background: #16213e; color: #eee; font-size: 1em; outline: none; }
  #send { margin-left: 8px; padding: 10px 20px; background: #e94560; color: #fff;
          border: none; border-radius: 6px; cursor: pointer; font-size: 1em; }
  #send:hover { background: #c73652; }
</style>
</head>
<body>
<div id="chat-log"></div>
<div id="input-area">
  <input id="input" type="text" placeholder="Type a message…" autofocus />
  <button id="send" onclick="sendMessage()">Send</button>
</div>
<script>
const log = document.getElementById('chat-log');
const input = document.getElementById('input');

function addMsg(role, text) {
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const meta = document.createElement('div');
  meta.className = 'meta';
  meta.textContent = role.toUpperCase() + ' · ' +
    new Date().toLocaleTimeString();
  div.appendChild(meta);
  div.appendChild(document.createTextNode(text));
  log.appendChild(div);
  log.scrollTop = log.scrollHeight;
}

async function sendMessage() {
  const text = input.value.trim();
  if (!text) return;
  input.value = '';
  addMsg('user', text);
  try {
    await fetch('/api/send', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({role: 'user', message: text})
    });
  } catch(e) {
    addMsg('agent', '[error: ' + e.message + ']');
  }
}

async function poll() {
  try {
    const r = await fetch('/api/poll');
    const msgs = await r.json();
    msgs.forEach(m => addMsg(m.role, m.message));
  } catch(e) {}
  setTimeout(poll, 1000);
}

input.addEventListener('keydown', e => { if (e.key === 'Enter') sendMessage(); });
poll();
</script>
</body>
</html>
"""

# ---------------------------------------------------------------------------
# Shared message store (thread-safe)
# ---------------------------------------------------------------------------

_messages = []
_lock = threading.Lock()
_last_seen_index = {}  # per-session index tracking not needed — broadcast model


def add_message(role: str, message: str):
    """Append a message to the shared log."""
    with _lock:
        _messages.append({
            "role": role,
            "message": message,
            "timestamp": datetime.now().isoformat(),
        })


def get_messages(since_idx: int = 0):
    """Return messages since the given index."""
    with _lock:
        return list(_messages[since_idx:])


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------

class ChatHandler(http.server.BaseHTTPRequestHandler):
    """Handles GET /, POST /api/send, GET /api/poll, GET /api/messages."""

    def log_message(self, format, *args):
        pass  # suppress noisy logging

    def _send_json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html, status=200):
        body = html.encode()
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self._send_html(HTML_PAGE)
        elif self.path.startswith("/api/poll"):
            # parse ?since=N
            idx = 0
            if "?" in self.path:
                qs = urllib.parse.urlparse(self.path).query
                params = urllib.parse.parse_qs(qs)
                idx = int(params.get("since", ["0"])[0])
            msgs = get_messages(idx)
            self._send_json(msgs)
        elif self.path == "/api/messages":
            self._send_json(get_messages())
        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/api/send":
            length = int(self.headers.get("Content-Length", 0))
            raw = self.rfile.read(length).decode()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                self._send_json({"error": "invalid json"}, 400)
                return
            role = payload.get("role", "agent")
            message = payload.get("message", "").strip()
            if not message:
                self._send_json({"error": "empty message"}, 400)
                return
            add_message(role, message)
            self._send_json({"status": "ok"})
        else:
            self._send_json({"error": "not found"}, 404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


# ---------------------------------------------------------------------------
# Reusable server (SO_REUSEADDR so restart is clean)
# ---------------------------------------------------------------------------

class ReusableTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True


def run_server():
    """Start the HTTP server (blocking)."""
    with ReusableTCPServer(("127.0.0.1", PORT), ChatHandler) as httpd:
        pid = os.getpid()
        with open(PID_FILE, "w") as f:
            f.write(str(pid))
        add_message("agent", f"Agent Chat server started on http://127.0.0.1:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            httpd.shutdown()
            if os.path.exists(PID_FILE):
                os.remove(PID_FILE)


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

def is_running():
    """Check if the server process is alive via PID file."""
    if not os.path.exists(PID_FILE):
        return False
    try:
        with open(PID_FILE) as f:
            pid = int(f.read().strip())
        os.kill(pid, 0)  # signal 0 = check existence
        return True
    except (OSError, ValueError):
        return False


def send_message(role: str, message: str):
    """POST a message to the running server."""
    data = json.dumps({"role": role, "message": message}).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/api/send",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            result = json.loads(resp.read())
            if result.get("status") == "ok":
                return True
            print(f"Server error: {result}", file=sys.stderr)
            return False
    except Exception as e:
        print(f"Failed to send message: {e}", file=sys.stderr)
        return False


def stop_server():
    """Kill the server via PID file."""
    if not os.path.exists(PID_FILE):
        print("No running server found (no PID file).")
        return
    with open(PID_FILE) as f:
        pid = int(f.read().strip())
    try:
        os.kill(pid, 15)  # SIGTERM
        print(f"Server (PID {pid}) stopped.")
    except ProcessLookupError:
        print(f"Process {pid} not found. Cleaning PID file.")
    except PermissionError:
        print(f"Permission denied killing PID {pid}.")
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


# ---------------------------------------------------------------------------
# proactive_agent integration
# ---------------------------------------------------------------------------

def proactive_notify(issue_description: str):
    """Called by proactive_agent when issues are found.

    Formats the issue as a chat message and delivers it to the user
    through the running server.
    """
    text = f"[Proactive Alert] {issue_description}"
    if is_running():
        send_message("agent", text)
    else:
        # queue it so the next poll picks it up
        add_message("agent", text)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Agent Chat Channel")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("start", help="Start the chat server")
    say_parser = sub.add_parser("say", help="Send a message")
    say_parser.add_argument("message", help="Message text to send")
    say_parser.add_argument("--role", default="agent",
                            help="Sender role (default: agent)")
    sub.add_parser("stop", help="Stop the running server")

    args = parser.parse_args()

    if args.command == "start":
        if is_running():
            print(f"Server already running on port {PORT}.")
            sys.exit(1)
        print(f"Starting Agent Chat on http://127.0.0.1:{PORT}")
        run_server()
    elif args.command == "say":
        if not is_running():
            print("Server is not running. Start it first with: "
                  "python3 agent_chat.py start")
            sys.exit(1)
        ok = send_message(args.role, args.message)
        if ok:
            print("Message sent.")
        else:
            print("Failed to send message.")
            sys.exit(1)
    elif args.command == "stop":
        stop_server()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
