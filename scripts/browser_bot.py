#!/usr/bin/env python3
"""
Browser Automation Bot via CDP (Chrome DevTools Protocol).

Auto-starts Chromium with --remote-debugging-port=9222 if not already running.
Uses websocket-client for CDP communication.

Usage:
    python3 browser_bot.py navigate <url>
    python3 browser_bot.py screenshot <output_path>
    python3 browser_bot.py extract [--selector <css>]
    python3 browser_bot.py click <selector>
    python3 browser_bot.py fill <selector> <text>
    python3 browser_bot.py type <selector> <text>
    python3 browser_bot.py eval <js_expression>
    python3 browser_bot.py new_tab <url>
    python3 browser_bot.py close_tab
    python3 browser_bot.py --status
"""

import sys
import os
import json
import time
import base64
import argparse
import subprocess
import urllib.request
import urllib.error

try:
    import websocket
except ImportError:
    print("ERROR: websocket-client not installed. Run: pip install websocket-client")
    sys.exit(1)

CDP_PORT = 9223
CDP_BASE = f"http://127.0.0.1:{CDP_PORT}"
CHROMIUM_CANDIDATES = [
    "/home/aorus/.cache/ms-playwright/chromium-1243/chrome-linux64/chrome",
    "google-chrome-stable",
    "google-chrome",
    "chromium-browser",
    "chromium",
    "/usr/bin/chromium-browser",
    "/usr/bin/chromium",
    "/usr/bin/google-chrome",
]


def find_chromium():
    """Find a working Chromium/Chrome binary."""
    for candidate in CHROMIUM_CANDIDATES:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
        # Check in PATH
        if not os.path.sep in candidate:
            try:
                result = subprocess.run(
                    ["which", candidate],
                    capture_output=True, text=True
                )
                if result.returncode == 0:
                    path = result.stdout.strip()
                    if path:
                        return path
            except Exception:
                pass
    return None


def is_chromium_running():
    """Check if Chromium is already listening on the CDP port."""
    try:
        resp = urllib.request.urlopen(f"{CDP_BASE}/json/version", timeout=2)
        data = json.loads(resp.read())
        return "webSocketDebuggerUrl" in data
    except Exception:
        return False


def start_chromium():
    """Start Chromium with remote debugging port."""
    binary = find_chromium()
    if not binary:
        print("ERROR: No Chrome/Chromium binary found.")
        print("Install Chromium or set path in CHROMIUM_CANDIDATES.")
        sys.exit(1)

    user_data_dir = os.path.expanduser("~/.hermes/cache/chromium-profile")
    os.makedirs(user_data_dir, exist_ok=True)

    cmd = [
        binary,
        f"--remote-debugging-port={CDP_PORT}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-gpu",
        "--disable-software-rasterizer",
        "--disable-dev-shm-usage",
        "--remote-allow-origins=*",
        "--window-size=1280,800",
        "about:blank",
    ]

    print(f"Starting Chromium: {binary}")
    subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for CDP to become ready
    for _ in range(30):
        if is_chromium_running():
            print("Chromium started, CDP ready.")
            return True
        time.sleep(0.5)

    print("ERROR: Chromium did not start within 15 seconds.")
    sys.exit(1)


def ensure_chromium():
    """Ensure Chromium is running, start if needed."""
    if not is_chromium_running():
        start_chromium()
    return True


def get_targets():
    """Get list of browser targets (tabs/pages)."""
    resp = urllib.request.urlopen(f"{CDP_BASE}/json")
    return json.loads(resp.read())


def get_page_target():
    """Get the first page-type target."""
    targets = get_targets()
    for t in targets:
        if t.get("type") == "page":
            return t
    # If no page target, create one
    resp = urllib.request.urlopen(f"{CDP_BASE}/json/new?about:blank")
    return json.loads(resp.read())


class CDPSession:
    """A CDP session connected to a browser target."""

    def __init__(self):
        target = get_page_target()
        ws_url = target["webSocketDebuggerUrl"]
        self.ws = websocket.create_connection(ws_url, timeout=30)
        self.msg_id = 0
        self._enable_domains()

    def _enable_domains(self):
        """Enable common CDP domains."""
        self.send("Page.enable")
        self.send("Runtime.enable")
        self.send("DOM.enable")

    def send(self, method, params=None):
        """Send a CDP command and wait for response."""
        self.msg_id += 1
        msg = {"id": self.msg_id, "method": method}
        if params:
            msg["params"] = params
        self.ws.send(json.dumps(msg))
        while True:
            resp = json.loads(self.ws.recv())
            if resp.get("id") == self.msg_id:
                if "error" in resp:
                    raise RuntimeError(f"CDP error: {resp['error']}")
                return resp.get("result", {})
            # Skip events

    def close(self):
        """Close the WebSocket connection."""
        try:
            self.ws.close()
        except Exception:
            pass


def cmd_status():
    """Show browser status."""
    chromium_path = find_chromium()
    running = is_chromium_running()
    print(f"Chromium binary: {chromium_path or 'NOT FOUND'}")
    print(f"CDP on port {CDP_PORT}: {'RUNNING' if running else 'NOT RUNNING'}")
    if running:
        targets = get_targets()
        print(f"Open targets: {len(targets)}")
        for t in targets:
            print(f"  [{t.get('type')}] {t.get('title', 'untitled')}")
    return chromium_path is not None


def cmd_navigate(url):
    """Navigate to a URL."""
    ensure_chromium()
    cdp = CDPSession()
    try:
        result = cdp.send("Page.navigate", {"url": url})
        # Wait for load
        time.sleep(2)
        print(json.dumps(result, indent=2))
    finally:
        cdp.close()


def cmd_screenshot(output_path):
    """Take a screenshot and save to file."""
    ensure_chromium()
    cdp = CDPSession()
    try:
        result = cdp.send("Page.captureScreenshot", {"format": "png"})
        img_data = base64.b64decode(result["data"])
        with open(output_path, "wb") as f:
            f.write(img_data)
        print(f"Screenshot saved: {output_path} ({len(img_data)} bytes)")
    finally:
        cdp.close()


def cmd_extract(selector=None):
    """Extract text content from the page."""
    ensure_chromium()
    cdp = CDPSession()
    try:
        if selector:
            # Get specific element text
            js = f"""
            (() => {{
                const el = document.querySelector('{selector}');
                return el ? el.innerText : null;
            }})()
            """
        else:
            # Get all body text
            js = "document.body ? document.body.innerText : ''"
        result = cdp.send("Runtime.evaluate", {"expression": js})
        text = result.get("result", {}).get("value", "")
        print(text)
    finally:
        cdp.close()


def cmd_click(selector):
    """Click an element by CSS selector."""
    ensure_chromium()
    cdp = CDPSession()
    try:
        js = f"""
        (() => {{
            const el = document.querySelector('{selector}');
            if (!el) return 'ELEMENT_NOT_FOUND';
            el.click();
            return 'CLICKED';
        }})()
        """
        result = cdp.send("Runtime.evaluate", {"expression": js})
        value = result.get("result", {}).get("value", "UNKNOWN")
        print(value)
    finally:
        cdp.close()


def cmd_fill(selector, text):
    """Fill an input element with text."""
    ensure_chromium()
    cdp = CDPSession()
    try:
        escaped_text = json.dumps(text)
        js = f"""
        (() => {{
            const el = document.querySelector('{selector}');
            if (!el) return 'ELEMENT_NOT_FOUND';
            el.focus();
            el.value = {escaped_text};
            el.dispatchEvent(new Event('input', {{ bubbles: true }}));
            el.dispatchEvent(new Event('change', {{ bubbles: true }}));
            return 'FILLED';
        }})()
        """
        result = cdp.send("Runtime.evaluate", {"expression": js})
        value = result.get("result", {}).get("value", "UNKNOWN")
        print(value)
    finally:
        cdp.close()


def cmd_type(selector, text):
    """Type text into an element character by character via CDP Input.dispatchKeyEvent."""
    ensure_chromium()
    cdp = CDPSession()
    try:
        # Focus element
        js = f"""
        (() => {{
            const el = document.querySelector('{selector}');
            if (!el) return 'ELEMENT_NOT_FOUND';
            el.focus();
            return 'FOCUSED';
        }})()
        """
        result = cdp.send("Runtime.evaluate", {"expression": js})
        if result.get("result", {}).get("value") != "FOCUSED":
            print("ELEMENT_NOT_FOUND")
            return

        for char in text:
            cdp.send("Input.dispatchKeyEvent", {
                "type": "keyDown",
                "text": char,
            })
            cdp.send("Input.dispatchKeyEvent", {
                "type": "keyUp",
                "text": char,
            })
        print("TYPED")
    finally:
        cdp.close()


def cmd_eval(js_expression):
    """Evaluate arbitrary JavaScript in the page."""
    ensure_chromium()
    cdp = CDPSession()
    try:
        result = cdp.send("Runtime.evaluate", {"expression": js_expression})
        value = result.get("result", {}).get("value", "")
        print(value)
    finally:
        cdp.close()


def cmd_new_tab(url="about:blank"):
    """Open a new tab with the given URL."""
    ensure_chromium()
    try:
        resp = urllib.request.urlopen(f"{CDP_BASE}/json/new?{url}")
        data = json.loads(resp.read())
        print(json.dumps(data, indent=2))
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        print(f"Error: {body}")


def cmd_close_tab():
    """Close the current page target."""
    targets = get_targets()
    page_targets = [t for t in targets if t.get("type") == "page"]
    if page_targets:
        last = page_targets[-1]
        try:
            urllib.request.urlopen(f"{CDP_BASE}/json/close/{last['id']}")
            print(f"Closed tab: {last.get('title', 'untitled')}")
        except Exception as e:
            print(f"Error closing tab: {e}")
    else:
        print("No page targets to close.")


def main():
    parser = argparse.ArgumentParser(
        description="Browser automation bot via CDP",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--status", action="store_true", help="Show browser status")
    parser.add_argument("--port", type=int, default=9222, help="CDP port (default: 9222)")
    sub = parser.add_subparsers(dest="command")

    p_nav = sub.add_parser("navigate", help="Navigate to URL")
    p_nav.add_argument("url", help="URL to navigate to")

    p_shot = sub.add_parser("screenshot", help="Take screenshot")
    p_shot.add_argument("output", help="Output file path")

    p_ext = sub.add_parser("extract", help="Extract text content")
    p_ext.add_argument("--selector", "-s", help="CSS selector for specific element")

    p_click = sub.add_parser("click", help="Click element")
    p_click.add_argument("selector", help="CSS selector")

    p_fill = sub.add_parser("fill", help="Fill input element")
    p_fill.add_argument("selector", help="CSS selector")
    p_fill.add_argument("text", help="Text to fill")

    p_type = sub.add_parser("type", help="Type text into element")
    p_type.add_argument("selector", help="CSS selector")
    p_type.add_argument("text", help="Text to type")

    p_eval = sub.add_parser("eval", help="Evaluate JavaScript")
    p_eval.add_argument("expression", help="JavaScript expression")

    p_tab = sub.add_parser("new_tab", help="Open new tab")
    p_tab.add_argument("url", nargs="?", default="about:blank", help="URL to open")

    p_close = sub.add_parser("close_tab", help="Close current tab")

    args = parser.parse_args()

    global CDP_PORT, CDP_BASE
    CDP_PORT = args.port
    CDP_BASE = f"http://127.0.0.1:{CDP_PORT}"

    if args.status or args.command is None:
        cmd_status()
        return

    commands = {
        "navigate": lambda: cmd_navigate(args.url),
        "screenshot": lambda: cmd_screenshot(args.output),
        "extract": lambda: cmd_extract(args.selector),
        "click": lambda: cmd_click(args.selector),
        "fill": lambda: cmd_fill(args.selector, args.text),
        "type": lambda: cmd_type(args.selector, args.text),
        "eval": lambda: cmd_eval(args.expression),
        "new_tab": lambda: cmd_new_tab(args.url),
        "close_tab": cmd_close_tab,
    }

    handler = commands.get(args.command)
    if handler:
        handler()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
