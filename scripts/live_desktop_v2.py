#!/usr/bin/env python3
"""ATSM Live Desktop Presence — Real-time agent state overlay.

A live desktop presence that shows the ATSM agent's current state, active task,
recent thoughts/actions, and system metrics. Works both with tkinter (GUI overlay)
and without (terminal-based fallback).

Usage:
    python3 live_desktop_v2.py          # Start the live overlay (or terminal mode)
    python3 live_desktop_v2.py --test   # Run self-test and exit
    python3 live_desktop_v2.py --status # Print current status and exit
    python3 live_desktop_v2.py --terminal # Force terminal mode (no GUI)

Status file: /tmp/atsm_status.json (updated by ATSM gateway/agent)
Status schema:
{
  "status": "idle|thinking|executing|error",
  "current_task": "string",
  "thoughts": [{"timestamp": "...", "text": "..."}, ...],
  "actions": [{"timestamp": "...", "name": "...", "result": "..."}, ...],
  "metrics": {"cpu_percent": 0.0, "memory_percent": 0.0, "gpu_percent": 0.0},
  "timestamp": "ISO8601"
}
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Constants ---
WINDOW_WIDTH = 420
WINDOW_HEIGHT = 360
UPDATE_INTERVAL_MS = 1500
PULSE_INTERVAL_MS = 80
SCROLL_INTERVAL_MS = 40
STATUS_FILE = Path("/tmp/atsm_status.json")

# --- Color scheme ---
COLORS = {
    "bg": "#0d1117",
    "bg_card": "#161b22",
    "fg": "#e6edf3",
    "fg_dim": "#7d8590",
    "fg_muted": "#6e7681",
    "border": "#30363d",
    "accent": "#58a6ff",
    "idle": "#8b949e",
    "thinking": "#d29922",
    "executing": "#3fb950",
    "error": "#f85149",
    "cpu": "#58a6ff",
    "mem": "#bc8cff",
    "gpu": "#f0883e",
    "success": "#3fb950",
    "failure": "#f85149",
}

# ANSI colors for terminal fallback
ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
    "bg_blue": "\033[44m",
    "clear": "\033[2J\033[H",
}


# --- Status file handling ---
def read_status() -> dict:
    """Read ATSM status from the status file."""
    if not STATUS_FILE.exists():
        return get_default_status()
    try:
        with open(STATUS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except (json.JSONDecodeError, OSError):
        return get_default_status()


def get_default_status() -> dict:
    """Return default status when no status file exists."""
    return {
        "status": "idle",
        "current_task": "",
        "thoughts": [],
        "actions": [],
        "metrics": {"cpu_percent": 0.0, "memory_percent": 0.0, "gpu_percent": 0.0},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def print_status():
    """Print current status to stdout and exit."""
    status = read_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    sys.exit(0)


# --- Self-test ---
def run_self_test():
    """Run a self-test verifying all imports and basic functionality."""
    errors = []
    has_tkinter = False

    # Test 1: tkinter import (optional)
    try:
        import tkinter as tk
        from tkinter import font as tkf
        has_tkinter = True
        print(f"[{ANSI['green']}PASS{ANSI['reset']}] tkinter import")
    except ImportError as e:
        print(f"[{ANSI['yellow']}WARN{ANSI['reset']}] tkinter import: {e} (terminal mode will be used)")

    # Test 2: status file read
    try:
        status = read_status()
        assert isinstance(status, dict), "status is not a dict"
        assert "status" in status, "missing 'status' key"
        print(f"[{ANSI['green']}PASS{ANSI['reset']}] status read: {status.get('status', '?')}")
    except Exception as e:
        errors.append(f"status read failed: {e}")
        print(f"[{ANSI['red']}FAIL{ANSI['reset']}] status read: {e}")

    # Test 3: default status structure
    try:
        ds = get_default_status()
        for key in ["status", "current_task", "thoughts", "actions", "metrics"]:
            assert key in ds, f"missing key: {key}"
        print(f"[{ANSI['green']}PASS{ANSI['reset']}] default status structure")
    except Exception as e:
        errors.append(f"default status: {e}")
        print(f"[{ANSI['red']}FAIL{ANSI['reset']}] default status: {e}")

    # Test 4: metrics structure
    try:
        ds = get_default_status()
        m = ds["metrics"]
        for k in ["cpu_percent", "memory_percent", "gpu_percent"]:
            assert k in m, f"missing metric: {k}"
        print(f"[{ANSI['green']}PASS{ANSI['reset']}] metrics structure")
    except Exception as e:
        errors.append(f"metrics structure: {e}")
        print(f"[{ANSI['red']}FAIL{ANSI['reset']}] metrics structure: {e}")

    # Test 5: status color mapping
    try:
        for s in ["idle", "thinking", "executing", "error"]:
            assert s in COLORS, f"missing color for status: {s}"
        print(f"[{ANSI['green']}PASS{ANSI['reset']}] status color mapping")
    except Exception as e:
        errors.append(f"color mapping: {e}")
        print(f"[{ANSI['red']}FAIL{ANSI['reset']}] color mapping: {e}")

    # Test 6: terminal mode rendering (basic)
    try:
        status = get_default_status()
        lines = render_terminal_frame(status)
        assert len(lines) > 0, "no output lines"
        print(f"[{ANSI['green']}PASS{ANSI['reset']}] terminal rendering ({len(lines)} lines)")
    except Exception as e:
        errors.append(f"terminal rendering: {e}")
        print(f"[{ANSI['red']}FAIL{ANSI['reset']}] terminal rendering: {e}")

    mode = "GUI" if has_tkinter else "TERMINAL"
    if errors:
        print(f"\n[RESULT] FAILED ({len(errors)} error(s)) — mode: {mode}")
        for err in errors:
            print(f"  - {err}")
        sys.exit(1)
    else:
        print(f"\n[RESULT] ALL TESTS PASSED — mode: {mode}")
        sys.exit(0)


def render_terminal_frame(status: dict) -> list[str]:
    """Render a single terminal frame for the live desktop display."""
    lines = []
    a = ANSI
    a_clear = ANSI["clear"]
    
    status_str = status.get("status", "idle")
    task = status.get("current_task", "Waiting for agent...")
    thoughts = status.get("thoughts", [])
    actions = status.get("actions", [])
    metrics = status.get("metrics", {})

    # Status color
    if status_str == "thinking":
        status_color = a["yellow"]
    elif status_str == "executing":
        status_color = a["green"]
    elif status_str == "error":
        status_color = a["red"]
    else:
        status_color = a["dim"]

    # Header
    lines.append(f"{a_clear}")
    lines.append(f"{a['bold']}{a['cyan']} ◈ ATSM{ANSI['reset']}  {a['dim']}Live Desktop Presence{ANSI['reset']}    {status_color}● {status_str.upper()}{ANSI['reset']}")
    lines.append(f"{a['dim']}{'─' * 60}{ANSI['reset']}")
    
    # Current task
    lines.append(f"{a['bold']} CURRENT TASK{ANSI['reset']}")
    lines.append(f" {a['cyan']}{task[:55]}{ANSI['reset']}")
    lines.append(f"{a['dim']}{'─' * 60}{ANSI['reset']}")

    # Thoughts
    lines.append(f"{a['bold']} AGENT THOUGHTS{ANSI['reset']}")
    if thoughts:
        for t in thoughts[-3:]:
            ts = t.get("timestamp", "").split("T")[1][:8] if "T" in t.get("timestamp", "") else ""
            text = t.get("text", str(t))[:50]
            lines.append(f" {a['dim']}{ts}{ANSI['reset']} {a['white']}{text}{ANSI['reset']}")
    else:
        lines.append(f" {a['dim']}No thoughts yet...{ANSI['reset']}")
    lines.append(f"{a['dim']}{'─' * 60}{ANSI['reset']}")

    # Actions
    lines.append(f"{a['bold']} LAST ACTIONS{ANSI['reset']}")
    if actions:
        for a_item in actions[-3:]:
            ts = a_item.get("timestamp", "").split("T")[1][:8] if "T" in a_item.get("timestamp", "") else ""
            name = a_item.get("name", str(a_item))[:35]
            result = a_item.get("result", "")
            icon_color = a["green"] if result in ("success", "ok", "True", True, 1) else a["red"] if result else a["dim"]
            icon = "✓" if icon_color == a["green"] else "✗" if icon_color == a["red"] else "→"
            lines.append(f" {a['dim']}{ts}{ANSI['reset']} {icon_color}{icon}{ANSI['reset']} {name}")
    else:
        lines.append(f" {a['dim']}No actions recorded{ANSI['reset']}")
    lines.append(f"{a['dim']}{'─' * 60}{ANSI['reset']}")

    # Metrics
    cpu = metrics.get("cpu_percent", 0)
    mem = metrics.get("memory_percent", 0)
    gpu = metrics.get("gpu_percent", 0)
    
    def bar(pct, width=20):
        filled = int(pct / 100 * width)
        return "█" * filled + "░" * (width - filled)
    
    lines.append(f"{a['bold']} METRICS{ANSI['reset']}")
    lines.append(f" {a['cyan']}CPU{ANSI['reset']} {bar(cpu)} {cpu:5.1f}%")
    lines.append(f" {a['magenta']}MEM{ANSI['reset']} {bar(mem)} {mem:5.1f}%")
    lines.append(f" {a['yellow']}GPU{ANSI['reset']} {bar(gpu)} {gpu:5.1f}%")
    lines.append(f"{a['dim']}{'─' * 60}{ANSI['reset']}")
    lines.append(f" {a['dim']}Status file: {STATUS_FILE}{ANSI['reset']}")
    lines.append(f" {a['dim']}Last update: {datetime.now().strftime('%H:%M:%S')}{ANSI['reset']}")

    return lines


def run_terminal_mode():
    """Run in terminal mode (no GUI) — continuously prints status to stdout."""
    print(f"{ANSI['clear']}", end="")
    try:
        while True:
            status = read_status()
            lines = render_terminal_frame(status)
            # Print frame
            print("\n".join(lines), end="", flush=True)
            time.sleep(UPDATE_INTERVAL_MS / 1000)
    except KeyboardInterrupt:
        print(f"\n\n{ANSI['dim']}Live desktop presence stopped.{ANSI['reset']}")


# --- Main Application (tkinter GUI) ---
class ATSMDesktopPresence:
    """Live desktop overlay for ATSM agent state display."""

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("ATSM")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # Try to set transparency (Windows/Linux with compositor)
        try:
            self.root.attributes("-alpha", 0.92)
        except Exception:
            pass

        # Position at bottom-right
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - WINDOW_WIDTH - 20
        y = screen_h - WINDOW_HEIGHT - 60
        self.root.geometry(f"+{x}+{y}")

        # Animation state
        self._pulse_alpha = 0
        self._pulse_direction = 1
        self._scroll_x = WINDOW_WIDTH

        # Data state
        self._status = get_default_status()

        self._build_ui()
        self._start_updates()

    def _build_ui(self):
        """Build the UI layout."""
        tk = self.tk
        main = tk.Frame(self.root, bg=COLORS["bg"], padx=12, pady=10)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Header Row: Title + Status ---
        header = tk.Frame(main, bg=COLORS["bg"])
        header.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            header, text="◈ ATSM", font=("Helvetica", 11, "bold"),
            fg=COLORS["accent"], bg=COLORS["bg"]
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            header, text="● IDLE", font=("Helvetica", 9, "bold"),
            fg=COLORS["idle"], bg=COLORS["bg"]
        )
        self.status_label.pack(side=tk.RIGHT)

        # --- Current Task Card ---
        task_card = tk.Frame(main, bg=COLORS["bg_card"], padx=8, pady=6,
                             highlightbackground=COLORS["border"],
                             highlightthickness=1)
        task_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            task_card, text="CURRENT TASK", font=("Helvetica", 7),
            fg=COLORS["fg_muted"], bg=COLORS["bg_card"]
        ).pack(anchor=tk.W)

        self.task_canvas = tk.Canvas(
            task_card, height=22, bg=COLORS["bg_card"],
            highlightthickness=0, bd=0
        )
        self.task_canvas.pack(fill=tk.X)
        self._task_text = "Waiting for agent..."

        # --- Live Thoughts Card ---
        thoughts_card = tk.Frame(main, bg=COLORS["bg_card"], padx=8, pady=6,
                                  highlightbackground=COLORS["border"],
                                  highlightthickness=1)
        thoughts_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            thoughts_card, text="AGENT THOUGHTS", font=("Helvetica", 7),
            fg=COLORS["fg_muted"], bg=COLORS["bg_card"]
        ).pack(anchor=tk.W)

        self.thoughts_frame = tk.Frame(thoughts_card, bg=COLORS["bg_card"])
        self.thoughts_frame.pack(fill=tk.X)

        self.thought_labels = []
        for i in range(3):
            lbl = tk.Label(
                self.thoughts_frame, text="", font=("Courier", 8),
                fg=COLORS["fg_dim"], bg=COLORS["bg_card"],
                anchor=tk.W, wraplength=380, justify=tk.LEFT
            )
            lbl.pack(fill=tk.X, pady=(1, 0))
            self.thought_labels.append(lbl)

        # --- Recent Actions ---
        actions_card = tk.Frame(main, bg=COLORS["bg_card"], padx=8, pady=6,
                                 highlightbackground=COLORS["border"],
                                 highlightthickness=1)
        actions_card.pack(fill=tk.X, pady=(0, 6))

        tk.Label(
            actions_card, text="LAST ACTIONS", font=("Helvetica", 7),
            fg=COLORS["fg_muted"], bg=COLORS["bg_card"]
        ).pack(anchor=tk.W)

        self.action_labels = []
        for i in range(3):
            lbl = tk.Label(
                actions_card, text="—", font=("Courier", 8),
                fg=COLORS["fg_dim"], bg=COLORS["bg_card"],
                anchor=tk.W
            )
            lbl.pack(fill=tk.X, pady=(1, 0))
            self.action_labels.append(lbl)

        # --- Metrics Bars ---
        metrics_card = tk.Frame(main, bg=COLORS["bg_card"], padx=8, pady=6,
                                 highlightbackground=COLORS["border"],
                                 highlightthickness=1)
        metrics_card.pack(fill=tk.X, side=tk.BOTTOM)

        # CPU
        cpu_row = tk.Frame(metrics_card, bg=COLORS["bg_card"])
        cpu_row.pack(fill=tk.X)
        tk.Label(cpu_row, text="CPU", font=("Helvetica", 7),
                 fg=COLORS["cpu"], bg=COLORS["bg_card"], width=4, anchor=tk.W
        ).pack(side=tk.LEFT)
        self.cpu_canvas = tk.Canvas(cpu_row, height=8, bg=COLORS["bg_card_alt"],
                                     highlightthickness=0, bd=0)
        self.cpu_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        self.cpu_bar = self.cpu_canvas.create_rectangle(
            0, 0, 0, 8, fill=COLORS["cpu"], outline=""
        )
        self.cpu_text = tk.Label(cpu_row, text="0%", font=("Courier", 8),
                                  fg=COLORS["cpu"], bg=COLORS["bg_card"],
                                  width=5, anchor=tk.E)
        self.cpu_text.pack(side=tk.RIGHT)

        # MEM
        mem_row = tk.Frame(metrics_card, bg=COLORS["bg_card"])
        mem_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(mem_row, text="MEM", font=("Helvetica", 7),
                 fg=COLORS["mem"], bg=COLORS["bg_card"], width=4, anchor=tk.W
        ).pack(side=tk.LEFT)
        self.mem_canvas = tk.Canvas(mem_row, height=8, bg=COLORS["bg_card_alt"],
                                     highlightthickness=0, bd=0)
        self.mem_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        self.mem_bar = self.mem_canvas.create_rectangle(
            0, 0, 0, 8, fill=COLORS["mem"], outline=""
        )
        self.mem_text = tk.Label(mem_row, text="0%", font=("Courier", 8),
                                  fg=COLORS["mem"], bg=COLORS["bg_card"],
                                  width=5, anchor=tk.E)
        self.mem_text.pack(side=tk.RIGHT)

        # GPU
        gpu_row = tk.Frame(metrics_card, bg=COLORS["bg_card"])
        gpu_row.pack(fill=tk.X, pady=(2, 0))
        tk.Label(gpu_row, text="GPU", font=("Helvetica", 7),
                 fg=COLORS["gpu"], bg=COLORS["bg_card"], width=4, anchor=tk.W
        ).pack(side=tk.LEFT)
        self.gpu_canvas = tk.Canvas(gpu_row, height=8, bg=COLORS["bg_card_alt"],
                                     highlightthickness=0, bd=0)
        self.gpu_canvas.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(4, 4))
        self.gpu_bar = self.gpu_canvas.create_rectangle(
            0, 0, 0, 8, fill=COLORS["gpu"], outline=""
        )
        self.gpu_text = tk.Label(gpu_row, text="0%", font=("Courier", 8),
                                  fg=COLORS["gpu"], bg=COLORS["bg_card"],
                                  width=5, anchor=tk.E)
        self.gpu_text.pack(side=tk.RIGHT)

    def _start_updates(self):
        """Start periodic UI updates."""
        self._update_status()
        self._pulse_animation()
        self._scroll_task()

    def _update_status(self):
        """Read status file and update UI."""
        self._status = read_status()
        status = self._status.get("status", "idle")
        task = self._status.get("current_task", "")
        thoughts = self._status.get("thoughts", [])
        actions = self._status.get("actions", [])
        metrics = self._status.get("metrics", {})

        # Update status label
        status_color = COLORS.get(status, COLORS["idle"])
        self.status_label.config(text=f"● {status.upper()}", fg=status_color)

        # Update task text
        self._task_text = task if task else "Waiting for agent task..."

        # Update thoughts
        if thoughts:
            for i, lbl in enumerate(self.thought_labels):
                idx = len(thoughts) - 3 + i
                if 0 <= idx < len(thoughts):
                    t = thoughts[idx]
                    ts = t.get("timestamp", "")
                    text = t.get("text", str(t))
                    if "T" in ts:
                        ts = ts.split("T")[1][:8]
                    lbl.config(text=f"  {ts}  {text}", fg=COLORS["fg_dim"])
                else:
                    lbl.config(text="", fg=COLORS["fg_dim"])
        else:
            for lbl in self.thought_labels:
                lbl.config(text="", fg=COLORS["fg_dim"])

        # Update actions
        if actions:
            for i, lbl in enumerate(self.action_labels):
                idx = len(actions) - 3 + i
                if 0 <= idx < len(actions):
                    a_item = actions[idx]
                    ts = a_item.get("timestamp", "")
                    name = a_item.get("name", str(a_item))
                    result = a_item.get("result", "")
                    if "T" in ts:
                        ts = ts.split("T")[1][:8]
                    icon = "✓" if result in ("success", "ok", "True", True, 1) else "✗" if result else "→"
                    color = COLORS["success"] if icon == "✓" else COLORS["failure"] if icon == "✗" else COLORS["fg_dim"]
                    lbl.config(text=f"  {ts}  {icon} {name}", fg=color)
                else:
                    lbl.config(text=" —", fg=COLORS["fg_muted"])
        else:
            for lbl in self.action_labels:
                lbl.config(text=" —", fg=COLORS["fg_muted"])

        # Update metric bars
        cpu_pct = metrics.get("cpu_percent", 0)
        mem_pct = metrics.get("memory_percent", 0)
        gpu_pct = metrics.get("gpu_percent", 0)

        self._draw_bar(self.cpu_canvas, self.cpu_bar, cpu_pct, COLORS["cpu"])
        self.cpu_text.config(text=f"{cpu_pct:.0f}%")

        self._draw_bar(self.mem_canvas, self.mem_bar, mem_pct, COLORS["mem"])
        self.mem_text.config(text=f"{mem_pct:.0f}%")

        self._draw_bar(self.gpu_canvas, self.gpu_bar, gpu_pct, COLORS["gpu"])
        self.gpu_text.config(text=f"{gpu_pct:.0f}%")

        # Schedule next update
        self.root.after(UPDATE_INTERVAL_MS, self._update_status)

    def _draw_bar(self, canvas, bar_id, percent, color):
        """Draw a horizontal progress bar on a canvas."""
        canvas.delete("bar")
        w = canvas.winfo_width()
        if w < 2:
            w = WINDOW_WIDTH - 80
        fill_w = max(0, min(w, int(w * percent / 100)))
        canvas.create_rectangle(0, 0, fill_w, 8, fill=color, outline="", tags="bar")

    def _pulse_animation(self):
        """Animate status dot with pulse effect when active."""
        status = self._status.get("status", "idle")
        if status in ("thinking", "executing"):
            self._pulse_alpha += self._pulse_direction * 4
            if self._pulse_alpha >= 20:
                self._pulse_direction = -1
            elif self._pulse_alpha <= 0:
                self._pulse_direction = 1
        self.root.after(PULSE_INTERVAL_MS, self._pulse_animation)

    def _scroll_task(self):
        """Scroll the task text horizontally."""
        canvas = self.task_canvas
        canvas.delete("all")
        w = canvas.winfo_width()
        if w < 2:
            w = WINDOW_WIDTH - 50

        text = self._task_text
        if text:
            self._scroll_x -= 1
            text_w = len(text) * 6
            if self._scroll_x < -text_w:
                self._scroll_x = w
            canvas.create_text(
                self._scroll_x, 11, text=text, font=("Courier", 9),
                fg=COLORS["accent"], anchor=self.tk.W
            )
        else:
            canvas.create_text(
                2, 11, text="Waiting...", font=("Courier", 9),
                fg=COLORS["fg_muted"], anchor=self.tk.W
            )

        self.root.after(SCROLL_INTERVAL_MS, self._scroll_task)

    def run(self):
        """Start the main loop."""
        self.root.mainloop()


# --- CLI ---
def main():
    if len(sys.argv) > 1:
        arg = sys.argv[1].lstrip("-")
        if arg in ("test", "self-test"):
            run_self_test()
            return
        if arg in ("status",):
            print_status()
            return
        if arg in ("terminal", "term", "t"):
            run_terminal_mode()
            return
        if arg in ("help", "h"):
            print(__doc__)
            return

    # Try to import tkinter for GUI
    try:
        import tkinter as tk
        has_tkinter = True
    except ImportError:
        has_tkinter = False

    if has_tkinter:
        app = ATSMDesktopPresence()
        app.run()
    else:
        print(f"{ANSI['yellow']}tkinter not available — running in terminal mode.{ANSI['reset']}")
        print(f"{ANSI['dim']}Press Ctrl+C to exit.{ANSI['reset']}\n")
        run_terminal_mode()


if __name__ == "__main__":
    main()
