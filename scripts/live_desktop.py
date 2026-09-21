#!/usr/bin/env python3
"""ATSM Live Desktop — Real-time always-on-top system status window.

Shows ATSM status, current task, last actions, and system metrics in a
compact 400x300 always-on-top window at the bottom-right of the screen.

CLI:
    python3 live_desktop.py          # Start the live desktop window
    python3 live_desktop.py --status # Print current status and exit

Status file: /tmp/atsm_status.json (updated by ATSM gateway/agent)
"""

import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Constants ---
WINDOW_WIDTH = 400
WINDOW_HEIGHT = 300
UPDATE_INTERVAL_MS = 2000  # 2 seconds
PULSE_INTERVAL_MS = 500
STATUS_FILE = Path("/tmp/atsm_status.json")

# --- Color scheme ---
COLORS = {
    "bg": "#1a1a2e",
    "fg": "#e0e0e0",
    "accent": "#00d4ff",
    "idle": "#888888",
    "thinking": "#ffd700",
    "executing": "#00ff88",
    "error": "#ff4444",
    "cpu": "#00d4ff",
    "mem": "#ff6b9d",
    "gpu": "#c084fc",
    "bar_bg": "#2a2a3e",
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
        "actions": [],
        "metrics": {
            "cpu_percent": 0.0,
            "memory_percent": 0.0,
            "gpu_percent": 0.0,
        },
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def print_status():
    """Print current status to stdout and exit."""
    status = read_status()
    print(json.dumps(status, indent=2, ensure_ascii=False))
    sys.exit(0)


# --- Main Application ---
class ATSMDesktop:
    """Live desktop window for ATSM status display."""

    def __init__(self):
        import tkinter as tk
        self.tk = tk
        self.root = tk.Tk()
        self.root.title("ATSM")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(False, False)
        self.root.attributes("-topmost", True)

        # Position at bottom-right
        screen_w = self.root.winfo_screenwidth()
        screen_h = self.root.winfo_screenheight()
        x = screen_w - WINDOW_WIDTH - 20
        y = screen_h - WINDOW_HEIGHT - 60  # leave room for taskbar
        self.root.geometry(f"+{x}+{y}")

        # Pulse state
        self._pulse_alpha = 0
        self._pulse_direction = 1

        # Data state
        self._status = get_default_status()

        self._build_ui()
        self._start_updates()

    def _build_ui(self):
        """Build the UI layout."""
        tk = self.tk
        # Main container with padding
        main = tk.Frame(self.root, bg=COLORS["bg"], padx=10, pady=8)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Header: Status ---
        header_frame = tk.Frame(main, bg=COLORS["bg"])
        header_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            header_frame,
            text="ATSM",
            font=("Helvetica", 10, "bold"),
            fg=COLORS["accent"],
            bg=COLORS["bg"],
        ).pack(side=tk.LEFT)

        self.status_label = tk.Label(
            header_frame,
            text="● IDLE",
            font=("Helvetica", 10, "bold"),
            fg=COLORS["idle"],
            bg=COLORS["bg"],
        )
        self.status_label.pack(side=tk.RIGHT)

        # --- Current Task (scrolling) ---
        task_frame = tk.Frame(main, bg=COLORS["bar_bg"], padx=4, pady=3)
        task_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            task_frame,
            text="TASK",
            font=("Helvetica", 7),
            fg="#888",
            bg=COLORS["bar_bg"],
        ).pack(anchor=tk.W)

        self.task_canvas = tk.Canvas(
            task_frame,
            height=20,
            bg=COLORS["bar_bg"],
            highlightthickness=0,
        )
        self.task_canvas.pack(fill=tk.X)
        self._task_text_id = None
        self._task_scroll_x = 0

        # --- Last 3 Actions ---
        actions_frame = tk.Frame(main, bg=COLORS["bg"])
        actions_frame.pack(fill=tk.X, pady=(0, 4))

        tk.Label(
            actions_frame,
            text="LAST ACTIONS",
            font=("Helvetica", 7),
            fg="#888",
            bg=COLORS["bg"],
        ).pack(anchor=tk.W)

        self.action_labels = []
        for i in range(3):
            lbl = tk.Label(
                actions_frame,
                text="—",
                font=("Courier", 8),
                fg=COLORS["fg"],
                bg=COLORS["bg"],
                anchor=tk.W,
            )
            lbl.pack(fill=tk.X)
            self.action_labels.append(lbl)

        # --- Metrics Bars ---
        metrics_frame = tk.Frame(main, bg=COLORS["bg"])
        metrics_frame.pack(fill=tk.X, side=tk.BOTTOM)

        # CPU
        self.cpu_label = tk.Label(
            metrics_frame,
            text="CPU",
            font=("Helvetica", 7),
            fg=COLORS["cpu"],
            bg=COLORS["bg"],
        )
        self.cpu_label.pack(anchor=tk.W)
        self.cpu_canvas = tk.Canvas(
            metrics_frame,
            height=10,
            bg=COLORS["bar_bg"],
            highlightthickness=0,
        )
        self.cpu_canvas.pack(fill=tk.X, pady=(0, 2))

        # MEM
        self.mem_label = tk.Label(
            metrics_frame,
            text="MEM",
            font=("Helvetica", 7),
            fg=COLORS["mem"],
            bg=COLORS["bg"],
        )
        self.mem_label.pack(anchor=tk.W)
        self.mem_canvas = tk.Canvas(
            metrics_frame,
            height=10,
            bg=COLORS["bar_bg"],
            highlightthickness=0,
        )
        self.mem_canvas.pack(fill=tk.X, pady=(0, 2))

        # GPU
        self.gpu_label = tk.Label(
            metrics_frame,
            text="GPU",
            font=("Helvetica", 7),
            fg=COLORS["gpu"],
            bg=COLORS["bg"],
        )
        self.gpu_label.pack(anchor=tk.W)
        self.gpu_canvas = tk.Canvas(
            metrics_frame,
            height=10,
            bg=COLORS["bar_bg"],
            highlightthickness=0,
        )
        self.gpu_canvas.pack(fill=tk.X)

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
        actions = self._status.get("actions", [])
        metrics = self._status.get("metrics", {})

        # Update status label
        status_color = COLORS.get(status, COLORS["idle"])
        self.status_label.config(text=f"● {status.upper()}", fg=status_color)

        # Update current task
        self._task_text = task if task else "No active task"

        # Update last 3 actions
        for i, lbl in enumerate(self.action_labels):
            if i < len(actions):
                action = actions[i]
                ts = action.get("timestamp", "")
                name = action.get("name", str(action))
                # Shorten timestamp
                if "T" in ts:
                    ts = ts.split("T")[1][:8]
                lbl.config(text=f" {ts}  {name}")
            else:
                lbl.config(text=" —")

        # Update metric bars
        cpu_pct = metrics.get("cpu_percent", 0)
        mem_pct = metrics.get("memory_percent", 0)
        gpu_pct = metrics.get("gpu_percent", 0)

        self._draw_bar(self.cpu_canvas, cpu_pct, COLORS["cpu"])
        self._draw_bar(self.mem_canvas, mem_pct, COLORS["mem"])
        self._draw_bar(self.gpu_canvas, gpu_pct, COLORS["gpu"])

        # Schedule next update
        self.root.after(UPDATE_INTERVAL_MS, self._update_status)

    def _draw_bar(self, canvas, percent: float, color: str):
        """Draw a horizontal progress bar on a canvas."""
        canvas.delete("bar")
        w = canvas.winfo_width()
        if w < 2:
            w = WINDOW_WIDTH - 40
        fill_w = max(0, min(w, int(w * percent / 100)))
        canvas.create_rectangle(
            0, 0, fill_w, 10, fill=color, outline="", tags="bar"
        )

    def _pulse_animation(self):
        """Animate status dot with pulse effect when active."""
        status = self._status.get("status", "idle")
        if status in ("thinking", "executing"):
            self._pulse_alpha += self._pulse_direction * 3
            if self._pulse_alpha >= 15:
                self._pulse_direction = -1
            elif self._pulse_alpha <= 0:
                self._pulse_direction = 1
        self.root.after(PULSE_INTERVAL_MS, self._pulse_animation)

    def _scroll_task(self):
        """Scroll the task text horizontally."""
        if hasattr(self, "_task_text"):
            canvas = self.task_canvas
            canvas.delete("all")
            w = canvas.winfo_width()
            if w < 2:
                w = WINDOW_WIDTH - 40

            text = self._task_text
            if text:
                self._task_scroll_x -= 1
                text_w = len(text) * 7
                if self._task_scroll_x < -text_w:
                    self._task_scroll_x = w
                canvas.create_text(
                    self._task_scroll_x,
                    10,
                    text=text,
                    font=("Courier", 9),
                    fg=COLORS["accent"],
                    anchor=self.tk.W,
                )
            else:
                canvas.create_text(
                    2, 10, text="Waiting...", font=("Courier", 9),
                    fg="#666", anchor=self.tk.W,
                )

        self.root.after(50, self._scroll_task)

    def run(self):
        """Start the main loop."""
        self.root.mainloop()


# --- CLI ---
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--status":
        print_status()

    # Try to import tkinter for GUI
    try:
        import tkinter as tk
    except ImportError:
        print(
            "ERROR: tkinter is required for the GUI.\n"
            "Install it with: sudo apt-get install python3-tk\n"
            "Or use: python3 live_desktop.py --status (reads status without GUI)",
            file=sys.stderr,
        )
        sys.exit(1)

    app = ATSMDesktop()
    app.run()


if __name__ == "__main__":
    main()
