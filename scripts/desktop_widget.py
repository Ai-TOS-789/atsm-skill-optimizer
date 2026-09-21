#!/usr/bin/env python3
"""
ATSM Desktop Widget - A floating desktop widget showing ATSM agent status.

Usage:
    python3 desktop_widget.py start    # Launch the widget
    python3 desktop_widget.py stop     # Stop the running widget
    python3 desktop_widget.py status   # Show whether the widget is running
    python3 desktop_widget.py update '<json>'  # Update the status file

The widget reads status from: ~/.hermes/atsm_widget_status.json
"""

import sys
import os
import json
import time
import signal
import threading

try:
    import tkinter as tk
    from tkinter import font as tkfont
except ImportError:
    tk = None
    tkfont = None

# ── Terminal fallback for headless environments ─────────────────────────────
HEADLESS_MODE = (tk is None)

# ── Configuration ────────────────────────────────────────────────────────────
WORKSPACE = os.path.expanduser("~/.hermes")
STATUS_FILE = os.path.join(WORKSPACE, "atsm_widget_status.json")
PID_FILE = os.path.join(WORKSPACE, "atsm_widget.pid")
WINDOW_WIDTH = 300
WINDOW_HEIGHT = 200
UPDATE_INTERVAL = 2000  # milliseconds


# ── System metrics helpers (no external deps) ────────────────────────────────
def _read_cpu_usage():
    """Read CPU usage percentage from /proc/stat (Linux)."""
    try:
        with open("/proc/stat", "r") as f:
            parts = f.readline().split()
        if parts[0] != "cpu":
            return None
        values = [int(x) for x in parts[1:]]
        idle = values[3]
        total = sum(values)
        return idle, total
    except Exception:
        return None


def get_cpu_percent():
    """Return current CPU usage as a float 0-100."""
    first = _read_cpu_usage()
    if first is None:
        return 0.0
    time.sleep(0.1)
    second = _read_cpu_usage()
    if second is None:
        return 0.0
    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1]
    if total_delta == 0:
        return 0.0
    return (1.0 - idle_delta / total_delta) * 100.0


def get_mem_percent():
    """Return current memory usage as a float 0-100."""
    try:
        mem_info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    mem_info[key] = int(val)
        total = mem_info.get("MemTotal", 0)
        available = mem_info.get("MemAvailable", 0)
        if total == 0:
            return 0.0
        return ((total - available) / total) * 100.0
    except Exception:
        return 0.0


# ── PID management ───────────────────────────────────────────────────────────
def write_pid():
    os.makedirs(os.path.dirname(PID_FILE), exist_ok=True)
    with open(PID_FILE, "w") as f:
        f.write(str(os.getpid()))


def remove_pid():
    if os.path.exists(PID_FILE):
        os.remove(PID_FILE)


def read_pid():
    if os.path.exists(PID_FILE):
        try:
            with open(PID_FILE, "r") as f:
                return int(f.read().strip())
        except (ValueError, OSError):
            return None
    return None


def is_running(pid):
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


# ── Widget UI ────────────────────────────────────────────────────────────────
class ATSMWidget:
    """Floating desktop widget showing ATSM status."""

    STATUS_COLORS = {
        "idle": "#4ecca3",
        "listening": "#f9ed69",
        "thinking": "#e94560",
        "processing": "#0f3460",
        "error": "#ff6b6b",
    }

    def __init__(self):
        self.root = tk.Tk()
        self.root.title("ATSM Widget")
        self.root.overrideredirect(True)  # Frameless window
        self.root.attributes("-topmost", True)  # Always on top
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(False, False)

        # Position: top-right corner with small margin
        screen_w = self.root.winfo_screenwidth()
        pos_x = screen_w - WINDOW_WIDTH - 10
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}+{pos_x}+10")

        self._running = True
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Handle SIGTERM gracefully
        signal.signal(signal.SIGTERM, lambda s, f: self._on_close())

    def _build_ui(self):
        """Construct the widget UI elements."""
        title_font = tkfont.Font(family="Helvetica", size=11, weight="bold")
        normal_font = tkfont.Font(family="Helvetica", size=9)
        small_font = tkfont.Font(family="Helvetica", size=8)

        # ── Header ───────────────────────────────────────────────────────
        header_frame = tk.Frame(self.root, bg="#1a1a2e")
        header_frame.pack(fill="x", padx=8, pady=(6, 0))

        tk.Label(
            header_frame, text="◆ ATSM", fg="#e94560", bg="#1a1a2e",
            font=title_font, anchor="w"
        ).pack(side="left")

        self.uptime_label = tk.Label(
            header_frame, text="", fg="#533483", bg="#1a1a2e",
            font=small_font, anchor="e"
        )
        self.uptime_label.pack(side="right")

        # ── Status ───────────────────────────────────────────────────────
        self.status_label = tk.Label(
            self.root, text="Status: idle", fg="#4ecca3", bg="#1a1a2e",
            font=normal_font, anchor="w"
        )
        self.status_label.pack(fill="x", padx=8, pady=(2, 0))

        # ── Current Task ─────────────────────────────────────────────────
        self.task_label = tk.Label(
            self.root, text="Task: —", fg="#a8a8b3", bg="#1a1a2e",
            font=small_font, anchor="w", wraplength=280, justify="left"
        )
        self.task_label.pack(fill="x", padx=8, pady=(2, 0))

        # ── Separator ────────────────────────────────────────────────────
        tk.Frame(self.root, height=1, bg="#16213e").pack(fill="x", padx=8, pady=5)

        # ── Recent Actions ───────────────────────────────────────────────
        tk.Label(
            self.root, text="Recent Actions", fg="#e94560", bg="#1a1a2e",
            font=small_font, anchor="w"
        ).pack(fill="x", padx=8)

        self.action_labels = []
        for _ in range(3):
            lbl = tk.Label(
                self.root, text="", fg="#7b68ee", bg="#1a1a2e",
                font=small_font, anchor="w", wraplength=280, justify="left"
            )
            lbl.pack(fill="x", padx=8)
            self.action_labels.append(lbl)

        # ── Separator ────────────────────────────────────────────────────
        tk.Frame(self.root, height=1, bg="#16213e").pack(fill="x", padx=8, pady=5)

        # ── System Metrics ───────────────────────────────────────────────
        metrics_frame = tk.Frame(self.root, bg="#1a1a2e")
        metrics_frame.pack(fill="x", padx=8, pady=(0, 4))

        # CPU row
        cpu_row = tk.Frame(metrics_frame, bg="#1a1a2e")
        cpu_row.pack(fill="x")
        tk.Label(
            cpu_row, text="CPU", fg="#a8a8b3", bg="#1a1a2e",
            font=small_font, width=4, anchor="w"
        ).pack(side="left")
        self.cpu_bar_canvas = tk.Canvas(
            cpu_row, height=8, bg="#16213e", highlightthickness=0
        )
        self.cpu_bar_canvas.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.cpu_bar_rect = self.cpu_bar_canvas.create_rectangle(
            0, 0, 0, 8, fill="#4ecca3", outline=""
        )
        self.cpu_text_label = tk.Label(
            cpu_row, text="0%", fg="#4ecca3", bg="#1a1a2e",
            font=small_font, width=6, anchor="e"
        )
        self.cpu_text_label.pack(side="right")

        # MEM row
        mem_row = tk.Frame(metrics_frame, bg="#1a1a2e")
        mem_row.pack(fill="x", pady=(2, 0))
        tk.Label(
            mem_row, text="MEM", fg="#a8a8b3", bg="#1a1a2e",
            font=small_font, width=4, anchor="w"
        ).pack(side="left")
        self.mem_bar_canvas = tk.Canvas(
            mem_row, height=8, bg="#16213e", highlightthickness=0
        )
        self.mem_bar_canvas.pack(side="left", fill="x", expand=True, padx=(4, 4))
        self.mem_bar_rect = self.mem_bar_canvas.create_rectangle(
            0, 0, 0, 8, fill="#0f3460", outline=""
        )
        self.mem_text_label = tk.Label(
            mem_row, text="0%", fg="#0f3460", bg="#1a1a2e",
            font=small_font, width=6, anchor="e"
        )
        self.mem_text_label.pack(side="right")

    def _on_close(self):
        """Handle window close / SIGTERM."""
        self._running = False
        remove_pid()
        self.root.quit()
        self.root.destroy()

    def _read_status(self):
        """Read and return the status dictionary from the status file."""
        try:
            if os.path.exists(STATUS_FILE):
                with open(STATUS_FILE, "r") as f:
                    return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
        return None

    def _update(self):
        """Refresh all UI elements with current data."""
        status = self._read_status()

        # ── ATSM Status ──────────────────────────────────────────────────
        if status:
            atsm_status = status.get("status", "idle")
            color = self.STATUS_COLORS.get(atsm_status, "#4ecca3")
            self.status_label.config(text=f"Status: {atsm_status}", fg=color)

            task = status.get("current_task", "—")
            self.task_label.config(text=f"Task: {task[:60]}")

            actions = status.get("recent_actions", [])
            for i, lbl in enumerate(self.action_labels):
                if i < len(actions):
                    action = actions[i]
                    ts = action.get("timestamp", "??")
                    name = action.get("name", "??")
                    result = action.get("result", "")
                    text = f"• [{ts}] {name}"
                    if result:
                        text += f" → {result}"
                    lbl.config(text=text, fg="#7b68ee")
                else:
                    lbl.config(text="", fg="#7b68ee")
        else:
            self.status_label.config(text="Status: idle", fg="#4ecca3")
            self.task_label.config(text="Task: —")
            for lbl in self.action_labels:
                lbl.config(text="")

        # ── System Metrics ───────────────────────────────────────────────
        cpu = get_cpu_percent()
        mem = get_mem_percent()

        # Update CPU bar
        cpu_w = self.cpu_bar_canvas.winfo_width()
        if cpu_w > 1:
            fill_w = max(1, int(cpu_w * cpu / 100))
            self.cpu_bar_canvas.coords(self.cpu_bar_rect, 0, 0, fill_w, 8)
            # Color shift based on usage
            if cpu > 80:
                self.cpu_bar_canvas.itemconfig(self.cpu_bar_rect, fill="#e94560")
            elif cpu > 50:
                self.cpu_bar_canvas.itemconfig(self.cpu_bar_rect, fill="#f9ed69")
            else:
                self.cpu_bar_canvas.itemconfig(self.cpu_bar_rect, fill="#4ecca3")
        self.cpu_text_label.config(text=f"{cpu:.0f}%")

        # Update MEM bar
        mem_w = self.mem_bar_canvas.winfo_width()
        if mem_w > 1:
            fill_w = max(1, int(mem_w * mem / 100))
            self.mem_bar_canvas.coords(self.mem_bar_rect, 0, 0, fill_w, 8)
            if mem > 80:
                self.mem_bar_canvas.itemconfig(self.mem_bar_rect, fill="#e94560")
            elif mem > 50:
                self.mem_bar_canvas.itemconfig(self.mem_bar_rect, fill="#f9ed69")
            else:
                self.mem_bar_canvas.itemconfig(self.mem_bar_rect, fill="#0f3460")
        self.mem_text_label.config(text=f"{mem:.0f}%")

    def _schedule_update(self):
        """Schedule the next UI refresh."""
        if self._running:
            self._update()
            self.root.after(UPDATE_INTERVAL, self._schedule_update)

    def run(self):
        """Start the widget main loop."""
        self._schedule_update()
        self.root.mainloop()


# ── CLI Commands ─────────────────────────────────────────────────────────────
def cmd_start():
    """Launch the widget."""
    pid = read_pid()
    if pid and is_running(pid):
        print("ATSM Widget is already running.")
        sys.exit(0)

    if HEADLESS_MODE:
        print("Warning: tkinter not available. Running in headless mode (status polling only).")
        print(f"Status file: {STATUS_FILE}")
        print("Use 'update' command to push status data.")
        # In headless mode, just keep the PID file and poll
        write_pid()
        try:
            while True:
                status = None
                try:
                    if os.path.exists(STATUS_FILE):
                        with open(STATUS_FILE, "r") as f:
                            status = json.load(f)
                except (json.JSONDecodeError, OSError):
                    pass
                if status:
                    print(f"\r[{time.strftime('%H:%M:%S')}] Status: {status.get('status','?')} | Task: {status.get('current_task','—')}", end="", flush=True)
                time.sleep(2)
        except KeyboardInterrupt:
            print("\nWidget stopped.")
            remove_pid()
        return

    write_pid()
    widget = ATSMWidget()
    widget.run()
    remove_pid()


def cmd_stop():
    """Stop the running widget."""
    pid = read_pid()
    if not pid:
        print("ATSM Widget is not running (no PID file found).")
        sys.exit(0)

    if not is_running(pid):
        print("ATSM Widget is not running (stale PID file removed).")
        remove_pid()
        return

    try:
        os.kill(pid, signal.SIGTERM)
        print(f"ATSM Widget stopped (PID: {pid}).")
    except ProcessLookupError:
        print("ATSM Widget was already stopped.")
        remove_pid()


def cmd_status():
    """Show whether the widget is running."""
    pid = read_pid()
    if pid and is_running(pid):
        print(f"ATSM Widget is running (PID: {pid}).")
    else:
        if pid:
            print("ATSM Widget is not running (stale PID file).")
            remove_pid()
        else:
            print("ATSM Widget is not running.")


def cmd_update():
    """Update the status file with provided JSON data."""
    if len(sys.argv) < 3:
        print("Usage: python3 desktop_widget.py update '<json_string>'")
        print('Example: python3 desktop_widget.py update \'{"status":"thinking","current_task":"analyzing code"}\'')
        sys.exit(1)

    try:
        data = json.loads(sys.argv[2])
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        sys.exit(1)

    os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
    with open(STATUS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    print("Status updated successfully.")


# ── Main Entry Point ─────────────────────────────────────────────────────────
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    cmd = sys.argv[1].lower().strip("-")

    if cmd in ("start",):
        cmd_start()
    elif cmd in ("stop",):
        cmd_stop()
    elif cmd in ("status",):
        cmd_status()
    elif cmd in ("update",):
        cmd_update()
    else:
        print(f"Unknown command: {sys.argv[1]}")
        print("Usage: python3 desktop_widget.py {start|stop|status|update}")
        sys.exit(1)


if __name__ == "__main__":
    main()
