#!/usr/bin/env python3
"""
System Monitor - tracks CPU, Memory, Disk, Network, and GPU metrics.

CLI usage:
    python3 sys_monitor.py snapshot   - Take a snapshot of all metrics
    python3 sys_monitor.py history N  - Show last N entries
    python3 sys_monitor.py check      - Check current metrics against alert thresholds
    python3 sys_monitor.py gpu        - Show GPU metrics only
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR.parent / "data"
METRICS_FILE = DATA_DIR / "sys_metrics.jsonl"

# Alert thresholds
ALERT_CPU = 90.0
ALERT_MEM = 90.0
ALERT_DISK = 85.0


def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def read_metrics_history(limit=None):
    """Read metrics history from JSONL file."""
    entries = []
    if METRICS_FILE.exists():
        with open(METRICS_FILE, "r") as f:
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


def write_metric(entry):
    """Append a metric entry to the JSONL file."""
    ensure_data_dir()
    with open(METRICS_FILE, "a") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def get_cpu_metrics():
    """Get CPU usage per core using /proc/stat."""
    try:
        with open("/proc/stat", "r") as f:
            lines = f.readlines()

        cores = []
        cpu_times = []
        for line in lines:
            if line.startswith("cpu"):
                parts = line.split()
                name = parts[0]
                times = [int(x) for x in parts[1:]]
                idle = times[3]
                total = sum(times)
                cpu_times.append((name, idle, total))
                if name != "cpu":
                    cores.append(name)

        # Need two readings for usage calculation
        time.sleep(0.5)

        with open("/proc/stat", "r") as f:
            lines2 = f.readlines()

        cpu_times2 = []
        for line in lines2:
            if line.startswith("cpu"):
                parts = line.split()
                name = parts[0]
                times = [int(x) for x in parts[1:]]
                idle = times[3]
                total = sum(times)
                cpu_times2.append((name, idle, total))

        # Calculate usage
        cpu_info = {"cores": [], "overall": {}}
        per_core = []

        for (name1, idle1, total1), (name2, idle2, total2) in zip(cpu_times, cpu_times2):
            delta_idle = idle2 - idle1
            delta_total = total2 - total1
            if delta_total == 0:
                usage = 0.0
            else:
                usage = ((delta_total - delta_idle) / delta_total) * 100

            core_data = {"core": name1, "usage_percent": round(usage, 1)}
            if name1 != "cpu":
                per_core.append(core_data)

            if name1 == "cpu":
                cpu_info["overall"] = core_data

        cpu_info["cores"] = per_core
        return cpu_info

    except Exception as e:
        return {"error": str(e), "cores": [], "overall": {}}


def get_memory_metrics():
    """Get memory usage from /proc/meminfo."""
    try:
        mem_info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip().split()[0]
                    mem_info[key] = int(value)

        total = mem_info.get("MemTotal", 0)
        available = mem_info.get("MemAvailable", 0)
        free = mem_info.get("MemFree", 0)
        used = total - available
        percent = (used / total * 100) if total > 0 else 0.0

        return {
            "total_kb": total,
            "used_kb": used,
            "available_kb": available,
            "total_gb": round(total / 1048576, 2),
            "used_gb": round(used / 1048576, 2),
            "available_gb": round(available / 1048576, 2),
            "usage_percent": round(percent, 1),
        }
    except Exception as e:
        return {"error": str(e)}


def get_disk_metrics():
    """Get disk usage per mount using df."""
    try:
        result = subprocess.run(
            ["df", "-B1", "--output=source,target,size,used,avail,pcent"],
            capture_output=True, text=True, timeout=10
        )
        mounts = []
        lines = result.stdout.strip().split("\n")
        headers = lines[0].split()

        for line in lines[1:]:
            parts = line.split()
            if len(parts) < 6:
                continue
            source = parts[0]
            target = parts[1]
            size = int(parts[2])
            used = int(parts[3])
            avail = int(parts[4])
            pcent = int(parts[5].rstrip("%"))

            mounts.append({
                "device": source,
                "mountpoint": target,
                "total_bytes": size,
                "used_bytes": used,
                "available_bytes": avail,
                "total_gb": round(size / 1073741824, 2),
                "used_gb": round(used / 1073741824, 2),
                "usage_percent": pcent,
            })

        return mounts
    except Exception as e:
        return [{"error": str(e)}]


def get_network_metrics():
    """Get network bytes sent/received from /proc/net/dev."""
    try:
        with open("/proc/net/dev", "r") as f:
            lines = f.readlines()

        interfaces = {}
        for line in lines[2:]:
            parts = line.split(":")
            if len(parts) != 2:
                continue
            iface = parts[0].strip()
            if iface == "lo":
                continue
            stats = parts[1].split()
            rx_bytes = int(stats[0])
            tx_bytes = int(stats[8])
            interfaces[iface] = {
                "rx_bytes": rx_bytes,
                "tx_bytes": tx_bytes,
                "rx_mb": round(rx_bytes / 1048576, 2),
                "tx_mb": round(tx_bytes / 1048576, 2),
            }

        return interfaces
    except Exception as e:
        return {"error": str(e)}


def get_gpu_metrics():
    """Get GPU metrics via nvidia-smi if available."""
    gpu_info = {"available": False, "gpus": []}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,temperature.gpu,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free,driver_version",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return gpu_info

        gpu_info["available"] = True
        for line in result.stdout.strip().split("\n"):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 8:
                gpu_info["gpus"].append({
                    "name": parts[0],
                    "temperature_c": int(parts[1]) if parts[1] != "N/A" else None,
                    "gpu_utilization_percent": float(parts[2]) if parts[2] != "N/A" else None,
                    "memory_utilization_percent": float(parts[3]) if parts[3] != "N/A" else None,
                    "memory_total_mb": float(parts[4]) if parts[4] != "N/A" else None,
                    "memory_used_mb": float(parts[5]) if parts[5] != "N/A" else None,
                    "memory_free_mb": float(parts[6]) if parts[6] != "N/A" else None,
                    "driver_version": parts[7],
                })
    except FileNotFoundError:
        gpu_info["available"] = False
        gpu_info["error"] = "nvidia-smi not found"
    except Exception as e:
        gpu_info["error"] = str(e)

    return gpu_info


def check_alerts(metrics):
    """Check metrics against alert thresholds."""
    alerts = []

    cpu_overall = metrics.get("cpu", {}).get("overall", {})
    if cpu_overall.get("usage_percent", 0) > ALERT_CPU:
        alerts.append(f"⚠ CPU usage {cpu_overall['usage_percent']}% exceeds {ALERT_CPU}%")

    mem = metrics.get("memory", {})
    if mem.get("usage_percent", 0) > ALERT_MEM:
        alerts.append(f"⚠ Memory usage {mem['usage_percent']}% exceeds {ALERT_MEM}%")

    for mount in metrics.get("disk", []):
        if mount.get("usage_percent", 0) > ALERT_DISK:
            alerts.append(
                f"⚠ Disk {mount.get('mountpoint', '?')} usage {mount['usage_percent']}% exceeds {ALERT_DISK}%"
            )

    return alerts


def cmd_snapshot():
    """Take a full system snapshot and store it."""
    timestamp = datetime.now().isoformat()

    metrics = {
        "timestamp": timestamp,
        "cpu": get_cpu_metrics(),
        "memory": get_memory_metrics(),
        "disk": get_disk_metrics(),
        "network": get_network_metrics(),
        "gpu": get_gpu_metrics(),
    }

    write_metric(metrics)

    # Print summary
    cpu = metrics["cpu"].get("overall", {}).get("usage_percent", "N/A")
    mem = metrics["memory"].get("usage_percent", "N/A")
    gpu_avail = metrics["gpu"].get("available", False)
    gpu_count = len(metrics["gpu"].get("gpus", []))

    print(f"=== System Snapshot ({timestamp}) ===")
    print(f"CPU Usage:    {cpu}%")
    print(f"Memory Usage: {mem}% ({metrics['memory'].get('used_gb', '?')}/{metrics['memory'].get('total_gb', '?')} GB)")
    print(f"GPU Available: {gpu_avail} ({gpu_count} GPUs)")
    print()

    # Per-core CPU
    cores = metrics["cpu"].get("cores", [])
    if cores:
        print("CPU Cores:")
        for c in cores:
            bar = "█" * int(c["usage_percent"] / 5) + "░" * (20 - int(c["usage_percent"] / 5))
            print(f"  {c['core']:>6}: {bar} {c['usage_percent']}%")

    # Disk
    print("\nDisk Usage:")
    for m in metrics["disk"]:
        if "error" in m:
            continue
        bar = "█" * int(m["usage_percent"] / 5) + "░" * (20 - int(m["usage_percent"] / 5))
        print(f"  {m['mountpoint']:>15}: {bar} {m['usage_percent']}% ({m['used_gb']}/{m['total_gb']} GB)")

    # Network
    print("\nNetwork:")
    for iface, net in metrics["network"].items():
        if iface == "error":
            continue
        print(f"  {iface}: TX={net['tx_mb']} MB, RX={net['rx_mb']} MB")

    # GPU
    if gpu_avail and gpu_count > 0:
        print("\nGPU:")
        for g in metrics["gpu"]["gpus"]:
            print(f"  {g['name']}: Temp={g['temperature_c']}°C, GPU={g['gpu_utilization_percent']}%, "
                  f"Mem={g['memory_used_mb']}/{g['memory_total_mb']} MB")

    # Alerts
    alerts = check_alerts(metrics)
    if alerts:
        print("\n🚨 ALERTS:")
        for a in alerts:
            print(f"  {a}")
    else:
        print("\n✓ All metrics within thresholds")

    print(f"\nSaved to {METRICS_FILE}")


def cmd_history(count):
    """Show last N history entries."""
    entries = read_metrics_history(limit=count)

    if not entries:
        print("No metrics history found.")
        return

    print(f"=== Last {len(entries)} Snapshots ===\n")

    for entry in entries:
        ts = entry.get("timestamp", "?")
        cpu = entry.get("cpu", {}).get("overall", {}).get("usage_percent", "N/A")
        mem = entry.get("memory", {}).get("usage_percent", "N/A")
        gpu = entry.get("gpu", {}).get("available", False)

        alerts = check_alerts(entry)
        alert_flag = " 🚨" if alerts else ""

        print(f"[{ts}] CPU={cpu}% | MEM={mem}% | GPU={'✓' if gpu else '✗'}{alert_flag}")


def cmd_check():
    """Quick check: current metrics against thresholds."""
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "cpu": get_cpu_metrics(),
        "memory": get_memory_metrics(),
        "disk": get_disk_metrics(),
        "network": get_network_metrics(),
    }

    cpu = metrics["cpu"].get("overall", {}).get("usage_percent", 0)
    mem = metrics["memory"].get("usage_percent", 0)

    alerts = check_alerts(metrics)

    print(f"=== System Check ({metrics['timestamp']}) ===")
    print(f"CPU:    {cpu}% {'⚠' if cpu > ALERT_CPU else '✓'}")
    print(f"Memory: {mem}% ({metrics['memory'].get('used_gb', '?')}/{metrics['memory'].get('total_gb', '?')} GB) {'⚠' if mem > ALERT_MEM else '✓'}")

    max_disk = max((d.get("usage_percent", 0) for d in metrics["disk"]), default=0)
    print(f"Disk:   {max_disk}% (max) {'⚠' if max_disk > ALERT_DISK else '✓'}")

    if alerts:
        print("\n🚨 ALERTS:")
        for a in alerts:
            print(f"  {a}")
        sys.exit(1)
    else:
        print("\n✓ All metrics within thresholds")
        sys.exit(0)


def cmd_gpu():
    """Show GPU metrics only."""
    gpu = get_gpu_metrics()

    if not gpu["available"]:
        print("GPU: nvidia-smi not available")
        if "error" in gpu:
            print(f"  ({gpu['error']})")
        return

    print(f"=== GPU Status ({len(gpu['gpus'])} GPUs) ===\n")
    for i, g in enumerate(gpu["gpus"]):
        print(f"GPU {i}: {g['name']}")
        print(f"  Driver Version: {g['driver_version']}")
        print(f"  Temperature:    {g['temperature_c']}°C")
        print(f"  GPU Utilization: {g['gpu_utilization_percent']}%")
        print(f"  Memory:         {g['memory_used_mb']}/{g['memory_total_mb']} MB "
              f"({g['memory_utilization_percent']}% util)")
        print(f"  Memory Free:    {g['memory_free_mb']} MB")
        print()


def main():
    if len(sys.argv) < 2:
        print("Usage: sys_monitor.py <snapshot|history|check|gpu>")
        print("  snapshot   - Take and store a full system snapshot")
        print("  history N  - Show last N snapshots")
        print("  check      - Check current metrics against alert thresholds")
        print("  gpu        - Show GPU metrics only")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "snapshot":
        cmd_snapshot()
    elif command == "history":
        count = int(sys.argv[2]) if len(sys.argv) > 2 else 30
        cmd_history(count)
    elif command == "check":
        cmd_check()
    elif command == "gpu":
        cmd_gpu()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
