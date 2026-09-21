#!/usr/bin/env python3
"""ATSM Plugin Auto-Installer.

Checks for missing but needed Python packages for ATSM subsystems,
auto-installs them via pip, updates ATSM priors, and tracks state.

Usage:
    python3 plugin_install.py check
    python3 plugin_install.py install
    python3 plugin_install.py list
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SKILL_DIR = Path(__file__).parent.parent
DATA_DIR = SKILL_DIR / "data"
PRIORS_FILE = DATA_DIR / "atsm_priors.json"
INSTALLED_FILE = DATA_DIR / "installed_plugins.json"

# Plugin registry: plugin_name -> pip_package (None if stdlib/system)
# Maps plugin identifier to the pip package that satisfies its requirement.
PLUGINS = {
    "embed_rank": {
        "description": "Embedding-based skill ranker (sentence-transformers)",
        "pip_package": "sentence-transformers",
        "import_name": "sentence_transformers",
    },
    "browser_bot": {
        "description": "Browser automation via CDP (websocket-client)",
        "pip_package": "websocket-client",
        "import_name": "websocket",
    },
    "sys_monitor": {
        "description": "System metrics monitor (psutil)",
        "pip_package": "psutil",
        "import_name": "psutil",
    },
    "desktop_widget": {
        "description": "Desktop GUI widget (tkinter)",
        "pip_package": None,  # stdlib / system package
        "import_name": "tkinter",
    },
}


# --- Utility ---
def ensure_data_dir():
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def load_installed() -> dict:
    """Load installed plugins tracking data."""
    if INSTALLED_FILE.exists():
        try:
            return json.loads(INSTALLED_FILE.read_text())
        except (json.JSONDecodeError, IOError):
            pass
    return {"plugins": {}, "meta": {"version": 1}}


def save_installed(data: dict):
    """Save installed plugins tracking data."""
    ensure_data_dir()
    data["meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
    INSTALLED_FILE.write_text(json.dumps(data, indent=2))


def is_importable(import_name: str) -> bool:
    """Check if a module can be imported."""
    # Special case: tkinter may be importable but need system package
    result = subprocess.run(
        [sys.executable, "-c", f"import {import_name}"],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0


def check_plugin(plugin_name: str) -> dict:
    """Check if a plugin's dependency is satisfied.

    Returns a dict with:
        - name: plugin name
        - available: bool (import succeeded)
        - status: "available" | "missing" | "stdlib_missing"
        - description: plugin description
        "pip_package": the pip package name or None
        "import_name": the module to import
    """
    plugin = PLUGINS[plugin_name]
    import_name = plugin["import_name"]
    pip_package = plugin["pip_package"]
    available = is_importable(import_name)

    if available:
        status = "available"
    elif pip_package is None:
        status = "stdlib_missing"
    else:
        status = "missing"

    return {
        "name": plugin_name,
        "available": available,
        "status": status,
        "description": plugin["description"],
        "pip_package": pip_package,
        "import_name": import_name,
    }


def install_pip_package(package_name: str) -> tuple[bool, str]:
    """Install a package via pip.

    Tries multiple strategies: --user, --break-system-packages, then plain install.
    Returns (success, message).
    """
    base_cmd = [sys.executable, "-m", "pip", "install"]
    strategies = [
        base_cmd + ["--user", package_name],
        base_cmd + ["--break-system-packages", package_name],
        base_cmd + [package_name],
    ]

    for cmd in strategies:
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                return True, f"✓ {package_name} installed successfully."
        except subprocess.TimeoutExpired:
            continue
        except FileNotFoundError:
            return False, "✗ pip not found."
        except Exception as e:
            return False, f"✗ Unexpected error: {e}"

    return False, f"✗ Failed to install {package_name} (tried --user, --break-system-packages, plain)"


def update_priors(plugin_names: list[str]) -> bool:
    """Update ATSM priors to reflect newly installed plugins.

    Adds prior entries for plugins so they can be ranked.
    """
    if not PRIORS_FILE.exists():
        return False

    try:
        with open(PRIORS_FILE, "r", encoding="utf-8") as f:
            priors = json.load(f)
    except (json.JSONDecodeError, IOError):
        return False

    # Ensure plugin priors exist (neutral Beta(1,1))
    updated = False
    for name in plugin_names:
        if name not in priors:
            priors[name] = {
                "alpha": 1.0,
                "beta": 1.0,
                "expected_success": 0.5,
                "observations": 0,
            }
            updated = True

    # Update metadata
    if "_meta" in priors:
        priors["_meta"]["updated_at"] = datetime.now(timezone.utc).isoformat()
        skills_count = len([k for k in priors if not k.startswith("_")])
        priors["_meta"]["skills_tracked"] = skills_count

    if updated:
        with open(PRIORS_FILE, "w", encoding="utf-8") as f:
            json.dump(priors, f, indent=2)
        return True
    return False


def record_install(plugin_name: str, success: bool, version: str = None):
    """Record a plugin installation in the tracking file."""
    data = load_installed()
    data["plugins"][plugin_name] = {
        "installed_at": datetime.now(timezone.utc).isoformat(),
        "success": success,
        "version": version,
    }
    save_installed(data)


# --- Commands ---
def cmd_check() -> dict:
    """Check all plugins and report availability.

    Returns a summary dict.
    """
    results = {}
    for name in PLUGINS:
        info = check_plugin(name)
        results[name] = info

    available = [n for n, r in results.items() if r["available"]]
    missing = [n for n, r in results.items() if not r["available"]]

    print("╔══════════════════════════════════════════════════════════╗")
    print("║           ATSM Plugin Dependency Check                  ║")
    print("╠══════════════════════════════════════════════════════════╣")

    for name, info in results.items():
        if info["available"]:
            icon = "✓"
            status_str = "available"
        elif info["status"] == "stdlib_missing":
            icon = "✗"
            status_str = "stdlib missing"
        else:
            icon = "✗"
            status_str = "missing (pip)"

        print(f"║  {icon} {info['description']:<50} {status_str:>10} ║")

    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Available: {len(available)}/{len(PLUGINS)}                                        ║")
    if missing:
        print(f"║  Missing:   {len(missing)} → run 'install' to fix                      ║")
    print("╚══════════════════════════════════════════════════════════╝")

    return results


def cmd_install(dry_run: bool = False) -> dict:
    """Install missing plugins.

    Returns a summary of actions taken.
    """
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          ATSM Plugin Auto-Installer                     ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    results = {}
    installed_names = []
    failed_names = []
    skipped_names = []

    for name, plugin in PLUGINS.items():
        info = check_plugin(name)
        pip_pkg = plugin["pip_package"]

        if info["available"]:
            print(f"  ✓ {name:20s} — already available, skipping")
            skipped_names.append(name)
            results[name] = {"status": "skipped", "action": "none"}
            continue

        if pip_pkg is None:
            # stdlib / system package — can't pip install
            print(f"  ✗ {name:20s} — requires system package (not pip-installable)")
            print(f"    → Install via system package manager (e.g. apt install python3-tk)")
            failed_names.append(name)
            results[name] = {"status": "failed", "reason": "system_package"}
            continue

        if dry_run:
            print(f"  ⊘ {name:20s} — would install: pip install {pip_pkg}")
            results[name] = {"status": "dry_run", "package": pip_pkg}
            continue

        print(f"  ↓ Installing {name:20s} ({pip_pkg})...")
        success, message = install_pip_package(pip_pkg)

        if success:
            # Verify import works now
            if is_importable(plugin["import_name"]):
                print(f"    ✓ Verified: import {plugin['import_name']} OK")
                record_install(name, True)
                installed_names.append(name)
                results[name] = {"status": "installed", "package": pip_pkg}
            else:
                print(f"    ✗ Installed but import failed — may need restart")
                record_install(name, False)
                failed_names.append(name)
                results[name] = {"status": "failed", "reason": "import_failed"}
        else:
            print(f"    {message}")
            record_install(name, False)
            failed_names.append(name)
            results[name] = {"status": "failed", "reason": message}

    # Update priors for newly installed plugins
    if installed_names and not dry_run:
        print()
        print("  Updating ATSM priors...")
        if update_priors(installed_names):
            print(f"  ✓ Added prior entries for: {', '.join(installed_names)}")
        else:
            print("  ⊘ No prior updates needed.")

    # Summary
    print()
    print("  ── Summary ─────────────────────────────────────────")
    print(f"    Installed: {len(installed_names)}")
    print(f"    Skipped (already available): {len(skipped_names)}")
    print(f"    Failed: {len(failed_names)}")

    return results


def cmd_list() -> dict:
    """List all tracked plugin installations."""
    data = load_installed()
    plugins = data.get("plugins", {})

    print("╔══════════════════════════════════════════════════════════╗")
    print("║          Installed ATSM Plugins                         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    if not plugins:
        print("  No plugins recorded yet. Run 'install' first.")
        return data

    print()
    print(f"  {'Plugin':<20} {'Status':<10} {'Date':<25} {'Version'}")
    print("  " + "─" * 70)

    for name, info in sorted(plugins.items()):
        status = "✓ ok" if info.get("success") else "✗ failed"
        date = info.get("installed_at", "?")[:19]  # trim to seconds
        version = info.get("version", "—") or "—"
        print(f"  {name:<20} {status:<10} {date:<25} {version}")

    print()
    print(f"  Total tracked: {len(plugins)}")
    if "meta" in data and "updated_at" in data["meta"]:
        print(f"  Last updated: {data['meta']['updated_at'][:19]}")

    return data


# --- Main ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "check":
        cmd_check()
    elif command == "install":
        dry_run = "--dry-run" in sys.argv
        cmd_install(dry_run=dry_run)
    elif command == "list":
        cmd_list()
    elif command in ("--help", "-h"):
        print(__doc__)
    else:
        print(f"Unknown command: {command}")
        print("Available commands: check, install, list")
        sys.exit(1)


if __name__ == "__main__":
    main()
