#!/usr/bin/env python3
"""ATSM Self-Update: Pull latest skill from GitHub and verify integrity.

Usage:
    python3 atsm_update.py [--repo URL] [--branch main]
"""

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_REPO = "https://github.com/ais的区域/atsm-skill-optimizer"
DEFAULT_BRANCH = "main"

SKILL_DIR = Path(__file__).parent.parent
MANIFEST_FILE = SKILL_DIR / "data" / "update_manifest.json"


def git_available() -> bool:
    return shutil.which("git") is not None


def compute_sha256(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file() and ".git" not in f.parts:
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def read_manifest() -> dict:
    if MANIFEST_FILE.exists():
        return json.loads(MANIFEST_FILE.read_text())
    return {}


def write_manifest(data: dict):
    MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_FILE.write_text(json.dumps(data, indent=2))


def update_via_git(repo_url: str, branch: str) -> bool:
    """Pull updates from git repo into skill directory."""
    if not git_available():
        print("Error: git not available")
        return False
    
    # If skill dir is not a git repo, clone into temp and copy
    git_dir = SKILL_DIR / ".git"
    if not git_dir.exists():
        print(f"Cloning {repo_url} into {SKILL_DIR}...")
        with tempfile.TemporaryDirectory() as tmp:
            result = subprocess.run(
                ["git", "clone", "--depth", "1", "--branch", branch, repo_url, tmp],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"Clone failed: {result.stderr}")
                return False
            # Copy everything except .git
            for item in Path(tmp).iterdir():
                if item.name == ".git":
                    continue
                dest = SKILL_DIR / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                if item.is_dir():
                    shutil.copytree(item, dest)
                else:
                    shutil.copy2(item, dest)
        return True
    
    # Otherwise fetch and reset
    print(f"Fetching updates from {repo_url}...")
    result = subprocess.run(
        ["git", "-C", str(SKILL_DIR), "pull", "--ff-only", "origin", branch],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"Pull failed: {result.stderr}")
        return False
    print(result.stdout.strip())
    return True


def verify_integrity() -> bool:
    """Basic integrity check: ensure core files exist and are valid."""
    required_files = ["SKILL.md", "scripts/atsm.py"]
    for f in required_files:
        path = SKILL_DIR / f
        if not path.exists():
            print(f"Integrity check failed: {f} missing")
            return False
    # Verify atsm.py parses
    result = subprocess.run(
        [sys.executable, "-c", f"import ast; ast.parse(open('{SKILL_DIR / 'scripts/atsm.py'}').read())"],
        capture_output=True
    )
    if result.returncode != 0:
        print("Integrity check failed: scripts/atsm.py has syntax errors")
        return False
    return True


def main():
    repo = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else DEFAULT_REPO
    branch = DEFAULT_BRANCH
    
    if "--branch" in sys.argv:
        idx = sys.argv.index("--branch")
        if idx + 1 < len(sys.argv):
            branch = sys.argv[idx + 1]
    
    if "--help" in sys.argv:
        print(__doc__)
        return
    
    print(f"ATSM Self-Update")
    print(f"  Repository: {repo}")
    print(f"  Branch: {branch}")
    print(f"  Target: {SKILL_DIR}")
    
    old_hash = compute_sha256(SKILL_DIR)
    
    success = update_via_git(repo, branch)
    if not success:
        print("Update failed — keeping current version")
        sys.exit(1)
    
    new_hash = compute_sha256(SKILL_DIR)
    
    if old_hash == new_hash:
        print("Already up to date.")
    else:
        print("Update applied successfully.")
        if verify_integrity():
            print("Integrity verified ✓")
            manifest = read_manifest()
            manifest["last_update"] = subprocess.run(
                ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"], capture_output=True, text=True
            ).stdout.strip()
            manifest["last_hash"] = new_hash
            write_manifest(manifest)
        else:
            print("WARNING: Integrity check failed — restore from backup recommended")
            sys.exit(1)


if __name__ == "__main__":
    main()
