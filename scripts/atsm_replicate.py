#!/usr/bin/env python3
"""ATSM Self-Replication Engine v1.0.

An agent can spawn a full copy of itself with complete knowledge transfer:
  clone     — snapshot all data, memory, priors, lessons into a clone bundle
  transfer  — send the clone bundle to a remote machine via rsync/scp
  bootstrap — start the clone on the target with full context restored
  verify    — clone reports back its capabilities and knowledge integrity

Knowledge transferred:
  - agent_memory.json       (issue history, learned lessons, cycle state)
  - atsm_db.jsonl           (skill outcome records)
  - atsm_priors.json        (Bayesian skill priors)
  - knowledge_graph.json    (skill relationship graph + learned lessons)
  - learned_lessons         (extracted from agent_memory + knowledge graph)

Usage:
    python3 atsm_replicate.py clone [--dir PATH]
    python3 atsm_replicate.py transfer user@host [--dir PATH] [--remote-dir PATH]
    python3 atsm_replicate.py bootstrap [--dir PATH]
    python3 atsm_replicate.py verify [--dir PATH]
"""

import argparse
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# --- Config ---
SCRIPTS_DIR = Path(__file__).parent.resolve()
SKILL_DIR = SCRIPTS_DIR.parent
DATA_DIR = SKILL_DIR / "data"
CLONE_BUNDLE_DIR = SKILL_DIR / "clone_bundle"

# Knowledge files required for full replication
KNOWLEDGE_FILES = [
    "agent_memory.json",
    "atsm_db.jsonl",
    "atsm_priors.json",
    "knowledge_graph.json",
]

# Additional context files (best-effort, non-fatal if missing)
AUXILIARY_FILES = [
    "agent_preferences.json",
    "evolve_prefs.json",
    "chains_db.jsonl",
    "goals_db.json",
    "swarm_db.jsonl",
    "swarm_stats.json",
]


def _read_json(path: Path) -> dict | list | None:
    """Safely read a JSON file; return None on failure."""
    if not path.exists():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file; return list of dicts."""
    records = []
    if not path.exists():
        return records
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def _compute_bundle_size(bundle_dir: Path) -> tuple[int, int]:
    """Return (file_count, total_bytes) for a directory."""
    total_bytes = 0
    file_count = 0
    for root, _dirs, files in os.walk(bundle_dir):
        for fn in files:
            fp = Path(root) / fn
            if fp.is_file():
                total_bytes += fp.stat().st_size
                file_count += 1
    return file_count, total_bytes


def _format_size(size_bytes: int) -> str:
    """Format bytes into human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def _manifest_hash(data: dict) -> str:
    """Compute a SHA-256 hash of the canonical JSON representation."""
    canonical = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]


# ---------------------------------------------------------------------------
# CLONE
# ---------------------------------------------------------------------------

def cmd_clone(bundle_dir: Path | None = None) -> dict:
    """Create a full clone bundle with all knowledge files."""
    if bundle_dir is None:
        bundle_dir = CLONE_BUNDLE_DIR

    # Clean previous bundle
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    data_bundle = bundle_dir / "data"
    data_bundle.mkdir()

    copied_files = []
    missing_files = []
    lessons = []
    manifest_entries = {}

    for fname in KNOWLEDGE_FILES:
        src = DATA_DIR / fname
        dst = data_bundle / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied_files.append(fname)

            data = _read_json(src) if fname.endswith(".json") else _read_jsonl(src)
            manifest_entries[fname] = _manifest_hash(data) if isinstance(data, (dict, list)) else "jsonl"

            if fname == "agent_memory.json" and isinstance(data, dict):
                lessons.extend(data.get("learned_lessons", []))
        else:
            missing_files.append(fname)

    # Auxiliary files (non-fatal)
    for fname in AUXILIARY_FILES:
        src = DATA_DIR / fname
        dst = data_bundle / fname
        if src.exists():
            shutil.copy2(src, dst)
            copied_files.append(fname)

    # Extract lessons from knowledge_graph if present
    kg_path = data_bundle / "knowledge_graph.json"
    if kg_path.exists():
        kg_data = _read_json(kg_path)
        if isinstance(kg_data, dict) and "learned_lessons" in kg_data:
            for lesson in kg_data["learned_lessons"]:
                if lesson not in lessons:
                    lessons.append(lesson)

    # Build replication manifest
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    file_count, total_bytes = _compute_bundle_size(bundle_dir)

    manifest = {
        "version": "1.0",
        "created_at": now,
        "source": str(SKILL_DIR),
        "files_copied": copied_files,
        "missing_files": missing_files,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "hashes": manifest_entries,
        "lessons_count": len(lessons),
        "lessons": lessons,
    }

    manifest_path = bundle_dir / "replication_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)

    print(f"[ATSM Replicate] Clone complete → {bundle_dir}")
    print(f"  Files copied:  {len(copied_files)}")
    print(f"  Missing:       {len(missing_files)} {missing_files}")
    print(f"  Lessons:       {len(lessons)}")
    print(f"  Bundle size:   {_format_size(total_bytes)} ({file_count} files)")

    return {
        "status": "ok",
        "bundle_dir": str(bundle_dir),
        "files_copied": copied_files,
        "missing_files": missing_files,
        "lessons_count": len(lessons),
        "total_bytes": total_bytes,
        "manifest": str(manifest_path),
    }


# ---------------------------------------------------------------------------
# TRANSFER
# ---------------------------------------------------------------------------

def cmd_transfer(remote_target: str, bundle_dir: Path | None = None,
                 remote_dir: str | None = None) -> dict:
    """Transfer clone bundle to remote machine via rsync or scp."""
    if bundle_dir is None:
        bundle_dir = CLONE_BUNDLE_DIR

    if not bundle_dir.exists():
        print(f"[ATSM Replicate] Bundle not found. Run 'clone' first.")
        return {"status": "error", "reason": "bundle_not_found"}

    manifest_path = bundle_dir / "replication_manifest.json"
    if not manifest_path.exists():
        print(f"[ATSM Replicate] Manifest missing. Run 'clone' first.")
        return {"status": "error", "reason": "manifest_missing"}

    manifest = _read_json(manifest_path)

    # Build transfer destination path
    if remote_dir:
        dest = f"{remote_target}:{remote_dir}"
    else:
        dest = f"{remote_target}:{bundle_dir.name}"

    print(f"[ATSM Replicate] Transferring clone to {dest}...")

    # Try rsync first, fall back to scp
    transfer_method = None
    transfer_output = ""

    try:
        result = subprocess.run(
            ["rsync", "-avz", "--progress", str(bundle_dir) + "/", dest + "/"],
            capture_output=True, text=True, timeout=300
        )
        transfer_output = result.stdout + result.stderr
        if result.returncode == 0:
            transfer_method = "rsync"
        else:
            raise subprocess.CalledProcessError(result.returncode, "rsync")
    except (FileNotFoundError, subprocess.CalledProcessError):
        # Fallback: tar + scp
        print("[ATSM Replicate] rsync unavailable, using tar+scp fallback...")
        tarball = bundle_dir.parent / f"{bundle_dir.name}.tar.gz"
        try:
            subprocess.run(
                ["tar", "-czf", str(tarball), "-C", str(bundle_dir.parent), bundle_dir.name],
                check=True, capture_output=True, timeout=60
            )
            remote_parent = remote_dir or "."
            result = subprocess.run(
                ["scp", str(tarball), f"{remote_target}:{remote_parent}/"],
                capture_output=True, text=True, timeout=300
            )
            transfer_output = result.stdout + result.stderr
            if result.returncode == 0:
                # Extract on remote side
                extract_cmd = f"cd {remote_parent} && tar -xzf {tarball.name}"
                subprocess.run(
                    ["ssh", remote_target, extract_cmd],
                    capture_output=True, text=True, timeout=60
                )
                transfer_method = "scp+tar"
            else:
                print(f"[ATSM Replicate] SCP failed: {transfer_output}")
                return {"status": "error", "reason": "scp_failed", "output": transfer_output}
        finally:
            if tarball.exists():
                tarball.unlink()

    if transfer_method:
        print(f"[ATSM Replicate] Transfer complete via {transfer_method}")
        print(f"  Destination:  {dest}")
        print(f"  Files:        {manifest.get('file_count', '?')}")
        print(f"  Size:         {_format_size(manifest.get('total_bytes', 0))}")
        return {
            "status": "ok",
            "method": transfer_method,
            "destination": dest,
            "bundle_dir": str(bundle_dir),
            "files": manifest.get("file_count"),
            "bytes": manifest.get("total_bytes"),
        }

    return {"status": "error", "reason": "no_transfer_method"}


# ---------------------------------------------------------------------------
# BOOTSTRAP
# ---------------------------------------------------------------------------

def cmd_bootstrap(bundle_dir: Path | None = None) -> dict:
    """Bootstrap clone: verify manifest, restore knowledge, report readiness."""
    if bundle_dir is None:
        bundle_dir = CLONE_BUNDLE_DIR

    if not bundle_dir.exists():
        return {"status": "error", "reason": "bundle_not_found"}

    manifest_path = bundle_dir / "replication_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        return {"status": "error", "reason": "manifest_corrupt"}

    data_dir = bundle_dir / "data"
    restored = []
    failed = []
    hash_mismatches = []

    for fname, expected_hash in manifest.get("hashes", {}).items():
        src = data_dir / fname
        if not src.exists():
            failed.append(fname)
            continue

        data = _read_json(src) if fname.endswith(".json") else _read_jsonl(src)
        if data is None:
            failed.append(fname)
            continue

        actual_hash = _manifest_hash(data) if isinstance(data, (dict, list)) else "jsonl"
        if expected_hash != "jsonl" and actual_hash != expected_hash:
            hash_mismatches.append({"file": fname, "expected": expected_hash, "actual": actual_hash})

        restored.append(fname)

    # Simulate agent start
    lessons = manifest.get("lessons", [])
    now = datetime.datetime.now(datetime.timezone.utc).isoformat()

    bootstrap_report = {
        "status": "ok",
        "bootstrapped_at": now,
        "bundle_dir": str(bundle_dir),
        "restored_files": restored,
        "failed_files": failed,
        "hash_mismatches": hash_mismatches,
        "lessons_loaded": len(lessons),
        "capabilities": {
            "knowledge_transfer": len(restored) > 0,
            "lesson_replay": len(lessons),
            "skill_ranking": "atsm_priors.json" in restored,
            "outcome_logging": "atsm_db.jsonl" in restored,
            "memory_restore": "agent_memory.json" in restored,
            "graph_reasoning": "knowledge_graph.json" in restored,
        },
        "agent_id": _manifest_hash({"bootstrapped_at": now, "bundle": str(bundle_dir)}),
    }

    # Write bootstrap report
    report_path = bundle_dir / "bootstrap_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(bootstrap_report, f, indent=2, default=str)

    print(f"[ATSM Replicate] Bootstrap complete")
    print(f"  Restored:     {restored}")
    print(f"  Failed:       {failed}")
    print(f"  Mismatches:   {len(hash_mismatches)}")
    print(f"  Lessons:      {len(lessons)}")
    print(f"  Agent ID:     {bootstrap_report['agent_id']}")
    print(f"  Capabilities:")
    for cap, val in bootstrap_report["capabilities"].items():
        print(f"    {cap}: {val}")

    return bootstrap_report


# ---------------------------------------------------------------------------
# VERIFY
# ---------------------------------------------------------------------------

def cmd_verify(bundle_dir: Path | None = None) -> dict:
    """Verify clone: check integrity, report capabilities, confirm readiness."""
    if bundle_dir is None:
        bundle_dir = CLONE_BUNDLE_DIR

    if not bundle_dir.exists():
        return {"status": "error", "reason": "bundle_not_found"}

    manifest_path = bundle_dir / "replication_manifest.json"
    manifest = _read_json(manifest_path)
    if manifest is None:
        return {"status": "error", "reason": "manifest_corrupt"}

    data_dir = bundle_dir / "data"
    checks = {}
    all_passed = True

    # 1. File presence
    for fname in manifest.get("files_copied", []):
        present = (data_dir / fname).exists()
        checks[f"file_present:{fname}"] = present
        if not present:
            all_passed = False

    # 2. Hash integrity
    for fname, expected_hash in manifest.get("hashes", {}).items():
        fp = data_dir / fname
        if not fp.exists():
            checks[f"hash:{fname}"] = False
            all_passed = False
            continue
        data = _read_json(fp) if fname.endswith(".json") else _read_jsonl(fp)
        actual_hash = _manifest_hash(data) if isinstance(data, (dict, list)) else "jsonl"
        passed = expected_hash == "jsonl" or actual_hash == expected_hash
        checks[f"hash:{fname}"] = passed
        if not passed:
            all_passed = False

    # 3. Lessons integrity
    lessons = manifest.get("lessons", [])
    checks["lessons_present"] = len(lessons) > 0
    checks["lesson_count"] = len(lessons)

    # 4. Capability check
    agent_memory = _read_json(data_dir / "agent_memory.json") or {}
    atsm_priors = _read_json(data_dir / "atsm_priors.json") or {}
    kg = _read_json(data_dir / "knowledge_graph.json") or {}

    capabilities = {
        "memory_cycles": agent_memory.get("cycle_count", 0),
        "memory_issues": len(agent_memory.get("issue_history", [])),
        "memory_lessons": len(agent_memory.get("learned_lessons", [])),
        "skills_tracked": atsm_priors.get("_meta", {}).get("skills_tracked", 0),
        "total_records": atsm_priors.get("_meta", {}).get("total_records", 0),
        "graph_nodes": len(kg.get("nodes", {})),
        "graph_edges": sum(
            len(neighbors) for node in kg.get("nodes", {}).values()
            for neighbors in [node.get("neighbors", {})]
        ),
    }

    now = datetime.datetime.now(datetime.timezone.utc).isoformat()
    verify_report = {
        "status": "ok" if all_passed else "degraded",
        "verified_at": now,
        "bundle_dir": str(bundle_dir),
        "all_checks_passed": all_passed,
        "checks": checks,
        "capabilities": capabilities,
        "lessons_transferred": len(lessons),
    }

    # Write verify report
    report_path = bundle_dir / "verify_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(verify_report, f, indent=2, default=str)

    print(f"[ATSM Replicate] Verify complete")
    print(f"  Status:       {'ALL PASSED' if all_passed else 'DEGRADED'}")
    print(f"  Checks:       {sum(1 for v in checks.values() if v is True)}/{len(checks)} passed")
    print(f"  Capabilities:")
    for k, v in capabilities.items():
        print(f"    {k}: {v}")
    print(f"  Lessons:      {len(lessons)} transferred")

    return verify_report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ATSM Self-Replication Engine — clone, transfer, bootstrap, verify"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # clone
    p_clone = subparsers.add_parser("clone", help="Create a full clone bundle")
    p_clone.add_argument("--dir", type=Path, default=None, help="Bundle output directory")

    # transfer
    p_transfer = subparsers.add_parser("transfer", help="Transfer clone to remote machine")
    p_transfer.add_argument("target", help="Remote target (user@host)")
    p_transfer.add_argument("--dir", type=Path, default=None, help="Bundle directory")
    p_transfer.add_argument("--remote-dir", default=None, help="Remote destination directory")

    # bootstrap
    p_bootstrap = subparsers.add_parser("bootstrap", help="Bootstrap clone with full context")
    p_bootstrap.add_argument("--dir", type=Path, default=None, help="Bundle directory")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify clone integrity and capabilities")
    p_verify.add_argument("--dir", type=Path, default=None, help="Bundle directory")

    args = parser.parse_args()

    if args.command == "clone":
        result = cmd_clone(args.dir)
    elif args.command == "transfer":
        result = cmd_transfer(args.target, args.dir, args.remote_dir)
    elif args.command == "bootstrap":
        result = cmd_bootstrap(args.dir)
    elif args.command == "verify":
        result = cmd_verify(args.dir)
    else:
        parser.print_help()
        sys.exit(1)

    if result.get("status") == "error":
        sys.exit(1)


if __name__ == "__main__":
    main()
