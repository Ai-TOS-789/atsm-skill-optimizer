#!/usr/bin/env python3
"""Hermes Bridge: sync Hermes skills to ATSM tracking.

Bridges the gap between Hermes skills (~/.hermes/skills/) and the ATSM
(Adaptive Task-Skill Matching) engine by:
  - Auto-discovering new skills and adding them with neutral priors
  - Detecting deleted skills and marking them as retired
  - Detecting updated skills (via content hash) and refreshing metadata
  - Validating skill files and ATSM entry consistency
  - Importing external skills into the Hermes ecosystem

Usage:
    python3 hermes_bridge.py sync [--dry-run]
    python3 hermes_bridge.py list [--all|--synced|--retired]
    python3 hermes_bridge.py validate
    python3 hermes_bridge.py import /path/to/skill
"""

import json
import hashlib
import os
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = Path(__file__).parent.parent / "data"
PRIORS_FILE = DATA_DIR / "atsm_priors.json"
BRIDGE_STATE_FILE = DATA_DIR / "hermes_bridge_state.json"

# Neutral prior for new skills (uniform Beta(1,1))
NEUTRAL_PRIOR = {"alpha": 1.0, "beta": 1.0, "expected_success": 0.5, "observations": 0}


# ---------------------------------------------------------------------------
# YAML frontmatter parser (reused from atsm.py logic, standalone)
# ---------------------------------------------------------------------------
def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter from SKILL.md content.

    Handles:
      - key: value
      - key: "quoted value"
      - key: |
        multi-line
        value
      - key: >
        folded value
    """
    if not content.startswith("---"):
        return {}

    end = content.find("---", 3)
    if end == -1:
        return {}

    fm_text = content[3:end]
    result = {}
    current_key = None
    current_value_lines = []
    in_block_scalar = False
    block_indent = 0

    for line in fm_text.splitlines():
        stripped = line.rstrip()

        # Skip empty lines outside block scalars
        if not in_block_scalar and not stripped:
            continue

        is_new_key = False
        if not in_block_scalar:
            match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$', stripped)
            if match and (not line[0].isspace() or line == stripped):
                is_new_key = True

        if is_new_key:
            # Save previous key
            if current_key:
                val = "\n".join(current_value_lines).strip()
                result[current_key] = val

            key = match.group(1)
            rest = match.group(2).strip()
            current_key = key

            if rest in ("|", ">", "|->", ">-"):
                in_block_scalar = True
                block_indent = None
                current_value_lines = []
            elif rest:
                result[key] = rest.strip('"').strip("'")
                current_key = None
                current_value_lines = []
            else:
                in_block_scalar = False
                current_value_lines = []

        elif in_block_scalar:
            if block_indent is None and stripped:
                block_indent = len(line) - len(line.lstrip())

            if block_indent is not None and stripped:
                if len(line) - len(line.lstrip()) >= block_indent:
                    current_value_lines.append(line[block_indent:])
                else:
                    in_block_scalar = False
            elif not stripped:
                current_value_lines.append("")

        else:
            if current_key and line[0].isspace():
                current_value_lines.append(stripped)

    # Save last key
    if current_key:
        val = "\n".join(current_value_lines).strip()
        result[current_key] = val

    return result


# ---------------------------------------------------------------------------
# Hermes Bridge core
# ---------------------------------------------------------------------------
class HermesBridge:
    """Syncs Hermes skills <-> ATSM priors."""

    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.state = self._load_json(
            BRIDGE_STATE_FILE,
            default={"skills": {}, "last_sync": None},
        )
        self.priors = self._load_json(
            PRIORS_FILE,
            default={
                "_meta": {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "decay_lambda": 0.01,
                    "prior_alpha": 1.0,
                    "prior_beta": 1.0,
                    "total_records": 0,
                    "skills_tracked": 0,
                },
                "_global": {"alpha": 1.0, "beta": 1.0},
            },
        )

    # --- I/O helpers ---
    @staticmethod
    def _load_json(path: Path, default: dict = None) -> dict:
        if path.exists():
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
        return default if default is not None else {}

    @staticmethod
    def _save_json(path: Path, data: dict):
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def _compute_hash(path: Path) -> str:
        return hashlib.md5(path.read_bytes()).hexdigest()

    # --- Skill scanning ---
    def scan_skills(self) -> dict:
        """Walk SKILLS_ROOT and return dict of name -> skill info."""
        skills = {}
        for skill_md in SKILLS_ROOT.rglob("SKILL.md"):
            try:
                content = skill_md.read_text(encoding="utf-8", errors="replace")
                name = skill_md.parent.name
                description = ""
                category = (
                    skill_md.parent.parent.name
                    if skill_md.parent.parent != SKILLS_ROOT
                    else ""
                )

                if content.startswith("---"):
                    fm = parse_frontmatter(content)
                    if "name" in fm:
                        name = fm["name"]
                    if "description" in fm:
                        description = fm["description"]

                skills[name] = {
                    "name": name,
                    "description": description,
                    "path": str(skill_md.parent),
                    "category": category,
                    "hash": self._compute_hash(skill_md),
                    "mtime": skill_md.stat().st_mtime,
                }
            except Exception:
                continue
        return skills

    # --- Commands ---
    def sync(self, dry_run: bool = False) -> dict:
        """Sync Hermes skills to ATSM entries.

        Returns dict with keys: added, updated, retired, unchanged.
        """
        current_skills = self.scan_skills()
        results = {"added": [], "updated": [], "retired": [], "unchanged": []}

        current_names = set(current_skills.keys())
        tracked_names = set(self.state.get("skills", {}).keys())

        # 1. New skills (not yet tracked)
        for name, info in sorted(current_skills.items()):
            if name not in tracked_names:
                results["added"].append(name)
                if not dry_run:
                    self.state["skills"][name] = {
                        "hash": info["hash"],
                        "status": "synced",
                        "added_at": datetime.now(timezone.utc).isoformat(),
                    }
                    self.priors[name] = dict(NEUTRAL_PRIOR)
            else:
                tracked = self.state["skills"][name]
                if tracked.get("hash") != info["hash"]:
                    # Content changed since last sync
                    results["updated"].append(name)
                    if not dry_run:
                        tracked["hash"] = info["hash"]
                        tracked["status"] = "synced"
                        tracked["updated_at"] = datetime.now(
                            timezone.utc
                        ).isoformat()
                else:
                    results["unchanged"].append(name)

        # 2. Deleted skills (tracked but no longer on disk)
        for name in sorted(tracked_names):
            if name not in current_names:
                results["retired"].append(name)
                if not dry_run:
                    self.state["skills"][name]["status"] = "retired"
                    self.state["skills"][name]["retired_at"] = datetime.now(
                        timezone.utc
                    ).isoformat()

        if not dry_run:
            self.state["last_sync"] = datetime.now(timezone.utc).isoformat()
            # Update _meta
            self.priors.setdefault("_meta", {})
            self.priors["_meta"]["updated_at"] = datetime.now(
                timezone.utc
            ).isoformat()
            self.priors["_meta"]["skills_tracked"] = len(
                [
                    n
                    for n in current_names
                    if n in self.priors and not n.startswith("_")
                ]
            )

            self._save_json(BRIDGE_STATE_FILE, self.state)
            self._save_json(PRIORS_FILE, self.priors)

        return results

    def list_skills(self, filter_status: str = "all") -> list[dict]:
        """Return list of tracked skills with their sync status."""
        skills = []
        current_skills = self.scan_skills()

        for name, tracked in self.state.get("skills", {}).items():
            status = tracked.get("status", "unknown")
            if filter_status != "all" and status != filter_status:
                continue

            skill_info = current_skills.get(name, {})
            prior = self.priors.get(name, {})

            skills.append(
                {
                    "name": name,
                    "status": status,
                    "description": skill_info.get("description", ""),
                    "category": skill_info.get("category", ""),
                    "path": skill_info.get("path", ""),
                    "expected_success": prior.get("expected_success", 0.5),
                    "observations": prior.get("observations", 0),
                    "last_sync": tracked.get(
                        "updated_at", tracked.get("added_at", "")
                    ),
                }
            )

        return skills

    def validate(self) -> dict:
        """Validate skill files and ATSM entries.

        Returns dict with keys: errors, warnings, valid.
        """
        errors = []
        warnings = []

        current_skills = self.scan_skills()

        for name, info in sorted(current_skills.items()):
            skill_md = Path(info["path"]) / "SKILL.md"

            # Check frontmatter validity
            try:
                content = skill_md.read_text(encoding="utf-8", errors="replace")
                if not content.startswith("---"):
                    errors.append(f"{name}: Missing YAML frontmatter")
                else:
                    fm = parse_frontmatter(content)
                    if "name" not in fm:
                        warnings.append(f"{name}: Missing 'name' in frontmatter")
                    if "description" not in fm:
                        warnings.append(
                            f"{name}: Missing 'description' in frontmatter"
                        )
            except Exception as e:
                errors.append(f"{name}: Failed to parse - {e}")

            # Check ATSM prior exists
            if name not in self.priors:
                warnings.append(f"{name}: No ATSM prior entry")

        # Check for orphaned ATSM entries (in priors but no skill on disk)
        for name in sorted(self.priors):
            if name.startswith("_"):
                continue
            if name not in current_skills:
                warnings.append(f"ATSM entry '{name}' has no corresponding skill")

        # Check for orphaned bridge state entries
        for name in sorted(self.state.get("skills", {})):
            if name not in current_skills and self.state["skills"][name].get(
                "status"
            ) != "retired":
                warnings.append(
                    f"Bridge state '{name}' not retired but skill missing"
                )

        return {
            "errors": errors,
            "warnings": warnings,
            "valid": len(errors) == 0,
        }

    def import_skill(self, source_path: str) -> dict:
        """Import a skill from an external path into Hermes skills.

        Args:
            source_path: Path to a directory containing SKILL.md, or to
                         the SKILL.md file itself.

        Returns dict with keys: success, name, path (or error).
        """
        source = Path(source_path).expanduser().resolve()

        if not source.exists():
            return {"success": False, "error": f"Path does not exist: {source}"}

        if source.is_file() and source.name == "SKILL.md":
            skill_dir = source.parent
        elif source.is_dir():
            skill_dir = source
            if not (skill_dir / "SKILL.md").exists():
                return {
                    "success": False,
                    "error": f"No SKILL.md found in {source}",
                }
        else:
            return {"success": False, "error": "Invalid skill path"}

        # Determine skill name from frontmatter
        content = (skill_dir / "SKILL.md").read_text(
            encoding="utf-8", errors="replace"
        )
        name = skill_dir.name
        if content.startswith("---"):
            fm = parse_frontmatter(content)
            if "name" in fm:
                name = fm["name"]

        dest = SKILLS_ROOT / name

        if dest.exists():
            return {"success": False, "error": f"Skill already exists: {dest}"}

        # Copy skill directory
        shutil.copytree(skill_dir, dest)

        # Register in bridge state and priors
        skill_md = dest / "SKILL.md"
        self.state["skills"][name] = {
            "hash": self._compute_hash(skill_md),
            "status": "synced",
            "added_at": datetime.now(timezone.utc).isoformat(),
            "imported_from": str(source),
        }
        self.priors[name] = dict(NEUTRAL_PRIOR)

        self._save_json(BRIDGE_STATE_FILE, self.state)
        self._save_json(PRIORS_FILE, self.priors)

        return {"success": True, "name": name, "path": str(dest)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    bridge = HermesBridge()

    # ---- sync ----
    if command == "sync":
        dry_run = "--dry-run" in sys.argv
        results = bridge.sync(dry_run=dry_run)

        if dry_run:
            print("=== DRY RUN (no changes written) ===\n")

        print("Sync Results:")
        print(f"  Added:     {len(results['added'])}")
        for s in results["added"]:
            print(f"    + {s}")
        print(f"  Updated:   {len(results['updated'])}")
        for s in results["updated"]:
            print(f"    ~ {s}")
        print(f"  Retired:   {len(results['retired'])}")
        for s in results["retired"]:
            print(f"    - {s}")
        print(f"  Unchanged: {len(results['unchanged'])}")
        total = len(results["added"]) + len(results["updated"]) + len(
            results["unchanged"]
        )
        print(f"\nTotal synced: {total}")

    # ---- list ----
    elif command == "list":
        filter_status = "all"
        if "--synced" in sys.argv:
            filter_status = "synced"
        elif "--retired" in sys.argv:
            filter_status = "retired"

        skills = bridge.list_skills(filter_status)

        print(f"\nHermes Bridge — {len(skills)} skills ({filter_status})")
        header = f"{'Name':<35} {'Status':<10} {'E[success]':<12} {'N':<4} {'Description'}"
        print(header)
        print("-" * len(header) + "-" * 20)
        for s in sorted(skills, key=lambda x: (x["status"], x["name"])):
            desc = (s["description"] or "")[:40]
            print(
                f"{s['name']:<35} {s['status']:<10} "
                f"{s['expected_success']:<12.3f} {s['observations']:<4} {desc}"
            )

    # ---- validate ----
    elif command == "validate":
        result = bridge.validate()

        print("\nValidation Results:")
        print(f"  Valid: {result['valid']}")

        if result["errors"]:
            print(f"\n  Errors ({len(result['errors'])}):")
            for e in result["errors"]:
                print(f"    ✗ {e}")

        if result["warnings"]:
            print(f"\n  Warnings ({len(result['warnings'])}):")
            for w in result["warnings"]:
                print(f"    ⚠ {w}")

        if result["valid"] and not result["warnings"]:
            print("  All skills valid and tracked.")

        sys.exit(0 if result["valid"] else 1)

    # ---- import ----
    elif command == "import":
        if len(sys.argv) < 3:
            print("Usage: hermes_bridge.py import /path/to/skill")
            sys.exit(1)

        source = sys.argv[2]
        result = bridge.import_skill(source)

        if result.get("success"):
            print(f"✓ Imported '{result['name']}' → {result['path']}")
        else:
            print(f"✗ Import failed: {result.get('error', 'Unknown error')}")
            sys.exit(1)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
