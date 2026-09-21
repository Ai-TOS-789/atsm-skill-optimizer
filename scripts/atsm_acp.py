#!/usr/bin/env python3
"""ATSM ACP Adapter: Agent Client Protocol for IDE integration.

Provides a clean, JSON-serializable interface for VS Code / Zed / JetBrains
extensions to interact with the ATSM skill optimizer.

Usage:
    python3 atsm_acp.py list
    python3 atsm_acp.py rank "task description"
    python3 atsm_acp.py record <skill_name> <0|1>
    python3 atsm_acp.py stats

Import:
    from atsm_acp import ATSMACP
    acp = ATSMACP()
    skills = acp.get_skills()
    ranked = acp.rank_skills("search papers")
    acp.record_outcome("arxiv", True)
    stats = acp.get_stats()
"""

import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Ensure we can import the existing atsm.py engine
# ---------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from atsm import (  # noqa: E402
    load_skills,
    load_db,
    compute_stats,
    rank_skills as _rank_skills,
    append_record as _append_record,
)


class ATSMACP:
    """Agent Client Protocol adapter for ATSM.

    All public methods return JSON-serializable dicts / lists so they can be
    consumed directly by IDE extensions via subprocess or embedded Python.
    """

    def __init__(self, dedup: bool = True):
        self.dedup = dedup

    # ------------------------------------------------------------------
    # Core ACP methods
    # ------------------------------------------------------------------

    def get_skills(self) -> list[dict]:
        """Return all skills with metadata.

        Returns:
            List of dicts with keys: name, description, path
        """
        skills = load_skills(dedup=self.dedup)
        return [
            {
                "name": s["name"],
                "description": s["description"],
                "path": s["path"],
            }
            for s in skills
        ]

    def rank_skills(self, task: str, top_k: int = 10) -> list[dict]:
        """Rank skills by relevance and expected success for a task.

        Args:
            task: Task description to rank against.
            top_k: Number of top results to return.

        Returns:
            List of dicts with keys: name, description, relevance,
            success_prob, final_score, observations
        """
        results, elapsed = _rank_skills(task, top_k=top_k, dedup=self.dedup)
        return [
            {
                "name": r["name"],
                "description": r["description"],
                "relevance": r["relevance"],
                "success_prob": r["success_prob"],
                "final_score": r["final_score"],
                "observations": r["observations"],
            }
            for r in results
        ]

    def record_outcome(self, skill: str, success: bool) -> dict:
        """Record a success/failure outcome for a skill.

        Args:
            skill: Skill name.
            success: True for success, False for failure.

        Returns:
            Dict with keys: skill, success, recorded
        """
        _append_record(skill, 1 if success else 0)
        return {
            "skill": skill,
            "success": success,
            "recorded": True,
        }

    def get_stats(self) -> dict[str, dict]:
        """Return per-skill statistics.

        Returns:
            Dict mapping skill name -> stats dict with keys:
            raw_successes, raw_failures, total, expected_success,
            confidence, effective_total
        """
        records = load_db()
        stats = compute_stats(records)
        # Convert to plain dicts (they already are, but ensure JSON-serializable)
        return {
            name: {
                "raw_successes": s["raw_successes"],
                "raw_failures": s["raw_failures"],
                "total": s["total"],
                "expected_success": round(s["expected_success"], 4),
                "confidence": round(s["confidence"], 4),
                "effective_total": round(s["effective_total"], 4),
            }
            for name, s in stats.items()
        }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    acp = ATSMACP()

    if command == "list":
        skills = acp.get_skills()
        print(json.dumps(skills, indent=2, ensure_ascii=False))

    elif command == "rank":
        if len(sys.argv) < 3:
            print("Usage: atsm_acp.py rank <task description>")
            sys.exit(1)
        task = " ".join(sys.argv[2:])
        results = acp.rank_skills(task)
        print(json.dumps(results, indent=2, ensure_ascii=False))

    elif command == "record":
        if len(sys.argv) < 4:
            print("Usage: atsm_acp.py record <skill_name> <0|1>")
            sys.exit(1)
        skill_name = sys.argv[2]
        success = sys.argv[3] in ("1", "true", "True", "yes")
        result = acp.record_outcome(skill_name, success)
        print(json.dumps(result, indent=2))

    elif command == "stats":
        stats = acp.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
