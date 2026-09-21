#!/usr/bin/env python3
"""Hierarchical Goal Planner with ATSM skill ranking.

Decomposes high-level goals into subgoals, assigns skills via ATSM ranking,
tracks progress, and learns from decomposition outcomes.

Usage:
    python3 goal_planner.py plan "research GPT papers and create presentation"
    python3 goal_planner.py status
    python3 goal_planner.py complete goal_1
    python3 goal_planner.py history
    python3 goal_planner.py learn goal_1 1  # record decomposition success/failure
"""

import json
import math
import os
import re
import sys
import time
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# --- Paths ---
SKILL_DIR = Path(__file__).parent.parent
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = SKILL_DIR / "data"
GOALS_DB_FILE = DATA_DIR / "goals_db.json"
HISTORY_FILE = DATA_DIR / "goal_history.json"
FEEDBACK_FILE = DATA_DIR / "decomposition_feedback.json"

# Import ATSM engine
ATSM_SCRIPT = SKILL_DIR.parent / "atsm-self-update" / "scripts" / "atsm.py"

# --- Config ---
MAX_SUBGOALS = 6
MIN_SKILLS_PER_SUBGOAL = 1
MAX_SKILLS_PER_SUBGOAL = 3
DECAY_LAMBDA = 0.01


# ============================================================
# Goal Tree Data Structures
# ============================================================

class GoalNode:
    """Represents a node in the goal tree."""

    def __init__(self, description: str, node_id: str = None,
                 status: str = "pending", parent: str = None):
        self.id = node_id or f"node_{uuid.uuid4().hex[:8]}"
        self.description = description
        self.status = status  # pending, in_progress, completed, failed
        self.parent = parent
        self.subgoals: list = []
        self.skills: list = []  # assigned ATSM-ranked skills
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = None
        self.outcome = None  # 1=success, 0=failure

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "status": self.status,
            "parent": self.parent,
            "subgoals": [s.to_dict() if isinstance(s, GoalNode) else s for s in self.subgoals],
            "skills": self.skills,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "outcome": self.outcome,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoalNode":
        node = cls(
            description=d["description"],
            node_id=d["id"],
            status=d.get("status", "pending"),
            parent=d.get("parent"),
        )
        node.subgoals = [cls.from_dict(s) for s in d.get("subgoals", [])]
        node.skills = d.get("skills", [])
        node.created_at = d.get("created_at", datetime.now(timezone.utc).isoformat())
        node.completed_at = d.get("completed_at")
        node.outcome = d.get("outcome")
        return node


class GoalTree:
    """Root goal with full decomposition tree."""

    def __init__(self, description: str, goal_id: str = None):
        self.id = goal_id or f"goal_{uuid.uuid4().hex[:8]}"
        self.description = description
        self.root: Optional[GoalNode] = None
        self.status = "pending"
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.completed_at = None
        self.decomposition_feedback = []  # learned outcomes

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "description": self.description,
            "root": self.root.to_dict() if self.root else None,
            "status": self.status,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "decomposition_feedback": self.decomposition_feedback,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GoalTree":
        tree = cls(description=d["description"], goal_id=d["id"])
        tree.root = GoalNode.from_dict(d["root"]) if d.get("root") else None
        tree.status = d.get("status", "pending")
        tree.created_at = d.get("created_at", datetime.now(timezone.utc).isoformat())
        tree.completed_at = d.get("completed_at")
        tree.decomposition_feedback = d.get("decomposition_feedback", [])
        return tree


# ============================================================
# Decomposition Templates
# ============================================================

# Templates based on common goal patterns. Each template maps
# action verbs to a set of subgoal generators.
DECOMPOSITION_TEMPLATES = {
    "research_and_create": {
        "triggers": ["research", "papers", "analyze", "study", "investigate"],
        "create_triggers": ["create", "presentation", "deck", "slides", "write", "report", "build"],
        "subgoals": [
            "Search and collect relevant materials",
            "Analyze and synthesize findings",
            "Create outline and structure",
            "Draft content",
            "Review and refine",
        ],
    },
    "research": {
        "triggers": ["research", "papers", "analyze", "study", "investigate", "survey"],
        "subgoals": [
            "Define research scope and questions",
            "Collect relevant materials and sources",
            "Analyze and synthesize findings",
            "Summarize key insights",
        ],
    },
    "build": {
        "triggers": ["build", "implement", "develop", "code", "create", "make"],
        "subgoals": [
            "Define requirements and scope",
            "Design architecture/structure",
            "Implement core functionality",
            "Test and validate",
            "Deploy/finalize",
        ],
    },
    "write": {
        "triggers": ["write", "draft", "compose", "document", "author"],
        "subgoals": [
            "Research and gather information",
            "Create outline",
            "Draft content",
            "Review and edit",
            "Finalize",
        ],
    },
    "data_analysis": {
        "triggers": ["data", "analyze", "visualize", "chart", "plot", "statistics"],
        "subgoals": [
            "Collect and clean data",
            "Explore and analyze patterns",
            "Create visualizations",
            "Draw conclusions and report",
        ],
    },
    "presentation": {
        "triggers": ["presentation", "slides", "deck", "present", "pitch"],
        "subgoals": [
            "Define audience and key message",
            "Research supporting content",
            "Create slide structure",
            "Design visuals",
            "Practice and refine",
        ],
    },
    "default": {
        "triggers": [],
        "subgoals": [
            "Understand requirements and scope",
            "Gather necessary resources",
            "Execute main work",
            "Review and validate",
            "Finalize and deliver",
        ],
    },
}


# ============================================================
# Decomposition Engine
# ============================================================

def detect_goal_type(goal_text: str) -> str:
    """Detect the type of goal based on keywords."""
    text_lower = goal_text.lower()
    scores = {}

    for template_name, template in DECOMPOSITION_TEMPLATES.items():
        if template_name == "default":
            continue
        score = 0
        for trigger in template.get("triggers", []):
            if trigger in text_lower:
                score += 1
        if "create_triggers" in template:
            for ct in template["create_triggers"]:
                if ct in text_lower:
                    score += 2  # higher weight for combined patterns
        scores[template_name] = score

    # Return the highest-scoring template
    if scores:
        best = max(scores, key=scores.get)
        if scores[best] > 0:
            return best
    return "default"


def decompose_goal(goal_text: str) -> list[str]:
    """Decompose a high-level goal into subgoals."""
    goal_type = detect_goal_type(goal_text)
    template = DECOMPOSITION_TEMPLATES[goal_type]

    subgoals = template["subgoals"]
    # Truncate to MAX_SUBGOALS
    return subgoals[:MAX_SUBGOALS]


# ============================================================
# ATSM Integration
# ============================================================

def rank_skills_for_subgoal(subgoal: str, top_k: int = MAX_SKILLS_PER_SUBGOAL) -> list[dict]:
    """Use ATSM engine to rank skills for a subgoal."""
    # Try to import ATSM module directly
    try:
        # Add ATSM script directory to path
        atsm_dir = str(ATSM_SCRIPT.parent)
        if atsm_dir not in sys.path:
            sys.path.insert(0, atsm_dir)

        # Import from atsm module location
        import importlib.util
        spec = importlib.util.spec_from_file_location("atsm", str(ATSM_SCRIPT))
        atsm_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(atsm_mod)

        results, elapsed = atsm_mod.rank_skills(subgoal, top_k=top_k)
        return results
    except Exception:
        pass

    # Fallback: call as subprocess
    import subprocess
    try:
        result = subprocess.run(
            [sys.executable, str(ATSM_SCRIPT), "rank", subgoal],
            capture_output=True, text=True, timeout=30,
            cwd=str(ATSM_SCRIPT.parent.parent),
        )
        if result.returncode == 0:
            return parse_atsm_output(result.stdout)
    except Exception:
        pass

    # Ultimate fallback: return empty list
    return []


def parse_atsm_output(output: str) -> list[dict]:
    """Parse ATSM rank output into structured data."""
    results = []
    lines = output.strip().split("\n")
    for line in lines:
        # Match lines like: "1     0.234  0.567  0.412      3    skill-name"
        match = re.match(
            r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(.+)$",
            line
        )
        if match:
            results.append({
                "rank": int(match.group(1)),
                "final_score": float(match.group(2)),
                "relevance": float(match.group(3)),
                "success_prob": float(match.group(4)),
                "observations": int(match.group(5)),
                "name": match.group(6).strip(),
            })
    return results


# ============================================================
# Goal Planning Pipeline
# ============================================================

def create_goal_tree(goal_text: str) -> GoalTree:
    """Create a complete goal tree with ATSM-ranked skills."""
    tree = GoalTree(description=goal_text)

    # Create root node
    root = GoalNode(
        description=goal_text,
        node_id=f"{tree.id}_root",
        status="pending",
    )

    # Decompose into subgoals
    subgoal_texts = decompose_goal(goal_text)

    for i, sg_text in enumerate(subgoal_texts):
        sg_node = GoalNode(
            description=sg_text,
            node_id=f"{tree.id}_sg_{i+1}",
            status="pending",
            parent=root.id,
        )

        # Rank skills for this subgoal
        skills = rank_skills_for_subgoal(sg_text)
        sg_node.skills = skills

        root.subgoals.append(sg_node)

    tree.root = root
    return tree


# ============================================================
# Persistence
# ============================================================

def load_goals_db() -> dict[str, GoalTree]:
    """Load all goals from database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not GOALS_DB_FILE.exists():
        return {}
    try:
        with open(GOALS_DB_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {gid: GoalTree.from_dict(gd) for gid, gd in data.items()}
    except (json.JSONDecodeError, KeyError):
        return {}


def save_goals_db(goals: dict[str, GoalTree]):
    """Save goals database."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GOALS_DB_FILE, "w", encoding="utf-8") as f:
        json.dump({gid: g.to_dict() for gid, g in goals.items()}, f, indent=2)


def load_history() -> list[dict]:
    """Load completed goals history."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_history(history: list[dict]):
    """Save history."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def load_feedback() -> dict:
    """Load decomposition feedback data."""
    if not FEEDBACK_FILE.exists():
        return {"decompositions": defaultdict(lambda: {"success": 0, "total": 0})}
    try:
        with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError:
        return {"decompositions": {}}


def save_feedback(feedback: dict):
    """Save decomposition feedback."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(FEEDBACK_FILE, "w", encoding="utf-8") as f:
        json.dump(feedback, f, indent=2)


# ============================================================
# Tree Display
# ============================================================

def display_tree(tree: GoalTree, verbose: bool = False):
    """Display a goal tree with ASCII art."""
    if not tree.root:
        print("  (empty tree)")
        return

    # Status symbols
    status_icon = {
        "pending": "○",
        "in_progress": "◐",
        "completed": "●",
        "failed": "✗",
    }

    root_icon = status_icon.get(tree.root.status, "?")
    print(f"\n  {root_icon} [GOAL] {tree.root.description}")
    print(f"     ID: {tree.id}")
    print(f"     Status: {tree.status}")
    print(f"     Created: {tree.created_at[:19]}")
    print()

    for i, sg in enumerate(tree.root.subgoals):
        sg_icon = status_icon.get(sg.status, "?")
        is_last = (i == len(tree.root.subgoals) - 1)
        branch = "└──" if is_last else "├──"
        spacer = "    " if is_last else "│   "

        print(f"  {branch} {sg_icon} [{sg.id}] {sg.description}")

        if sg.skills:
            for j, skill in enumerate(sg.skills):
                skill_last = (j == len(sg.skills) - 1)
                skill_branch = "└──" if skill_last else "├──"
                score = skill.get("final_score", 0)
                p_success = skill.get("success_prob", 0)
                print(f"  {spacer}   {skill_branch} {skill['name']:<30} "
                      f"score={score:.3f} P(success)={p_success:.2f}")
        elif verbose:
            print(f"  {spacer}   └── (no skills ranked)")

        if not is_last:
            print(f"  │")

    print()


def display_goal_summary(goal: GoalTree):
    """Display compact goal summary."""
    status_icon = {
        "pending": "○",
        "in_progress": "◐",
        "completed": "●",
        "failed": "✗",
    }
    icon = status_icon.get(goal.status, "?")

    # Count subgoals by status
    sg_counts = Counter()
    if goal.root:
        for sg in goal.root.subgoals:
            sg_counts[sg.status] += 1
        total_sgs = len(goal.root.subgoals)
    else:
        total_sgs = 0

    sg_detail = " ".join(f"{s}:{c}" for s, c in sorted(sg_counts.items()))
    print(f"  {icon} {goal.id} — {goal.description[:50]}{'...' if len(goal.description) > 50 else ''}")
    print(f"     Subgoals: {total_sgs} ({sg_detail})  Status: {goal.status}")


# ============================================================
# Progress Tracking
# ============================================================

def update_status(node: GoalNode, new_status: str):
    """Update node status and propagate to parent."""
    old_status = node.status
    node.status = new_status

    if new_status == "completed":
        node.completed_at = datetime.now(timezone.utc).isoformat()
        node.outcome = 1
    elif new_status == "failed":
        node.completed_at = datetime.now(timezone.utc).isoformat()
        node.outcome = 0

    return old_status != new_status


def check_subtree_complete(node: GoalNode) -> bool:
    """Check if all children of a node are completed."""
    if not node.subgoals:
        return node.status == "completed"
    return all(sg.status == "completed" for sg in node.subgoals)


def propagate_completion(goals: dict[str, GoalTree]):
    """After marking a subgoal, check if parent should be marked complete."""
    for goal in goals.values():
        if goal.root and goal.root.subgoals:
            if check_subtree_complete(goal.root):
                goal.root.status = "completed"
                goal.status = "completed"
                goal.completed_at = datetime.now(timezone.utc).isoformat()


# ============================================================
# Learning System
# ============================================================

def record_decomposition_outcome(goal_id: str, success: int):
    """Record whether a decomposition pattern worked well."""
    goals = load_goals_db()
    if goal_id not in goals:
        print(f"Goal {goal_id} not found.")
        return

    goal = goals[goal_id]
    goal_type = detect_goal_type(goal.description)

    feedback = load_feedback()
    decomps = feedback.get("decompositions", {})

    if goal_type not in decomps:
        decomps[goal_type] = {"success": 0, "total": 0, "history": []}

    decomps[goal_type]["total"] += 1
    decomps[goal_type]["success"] += success
    decomps[goal_type]["history"].append({
        "goal_id": goal_id,
        "outcome": success,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })

    # Keep only last 20 entries
    decomps[goal_type]["history"] = decomps[goal_type]["history"][-20:]

    feedback["decompositions"] = decomps
    save_feedback(feedback)

    # Also record individual skill outcomes if goal succeeded
    if success == 1 and goal.root:
        for sg in goal.root.subgoals:
            if sg.skills and sg.status == "completed":
                # Record the top skill as successful
                top_skill = sg.skills[0]
                try:
                    atsm_dir = str(ATSM_SCRIPT.parent)
                    if atsm_dir not in sys.path:
                        sys.path.insert(0, atsm_dir)
                    import importlib.util
                    spec = importlib.util.spec_from_file_location("atsm", str(ATSM_SCRIPT))
                    atsm_mod = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(atsm_mod)
                    atsm_mod.append_record(top_skill["name"], 1)
                except Exception:
                    pass  # Silently skip ATSM recording

    print(f"  ✓ Recorded decomposition outcome for '{goal_type}': {'success' if success else 'failure'}")


def get_decomposition_stats() -> dict:
    """Get learning statistics about decomposition patterns."""
    feedback = load_feedback()
    decomps = feedback.get("decompositions", {})

    stats = {}
    for dtype, data in decomps.items():
        total = data.get("total", 0)
        success = data.get("success", 0)
        rate = success / total if total > 0 else 0.0
        stats[dtype] = {
            "success_rate": round(rate, 3),
            "total_uses": total,
            "successes": success,
        }
    return stats


# ============================================================
# CLI Commands
# ============================================================

def cmd_plan(args):
    """Plan a new goal: decompose, rank skills, show tree."""
    if not args:
        print("Usage: goal_planner.py plan <goal description>")
        print("Example: goal_planner.py plan \"research GPT papers and create presentation\"")
        sys.exit(1)

    goal_text = " ".join(args)

    print(f"\n🔍 Analyzing goal: '{goal_text}'")

    # Detect goal type
    goal_type = detect_goal_type(goal_text)
    print(f"   Detected type: {goal_type}")

    # Create tree
    print("   Decomposing into subgoals...")
    tree = create_goal_tree(goal_text)

    # Show skills ranking summary
    print("   Ranking skills with ATSM...")
    total_skills = 0
    for sg in tree.root.subgoals:
        total_skills += len(sg.skills)
    print(f"   Assigned {total_skills} skills across {len(tree.root.subgoals)} subgoals")

    # Display the tree
    print("\n📋 Goal Tree:")
    display_tree(tree, verbose=True)

    # Save
    goals = load_goals_db()
    goals[tree.id] = tree
    save_goals_db(goals)

    # Show decomposition learning stats
    stats = get_decomposition_stats()
    if goal_type in stats:
        s = stats[goal_type]
        print(f"   ℹ Decomposition '{goal_type}' has {s['success_rate']:.0%} success rate "
              f"({s['successes']}/{s['total_uses']} uses)")

    print(f"✅ Goal saved as: {tree.id}")
    print(f"   Next: python3 goal_planner.py status")
    print(f"        python3 goal_planner.py complete {tree.id}_sg_1")


def cmd_status(args):
    """Show status of all goals."""
    goals = load_goals_db()

    if not goals:
        print("\n📋 No goals yet. Use 'plan' to create one.")
        print("   python3 goal_planner.py plan \"your goal here\"")
        return

    # Filter active goals
    active = {gid: g for gid, g in goals.items() if g.status != "completed"}
    completed = {gid: g for gid, g in goals.items() if g.status == "completed"}

    if active:
        print(f"\n📋 Active Goals ({len(active)}):")
        print("─" * 70)
        for gid, goal in sorted(active.items(), key=lambda x: x[1].created_at, reverse=True):
            display_goal_summary(goal)

    if completed:
        print(f"\n✅ Completed Goals ({len(completed)}):")
        print("─" * 70)
        for gid, goal in sorted(completed.items(), key=lambda x: x[1].completed_at or "", reverse=True):
            display_goal_summary(goal)

    if not active and not completed:
        print("\n📋 No goals to show.")


def cmd_complete(args):
    """Mark a goal or subgoal as complete."""
    if not args:
        print("Usage: goal_planner.py complete <node_id>")
        print("Example: python3 goal_planner.py complete goal_abc12345_sg_1")
        sys.exit(1)

    node_id = args[0]
    goals = load_goals_db()

    # Find the node
    found = False
    for goal in goals.values():
        if not goal.root:
            continue

        # Check root
        if goal.root.id == node_id:
            update_status(goal.root, "completed")
            goal.status = "completed"
            goal.completed_at = datetime.now(timezone.utc).isoformat()
            found = True
            print(f"✅ Goal '{goal.id}' marked as completed!")
            break

        # Check subgoals
        for sg in goal.root.subgoals:
            if sg.id == node_id:
                update_status(sg, "completed")
                found = True
                print(f"✅ Subgoal '{sg.description}' marked as completed!")

                # Check if all subgoals done
                if all(s.status == "completed" for s in goal.root.subgoals):
                    goal.root.status = "completed"
                    goal.status = "completed"
                    goal.completed_at = datetime.now(timezone.utc).isoformat()
                    print(f"🎉 Entire goal '{goal.id}' is now complete!")
                break

        if found:
            save_goals_db(goals)
            break

    if not found:
        print(f"Node '{node_id}' not found.")
        print("Available nodes:")
        for goal in goals.values():
            if goal.root:
                print(f"  {goal.root.id} (root: {goal.description[:40]})")
                for sg in goal.root.subgoals:
                    print(f"  {sg.id} ({sg.description[:40]})")
        sys.exit(1)


def cmd_fail(args):
    """Mark a goal or subgoal as failed."""
    if not args:
        print("Usage: goal_planner.py fail <node_id>")
        sys.exit(1)

    node_id = args[0]
    goals = load_goals_db()

    found = False
    for goal in goals.values():
        if not goal.root:
            continue

        if goal.root.id == node_id:
            update_status(goal.root, "failed")
            goal.status = "failed"
            found = True
            print(f"❌ Goal '{goal.id}' marked as failed.")
            break

        for sg in goal.root.subgoals:
            if sg.id == node_id:
                update_status(sg, "failed")
                found = True
                print(f"❌ Subgoal '{sg.description}' marked as failed.")
                break

        if found:
            save_goals_db(goals)
            break

    if not found:
        print(f"Node '{node_id}' not found.")
        sys.exit(1)


def cmd_tree(args):
    """Show full tree for a specific goal."""
    goals = load_goals_db()

    if not goals:
        print("No goals yet.")
        return

    # If goal_id provided, show that one; show most recent
    if args:
        goal_id = args[0]
        if goal_id in goals:
            display_tree(goals[goal_id], verbose=True)
        else:
            print(f"Goal '{goal_id}' not found.")
    else:
        # Show most recent
        latest = max(goals.values(), key=lambda g: g.created_at)
        display_tree(latest, verbose=True)


def cmd_history(args):
    """Show completed goals history."""
    history = load_history()
    goals = load_goals_db()

    # Add completed goals to history display
    completed = [g for g in goals.values() if g.status == "completed"]

    if not completed and not history:
        print("\n📜 No completed goals yet.")
        return

    print(f"\n📜 Goal History:")
    print("─" * 70)

    # Show from goals DB (current session)
    for goal in sorted(completed, key=lambda g: g.completed_at or "", reverse=True):
        end_time = goal.completed_at[:19] if goal.completed_at else "?"
        start_time = goal.created_at[:19]
        duration = "?"
        if goal.completed_at:
            try:
                start = datetime.fromisoformat(goal.created_at)
                end = datetime.fromisoformat(goal.completed_at)
                delta = (end - start).total_seconds()
                if delta < 60:
                    duration = f"{delta:.0f}s"
                elif delta < 3600:
                    duration = f"{delta/60:.1f}m"
                else:
                    duration = f"{delta/3600:.1f}h"
            except Exception:
                duration = "?"

        print(f"  ● {goal.description[:50]}")
        print(f"    ID: {goal.id}  Completed: {end_time}  Duration: {duration}")

        # Show skills used
        if goal.root:
            skills_used = set()
            for sg in goal.root.subgoals:
                for skill in sg.skills[:2]:
                    skills_used.add(skill["name"])
            if skills_used:
                print(f"    Skills: {', '.join(sorted(skills_used)[:5])}")
        print()

    # Show decomposition learning stats
    stats = get_decomposition_stats()
    if stats:
        print("📊 Decomposition Learning:")
        print("─" * 50)
        for dtype, s in sorted(stats.items(), key=lambda x: x[1]["success_rate"], reverse=True):
            bar_len = int(s["success_rate"] * 20)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            print(f"  {dtype:<25} {bar} {s['success_rate']:.0%} ({s['successes']}/{s['total_uses']})")


def cmd_learn(args):
    """Record whether a decomposition pattern was successful."""
    if len(args) < 2:
        print("Usage: goal_planner.py learn <goal_id> <0|1>")
        print("  goal_id: the goal to record feedback for")
        print("  0 or 1: 0=decomposition failed, 1=decomposition succeeded")
        sys.exit(1)

    goal_id = args[0]
    try:
        success = int(args[1])
        if success not in (0, 1):
            raise ValueError
    except ValueError:
        print("Outcome must be 0 (failure) or 1 (success)")
        sys.exit(1)

    record_decomposition_outcome(goal_id, success)


def cmd_delete(args):
    """Delete a goal."""
    if not args:
        print("Usage: goal_planner.py delete <goal_id>")
        sys.exit(1)

    goal_id = args[0]
    goals = load_goals_db()

    if goal_id in goals:
        del goals[goal_id]
        save_goals_db(goals)
        print(f"🗑 Goal '{goal_id}' deleted.")
    else:
        print(f"Goal '{goal_id}' not found.")


def cmd_help(args):
    """Show help."""
    print("""
📋 Hierarchical Goal Planner — ATSM-powered goal decomposition

Commands:
  plan <description>      Decompose a goal into subgoals with ranked skills
  status                  Show all goals and their progress
  tree [goal_id]          Display full goal tree (most recent if no ID)
  complete <node_id>      Mark a goal/subgoal as completed
  fail <node_id>          Mark a goal/subgoal as failed
  learn <goal_id> <0|1>   Record decomposition success/failure for learning
  history                 Show completed goals and learning stats
  delete <goal_id>        Remove a goal
  help                    Show this help

Examples:
  python3 goal_planner.py plan "research GPT papers and create presentation"
  python3 goal_planner.py status
  python3 goal_planner.py complete goal_abc12345_sg_1
  python3 goal_planner.py learn goal_abc12345 1
  python3 goal_planner.py history
""")


# ============================================================
# Main
# ============================================================

def main():
    if len(sys.argv) < 2:
        cmd_help([])
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    commands = {
        "plan": cmd_plan,
        "status": cmd_status,
        "tree": cmd_tree,
        "complete": cmd_complete,
        "fail": cmd_fail,
        "learn": cmd_learn,
        "history": cmd_history,
        "delete": cmd_delete,
        "help": cmd_help,
    }

    handler = commands.get(command)
    if handler:
        handler(args)
    else:
        print(f"Unknown command: {command}")
        cmd_help([])
        sys.exit(1)


if __name__ == "__main__":
    main()
