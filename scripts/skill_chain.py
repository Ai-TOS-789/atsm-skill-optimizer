#!/usr/bin/env python3
"""Skill Chain Planner — parses compound commands into ordered skill sequences.

Parses commands like:
  "วิจัย arXiv paper เรื่อง GPT → สรุป → สร้างสไลด์"
  "read email → extract action items → create tasks"

Stages are separated by arrows (→, ->). For each stage, ATSM ranks skills by
relevance + success history. The planner outputs an ordered chain, executes it
(step-by-step), records outcomes per step, and learns from failures.

Usage:
    python3 skill_chain.py plan  "task description"
    python3 skill_chain.py run   "task description"
    python3 skill_chain.py history
    python3 skill_chain.py record <step_id> <0|1>
"""

import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = Path(__file__).parent.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"
PRIORS_FILE = DATA_DIR / "atsm_priors.json"
CHAINS_FILE = DATA_DIR / "chains_db.jsonl"

# Algorithm weights (mirrors atsm.py)
ALPHA = 0.6
BETA = 0.4
PRIOR_SUCCESS = 1.0
PRIOR_FAILURE = 1.0
DECAY_LAMBDA = 0.01
MIN_OBSERVATIONS = 3

# Chain learning
AVOID_PENALTY = 0.3  # multiplier for skill pairs that historically co-failed
SESSION_TIMEOUT = 300  # seconds between chains to count as new session


# --- Text processing ---
import re as _re
TOKEN_RE = _re.compile(r"[a-zA-Z0-9_\u0E00-\u0E7F]+", _re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization supporting Thai and English."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def tf_idf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
    """Compute TF-IDF vectors."""
    n = len(docs)
    df = Counter()
    for doc in docs:
        df.update(set(doc))
    idf = {term: math.log((n + 1) / (1 + freq)) + 1 for term, freq in df.items()}
    vectors = []
    for doc in docs:
        tf = Counter(doc)
        total = len(doc) or 1
        vec = {term: (count / total) * idf.get(term, 0) for term, count in tf.items()}
        vectors.append(vec)
    return vectors


def cosine_sim(a: dict[str, float], b: dict[str, float]) -> float:
    """Cosine similarity."""
    if not a or not b:
        return 0.0
    common = set(a) & set(b)
    if not common:
        return 0.0
    dot = sum(a[t] * b[t] for t in common)
    norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
    norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def keyword_overlap(a: list[str], b: list[str]) -> float:
    """Jaccard-like overlap."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --- YAML frontmatter parser ---
def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter robustly."""
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
        if not in_block_scalar and not stripped:
            continue
        is_new_key = False
        if not in_block_scalar:
            match = _re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$', stripped)
            if match and (not line[0].isspace() or line == stripped):
                is_new_key = True
        if is_new_key:
            if current_key:
                val = "\n".join(current_value_lines).strip()
                result[current_key] = val
            key = match.group(1)
            rest = match.group(2).strip()
            current_key = key
            if rest == "|" or rest == ">" or rest == "|-" or rest == ">-":
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
    if current_key:
        val = "\n".join(current_value_lines).strip()
        result[current_key] = val
    return result


# --- Skill loading ---
def load_skills() -> list[dict]:
    """Load all skills with descriptions."""
    skills = []
    if not SKILLS_ROOT.exists():
        return skills
    for skill_dir in SKILLS_ROOT.rglob("SKILL.md"):
        try:
            content = skill_dir.read_text(encoding="utf-8", errors="replace")
            name = skill_dir.parent.name
            description = ""
            if content.startswith("---"):
                fm = parse_frontmatter(content)
                if "name" in fm:
                    name = fm["name"]
                if "description" in fm:
                    description = fm["description"]
            skills.append({
                "name": name,
                "description": description,
                "path": str(skill_dir.parent),
                "tokens": tokenize(name + " " + description),
            })
        except Exception:
            continue
    return skills


# --- Database (mirrors atsm.py) ---
def load_db() -> list[dict]:
    """Load outcome records."""
    records = []
    if not DB_FILE.exists():
        return records
    with open(DB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def compute_stats(records: list[dict]) -> dict[str, dict]:
    """Compute per-skill success statistics with recency decay."""
    now = datetime.now(timezone.utc)
    stats = defaultdict(lambda: {
        "successes": 0.0, "failures": 0.0, "total": 0,
        "raw_successes": 0, "raw_failures": 0, "effective_total": 0.0
    })
    for rec in records:
        skill = rec["skill"]
        success = rec["success"]
        ts = datetime.fromisoformat(rec["timestamp"])
        age_days = max((now - ts).total_seconds() / 86400, 0)
        weight = math.exp(-DECAY_LAMBDA * age_days)
        s = stats[skill]
        s["total"] += 1
        s["effective_total"] += weight
        if success:
            s["successes"] += weight
            s["raw_successes"] += 1
        else:
            s["failures"] += weight
            s["raw_failures"] += 1
    for skill, s in stats.items():
        alpha_post = PRIOR_SUCCESS + s["successes"]
        beta_post = PRIOR_FAILURE + s["failures"]
        s["expected_success"] = alpha_post / (alpha_post + beta_post)
        s["confidence"] = min(s["effective_total"] / MIN_OBSERVATIONS, 1.0)
    return dict(stats)


def rank_skills(task: str, top_k: int = 10, avoid_pairs: set = None) -> list[dict]:
    """Rank skills by relevance × expected success for a task.
    
    Args:
        task: stage description
        top_k: max results
        avoid_pairs: set of (prev_skill, skill) tuples to penalize
    """
    skills = load_skills()
    if not skills:
        return []
    records = load_db()
    stats = compute_stats(records)
    task_tokens = tokenize(task)
    all_tokens = [s["tokens"] for s in skills] + [task_tokens]
    vectors = tf_idf_vectors(all_tokens)
    task_vec = vectors[-1]
    skill_vectors = vectors[:-1]
    
    results = []
    for i, skill in enumerate(skills):
        relevance = ALPHA * cosine_sim(skill_vectors[i], task_vec) + BETA * keyword_overlap(skill["tokens"], task_tokens)
        s = stats.get(skill["name"])
        if s and s["total"] > 0:
            prior_mean = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
            success_prob = s["confidence"] * s["expected_success"] + (1 - s["confidence"]) * prior_mean
        else:
            success_prob = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
        
        final_score = relevance * success_prob
        
        # Apply chain-avoidance penalty
        if avoid_pairs:
            for prev_skill, curr_skill in avoid_pairs:
                if curr_skill == skill["name"]:
                    final_score *= AVOID_PENALTY
        
        results.append({
            "name": skill["name"],
            "description": skill["description"],
            "relevance": round(relevance, 4),
            "success_prob": round(success_prob, 4),
            "final_score": round(final_score, 4),
            "observations": s["total"] if s else 0,
        })
    
    results.sort(key=lambda x: x["final_score"], reverse=True)
    return results[:top_k]


# --- Chain DB ---
def load_chains() -> list[dict]:
    """Load chain execution records."""
    chains = []
    if not CHAINS_FILE.exists():
        return chains
    with open(CHAINS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    chains.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return chains


def append_chain(chain: dict):
    """Append a chain record."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(CHAINS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(chain, ensure_ascii=False) + "\n")


def update_chain_step(chain_id: str, step_idx: int, success: bool):
    """Update a specific step's outcome in the chain DB."""
    chains = load_chains()
    for chain in chains:
        if chain.get("chain_id") == chain_id:
            for step in chain["steps"]:
                if step["index"] == step_idx:
                    step["success"] = success
                    step["completed_at"] = datetime.now(timezone.utc).isoformat()
            # Recompute chain success
            chain["overall_success"] = all(
                s.get("success") for s in chain["steps"] if s.get("success") is not None
            )
            break
    # Rewrite
    with open(CHAINS_FILE, "w", encoding="utf-8") as f:
        for chain in chains:
            f.write(json.dumps(chain, ensure_ascii=False) + "\n")


# --- Chain learning: extract failure pairs ---
def get_avoid_pairs() -> set:
    """Extract skill pairs to avoid based on chain failure history.
    
    If chain X→Y→Z failed at Y, we mark (X, Y) as a pair to avoid.
    Returns set of (prev_skill, failed_skill) tuples.
    """
    chains = load_chains()
    fail_pairs = Counter()
    
    for chain in chains:
        steps = chain.get("steps", [])
        for i, step in enumerate(steps):
            if step.get("success") is False and i > 0:
                prev_skill = steps[i - 1]["skill"]
                curr_skill = step["skill"]
                fail_pairs[(prev_skill, curr_skill)] += 1
    
    # Only avoid pairs that failed 2+ times
    return {pair for pair, count in fail_pairs.items() if count >= 2}


# --- Command parsing ---
def parse_stages(command: str) -> list[str]:
    """Parse compound command into stages by arrow separators."""
    # Split on arrow patterns
    parts = _re.split(r'\s*(?:→|->|=>|⟶|➜|»|>>)\s*', command)
    # Clean up
    stages = [p.strip() for p in parts if p.strip()]
    return stages


# --- Main: plan ---
def cmd_plan(command: str):
    """Plan a skill chain from a compound command."""
    stages = parse_stages(command)
    
    if len(stages) <= 1:
        # No arrows — treat entire command as single task
        stages = [command]
    
    avoid_pairs = get_avoid_pairs()
    chain = []
    
    print(f"\n{'═' * 60}")
    print(f"  ⛓  SKILL CHAIN PLANNER")
    print(f"{'═' * 60}")
    print(f"\n  Command: {command}")
    print(f"  Stages:  {len(stages)}")
    print(f"  Avoid pairs: {len(avoid_pairs)}")
    print(f"{'─' * 60}")
    
    for i, stage in enumerate(stages):
        results = rank_skills(stage, top_k=5, avoid_pairs=avoid_pairs if i > 0 else None)
        if results:
            best = results[0]
            chain.append(best)
            print(f"\n  Step {i + 1}: {best['name']} (P={best['final_score']:.2f}) - {stage}")
            print(f"           relevance={best['relevance']:.2f} | success_prob={best['success_prob']:.2f} | N={best['observations']}")
            if best["description"]:
                print(f"           └─ {best['description'][:60]}")
            
            # Show alternatives
            if len(results) > 1:
                alts = ", ".join(f"{r['name']} ({r['final_score']:.2f})" for r in results[1:3])
                print(f"           alternatives: {alts}")
        else:
            print(f"\n  Step {i + 1}: ??? - {stage} (no matching skill)")
            chain.append({"name": "unknown", "description": "", "final_score": 0, "success_prob": 0, "relevance": 0, "observations": 0})
    
    print(f"\n{'─' * 60}")
    chain_names = " → ".join(c["name"] for c in chain)
    avg_prob = sum(c["final_score"] for c in chain) / len(chain) if chain else 0
    print(f"  Chain: {chain_names}")
    print(f"  Avg P(score): {avg_prob:.2f}")
    print(f"{'═' * 60}")
    
    return chain


# --- Main: run ---
def cmd_run(command: str, simulate: bool = True):
    """Plan and execute a skill chain."""
    stages = parse_stages(command)
    
    if len(stages) <= 1:
        stages = [command]
    
    avoid_pairs = get_avoid_pairs()
    chain_steps = []
    chain_id = f"chain_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.getpid()}_{int(random.random()*10000)}"
    
    print(f"\n{'═' * 60}")
    print(f"  ⛓  SKILL CHAIN EXECUTOR")
    print(f"{'═' * 60}")
    print(f"\n  Chain ID: {chain_id}")
    print(f"  Command:  {command}")
    print(f"{'─' * 60}")
    
    # Plan each step
    for i, stage in enumerate(stages):
        results = rank_skills(stage, top_k=5, avoid_pairs=avoid_pairs if i > 0 else None)
        if results:
            best = results[0]
            chain_steps.append({
                "index": i,
                "skill": best["name"],
                "description": best["description"],
                "stage": stage,
                "score": best["final_score"],
                "success_prob": best["success_prob"],
                "success": None,
                "started_at": None,
                "completed_at": None,
            })
        else:
            chain_steps.append({
                "index": i,
                "skill": "unknown",
                "description": "",
                "stage": stage,
                "score": 0,
                "success_prob": 0,
                "success": None,
                "started_at": None,
                "completed_at": None,
            })
    
    # Execute
    all_success = True
    for i, step in enumerate(chain_steps):
        step["started_at"] = datetime.now(timezone.utc).isoformat()
        
        print(f"\n  ┌─ Step {i + 1}: {step['skill']}")
        print(f"  │  Task: {step['stage']}")
        print(f"  │  P(success): {step['success_prob']:.2f}")
        
        if simulate:
            # Simulate: random outcome based on skill's success_prob
            # Use a deterministic-ish but varied outcome
            roll = random.random()
            success = roll < step["success_prob"]
            print(f"  │  ... executing {'✓' if success else '✗'}")
        else:
            # In non-simulate mode, we'd invoke the skill here
            # For now, mark as pending
            success = None
            print(f"  │  → Execute this skill, then run:")
            print(f"  │    python3 skill_chain.py record {chain_id} {i} <0|1>")
        
        step["success"] = success
        step["completed_at"] = datetime.now(timezone.utc).isoformat()
        
        if success is False:
            all_success = False
            print(f"  └─ ✗ FAILED")
            # Stop chain on failure? Or continue?
            # Let's record but continue to show full picture
        elif success is True:
            print(f"  └─ ✓ OK")
        else:
            print(f"  └─ ⏸ PENDING")
    
    # Record chain
    chain_record = {
        "chain_id": chain_id,
        "command": command,
        "stages": stages,
        "steps": chain_steps,
        "overall_success": all_success,
        "simulated": simulate,
        "executed_at": datetime.now(timezone.utc).isoformat(),
        "avoid_pairs_used": list(avoid_pairs),
    }
    append_chain(chain_record)
    
    print(f"\n{'─' * 60}")
    chain_names = " → ".join(c["skill"] for c in chain_steps)
    status = "✓ SUCCESS" if all_success else "✗ PARTIAL/FAILURE" if any(s["success"] is False for s in chain_steps) else "⏸ PENDING"
    print(f"  Chain: {chain_names}")
    print(f"  Status: {status}")
    print(f"  Recorded: {chain_id}")
    print(f"{'═' * 60}")
    
    return chain_record


# --- Main: history ---
def cmd_history(limit: int = 10):
    """Show chain execution history."""
    chains = load_chains()
    
    if not chains:
        print("\n  No chain history yet.")
        print("  Run: python3 skill_chain.py run \"your task\"")
        return
    
    print(f"\n{'═' * 60}")
    print(f"  ⛓  CHAIN EXECUTION HISTORY")
    print(f"{'═' * 60}")
    print(f"\n  Total chains: {len(chains)}")
    
    # Summary stats
    completed = [c for c in chains if all(s.get("success") is not None for s in c["steps"])]
    successful = [c for c in completed if c["overall_success"]]
    failed = [c for c in completed if not c["overall_success"]]
    print(f"  Completed: {len(completed)} | Success: {len(successful)} | Failed: {len(failed)}")
    
    # Show recent chains
    recent = sorted(chains, key=lambda c: c["executed_at"], reverse=True)[:limit]
    
    print(f"\n  {'─' * 56}")
    for chain in recent:
        chain_id = chain["chain_id"]
        steps = chain["steps"]
        step_str = " → ".join(
            f"{'✓' if s['success'] else '✗' if s['success'] is False else '?'}{s['skill']}"
            for s in steps
        )
        status = "✓" if chain["overall_success"] else "✗" if any(s["success"] is False for s in steps) else "⏸"
        ts = chain["executed_at"][:19].replace("T", " ")
        print(f"  [{status}] {chain_id}")
        print(f"      {step_str}")
        print(f"      {ts}")
        print(f"  {'─' * 56}")
    
    # Show avoid pairs
    avoid = get_avoid_pairs()
    if avoid:
        print(f"\n  ⚠ Avoid pairs (learned from failures):")
        for prev, curr in avoid:
            print(f"    {prev} → {curr}  (penalized)")
    
    print(f"{'═' * 60}")


# --- Main: record ---
def cmd_record(chain_id: str, step_idx: int, success: int):
    """Record outcome for a specific chain step."""
    update_chain_step(chain_id, step_idx, bool(success))
    print(f"  Recorded: chain={chain_id} step={step_idx} → {'success' if success else 'failure'}")


# --- CLI ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "plan":
        if len(sys.argv) < 3:
            print("Usage: skill_chain.py plan <compound command>")
            sys.exit(1)
        cmd_plan(" ".join(sys.argv[2:]))
    
    elif command == "run":
        if len(sys.argv) < 3:
            print("Usage: skill_chain.py run <compound command>")
            sys.exit(1)
        cmd_run(" ".join(sys.argv[2:]), simulate=True)
    
    elif command == "record":
        if len(sys.argv) < 5:
            print("Usage: skill_chain.py record <chain_id> <step_idx> <0|1>")
            sys.exit(1)
        cmd_record(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
    
    elif command == "history":
        limit = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        cmd_history(limit)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
