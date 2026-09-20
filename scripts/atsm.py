#!/usr/bin/env python3
"""ATSM: Adaptive Task-Skill Matching engine.

Ranks skills by relevance to a task and historical success rate.
Uses Bayesian Beta-Binomial update with exponential recency decay.

Usage:
    python3 atsm.py rank "task description"
    python3 atsm.py record <skill_name> <0|1>
    python3 atsm.py stats
"""

import json
import math
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = Path(__file__).parent.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"
PRIORS_FILE = DATA_DIR / "atsm_priors.json"

# Algorithm weights
ALPHA = 0.6          # weight for cosine similarity
BETA = 0.4           # weight for keyword overlap
PRIOR_SUCCESS = 1.0  # Beta prior alpha (pseudo-counts)
PRIOR_FAILURE = 1.0  # Beta prior beta (pseudo-counts)
DECAY_LAMBDA = 0.01  # recency decay per day
MIN_OBSERVATIONS = 3 # min observations before trusting success rate


# --- Text processing ---
TOKEN_RE = re.compile(r"[a-zA-Z0-9_\u0E00-\u0E7F]+", re.UNICODE)

def tokenize(text: str) -> list[str]:
    """Lowercase tokenization supporting Thai and English."""
    return [t.lower() for t in TOKEN_RE.findall(text)]

def tf_idf_vectors(docs: list[list[str]]) -> list[dict[str, float]]:
    """Compute TF-IDF vectors for a list of tokenized documents."""
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
    """Cosine similarity between two sparse vectors."""
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
    """Jaccard-like overlap between two token lists."""
    sa, sb = set(a), set(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


# --- Skill loading ---
def load_skills() -> list[dict]:
    """Load all skills with their descriptions and categories."""
    skills = []
    if not SKILLS_ROOT.exists():
        return skills
    for skill_dir in SKILLS_ROOT.rglob("SKILL.md"):
        try:
            content = skill_dir.read_text(encoding="utf-8", errors="replace")
            # Parse frontmatter
            name = skill_dir.parent.name
            description = ""
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    fm = content[3:end]
                    for line in fm.splitlines():
                        if line.strip().startswith("name:"):
                            name = line.split(":", 1)[1].strip().strip('"').strip("'")
                        elif line.strip().startswith("description:"):
                            description = line.split(":", 1)[1].strip().strip('"').strip("'")
            skills.append({
                "name": name,
                "description": description,
                "path": str(skill_dir.parent),
                "tokens": tokenize(name + " " + description),
            })
        except Exception:
            continue
    return skills


# --- Database ---
def load_db() -> list[dict]:
    """Load outcome records."""
    if not DB_FILE.exists():
        return []
    records = []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records

def append_record(skill_name: str, success: int):
    """Append an outcome record."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "skill": skill_name,
        "success": int(success),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


# --- Statistics ---
def compute_stats(records: list[dict]) -> dict[str, dict]:
    """Compute per-skill success statistics with recency decay."""
    now = datetime.now(timezone.utc)
    stats = defaultdict(lambda: {"successes": 0.0, "failures": 0.0, "total": 0, "raw_successes": 0, "raw_failures": 0})
    
    for rec in records:
        skill = rec["skill"]
        success = rec["success"]
        ts = datetime.fromisoformat(rec["timestamp"])
        age_days = max((now - ts).total_seconds() / 86400, 0)
        weight = math.exp(-DECAY_LAMBDA * age_days)
        
        s = stats[skill]
        s["total"] += 1
        if success:
            s["successes"] += weight
            s["raw_successes"] += 1
        else:
            s["failures"] += weight
            s["raw_failures"] += 1
    
    # Compute Beta posterior mean
    for skill, s in stats.items():
        alpha_post = PRIOR_SUCCESS + s["successes"]
        beta_post = PRIOR_FAILURE + s["failures"]
        s["expected_success"] = alpha_post / (alpha_post + beta_post)
        s["confidence"] = min(s["total"] / MIN_OBSERVATIONS, 1.0)
    
    return dict(stats)


# --- Ranking ---
def rank_skills(task: str, top_k: int = 10) -> list[dict]:
    """Rank skills by relevance and expected success for a task."""
    skills = load_skills()
    if not skills:
        return []
    
    records = load_db()
    stats = compute_stats(records)
    
    task_tokens = tokenize(task)
    
    # Compute TF-IDF for all skill descriptions + task
    all_tokens = [s["tokens"] for s in skills] + [task_tokens]
    vectors = tf_idf_vectors(all_tokens)
    task_vec = vectors[-1]
    skill_vectors = vectors[:-1]
    
    results = []
    for i, skill in enumerate(skills):
        relevance = ALPHA * cosine_sim(skill_vectors[i], task_vec) + BETA * keyword_overlap(skill["tokens"], task_tokens)
        
        s = stats.get(skill["name"])
        if s and s["total"] > 0:
            # Blend prior with observed success based on confidence
            prior_mean = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
            success_prob = s["confidence"] * s["expected_success"] + (1 - s["confidence"]) * prior_mean
        else:
            success_prob = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
        
        final_score = relevance * success_prob
        
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


# --- CLI ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "rank":
        if len(sys.argv) < 3:
            print("Usage: atsm.py rank <task description>")
            sys.exit(1)
        task = " ".join(sys.argv[2:])
        results = rank_skills(task)
        if not results:
            print("No skills found.")
            return
        print(f"\n{'Rank':<5} {'Score':<8} {'Rel':<6} {'P(success)':<11} {'N':<4} {'Skill'}")
        print("-" * 70)
        for i, r in enumerate(results, 1):
            print(f"{i:<5} {r['final_score']:<8} {r['relevance']:<6} {r['success_prob']:<11} {r['observations']:<4} {r['name']}")
            if r["description"]:
                print(f"      └─ {r['description'][:60]}")
    
    elif command == "record":
        if len(sys.argv) < 4:
            print("Usage: atsm.py record <skill_name> <0|1>")
            sys.exit(1)
        skill_name = sys.argv[2]
        success = int(sys.argv[3])
        if success not in (0, 1):
            print("Success must be 0 or 1")
            sys.exit(1)
        append_record(skill_name, success)
        print(f"Recorded: {skill_name} → {'success' if success else 'failure'}")
    
    elif command == "stats":
        records = load_db()
        stats = compute_stats(records)
        if not stats:
            print("No records yet. Use 'record' to log outcomes.")
            return
        print(f"\n{'Skill':<35} {'Successes':<10} {'Failures':<10} {'E[success]':<12} {'N':<5} {'Conf':<6}")
        print("-" * 85)
        for name, s in sorted(stats.items(), key=lambda x: x[1].get("expected_success", 0), reverse=True):
            print(f"{name:<35} {s['raw_successes']:<10} {s['raw_failures']:<10} "
                  f"{s['expected_success']:<12.3f} {s['total']:<5} {s['confidence']:<6.2f}")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
