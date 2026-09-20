#!/usr/bin/env python3
"""ATSM: Adaptive Task-Skill Matching engine v1.2.

Ranks skills by relevance to a task and historical success rate.
Uses Bayesian Beta-Binomial update with exponential recency decay.

Improvements in v1.2 (multi-agent awareness):
  - agent_id parameter on rank_skills() and record()
  - Cross-agent learning: successes from other agents boost rankings
  - --agent CLI flag for rank and record commands

Improvements in v1.1:
  - Caching layer: skills, DB, TF-IDF vectors cached with mtime-based invalidation
  - Confidence uses effective (decayed) sample size, not raw count
  - Robust YAML frontmatter parser for multi-line descriptions

Usage:
    python3 atsm.py rank [--dedup] [--agent NAME] "task description"
    python3 atsm.py record [--agent NAME] <skill_name> <0|1>
    python3 atsm.py stats
    python3 atsm.py benchmark
"""

import json
import math
import os
import re
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
CACHE_FILE = DATA_DIR / "atsm_cache.json"

# Algorithm weights
ALPHA = 0.6          # weight for cosine similarity
BETA = 0.4           # weight for keyword overlap
PRIOR_SUCCESS = 1.0  # Beta prior alpha (pseudo-counts)
PRIOR_FAILURE = 1.0  # Beta prior beta (pseudo-counts)
DECAY_LAMBDA = 0.01  # recency decay per day
MIN_OBSERVATIONS = 3 # min observations before trusting success rate

# Multi-agent awareness
CROSS_AGENT_WEIGHT = 0.3  # how much cross-agent success influences ranking (0-1)
DEFAULT_AGENT = "default" # default agent identifier


# --- Caching ---
class ATSMCache:
    """mtime-based cache for skills, DB records, and TF-IDF vectors."""
    
    def __init__(self):
        self.skills = None
        self.skills_mtime = 0
        self.records = None
        self.records_mtime = 0
        self.vectors = None
        self.vectors_mtime = 0
    
    def _max_skill_mtime(self) -> float:
        """Get the latest mtime across all SKILL.md files."""
        max_mtime = 0.0
        if not SKILLS_ROOT.exists():
            return max_mtime
        for skill_dir in SKILLS_ROOT.rglob("SKILL.md"):
            try:
                m = skill_dir.stat().st_mtime
                if m > max_mtime:
                    max_mtime = m
            except OSError:
                continue
        return max_mtime
    
    def _db_mtime(self) -> float:
        """Get DB file mtime."""
        try:
            return DB_FILE.stat().st_mtime
        except OSError:
            return 0.0
    
    def get_skills(self):
        current_mtime = self._max_skill_mtime()
        if self.skills is None or current_mtime > self.skills_mtime:
            self.skills = None  # force reload
        return self.skills, current_mtime
    
    def set_skills(self, skills, mtime):
        self.skills = skills
        self.skills_mtime = mtime
    
    def get_records(self):
        current_mtime = self._db_mtime()
        if self.records is None or current_mtime > self.records_mtime:
            self.records = None
        return self.records, current_mtime
    
    def set_records(self, records, mtime):
        self.records = records
        self.records_mtime = mtime
    
    def get_vectors(self, skills_mtime):
        if self.vectors is None or skills_mtime != self.vectors_mtime:
            self.vectors = None
        return self.vectors
    
    def set_vectors(self, vectors, mtime):
        self.vectors = vectors
        self.vectors_mtime = mtime


_cache = ATSMCache()


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


# --- YAML frontmatter parser ---
def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter robustly, handling multi-line values.
    
    Supports:
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
        
        # Check if this is a new top-level key
        # Top-level: non-indented word followed by ':'
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
            
            if rest == "|" or rest == ">" or rest == "|-" or rest == ">-":
                # Block scalar
                in_block_scalar = True
                block_indent = None  # determined by first content line
                current_value_lines = []
            elif rest:
                # Single-line value
                result[key] = rest.strip('"').strip("'")
                current_key = None
                current_value_lines = []
            else:
                # Empty value, might be multi-line
                in_block_scalar = False
                current_value_lines = []
        
        elif in_block_scalar:
            # Determine block indent from first content line
            if block_indent is None and stripped:
                block_indent = len(line) - len(line.lstrip())
            
            if block_indent is not None and stripped:
                # Strip the block indent
                if len(line) - len(line.lstrip()) >= block_indent:
                    current_value_lines.append(line[block_indent:])
                else:
                    # Dedented line ends block scalar
                    in_block_scalar = False
            elif not stripped:
                current_value_lines.append("")
        
        else:
            # Continuation line (indented) for regular value
            if current_key and line[0].isspace():
                current_value_lines.append(stripped)
    
    # Save last key
    if current_key:
        val = "\n".join(current_value_lines).strip()
        result[current_key] = val
    
    return result


# --- Skill loading ---
def load_skills(dedup: bool = True) -> list[dict]:
    """Load all skills with their descriptions and categories (cached).

    Args:
        dedup: If True, deduplicate skills by name (keep first encountered).
    """
    cached, current_mtime = _cache.get_skills()
    if cached is not None:
        return cached

    skills = []
    if not SKILLS_ROOT.exists():
        return skills

    seen_names = set()
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

            # Deduplicate by skill name (keep first encountered)
            if dedup:
                if name in seen_names:
                    continue
                seen_names.add(name)

            skills.append({
                "name": name,
                "description": description,
                "path": str(skill_dir.parent),
                "tokens": tokenize(name + " " + description),
            })
        except Exception:
            continue

    _cache.set_skills(skills, current_mtime)
    return skills


# --- Database ---
def load_db() -> list[dict]:
    """Load outcome records (cached)."""
    cached, current_mtime = _cache.get_records()
    if cached is not None:
        return cached
    
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
    
    _cache.set_records(records, current_mtime)
    return records

def append_record(skill_name: str, success: int, agent_id: str = DEFAULT_AGENT):
    """Append an outcome record and invalidate record cache.

    Args:
        skill_name: Name of the skill.
        success: 1 for success, 0 for failure.
        agent_id: Identifier of the agent reporting the outcome.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    record = {
        "skill": skill_name,
        "success": int(success),
        "agent": agent_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    # Invalidate cache
    _cache.records = None


# --- Statistics ---
def compute_stats(records: list[dict], agent_id: str = None) -> dict[str, dict]:
    """Compute per-skill success statistics with recency decay.
    
    FIX v1.1: Uses effective_total (sum of decayed weights) for confidence
    instead of raw record count. This correctly handles stale observations.

    Args:
        records: List of outcome records.
        agent_id: If provided, filter records to only this agent.
    
    Returns:
        Dict mapping skill name -> stats dict.
    """
    now = datetime.now(timezone.utc)
    stats = defaultdict(lambda: {
        "successes": 0.0, "failures": 0.0, "total": 0,
        "raw_successes": 0, "raw_failures": 0, "effective_total": 0.0
    })
    
    for rec in records:
        # Filter by agent if specified
        if agent_id is not None and rec.get("agent", DEFAULT_AGENT) != agent_id:
            continue
        
        skill = rec["skill"]
        success = rec["success"]
        ts = datetime.fromisoformat(rec["timestamp"])
        age_days = max((now - ts).total_seconds() / 86400, 0)
        weight = math.exp(-DECAY_LAMBDA * age_days)
        
        s = stats[skill]
        s["total"] += 1
        s["effective_total"] += weight  # v1.1: decayed effective sample size
        if success:
            s["successes"] += weight
            s["raw_successes"] += 1
        else:
            s["failures"] += weight
            s["raw_failures"] += 1
    
    # Compute Beta posterior mean with confidence based on effective sample size
    for skill, s in stats.items():
        alpha_post = PRIOR_SUCCESS + s["successes"]
        beta_post = PRIOR_FAILURE + s["failures"]
        s["expected_success"] = alpha_post / (alpha_post + beta_post)
        # v1.1: confidence from effective (decayed) sample size, not raw count
        s["confidence"] = min(s["effective_total"] / MIN_OBSERVATIONS, 1.0)
    
    return dict(stats)


# --- Ranking ---
def rank_skills(task: str, top_k: int = 10, dedup: bool = True, agent_id: str = DEFAULT_AGENT) -> tuple[list[dict], float]:
    """Rank skills by relevance and expected success for a task.

    Multi-agent awareness (v1.2):
    - Computes per-agent stats and cross-agent stats separately.
    - Cross-agent successes boost the success probability via weighted blend.
    - If agent A succeeds with skill X, agent B also benefits.

    Args:
        task: Task description to rank against.
        top_k: Number of top results to return.
        dedup: If True, deduplicate skills by name (keep first encountered).
        agent_id: Agent requesting the ranking (for personalized + cross-agent learning).

    Returns (results, elapsed_seconds).
    """
    start = time.time()
    
    skills = load_skills(dedup=dedup)
    if not skills:
        return [], time.time() - start
    
    records = load_db()
    
    # Compute stats for the requesting agent and for all agents (cross-agent)
    agent_stats = compute_stats(records, agent_id=agent_id)
    global_stats = compute_stats(records, agent_id=None)
    
    # Track which agents have used each skill (for display)
    skill_agents = defaultdict(set)
    for rec in records:
        skill_agents[rec["skill"]].add(rec.get("agent", DEFAULT_AGENT))
    
    task_tokens = tokenize(task)
    
    # Compute TF-IDF for all skill descriptions + task
    all_tokens = [s["tokens"] for s in skills] + [task_tokens]
    vectors = tf_idf_vectors(all_tokens)
    task_vec = vectors[-1]
    skill_vectors = vectors[:-1]
    
    results = []
    for i, skill in enumerate(skills):
        relevance = ALPHA * cosine_sim(skill_vectors[i], task_vec) + BETA * keyword_overlap(skill["tokens"], task_tokens)
        
        # Per-agent stats
        s = agent_stats.get(skill["name"])
        if s and s["total"] > 0:
            prior_mean = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
            agent_success_prob = s["confidence"] * s["expected_success"] + (1 - s["confidence"]) * prior_mean
            agent_confidence = s["confidence"]
        else:
            agent_success_prob = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
            agent_confidence = 0.0
        
        # Cross-agent stats (exclude current agent to avoid double-counting)
        gs = global_stats.get(skill["name"])
        other_agents = skill_agents.get(skill["name"], set()) - {agent_id}
        
        if gs and gs["total"] > 0 and other_agents:
            prior_mean = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
            cross_success_prob = gs["confidence"] * gs["expected_success"] + (1 - gs["confidence"]) * prior_mean
        else:
            cross_success_prob = PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE)
        
        # Blend: weighted combination of agent-specific and cross-agent success
        # If agent has high confidence, rely more on own history; otherwise lean on cross-agent
        if agent_confidence > 0:
            w = agent_confidence * (1 - CROSS_AGENT_WEIGHT)
            success_prob = w * agent_success_prob + (1 - w) * cross_success_prob
        else:
            # No agent-specific data: use cross-agent (or prior if no data at all)
            success_prob = CROSS_AGENT_WEIGHT * cross_success_prob + (1 - CROSS_AGENT_WEIGHT) * (PRIOR_SUCCESS / (PRIOR_SUCCESS + PRIOR_FAILURE))
        
        final_score = relevance * success_prob
        
        # Count agents who successfully used this skill recently
        recent_agents = set()
        now = datetime.now(timezone.utc)
        for rec in records:
            if rec["skill"] == skill["name"] and rec["success"] == 1:
                ts = datetime.fromisoformat(rec["timestamp"])
                age_days = (now - ts).total_seconds() / 86400
                if age_days <= 30:  # within 30 days
                    recent_agents.add(rec.get("agent", DEFAULT_AGENT))
        recent_agents -= {agent_id}  # exclude self
        
        results.append({
            "name": skill["name"],
            "description": skill["description"],
            "relevance": round(relevance, 4),
            "success_prob": round(success_prob, 4),
            "agent_success_prob": round(agent_success_prob, 4),
            "cross_success_prob": round(cross_success_prob, 4),
            "final_score": round(final_score, 4),
            "observations": s["total"] if s else 0,
            "agent_observations": s["total"] if s else 0,
            "cross_agents": len(recent_agents),
            "total_agents": len(skill_agents.get(skill["name"], set())),
        })
    
    results.sort(key=lambda x: x["final_score"], reverse=True)
    elapsed = time.time() - start
    return results[:top_k], elapsed


# --- CLI ---
def parse_agent_flag(args: list[str]) -> tuple[list[str], str]:
    """Extract --agent flag from args list, return remaining args and agent_id."""
    agent_id = DEFAULT_AGENT
    if "--agent" in args:
        idx = args.index("--agent")
        if idx + 1 < len(args):
            agent_id = args[idx + 1]
            # Remove --agent and its value from args
            args = args[:idx] + args[idx + 2:]
        else:
            print("Error: --agent requires a value")
            sys.exit(1)
    return args, agent_id


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "rank":
        if len(sys.argv) < 3:
            print("Usage: atsm.py rank [--dedup] [--agent NAME] <task description>")
            sys.exit(1)
        # Parse flags
        task_args = sys.argv[2:]
        # Extract --agent flag
        task_args, agent_id = parse_agent_flag(task_args)
        # Check for --dedup flag
        dedup = True
        if "--dedup" in task_args:
            dedup = True
            task_args.remove("--dedup")
        elif "--no-dedup" in task_args:
            dedup = False
            task_args.remove("--no-dedup")
        task = " ".join(task_args)
        results, elapsed = rank_skills(task, dedup=dedup, agent_id=agent_id)
        if not results:
            print("No skills found.")
            return
        print(f"\n{'Rank':<5} {'Score':<8} {'Rel':<6} {'P(success)':<11} {'N':<4} {'Cross':<6} {'Skill'}")
        print("-" * 80)
        for i, r in enumerate(results, 1):
            cross_info = f"({r['cross_agents']}a)" if r['cross_agents'] > 0 else ""
            print(f"{i:<5} {r['final_score']:<8} {r['relevance']:<6} {r['success_prob']:<11} {r['observations']:<4} {cross_info:<6} {r['name']}")
            if r["description"]:
                print(f"      └─ {r['description'][:60]}")
        print(f"\n  Agent: {agent_id}")
        print(f"  ⚡ {elapsed*1000:.2f}ms")
    
    elif command == "record":
        if len(sys.argv) < 4:
            print("Usage: atsm.py record [--agent NAME] <skill_name> <0|1>")
            sys.exit(1)
        # Parse flags
        record_args = sys.argv[2:]
        record_args, agent_id = parse_agent_flag(record_args)
        if len(record_args) < 2:
            print("Usage: atsm.py record [--agent NAME] <skill_name> <0|1>")
            sys.exit(1)
        skill_name = record_args[0]
        success = int(record_args[1])
        if success not in (0, 1):
            print("Success must be 0 or 1")
            sys.exit(1)
        append_record(skill_name, success, agent_id=agent_id)
        print(f"Recorded [{agent_id}]: {skill_name} → {'success' if success else 'failure'}")
    
    elif command == "stats":
        records = load_db()
        stats = compute_stats(records)
        if not stats:
            print("No records yet. Use 'record' to log outcomes.")
            return
        print(f"\n{'Skill':<35} {'S':<4} {'F':<4} {'E[success]':<12} {'N':<4} {'N_eff':<7} {'Conf':<6}")
        print("-" * 85)
        for name, s in sorted(stats.items(), key=lambda x: x[1].get("expected_success", 0), reverse=True):
            print(f"{name:<35} {s['raw_successes']:<4} {s['raw_failures']:<4} "
                  f"{s['expected_success']:<12.3f} {s['total']:<4} "
                  f"{s['effective_total']:<7.2f} {s['confidence']:<6.2f}")
        
        # Multi-agent breakdown
        agent_counts = defaultdict(int)
        for rec in records:
            agent_counts[rec.get("agent", DEFAULT_AGENT)] += 1
        if len(agent_counts) > 1:
            print(f"\n  Agents: {dict(agent_counts)}")
    
    elif command == "benchmark":
        """Run rank 10x and report timing."""
        task = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "search and summarize academic papers"
        # Warm up cache
        rank_skills(task)
        # Benchmark
        times = []
        for _ in range(10):
            _, elapsed = rank_skills(task)
            times.append(elapsed)
        print(f"\nBenchmark: 10 runs of '{task}'")
        print(f"  avg: {sum(times)/len(times)*1000:.2f}ms")
        print(f"  min: {min(times)*1000:.2f}ms")
        print(f"  max: {max(times)*1000:.2f}ms")
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
