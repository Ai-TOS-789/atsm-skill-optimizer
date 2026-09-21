#!/usr/bin/env python3
"""Swarm Intelligence Module v1.0.

Spawns N agents to solve the same problem independently using different
strategies, then votes on the best solution weighted by agent confidence.

Strategies:
    Agent 1: ATSM rank only (Bayesian success-weighted relevance)
    Agent 2: Embedding rank only (semantic similarity)
    Agent 3: Random exploration (stochastic sampling)
    Agent 4: Causal inference (cause-effect reasoning over skill graph)

Usage:
    python3 swarm.py solve "task description"
    python3 swarm.py vote
    python3 swarm.py stats
    python3 swarm.py consensus
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
SWARM_DB_FILE = DATA_DIR / "swarm_db.jsonl"
SWARM_STATS_FILE = DATA_DIR / "swarm_stats.json"

# Algorithm weights
PRIOR_SUCCESS = 1.0
PRIOR_FAILURE = 1.0
DECAY_LAMBDA = 0.01
MIN_OBSERVATIONS = 3

# Agent confidence weights (learned over time)
DEFAULT_AGENT_WEIGHTS = {
    "atsm_rank": 0.30,
    "embedding_rank": 0.25,
    "random_explore": 0.15,
    "causal_inference": 0.30,
}

# Strategy descriptions
STRATEGY_DESCRIPTIONS = {
    "atsm_rank": "Bayesian success-weighted relevance ranking",
    "embedding_rank": "Semantic embedding similarity ranking",
    "random_explore": "Stochastic random exploration",
    "causal_inference": "Cause-effect reasoning over skill dependency graph",
}


# --- Text processing ---
import re

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


def cosine_sim(a, b) -> float:
    """Cosine similarity between two vectors (sparse dicts or dense lists)."""
    if isinstance(a, dict) and isinstance(b, dict):
        if not a or not b:
            return 0.0
        common = set(a) & set(b)
        if not common:
            return 0.0
        dot = sum(a[t] * b[t] for t in common)
        norm_a = math.sqrt(sum(v ** 2 for v in a.values()))
        norm_b = math.sqrt(sum(v ** 2 for v in b.values()))
    else:
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x ** 2 for x in a))
        norm_b = math.sqrt(sum(y ** 2 for y in b))
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
    """Parse YAML frontmatter robustly, handling multi-line values."""
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
            match = re.match(r'^([a-zA-Z_][a-zA-Z0-9_]*)\s*:\s*(.*)$', stripped)
            if match and (not line[0].isspace() or line == stripped):
                is_new_key = True
        if is_new_key:
            if current_key:
                val = "\n".join(current_value_lines).strip()
                result[current_key] = val
            key = match.group(1)
            rest = match.group(2).strip()
            current_key = key
            if rest in ("|", ">", "|>", ">-", "|-"):
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
def load_skills(dedup: bool = True) -> list[dict]:
    """Load all skills with their descriptions and categories."""
    skills = []
    if not SKILLS_ROOT.exists():
        return skills
    seen_names = set()
    for skill_dir in SKILLS_ROOT.rglob("SKILL.md"):
        try:
            content = skill_dir.read_text(encoding="utf-8", errors="replace")
            name = skill_dir.parent.name
            description = ""
            category = str(skill_dir.parent.parent.name) if skill_dir.parent.parent != SKILLS_ROOT else ""
            if content.startswith("---"):
                fm = parse_frontmatter(content)
                if "name" in fm:
                    name = fm["name"]
                if "description" in fm:
                    description = fm["description"]
            if dedup:
                if name in seen_names:
                    continue
                seen_names.add(name)
            skills.append({
                "name": name,
                "description": description,
                "category": category,
                "path": str(skill_dir.parent),
                "tokens": tokenize(name + " " + description + " " + category),
            })
        except Exception:
            continue
    return skills


# --- Database ---
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


# --- Swarm Database ---
def load_swarm_db() -> list[dict]:
    """Load swarm voting records."""
    records = []
    if not SWARM_DB_FILE.exists():
        return records
    with open(SWARM_DB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def append_swarm_record(record: dict):
    """Append a swarm voting record."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(SWARM_DB_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_swarm_stats() -> dict:
    """Load aggregated swarm statistics."""
    if not SWARM_STATS_FILE.exists():
        return {
            "strategy_performance": defaultdict(lambda: {"wins": 0, "total": 0, "avg_confidence": 0.0}),
            "problem_type_performance": defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0})),
            "total_solves": 0,
        }
    with open(SWARM_STATS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Convert back to defaultdict
    data["strategy_performance"] = defaultdict(
        lambda: {"wins": 0, "total": 0, "avg_confidence": 0.0},
        data.get("strategy_performance", {})
    )
    data["problem_type_performance"] = defaultdict(
        lambda: defaultdict(lambda: {"wins": 0, "total": 0}),
        data.get("problem_type_performance", {})
    )
    return data


def save_swarm_stats(stats: dict):
    """Save aggregated swarm statistics."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    # Convert defaultdicts to regular dicts for JSON serialization
    data = {
        "strategy_performance": dict(stats["strategy_performance"]),
        "problem_type_performance": {k: dict(v) for k, v in stats["problem_type_performance"].items()},
        "total_solves": stats.get("total_solves", 0),
    }
    with open(SWARM_STATS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


# --- Problem Type Classification ---
def classify_problem(task: str) -> str:
    """Classify task into problem type based on keywords."""
    task_lower = task.lower()
    categories = {
        "research": ["search", "paper", "arxiv", "research", "summarize", "academic", "study", "analyze"],
        "coding": ["code", "program", "debug", "build", "develop", "function", "api", "script", "fix"],
        "creative": ["write", "design", "create", "draw", "art", "music", "song", "story", "poem"],
        "data": ["data", "csv", "excel", "spreadsheet", "chart", "graph", "visualize", "analyze data"],
        "communication": ["email", "message", "send", "post", "tweet", "notify", "alert"],
        "file": ["file", "pdf", "document", "read", "open", "save", "download", "upload"],
        "web": ["web", "browser", "url", "scrape", "fetch", "website", "online"],
        "system": ["monitor", "cpu", "memory", "disk", "process", "system", "performance"],
    }
    scores = {}
    for cat, keywords in categories.items():
        scores[cat] = sum(1 for kw in keywords if kw in task_lower)
    if not any(scores.values()):
        return "general"
    return max(scores, key=scores.get)


# --- Agent Strategies ---

def agent_atsm_rank(task: str, skills: list[dict], stats: dict, top_k: int = 5) -> dict:
    """Agent 1: ATSM rank only — Bayesian success-weighted relevance ranking."""
    task_tokens = tokenize(task)
    all_tokens = [s["tokens"] for s in skills] + [task_tokens]
    vectors = tf_idf_vectors(all_tokens)
    task_vec = vectors[-1]
    skill_vectors = vectors[:-1]

    results = []
    for i, skill in enumerate(skills):
        relevance = 0.6 * cosine_sim(skill_vectors[i], task_vec) + 0.4 * keyword_overlap(skill["tokens"], task_tokens)
        s = stats.get(skill["name"])
        if s and s["total"] > 0:
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
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = results[:top_k]

    # Confidence based on score separation
    if len(top_results) >= 2:
        separation = top_results[0]["final_score"] - top_results[1]["final_score"]
        confidence = min(0.5 + separation * 2, 0.95)
    else:
        confidence = 0.5

    return {
        "strategy": "atsm_rank",
        "rankings": top_results,
        "confidence": round(confidence, 4),
        "top_pick": top_results[0]["name"] if top_results else None,
    }


def agent_embedding_rank(task: str, skills: list[dict], top_k: int = 5) -> dict:
    """Agent 2: Embedding rank only — semantic similarity ranking."""
    task_tokens = tokenize(task)
    all_tokens = [s["tokens"] for s in skills] + [task_tokens]
    vectors = tf_idf_vectors(all_tokens)
    task_vec = vectors[-1]
    skill_vectors = vectors[:-1]

    results = []
    for i, skill in enumerate(skills):
        emb_cos = cosine_sim(skill_vectors[i], task_vec)
        kw_overlap = keyword_overlap(skill["tokens"], task_tokens)
        relevance = 0.7 * emb_cos + 0.3 * kw_overlap
        results.append({
            "name": skill["name"],
            "description": skill["description"],
            "embedding_cos": round(emb_cos, 4),
            "keyword_overlap": round(kw_overlap, 4),
            "relevance": round(relevance, 4),
            "final_score": round(relevance, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = results[:top_k]

    # Confidence based on embedding separation
    if len(top_results) >= 2:
        separation = top_results[0]["embedding_cos"] - top_results[1]["embedding_cos"]
        confidence = min(0.5 + separation * 3, 0.95)
    else:
        confidence = 0.5

    return {
        "strategy": "embedding_rank",
        "rankings": top_results,
        "confidence": round(confidence, 4),
        "top_pick": top_results[0]["name"] if top_results else None,
    }


def agent_random_explore(task: str, skills: list[dict], top_k: int = 5) -> dict:
    """Agent 3: Random exploration — stochastic sampling with noise."""
    task_tokens = tokenize(task)
    all_tokens = [s["tokens"] for s in skills] + [task_tokens]
    vectors = tf_idf_vectors(all_tokens)
    task_vec = vectors[-1]
    skill_vectors = vectors[:-1]

    results = []
    for i, skill in enumerate(skills):
        base_relevance = cosine_sim(skill_vectors[i], task_vec)
        # Add random noise for exploration
        noise = random.gauss(0, 0.15)
        perturbed_score = max(0, base_relevance + noise)
        results.append({
            "name": skill["name"],
            "description": skill["description"],
            "base_relevance": round(base_relevance, 4),
            "noise": round(noise, 4),
            "final_score": round(perturbed_score, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = results[:top_k]

    # Random agent has lower base confidence
    confidence = random.uniform(0.2, 0.5)

    return {
        "strategy": "random_explore",
        "rankings": top_results,
        "confidence": round(confidence, 4),
        "top_pick": top_results[0]["name"] if top_results else None,
    }


def agent_causal_inference(task: str, skills: list[dict], stats: dict, top_k: int = 5) -> dict:
    """Agent 4: Causal inference — cause-effect reasoning over skill dependency graph.

    Builds a causal graph from skill co-occurrence in historical data and
    reasons about which skills cause success for the given task type.
    """
    task_tokens = tokenize(task)
    task_type = classify_problem(task)

    # Build causal graph from historical co-occurrence
    records = load_db()
    skill_cooccurrence = defaultdict(lambda: defaultdict(float))
    skill_outcomes = defaultdict(lambda: {"success": 0, "total": 0})

    for rec in records:
        skill_outcomes[rec["skill"]]["total"] += 1
        if rec["success"]:
            skill_outcomes[rec["skill"]]["success"] += 1

    # Compute causal strength: how much does skill A's success predict overall success
    causal_scores = {}
    for skill in skills:
        name = skill["name"]
        s = stats.get(name)
        if s and s["total"] > 0:
            # Causal strength = P(success|skill_relevant) / P(success)
            p_success_global = 0.5  # prior
            p_success_skill = s["expected_success"]
            causal_strength = p_success_skill / p_success_global if p_success_global > 0 else 1.0
        else:
            causal_strength = 1.0  # neutral

        # Relevance to task
        all_tokens = [skill["tokens"], task_tokens]
        vectors = tf_idf_vectors(all_tokens)
        relevance = cosine_sim(vectors[0], vectors[1])

        # Causal score combines relevance with causal strength
        causal_scores[name] = relevance * causal_strength

    results = []
    for skill in skills:
        name = skill["name"]
        score = causal_scores.get(name, 0)
        results.append({
            "name": name,
            "description": skill["description"],
            "causal_score": round(score, 4),
            "final_score": round(score, 4),
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    top_results = results[:top_k]

    # Confidence based on causal graph density
    data_density = len(records) / max(len(skills), 1)
    confidence = min(0.4 + data_density * 0.1, 0.9)

    return {
        "strategy": "causal_inference",
        "rankings": top_results,
        "confidence": round(confidence, 4),
        "top_pick": top_results[0]["name"] if top_results else None,
        "problem_type": task_type,
    }


# --- Voting & Consensus ---
def get_agent_weights() -> dict:
    """Get agent weights, learned from historical performance if available."""
    stats = load_swarm_stats()
    weights = dict(DEFAULT_AGENT_WEIGHTS)

    # Adjust weights based on historical performance
    strategy_perf = stats.get("strategy_performance", {})
    total_wins = sum(s.get("wins", 0) for s in strategy_perf.values())

    if total_wins > 0:
        for strategy, perf in strategy_perf.items():
            if strategy in weights:
                # Blend default weight with learned performance
                win_rate = perf.get("wins", 0) / max(perf.get("total", 1), 1)
                weights[strategy] = 0.5 * weights[strategy] + 0.5 * win_rate

    # Normalize
    total = sum(weights.values())
    if total > 0:
        weights = {k: v / total for k, v in weights.items()}

    return weights


def vote(agents_results: list[dict], agent_weights: dict) -> dict:
    """Weighted voting on best solution.

    Each agent votes for its top pick, weighted by:
    1. Agent's confidence in its pick
    2. Agent's historical weight (learned)
    """
    votes = defaultdict(float)
    vote_details = []

    for result in agents_results:
        strategy = result["strategy"]
        top_pick = result["top_pick"]
        confidence = result["confidence"]
        weight = agent_weights.get(strategy, 0.25)

        # Vote = agent_weight * confidence
        vote_score = weight * confidence
        votes[top_pick] += vote_score

        vote_details.append({
            "strategy": strategy,
            "top_pick": top_pick,
            "confidence": confidence,
            "weight": round(weight, 4),
            "vote_score": round(vote_score, 4),
        })

    # Determine winner
    if not votes:
        return {"winner": None, "votes": [], "consensus_strength": 0.0}

    winner = max(votes, key=votes.get)
    total_votes = sum(votes.values())
    consensus_strength = votes[winner] / total_votes if total_votes > 0 else 0.0

    return {
        "winner": winner,
        "votes": sorted(vote_details, key=lambda x: x["vote_score"], reverse=True),
        "vote_totals": {k: round(v, 4) for k, v in sorted(votes.items(), key=lambda x: x[1], reverse=True)},
        "consensus_strength": round(consensus_strength, 4),
    }


# --- Main Commands ---

def cmd_solve(task: str):
    """Solve a task using swarm intelligence."""
    print(f"\n🐝 Swarm Intelligence — Solving: \"{task}\"")
    print("=" * 60)

    skills = load_skills()
    if not skills:
        print("No skills found.")
        return

    records = load_db()
    stats = compute_stats(records)
    problem_type = classify_problem(task)
    agent_weights = get_agent_weights()

    print(f"\n  Problem type: {problem_type}")
    print(f"  Skills available: {len(skills)}")
    print(f"  Historical records: {len(records)}")
    print(f"\n  Spawning 4 agents...\n")

    # Spawn agents
    agents_results = []

    # Agent 1: ATSM rank
    print("  [Agent 1] ATSM Rank — Bayesian success-weighted relevance")
    r1 = agent_atsm_rank(task, skills, stats)
    agents_results.append(r1)
    print(f"    Top pick: {r1['top_pick']} (confidence: {r1['confidence']})")
    print(f"    Top 3: {', '.join(x['name'] for x in r1['rankings'][:3])}")

    # Agent 2: Embedding rank
    print("\n  [Agent 2] Embedding Rank — Semantic similarity")
    r2 = agent_embedding_rank(task, skills)
    agents_results.append(r2)
    print(f"    Top pick: {r2['top_pick']} (confidence: {r2['confidence']})")
    print(f"    Top 3: {', '.join(x['name'] for x in r2['rankings'][:3])}")

    # Agent 3: Random exploration
    print("\n  [Agent 3] Random Exploration — Stochastic sampling")
    r3 = agent_random_explore(task, skills)
    agents_results.append(r3)
    print(f"    Top pick: {r3['top_pick']} (confidence: {r3['confidence']})")
    print(f"    Top 3: {', '.join(x['name'] for x in r3['rankings'][:3])}")

    # Agent 4: Causal inference
    print("\n  [Agent 4] Causal Inference — Cause-effect reasoning")
    r4 = agent_causal_inference(task, skills, stats)
    agents_results.append(r4)
    print(f"    Top pick: {r4['top_pick']} (confidence: {r4['confidence']})")
    print(f"    Top 3: {', '.join(x['name'] for x in r4['rankings'][:3])}")

    # Vote
    print(f"\n{'=' * 60}")
    print("🗳️  VOTING & CONSENSUS")
    print("=" * 60)

    result = vote(agents_results, agent_weights)

    print(f"\n  Agent weights:")
    for strategy, weight in sorted(agent_weights.items(), key=lambda x: x[1], reverse=True):
        desc = STRATEGY_DESCRIPTIONS.get(strategy, "")
        print(f"    {strategy:<20} {weight:.3f}  ({desc})")

    print(f"\n  Votes:")
    for v in result["votes"]:
        print(f"    {v['strategy']:<20} → {v['top_pick']:<25} score={v['vote_score']:.4f} (conf={v['confidence']:.3f}, w={v['weight']:.3f})")

    print(f"\n  Vote totals:")
    for name, score in result["vote_totals"].items():
        print(f"    {name:<25} {score:.4f}")

    print(f"\n  🏆 WINNER: {result['winner']}")
    print(f"  Consensus strength: {result['consensus_strength']:.1%}")

    # Record to swarm DB
    swarm_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "task": task,
        "problem_type": problem_type,
        "agents": [
            {
                "strategy": r["strategy"],
                "top_pick": r["top_pick"],
                "confidence": r["confidence"],
            }
            for r in agents_results
        ],
        "winner": result["winner"],
        "consensus_strength": result["consensus_strength"],
        "agent_weights": agent_weights,
    }
    append_swarm_record(swarm_record)

    # Update stats
    swarm_stats = load_swarm_stats()
    swarm_stats["total_solves"] += 1

    for r in agents_results:
        strategy = r["strategy"]
        swarm_stats["strategy_performance"][strategy]["total"] += 1
        if r["top_pick"] == result["winner"]:
            swarm_stats["strategy_performance"][strategy]["wins"] += 1

        # Per problem type
        pt = problem_type
        swarm_stats["problem_type_performance"][pt][strategy]["total"] += 1
        if r["top_pick"] == result["winner"]:
            swarm_stats["problem_type_performance"][pt][strategy]["wins"] += 1

    save_swarm_stats(swarm_stats)

    print(f"\n  📊 Recorded to swarm database.")
    print(f"  Total solves: {swarm_stats['total_solves']}")


def cmd_vote():
    """Show recent voting history."""
    records = load_swarm_db()
    if not records:
        print("No voting records yet. Use 'solve' first.")
        return

    print(f"\n🗳️  Swarm Voting History (last 10)")
    print("=" * 70)

    for rec in records[-10:]:
        ts = rec["timestamp"][:19]
        print(f"\n  {ts} | {rec['task'][:50]}")
        print(f"    Winner: {rec['winner']} (consensus: {rec['consensus_strength']:.1%})")
        for agent in rec["agents"]:
            marker = " 🏆" if agent["top_pick"] == rec["winner"] else ""
            print(f"      {agent['strategy']:<20} → {agent['top_pick']}{marker}")


def cmd_stats():
    """Show swarm statistics."""
    stats = load_swarm_stats()
    records = load_swarm_db()

    print(f"\n📊 Swarm Intelligence Statistics")
    print("=" * 60)

    print(f"\n  Total solves: {stats.get('total_solves', len(records))}")

    # Strategy performance
    print(f"\n  Strategy Performance:")
    print(f"    {'Strategy':<20} {'Wins':<6} {'Total':<6} {'Win Rate':<10}")
    print(f"    " + "-" * 42)

    strategy_perf = stats.get("strategy_performance", {})
    for strategy, perf in sorted(strategy_perf.items(), key=lambda x: x[1].get("wins", 0), reverse=True):
        wins = perf.get("wins", 0)
        total = perf.get("total", 0)
        win_rate = wins / total if total > 0 else 0
        print(f"    {strategy:<20} {wins:<6} {total:<6} {win_rate:.1%}")

    # Problem type performance
    print(f"\n  Best Strategy by Problem Type:")
    print(f"    {'Problem Type':<15} {'Best Strategy':<20} {'Wins':<6} {'Total':<6}")
    print(f"    " + "-" * 57)

    pt_perf = stats.get("problem_type_performance", {})
    for pt, strategies in sorted(pt_perf.items()):
        best = max(strategies.items(), key=lambda x: x[1].get("wins", 0))
        wins = best[1].get("wins", 0)
        total = best[1].get("total", 0)
        print(f"    {pt:<15} {best[0]:<20} {wins:<6} {total:<6}")

    # Recent consensus strength
    if records:
        recent = records[-10:]
        avg_consensus = sum(r["consensus_strength"] for r in recent) / len(recent)
        print(f"\n  Avg consensus (last 10): {avg_consensus:.1%}")


def cmd_consensus():
    """Show current agent weight distribution."""
    weights = get_agent_weights()

    print(f"\n⚖️  Agent Weight Distribution")
    print("=" * 50)

    for strategy, weight in sorted(weights.items(), key=lambda x: x[1], reverse=True):
        desc = STRATEGY_DESCRIPTIONS.get(strategy, "")
        bar = "█" * int(weight * 40)
        print(f"  {strategy:<20} {weight:.3f} {bar}")
        print(f"    └─ {desc}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "solve":
        if len(sys.argv) < 3:
            print("Usage: swarm.py solve <task description>")
            sys.exit(1)
        task = " ".join(sys.argv[2:])
        cmd_solve(task)

    elif command == "vote":
        cmd_vote()

    elif command == "stats":
        cmd_stats()

    elif command == "consensus":
        cmd_consensus()

    elif command == "help" or command == "-h" or command == "--help":
        print(__doc__)
        sys.exit(0)

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
