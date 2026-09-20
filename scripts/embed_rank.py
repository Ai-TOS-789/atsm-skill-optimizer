#!/usr/bin/env python3
"""Embedding-based Skill Ranker v1.0.

Ranks skills by semantic relevance to a task using sentence-transformers
embeddings, combined with ATSM's Bayesian success probability.

Falls back to TF-IDF (same as atsm.py) if sentence-transformers is not installed.

Usage:
    python3 embed_rank.py "task description"
    python3 embed_rank.py --install
    python3 embed_rank.py --tfidf "task description"   # force TF-IDF fallback
    python3 embed_rank.py --stats
"""

import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = Path(__file__).parent.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"

# Algorithm weights
ALPHA = 0.7  # weight for embedding cosine similarity
BETA = 0.3   # weight for keyword overlap
PRIOR_SUCCESS = 1.0
PRIOR_FAILURE = 1.0
DECAY_LAMBDA = 0.01
MIN_OBSERVATIONS = 3

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
        # Dense vectors
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
                "path": str(skill_dir.parent),
                "tokens": tokenize(name + " " + description),
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


# --- Embedding backend ---
class EmbeddingBackend:
    """Abstract embedding backend."""

    def encode(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class TransformerBackend(EmbeddingBackend):
    """sentence-transformers backend."""

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()


class TfIdfBackend(EmbeddingBackend):
    """TF-IDF fallback backend."""

    def __init__(self, skill_docs: list[list[str]]):
        self.skill_docs = skill_docs
        n = len(skill_docs)
        df = Counter()
        for doc in skill_docs:
            df.update(set(doc))
        self.idf = {term: math.log((n + 1) / (1 + freq)) + 1 for term, freq in df.items()}

    def encode(self, texts: list[str]) -> list[dict[str, float]]:
        results = []
        for text in texts:
            tokens = tokenize(text)
            tf = Counter(tokens)
            total = len(tokens) or 1
            vec = {term: (count / total) * self.idf.get(term, 0) for term, count in tf.items()}
            results.append(vec)
        return results


def get_backend(skills: list[dict]) -> tuple[EmbeddingBackend, str]:
    """Get the best available embedding backend.

    Returns (backend, name) where name is "sentence-transformers" or "tfidf".
    """
    try:
        backend = TransformerBackend()
        return backend, "sentence-transformers"
    except ImportError:
        skill_docs = [s["tokens"] for s in skills]
        return TfIdfBackend(skill_docs), "tfidf"


# --- Ranking ---
def rank_skills(task: str, top_k: int = 10, dedup: bool = True, force_tfidf: bool = False) -> tuple[list[dict], float, str]:
    """Rank skills by relevance and expected success for a task.

    Returns (results, elapsed_seconds, backend_name).
    """
    start = time.time()
    skills = load_skills(dedup=dedup)
    if not skills:
        return [], time.time() - start, "none"

    records = load_db()
    stats = compute_stats(records)
    task_tokens = tokenize(task)

    # Get embedding backend
    if force_tfidf:
        skill_docs = [s["tokens"] for s in skills]
        backend = TfIdfBackend(skill_docs)
        backend_name = "tfidf"
    else:
        backend, backend_name = get_backend(skills)

    # Compute embeddings
    if backend_name == "sentence-transformers":
        all_texts = [s["name"] + " " + s["description"] for s in skills] + [task]
        all_vecs = backend.encode(all_texts)
        task_vec = all_vecs[-1]
        skill_vecs = all_vecs[:-1]
    else:
        # TF-IDF: encode skills + task together for shared vocabulary
        all_docs = [s["tokens"] for s in skills] + [task_tokens]
        all_vecs = tf_idf_vectors(all_docs)
        task_vec = all_vecs[-1]
        skill_vecs = all_vecs[:-1]

    results = []
    for i, skill in enumerate(skills):
        emb_cos = cosine_sim(skill_vecs[i], task_vec)
        kw_overlap = keyword_overlap(skill["tokens"], task_tokens)
        relevance = ALPHA * emb_cos + BETA * kw_overlap

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
            "embedding_cos": round(emb_cos, 4),
            "keyword_overlap": round(kw_overlap, 4),
            "success_prob": round(success_prob, 4),
            "final_score": round(final_score, 4),
            "observations": s["total"] if s else 0,
        })

    results.sort(key=lambda x: x["final_score"], reverse=True)
    elapsed = time.time() - start
    return results[:top_k], elapsed, backend_name


# --- CLI ---
def cmd_install():
    """Install sentence-transformers."""
    print("Installing sentence-transformers...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])
        print("✓ sentence-transformers installed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"✗ Installation failed: {e}", file=sys.stderr)
        sys.exit(1)


def cmd_stats():
    """Show database statistics."""
    records = load_db()
    from collections import defaultdict
    skill_counts = defaultdict(lambda: {"success": 0, "failure": 0})
    for rec in records:
        if rec["success"]:
            skill_counts[rec["skill"]]["success"] += 1
        else:
            skill_counts[rec["skill"]]["failure"] += 1
    if not skill_counts:
        print("No records yet.")
        return
    print(f"\n{'Skill':<40} {'S':<4} {'F':<4}")
    print("-" * 52)
    for name, counts in sorted(skill_counts.items()):
        print(f"{name:<40} {counts['success']:<4} {counts['failure']:<4}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "--install":
        cmd_install()
        return

    if command == "--stats":
        cmd_stats()
        return

    force_tfidf = False
    if command == "--tfidf":
        force_tfidf = True
        if len(sys.argv) < 3:
            print("Usage: embed_rank.py --tfidf <task description>")
            sys.exit(1)
        task = " ".join(sys.argv[2:])
    elif command == "--help" or command == "-h":
        print(__doc__)
        sys.exit(0)
    else:
        # Treat as task description (support quoted or unquoted)
        task = " ".join(sys.argv[1:])

    results, elapsed, backend_name = rank_skills(task, force_tfidf=force_tfidf)
    if not results:
        print("No skills found.")
        return

    print(f"\n  Backend: {backend_name}")
    print(f"\n{'Rank':<5} {'Score':<8} {'Rel':<6} {'P(success)':<11} {'N':<4} {'Skill'}")
    print("-" * 70)
    for i, r in enumerate(results, 1):
        print(f"{i:<5} {r['final_score']:<8} {r['relevance']:<6} {r['success_prob']:<11} {r['observations']:<4} {r['name']}")
        if r["description"]:
            print(f"      └─ {r['description'][:60]}")
    print(f"\n  ⚡ {elapsed*1000:.2f}ms")


if __name__ == "__main__":
    main()
