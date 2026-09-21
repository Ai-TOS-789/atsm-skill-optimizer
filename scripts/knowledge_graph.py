#!/usr/bin/env python3
"""Knowledge Graph for ATSM skills.

Builds a graph of skills and their relationships from ATSM history:
- co-occurrence: skills that appear together in rankings
- sequence: skill A → skill B (chain patterns)
- similarity: TF-IDF cosine similarity between descriptions
- causal: skill A success → skill B success (Granger causality)

Usage:
    python3 knowledge_graph.py build
    python3 knowledge_graph.py neighbors <skill>
    python3 knowledge_graph.py causal <skill>
    python3 knowledge_graph.py path <from_skill> <to_skill>
    python3 knowledge_graph.py communities
"""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = Path(__file__).parent.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"
GRAPH_FILE = DATA_DIR / "knowledge_graph.json"

# --- Text processing ---
import re
TOKEN_RE = re.compile(r"[a-zA-Z0-9_\u0E00-\u0E7F]+", re.UNICODE)


def tokenize(text: str) -> list:
    """Lowercase tokenization supporting Thai and English."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def tf_idf_vectors(docs: list) -> list:
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


def cosine_sim(a: dict, b: dict) -> float:
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


# --- Skill loading ---
def parse_frontmatter(content: str) -> dict:
    """Parse YAML frontmatter."""
    if not content.startswith("---"):
        return {}
    end = content.find("---", 3)
    if end == -1:
        return {}
    fm_text = content[3:end]
    result = {}
    for line in fm_text.splitlines():
        if ":" in line and not line.startswith(" "):
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip().strip('"').strip("'")
    return result


def load_skills() -> list:
    """Load all skills with their descriptions."""
    skills = []
    if not SKILLS_ROOT.exists():
        return skills
    seen = set()
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
            if name in seen:
                continue
            seen.add(name)
            skills.append({
                "name": name,
                "description": description,
                "tokens": tokenize(name + " " + description),
            })
        except Exception:
            continue
    return skills


# --- Database ---
def load_db() -> list:
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


# --- Graph building ---
def build_cooccurrence(records: list, skills: list) -> dict:
    """Build co-occurrence edges from records that appear in temporal proximity."""
    skill_names = {s["name"] for s in skills}
    # Group records by timestamp proximity (within 60 seconds)
    records_sorted = sorted(records, key=lambda r: r.get("timestamp", ""))
    groups = []
    current_group = []
    for rec in records_sorted:
        if not current_group:
            current_group.append(rec)
        else:
            t1 = datetime.fromisoformat(current_group[-1]["timestamp"])
            t2 = datetime.fromisoformat(rec["timestamp"])
            if abs((t2 - t1).total_seconds()) <= 60:
                current_group.append(rec)
            else:
                groups.append(current_group)
                current_group = [rec]
    if current_group:
        groups.append(current_group)

    cooccurrence = Counter()
    for group in groups:
        skills_in_group = set()
        for rec in group:
            if rec.get("skill") in skill_names:
                skills_in_group.add(rec["skill"])
        skills_list = sorted(skills_in_group)
        for i in range(len(skills_list)):
            for j in range(i + 1, len(skills_list)):
                pair = (skills_list[i], skills_list[j])
                cooccurrence[pair] += 1

    return dict(cooccurrence)


def build_sequence(records: list, skills: list) -> dict:
    """Build sequence edges (skill A → skill B) from temporal ordering."""
    skill_names = {s["name"] for s in skills}
    records_sorted = sorted(records, key=lambda r: r.get("timestamp", ""))

    sequence = Counter()
    for i in range(len(records_sorted) - 1):
        s1 = records_sorted[i].get("skill")
        s2 = records_sorted[i + 1].get("skill")
        if s1 in skill_names and s2 in skill_names and s1 != s2:
            sequence[(s1, s2)] += 1

    return dict(sequence)


def build_similarity(skills: list) -> dict:
    """Build similarity edges using TF-IDF cosine similarity."""
    all_tokens = [s["tokens"] for s in skills]
    vectors = tf_idf_vectors(all_tokens)

    similarity = {}
    for i in range(len(skills)):
        for j in range(i + 1, len(skills)):
            sim = cosine_sim(vectors[i], vectors[j])
            if sim > 0.1:
                pair = (skills[i]["name"], skills[j]["name"])
                similarity[pair] = round(sim, 4)

    return similarity


def build_causal(records: list, skills: list) -> dict:
    """Build causal edges using Granger causality approximation.
    
    Simplified approach: if skill A's success at time t correlates with
    skill B's success at time t+1 more often than expected by chance,
    we say A Granger-causes B.
    """
    skill_names = {s["name"] for s in skills}
    records_sorted = sorted(records, key=lambda r: r.get("timestamp", ""))

    # Build success/failure sequences per skill
    skill_sequence = defaultdict(list)
    for rec in records_sorted:
        s = rec.get("skill")
        if s in skill_names:
            skill_sequence[s].append(rec.get("success", 0))

    causal = {}
    skill_list = sorted(skill_names)
    for i, s1 in enumerate(skill_list):
        for j, s2 in enumerate(skill_list):
            if i == j:
                continue
            seq1 = skill_sequence.get(s1, [])
            seq2 = skill_sequence.get(s2, [])
            if len(seq1) < 3 or len(seq2) < 3:
                continue
            # Check if s1's success predicts s2's success
            # Use lag-1 correlation
            min_len = min(len(seq1), len(seq2)) - 1
            if min_len < 2:
                continue
            # Count: s1 success at t AND s2 success at t+1
            both = sum(1 for k in range(min_len) if seq1[k] == 1 and seq2[k + 1] == 1)
            s1_success = sum(1 for k in range(min_len) if seq1[k] == 1)
            s2_success = sum(1 for k in range(1, min_len + 1) if seq2[k] == 1)
            
            if s1_success == 0 or min_len == 0:
                continue
            # Conditional probability: P(s2 success | s1 success) vs P(s2 success)
            p_s2_given_s1 = both / s1_success
            p_s2 = s2_success / min_len if min_len > 0 else 0
            
            # Granger causality: does s1 help predict s2?
            if p_s2_given_s1 > p_s2 + 0.1 and both >= 2:
                causal[(s1, s2)] = round(p_s2_given_s1 - p_s2, 4)

    return causal


def build_graph() -> dict:
    """Build the complete knowledge graph from ATSM history."""
    skills = load_skills()
    records = load_db()

    print(f"  Loaded {len(skills)} skills, {len(records)} records")

    cooccurrence = build_cooccurrence(records, skills)
    sequence = build_sequence(records, skills)
    similarity = build_similarity(skills)
    causal = build_causal(records, skills)

    # Build adjacency list
    nodes = {}
    for s in skills:
        nodes[s["name"]] = {
            "name": s["name"],
            "description": s["description"],
            "neighbors": {},
        }

    # Add edges
    for (a, b), weight in cooccurrence.items():
        if a in nodes and b in nodes:
            nodes[a]["neighbors"][b] = nodes[a]["neighbors"].get(b, {})
            nodes[a]["neighbors"][b]["co-occurrence"] = weight
            nodes[b]["neighbors"][a] = nodes[b]["neighbors"].get(a, {})
            nodes[b]["neighbors"][a]["co-occurrence"] = weight

    for (a, b), weight in sequence.items():
        if a in nodes and b in nodes:
            nodes[a]["neighbors"][b] = nodes[a]["neighbors"].get(b, {})
            nodes[a]["neighbors"][b]["sequence"] = weight

    for (a, b), weight in similarity.items():
        if a in nodes and b in nodes:
            nodes[a]["neighbors"][b] = nodes[a]["neighbors"].get(b, {})
            nodes[a]["neighbors"][b]["similarity"] = weight
            nodes[b]["neighbors"][a] = nodes[b]["neighbors"].get(a, {})
            nodes[b]["neighbors"][a]["similarity"] = weight

    for (a, b), weight in causal.items():
        if a in nodes and b in nodes:
            nodes[a]["neighbors"][b] = nodes[a]["neighbors"].get(b, {})
            nodes[a]["neighbors"][b]["causal"] = weight

    graph = {
        "meta": {
            "built_at": datetime.now(timezone.utc).isoformat(),
            "num_skills": len(skills),
            "num_records": len(records),
            "num_cooccurrence_edges": len(cooccurrence),
            "num_sequence_edges": len(sequence),
            "num_similarity_edges": len(similarity),
            "num_causal_edges": len(causal),
        },
        "nodes": nodes,
    }

    # Save to file
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(GRAPH_FILE, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    return graph


def load_graph() -> dict:
    """Load graph from file."""
    if not GRAPH_FILE.exists():
        return build_graph()
    with open(GRAPH_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


# --- API ---
def get_neighbors(skill: str) -> list:
    """Get related skills with relationship types."""
    graph = load_graph()
    if skill not in graph["nodes"]:
        return []
    neighbors = graph["nodes"][skill]["neighbors"]
    results = []
    for name, rels in neighbors.items():
        results.append({
            "skill": name,
            "relationships": rels,
        })
    # Sort by total weight
    results.sort(key=lambda x: sum(x["relationships"].values()), reverse=True)
    return results


def get_causal_skills(skill: str) -> list:
    """Get skills that cause this skill to succeed (Granger causality)."""
    graph = load_graph()
    if skill not in graph["nodes"]:
        return []
    causal = []
    for name, node in graph["nodes"].items():
        if skill in node["neighbors"] and "causal" in node["neighbors"][skill]:
            causal.append({
                "skill": name,
                "causal_strength": node["neighbors"][skill]["causal"],
            })
    causal.sort(key=lambda x: x["causal_strength"], reverse=True)
    return causal


def get_skill_path(from_skill: str, to_skill: str) -> list:
    """Find shortest path between two skills using BFS."""
    graph = load_graph()
    if from_skill not in graph["nodes"] or to_skill not in graph["nodes"]:
        return []

    # BFS
    from collections import deque
    queue = deque([(from_skill, [from_skill])])
    visited = {from_skill}

    while queue:
        current, path = queue.popleft()
        if current == to_skill:
            return path
        for neighbor in graph["nodes"][current]["neighbors"]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, path + [neighbor]))

    return []


def get_communities() -> list:
    """Detect skill clusters using simple label propagation."""
    graph = load_graph()
    nodes = graph["nodes"]
    if not nodes:
        return []

    # Initialize each node with its own label
    labels = {name: i for i, name in enumerate(nodes)}

    # Iterate
    for _ in range(10):
        changed = False
        for name in nodes:
            neighbor_labels = Counter()
            for neighbor, rels in nodes[name]["neighbors"].items():
                weight = sum(rels.values())
                neighbor_labels[labels[neighbor]] += weight
            if neighbor_labels:
                best_label = neighbor_labels.most_common(1)[0][0]
                if best_label != labels[name]:
                    labels[name] = best_label
                    changed = True
        if not changed:
            break

    # Group by label
    communities = defaultdict(list)
    for name, label in labels.items():
        communities[label].append(name)

    return [{"id": k, "skills": sorted(v)} for k, v in communities.items() if len(v) > 1]


# --- CLI ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "build":
        print("Building knowledge graph...")
        graph = build_graph()
        meta = graph["meta"]
        print(f"  Skills: {meta['num_skills']}")
        print(f"  Records: {meta['num_records']}")
        print(f"  Co-occurrence edges: {meta['num_cooccurrence_edges']}")
        print(f"  Sequence edges: {meta['num_sequence_edges']}")
        print(f"  Similarity edges: {meta['num_similarity_edges']}")
        print(f"  Causal edges: {meta['num_causal_edges']}")
        print(f"  Saved to: {GRAPH_FILE}")

    elif command == "neighbors":
        if len(sys.argv) < 3:
            print("Usage: knowledge_graph.py neighbors <skill>")
            sys.exit(1)
        skill = sys.argv[2]
        neighbors = get_neighbors(skill)
        if not neighbors:
            print(f"No neighbors found for '{skill}'")
            return
        print(f"\nNeighbors of '{skill}':")
        print(f"{'Skill':<30} {'Relationships'}")
        print("-" * 60)
        for n in neighbors:
            rels = ", ".join(f"{k}={v}" for k, v in n["relationships"].items())
            print(f"{n['skill']:<30} {rels}")

    elif command == "causal":
        if len(sys.argv) < 3:
            print("Usage: knowledge_graph.py causal <skill>")
            sys.exit(1)
        skill = sys.argv[2]
        causal = get_causal_skills(skill)
        if not causal:
            print(f"No causal predecessors found for '{skill}'")
            return
        print(f"\nSkills that cause '{skill}' to succeed:")
        print(f"{'Skill':<30} {'Causal Strength'}")
        print("-" * 50)
        for c in causal:
            print(f"{c['skill']:<30} {c['causal_strength']}")

    elif command == "path":
        if len(sys.argv) < 4:
            print("Usage: knowledge_graph.py path <from_skill> <to_skill>")
            sys.exit(1)
        from_skill = sys.argv[2]
        to_skill = sys.argv[3]
        path = get_skill_path(from_skill, to_skill)
        if not path:
            print(f"No path found from '{from_skill}' to '{to_skill}'")
            return
        print(f"\nPath from '{from_skill}' to '{to_skill}':")
        print(" → ".join(path))

    elif command == "communities":
        communities = get_communities()
        if not communities:
            print("No communities found")
            return
        print(f"\nSkill Communities ({len(communities)} clusters):")
        for i, comm in enumerate(communities, 1):
            print(f"\n  Cluster {i}: {', '.join(comm['skills'])}")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
