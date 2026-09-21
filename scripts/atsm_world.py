#!/usr/bin/env python3
"""ATSM World Model: Internal simulation engine for outcome prediction.

Predicts skill execution outcomes before running them using:
- Monte Carlo simulation of skill runs
- Markov chain state transitions
- Counterfactual analysis (what-if scenarios)
- System state awareness (CPU, mem, disk, network)
- Time-of-day pattern recognition

Usage:
    python3 atsm_world.py predict "skill_name"
    python3 atsm_world.py simulate "task" --runs 100
    python3 atsm_world.py counterfactual "skill_A" "skill_B"
    python3 atsm_world.py state
"""

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional

# --- Config ---
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = Path(__file__).parent.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"
PRIORS_FILE = DATA_DIR / "atsm_priors.json"
WORLD_STATE_FILE = DATA_DIR / "atsm_world_state.json"

# Simulation parameters
DEFAULT_MONTE_CARLO_RUNS = 1000
MARKOV_DECAY = 0.95
RESOURCE_DECAY = 0.01
TIME_BUCKETS = 24  # hourly buckets for time-of-day patterns

# Resource cost profiles per skill category (CPU%, Mem MB, Disk IO MB, Network KB)
RESOURCE_PROFILES = {
    "research": {"cpu": 15, "mem": 128, "disk": 5, "net": 512},
    "productivity": {"cpu": 10, "mem": 64, "disk": 2, "net": 256},
    "creative": {"cpu": 25, "mem": 256, "disk": 10, "net": 1024},
    "media": {"cpu": 20, "mem": 192, "disk": 8, "net": 2048},
    "software-development": {"cpu": 30, "mem": 320, "disk": 15, "net": 512},
    "social-media": {"cpu": 8, "mem": 48, "disk": 1, "net": 128},
    "web": {"cpu": 12, "mem": 96, "disk": 3, "net": 768},
    "note-taking": {"cpu": 5, "mem": 32, "disk": 1, "net": 64},
    "email": {"cpu": 8, "mem": 64, "disk": 2, "net": 384},
    "default": {"cpu": 10, "mem": 64, "disk": 2, "net": 256},
}

# Duration profiles per skill category (seconds)
DURATION_PROFILES = {
    "research": {"mean": 8.0, "std": 3.0},
    "productivity": {"mean": 5.0, "std": 2.0},
    "creative": {"mean": 15.0, "std": 5.0},
    "media": {"mean": 12.0, "std": 4.0},
    "software-development": {"mean": 20.0, "std": 8.0},
    "social-media": {"mean": 4.0, "std": 1.5},
    "web": {"mean": 6.0, "std": 2.5},
    "note-taking": {"mean": 3.0, "std": 1.0},
    "email": {"mean": 4.0, "std": 1.5},
    "default": {"mean": 5.0, "std": 2.0},
}


# --- System State ---
def get_system_state() -> dict[str, Any]:
    """Get current system state: CPU, memory, disk, network."""
    state = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cpu_percent": _get_cpu_percent(),
        "mem_percent": _get_mem_percent(),
        "mem_available_mb": _get_mem_available(),
        "disk_percent": _get_disk_percent(),
        "disk_free_gb": _get_disk_free(),
        "network_kb_s": _get_network_rate(),
        "load_avg": _get_load_avg(),
        "hour_of_day": datetime.now().hour,
        "day_of_week": datetime.now().weekday(),
    }
    return state


def _get_cpu_percent() -> float:
    """Get CPU usage percentage."""
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        fields = line.split()
        if fields[0] == "cpu":
            idle = int(fields[4])
            total = sum(int(x) for x in fields[1:])
            if total > 0:
                return round((1 - idle / total) * 100, 1)
    except (FileNotFoundError, ValueError, IndexError):
        pass
    return 0.0


def _get_mem_percent() -> float:
    """Get memory usage percentage."""
    try:
        mem_info = {}
        with open("/proc/meminfo", "r") as f:
            for line in f:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip()
                    val = parts[1].strip().split()[0]
                    mem_info[key] = int(val)
        total = mem_info.get("MemTotal", 0)
        available = mem_info.get("MemAvailable", 0)
        if total > 0:
            return round((1 - available / total) * 100, 1)
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def _get_mem_available() -> float:
    """Get available memory in MB."""
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except (FileNotFoundError, ValueError):
        pass
    return 0.0


def _get_disk_percent() -> float:
    """Get disk usage percentage for root partition."""
    try:
        import shutil
        usage = shutil.disk_usage("/")
        return round((usage.used / usage.total) * 100, 1)
    except Exception:
        return 0.0


def _get_disk_free() -> float:
    """Get free disk space in GB."""
    try:
        import shutil
        usage = shutil.disk_usage("/")
        return round(usage.free / (1024**3), 2)
    except Exception:
        return 0.0


def _get_network_rate() -> float:
    """Get network I/O rate in KB/s."""
    try:
        with open("/proc/net/dev", "r") as f:
            lines = f.readlines()[2:]  # skip headers
        total_bytes = 0
        for line in lines:
            parts = line.split()
            if len(parts) >= 10 and parts[0].startswith(("eth", "wlan", "en", "wl")):
                total_bytes += int(parts[9])  # receive bytes
        return round(total_bytes / 1024, 1)
    except (FileNotFoundError, ValueError):
        return 0.0


def _get_load_avg() -> list[float]:
    """Get system load average."""
    try:
        return list(os.getloadavg())
    except OSError:
        return [0.0, 0.0, 0.0]


# --- Historical Data ---
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


def load_priors() -> dict:
    """Load skill priors."""
    if not PRIORS_FILE.exists():
        return {}
    with open(PRIORS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def load_world_state() -> dict:
    """Load persisted world state."""
    if not WORLD_STATE_FILE.exists():
        return _default_world_state()
    with open(WORLD_STATE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_world_state(state: dict):
    """Persist world state."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with open(WORLD_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _default_world_state() -> dict:
    """Create default world state."""
    return {
        "markov_transitions": {},
        "time_patterns": {},
        "resource_history": [],
        "simulation_log": [],
        "last_updated": datetime.now(timezone.utc).isoformat(),
    }


# --- Skill Loading ---
TOKEN_RE = re.compile(r"[a-zA-Z0-9_\u0E00-\u0E7F]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def load_skills() -> list[dict]:
    """Load all skills with metadata."""
    skills = []
    if not SKILLS_ROOT.exists():
        return skills

    for skill_dir in SKILLS_ROOT.rglob("SKILL.md"):
        try:
            content = skill_dir.read_text(encoding="utf-8", errors="replace")
            name = skill_dir.parent.name
            description = ""
            category = skill_dir.parent.parent.name if skill_dir.parent.parent != SKILLS_ROOT else "default"

            if content.startswith("---"):
                fm = _parse_frontmatter(content)
                if "name" in fm:
                    name = fm["name"]
                if "description" in fm:
                    description = fm["description"]

            skills.append({
                "name": name,
                "description": description,
                "category": category,
                "path": str(skill_dir.parent),
                "tokens": tokenize(name + " " + description),
            })
        except Exception:
            continue

    return skills


def _parse_frontmatter(content: str) -> dict:
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


# --- Time-of-Day Patterns ---
def compute_time_patterns(records: list[dict]) -> dict[str, dict[int, dict[str, float]]]:
    """Compute success rates by hour of day for each skill."""
    patterns = defaultdict(lambda: defaultdict(lambda: {"successes": 0, "total": 0}))

    for rec in records:
        skill = rec["skill"]
        success = rec["success"]
        ts = datetime.fromisoformat(rec["timestamp"])
        hour = ts.hour

        patterns[skill][hour]["total"] += 1
        if success:
            patterns[skill][hour]["successes"] += 1

    # Convert to rates
    result = {}
    for skill, hours in patterns.items():
        result[skill] = {}
        for hour, data in hours.items():
            rate = data["successes"] / data["total"] if data["total"] > 0 else 0.5
            result[skill][str(hour)] = {
                "rate": round(rate, 3),
                "samples": data["total"],
            }
    return result


# --- Markov Chain ---
def build_markov_chain(records: list[dict]) -> dict[str, dict[str, float]]:
    """Build Markov transition matrix from skill execution sequences."""
    transitions = defaultdict(lambda: defaultdict(int))

    # Sort records by timestamp to build sequences
    sorted_records = sorted(records, key=lambda r: r["timestamp"])

    for i in range(len(sorted_records) - 1):
        current = sorted_records[i]["skill"]
        next_skill = sorted_records[i + 1]["skill"]
        transitions[current][next_skill] += 1

    # Normalize to probabilities
    matrix = {}
    for skill, next_skills in transitions.items():
        total = sum(next_skills.values())
        matrix[skill] = {k: round(v / total, 4) for k, v in next_skills.items()}

    return matrix


def predict_markov_next(current_skill: str, matrix: dict, steps: int = 1) -> dict[str, float]:
    """Predict next skill probabilities after N steps."""
    if current_skill not in matrix:
        return {}

    probs = {current_skill: 1.0}
    for _ in range(steps):
        new_probs = defaultdict(float)
        for skill, p in probs.items():
            if skill in matrix:
                for next_skill, trans_p in matrix[skill].items():
                    new_probs[next_skill] += p * trans_p
        probs = dict(new_probs)

    return probs


# --- Monte Carlo Simulation ---
def monte_carlo_predict(
    skill_name: str,
    system_state: dict,
    records: list[dict],
    priors: dict,
    n_runs: int = DEFAULT_MONTE_CARLO_RUNS,
) -> dict[str, Any]:
    """Run Monte Carlo simulation to predict skill outcome."""
    # Get base success probability from priors/history
    base_prob = _get_base_success_prob(skill_name, records, priors)

    # Apply system state modifiers
    modifier = _compute_system_modifier(system_state)

    # Apply time-of-day modifier
    time_mod = _compute_time_modifier(skill_name, records, system_state["hour_of_day"])

    # Adjusted probability
    adjusted_prob = min(max(base_prob * modifier * time_mod, 0.01), 0.99)

    # Run simulation
    successes = 0
    durations = []
    resource_samples = []

    skill_info = _get_skill_info(skill_name)
    category = skill_info.get("category", "default")
    dur_profile = DURATION_PROFILES.get(category, DURATION_PROFILES["default"])
    res_profile = RESOURCE_PROFILES.get(category, RESOURCE_PROFILES["default"])

    for _ in range(n_runs):
        # Simulate success/failure
        if random.random() < adjusted_prob:
            successes += 1

        # Simulate duration (log-normal distribution)
        mean_log = math.log(dur_profile["mean"])
        std_log = dur_profile["std"] / dur_profile["mean"]
        duration = random.lognormvariate(mean_log, std_log)
        durations.append(duration)

        # Simulate resource consumption
        cpu = res_profile["cpu"] * (0.8 + 0.4 * random.random())
        mem = res_profile["mem"] * (0.8 + 0.4 * random.random())
        disk = res_profile["disk"] * (0.5 + random.random())
        net = res_profile["net"] * (0.5 + random.random())
        resource_samples.append({"cpu": cpu, "mem": mem, "disk": disk, "net": net})

    # Aggregate results
    avg_resources = {
        "cpu_percent": round(sum(r["cpu"] for r in resource_samples) / n_runs, 1),
        "mem_mb": round(sum(r["mem"] for r in resource_samples) / n_runs, 1),
        "disk_io_mb": round(sum(r["disk"] for r in resource_samples) / n_runs, 1),
        "network_kb": round(sum(r["net"] for r in resource_samples) / n_runs, 1),
    }

    return {
        "skill": skill_name,
        "n_runs": n_runs,
        "predicted_success_rate": round(successes / n_runs, 4),
        "base_probability": round(base_prob, 4),
        "system_modifier": round(modifier, 4),
        "time_modifier": round(time_mod, 4),
        "adjusted_probability": round(adjusted_prob, 4),
        "expected_duration_s": round(sum(durations) / n_runs, 2),
        "duration_p50_s": round(sorted(durations)[n_runs // 2], 2),
        "duration_p95_s": round(sorted(durations)[int(n_runs * 0.95)], 2),
        "expected_resources": avg_resources,
        "confidence_interval_95": (
            round(max(0, adjusted_prob - 1.96 * math.sqrt(adjusted_prob * (1 - adjusted_prob) / n_runs)), 4),
            round(min(1, adjusted_prob + 1.96 * math.sqrt(adjusted_prob * (1 - adjusted_prob) / n_runs)), 4),
        ),
    }


def _get_base_success_prob(skill_name: str, records: list[dict], priors: dict) -> float:
    """Get base success probability from history or priors."""
    # Check priors first
    if skill_name in priors:
        skill_prior = priors[skill_name]
        if "expected_success" in skill_prior:
            return skill_prior["expected_success"]

    # Compute from records
    skill_records = [r for r in records if r["skill"] == skill_name]
    if not skill_records:
        return 0.5  # uniform prior

    successes = sum(1 for r in skill_records if r["success"])
    return successes / len(skill_records)


def _compute_system_modifier(state: dict) -> float:
    """Compute success probability modifier based on system state."""
    modifier = 1.0

    # CPU penalty
    cpu = state.get("cpu_percent", 0)
    if cpu > 90:
        modifier *= 0.5
    elif cpu > 70:
        modifier *= 0.8
    elif cpu > 50:
        modifier *= 0.95

    # Memory penalty
    mem = state.get("mem_percent", 0)
    if mem > 90:
        modifier *= 0.6
    elif mem > 80:
        modifier *= 0.85
    elif mem > 60:
        modifier *= 0.95

    # Disk penalty
    disk = state.get("disk_percent", 0)
    if disk > 95:
        modifier *= 0.7
    elif disk > 85:
        modifier *= 0.9

    # Load penalty
    load = state.get("load_avg", [0])[0]
    if load > 4.0:
        modifier *= 0.7
    elif load > 2.0:
        modifier *= 0.9

    return modifier


def _compute_time_modifier(skill_name: str, records: list[dict], hour: int) -> float:
    """Compute time-of-day modifier for a skill."""
    skill_records = [r for r in records if r["skill"] == skill_name]
    if len(skill_records) < 5:
        return 1.0  # Not enough data

    # Get success rate for this hour vs overall
    hour_records = [r for r in skill_records if datetime.fromisoformat(r["timestamp"]).hour == hour]
    if len(hour_records) < 2:
        return 1.0

    hour_success = sum(1 for r in hour_records if r["success"]) / len(hour_records)
    overall_success = sum(1 for r in skill_records if r["success"]) / len(skill_records)

    if overall_success == 0:
        return 1.0

    modifier = hour_success / overall_success
    return min(max(modifier, 0.5), 1.5)  # Clamp to reasonable range


def _get_skill_info(skill_name: str) -> dict:
    """Get skill metadata."""
    skills = load_skills()
    for skill in skills:
        if skill["name"] == skill_name:
            return skill
    return {"name": skill_name, "category": "default"}


# --- Counterfactual Analysis ---
def counterfactual_analysis(
    skill_a: str,
    skill_b: str,
    system_state: dict,
    records: list[dict],
    priors: dict,
    n_runs: int = 500,
) -> dict[str, Any]:
    """Compare outcomes of skill_A vs skill_B (what-if analysis)."""
    result_a = monte_carlo_predict(skill_a, system_state, records, priors, n_runs)
    result_b = monte_carlo_predict(skill_b, system_state, records, priors, n_runs)

    # Compute expected value difference
    ev_a = result_a["predicted_success_rate"] / max(result_a["expected_duration_s"], 0.1)
    ev_b = result_b["predicted_success_rate"] / max(result_b["expected_duration_s"], 0.1)

    # Resource efficiency
    res_eff_a = result_a["predicted_success_rate"] / max(result_a["expected_resources"]["cpu_percent"], 0.1)
    res_eff_b = result_b["predicted_success_rate"] / max(result_b["expected_resources"]["cpu_percent"], 0.1)

    return {
        "skill_A": result_a,
        "skill_B": result_b,
        "comparison": {
            "success_rate_diff": round(result_a["predicted_success_rate"] - result_b["predicted_success_rate"], 4),
            "duration_diff_s": round(result_a["expected_duration_s"] - result_b["expected_duration_s"], 2),
            "efficiency_A": round(ev_a, 4),
            "efficiency_B": round(ev_b, 4),
            "resource_efficiency_A": round(res_eff_a, 4),
            "resource_efficiency_B": round(res_eff_b, 4),
            "recommendation": skill_a if ev_a > ev_b else skill_b,
            "confidence": round(abs(ev_a - ev_b) / max(ev_a + ev_b, 0.001), 4),
        },
    }


# --- Prediction API ---
def predict_skill(
    skill_name: str,
    system_state: Optional[dict] = None,
    n_runs: int = DEFAULT_MONTE_CARLO_RUNS,
) -> dict[str, Any]:
    """Full prediction for a skill: success probability, duration, resources."""
    if system_state is None:
        system_state = get_system_state()

    records = load_db()
    priors = load_priors()

    result = monte_carlo_predict(skill_name, system_state, records, priors, n_runs)

    # Add Markov chain prediction for next skill
    markov = build_markov_chain(records)
    next_skills = predict_markov_next(skill_name, markov, steps=1)
    result["predicted_next_skills"] = dict(sorted(next_skills.items(), key=lambda x: x[1], reverse=True)[:5])

    # Add time pattern info
    time_patterns = compute_time_patterns(records)
    if skill_name in time_patterns:
        hour_key = str(system_state["hour_of_day"])
        if hour_key in time_patterns[skill_name]:
            result["time_pattern"] = time_patterns[skill_name][hour_key]

    return result


# --- CLI ---
def cmd_predict(args):
    """Predict success for a skill."""
    skill_name = args.skill_name
    n_runs = getattr(args, "runs", DEFAULT_MONTE_CARLO_RUNS)

    result = predict_skill(skill_name, n_runs=n_runs)

    print(f"\n{'='*60}")
    print(f"  ATSM World Model Prediction: {skill_name}")
    print(f"{'='*60}")
    print(f"  System State:")
    print(f"    CPU: {result.get('system_modifier', 1.0) * 100:.0f}% load modifier")
    print(f"    Hour: {datetime.now().hour}:00")
    print(f"")
    print(f"  Prediction ({result['n_runs']} Monte Carlo runs):")
    print(f"    P(success) = {result['predicted_success_rate']:.2%}")
    print(f"    95% CI: [{result['confidence_interval_95'][0]:.2%}, {result['confidence_interval_95'][1]:.2%}]")
    print(f"    Base probability: {result['base_probability']:.2%}")
    print(f"    System modifier: {result['system_modifier']:.2f}x")
    print(f"    Time modifier: {result['time_modifier']:.2f}x")
    print(f"")
    print(f"  Expected Duration:")
    print(f"    Mean: {result['expected_duration_s']:.1f}s")
    print(f"    Median: {result['duration_p50_s']:.1f}s")
    print(f"    P95: {result['duration_p95_s']:.1f}s")
    print(f"")
    print(f"  Expected Resources:")
    res = result["expected_resources"]
    print(f"    CPU: {res['cpu_percent']:.1f}%")
    print(f"    Memory: {res['mem_mb']:.0f} MB")
    print(f"    Disk I/O: {res['disk_io_mb']:.1f} MB")
    print(f"    Network: {res['network_kb']:.0f} KB")

    if result.get("predicted_next_skills"):
        print(f"\n  Predicted Next Skills (Markov):")
        for skill, prob in list(result["predicted_next_skills"].items())[:3]:
            print(f"    {skill}: {prob:.1%}")

    if result.get("time_pattern"):
        tp = result["time_pattern"]
        print(f"\n  Time-of-Day Pattern:")
        print(f"    This hour success rate: {tp['rate']:.1%} (n={tp['samples']})")

    print(f"{'='*60}\n")
    return result


def cmd_simulate(args):
    """Simulate a task with Monte Carlo."""
    task = args.task
    n_runs = getattr(args, "runs", 100)

    print(f"\n{'='*60}")
    print(f"  ATSM World Model Simulation")
    print(f"  Task: \"{task}\"")
    print(f"  Runs: {n_runs}")
    print(f"{'='*60}")

    # Rank skills for the task
    from atsm import rank_skills
    rankings, elapsed = rank_skills(task, top_k=5)

    if not rankings:
        print("  No matching skills found.")
        return

    system_state = get_system_state()
    records = load_db()
    priors = load_priors()

    print(f"\n  Top skill candidates:")
    print(f"  {'Rank':<5} {'Skill':<25} {'P(success)':<12} {'E[dur]s':<10} {'E[CPU%]':<10}")
    print(f"  {'-'*62}")

    for i, r in enumerate(rankings, 1):
        sim = monte_carlo_predict(r["name"], system_state, records, priors, n_runs)
        print(f"  {i:<5} {r['name']:<25} {sim['predicted_success_rate']:<12.2%} "
              f"{sim['expected_duration_s']:<10.1f} {sim['expected_resources']['cpu_percent']:<10.1f}")

    # Best choice
    best = rankings[0]
    best_sim = monte_carlo_predict(best["name"], system_state, records, priors, n_runs)
    print(f"\n  ✅ Recommended: {best['name']}")
    print(f"     P(success) = {best_sim['predicted_success_rate']:.2%}")
    print(f"     E[duration] = {best_sim['expected_duration_s']:.1f}s")
    print(f"     E[CPU] = {best_sim['expected_resources']['cpu_percent']:.1f}%")
    print(f"{'='*60}\n")


def cmd_counterfactual(args):
    """Compare two skills."""
    skill_a = args.skill_A
    skill_b = args.skill_B
    n_runs = getattr(args, "runs", 500)

    system_state = get_system_state()
    records = load_db()
    priors = load_priors()

    result = counterfactual_analysis(skill_a, skill_b, system_state, records, priors, n_runs)

    print(f"\n{'='*60}")
    print(f"  ATSM Counterfactual Analysis")
    print(f"  {skill_a} vs {skill_b}")
    print(f"{'='*60}")

    for label, key in [(skill_a, "skill_A"), (skill_b, "skill_B")]:
        r = result[key]
        print(f"\n  {label}:")
        print(f"    P(success) = {r['predicted_success_rate']:.2%}")
        print(f"    E[duration] = {r['expected_duration_s']:.1f}s")
        print(f"    E[CPU] = {r['expected_resources']['cpu_percent']:.1f}%")
        print(f"    E[Mem] = {r['expected_resources']['mem_mb']:.0f}MB")

    comp = result["comparison"]
    print(f"\n  Comparison:")
    print(f"    Δ P(success) = {comp['success_rate_diff']:+.2%}")
    print(f"    Δ Duration = {comp['duration_diff_s']:+.1f}s")
    print(f"    Efficiency A = {comp['efficiency_A']:.4f}")
    print(f"    Efficiency B = {comp['efficiency_B']:.4f}")
    print(f"")
    print(f"  ✅ Recommendation: {comp['recommendation']}")
    print(f"     Confidence: {comp['confidence']:.2%}")
    print(f"{'='*60}\n")


def cmd_state(args):
    """Show current world state."""
    system_state = get_system_state()
    world_state = load_world_state()
    records = load_db()
    priors = load_priors()

    print(f"\n{'='*60}")
    print(f"  ATSM World State")
    print(f"{'='*60}")

    print(f"\n  System State:")
    print(f"    CPU: {system_state['cpu_percent']:.1f}%")
    print(f"    Memory: {system_state['mem_percent']:.1f}% ({system_state['mem_available_mb']:.0f} MB free)")
    print(f"    Disk: {system_state['disk_percent']:.1f}% ({system_state['disk_free_gb']:.1f} GB free)")
    print(f"    Load: {system_state['load_avg']}")
    print(f"    Time: {system_state['hour_of_day']}:00 (day {system_state['day_of_week']})")

    print(f"\n  Historical Data:")
    print(f"    Total records: {len(records)}")
    print(f"    Skills tracked: {len(priors) - 2}")  # exclude _meta and _global

    # Markov chain summary
    markov = build_markov_chain(records)
    if markov:
        print(f"\n  Markov Chain:")
        print(f"    States: {len(markov)}")
        top_transitions = []
        for skill, nexts in markov.items():
            for next_skill, prob in nexts.items():
                top_transitions.append((skill, next_skill, prob))
        top_transitions.sort(key=lambda x: x[2], reverse=True)
        print(f"    Top transitions:")
        for src, dst, prob in top_transitions[:5]:
            print(f"      {src} → {dst}: {prob:.1%}")

    # Time patterns summary
    time_patterns = compute_time_patterns(records)
    if time_patterns:
        print(f"\n  Time-of-Day Patterns:")
        print(f"    Skills with patterns: {len(time_patterns)}")

    print(f"\n  World State File: {WORLD_STATE_FILE}")
    print(f"  Last Updated: {world_state.get('last_updated', 'never')}")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(
        description="ATSM World Model: Internal simulation for outcome prediction"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Predict success for a skill")
    predict_parser.add_argument("skill_name", help="Name of the skill to predict")
    predict_parser.add_argument("--runs", type=int, default=DEFAULT_MONTE_CARLO_RUNS,
                                help="Number of Monte Carlo runs")

    # simulate
    simulate_parser = subparsers.add_parser("simulate", help="Simulate a task")
    simulate_parser.add_argument("task", help="Task description")
    simulate_parser.add_argument("--runs", type=int, default=100,
                                 help="Number of Monte Carlo runs per skill")

    # counterfactual
    cf_parser = subparsers.add_parser("counterfactual", help="Compare two skills")
    cf_parser.add_argument("skill_A", help="First skill")
    cf_parser.add_argument("skill_B", help="Second skill")
    cf_parser.add_argument("--runs", type=int, default=500,
                           help="Number of Monte Carlo runs per skill")

    # state
    subparsers.add_parser("state", help="Show current world state")

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    # Ensure data directory exists
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.command == "predict":
        cmd_predict(args)
    elif args.command == "simulate":
        cmd_simulate(args)
    elif args.command == "counterfactual":
        cmd_counterfactual(args)
    elif args.command == "state":
        cmd_state(args)


if __name__ == "__main__":
    main()
