#!/usr/bin/env python3
"""Multi-Model Orchestrator — routes tasks to the optimal model.

Maintains a model pool (local GGUF + cloud free tier), picks based on
task characteristics, tracks cost/latitude, auto-fallback on failure,
and integrates with ATSM to learn which model works best per skill.

Usage:
    python3 model_router.py list
    python3 model_router.py pick "task description"
    python3 model_router.py pick --skill arxiv "search and summarize papers"
    python3 model_router.py record <model> <0|1> [--skill NAME] [--latency MS] [--cost USD]]
    python3 model_router.py stats
    python3 model_router.py fallback  # show fallback chain for a task

Model selection logic:
    1. Thai/multilingual tokens detected → multilingual model
    2. Complexity score high → larger cloud model
    3. Simple/fast → local GGUF
    4. ATSM learning override if we have history for the skill
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
DATA_DIR = Path(__file__).parent.parent / "data"
MODELS_FILE = DATA_DIR / "model_router_models.json"
STATS_FILE = DATA_DIR / "model_router_stats.jsonl"
PRIORS_FILE = DATA_DIR / "model_router_priors.json"
ATSM_DB_FILE = DATA_DIR / "atsm_db.jsonl"
ATSM_PRIORS_FILE = DATA_DIR / "atsm_priors.json"

# Default model pool
DEFAULT_MODELS = {
    "local": {
        "id": "hermes-4-36b-q3_k_m",
        "provider": "local",
        "engine": "llama.cpp",
        "quant": "Q3_K_M",
        "params": 36,
        "ctx": 8192,
        "cost_per_1k": 0.0,
        "speed": "medium",
        "strengths": ["simple", "fast", "private", "offline"],
        "supports_thai": False,
        "endpoint": None,  # local llama-server or CLI
    },
    "cloud_free_qwen": {
        "id": "qwen/qwen-2.5-72b-instruct:free",
        "provider": "openrouter",
        "engine": "openrouter",
        "quant": "fp16",
        "params": 72,
        "ctx": 32768,
        "cost_per_1k": 0.0,
        "speed": "medium",
        "strengths": ["complex", "reasoning", "multilingual", "thai"],
        "supports_thai": True,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    },
    "cloud_free_llama": {
        "id": "meta-llama/llama-3.1-70b-instruct:free",
        "provider": "openrouter",
        "engine": "openrouter",
        "quant": "fp16",
        "params": 70,
        "ctx": 16384,
        "cost_per_1k": 0.0,
        "speed": "fast",
        "strengths": ["complex", "reasoning", "code"],
        "supports_thai": False,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    },
    "cloud_free_gemini": {
        "id": "google/gemini-2.0-flash-001:free",
        "provider": "openrouter",
        "engine": "openrouter",
        "quant": "fp16",
        "params": 200,
        "ctx": 8192,
        "cost_per_1k": 0.0,
        "speed": "very_fast",
        "strengths": ["complex", "reasoning", "speed"],
        "supports_thai": True,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    },
    "cloud_free_mistral": {
        "id": "mistralai/mistral-7b-instruct:free",
        "provider": "openrouter",
        "engine": "openrouter",
        "quant": "fp16",
        "params": 7,
        "ctx": 32768,
        "cost_per_1k": 0.0,
        "speed": "very_fast",
        "strengths": ["simple", "fast", "cheap"],
        "supports_thai": False,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    },
    "cloud_free_phi": {
        "id": "microsoft/phi-3-medium-128k-instruct:free",
        "provider": "openrouter",
        "engine": "openrouter",
        "quant": "fp16",
        "params": 14,
        "ctx": 128000,
        "cost_per_1k": 0.0,
        "speed": "fast",
        "strengths": ["long_context", "reasoning"],
        "supports_thai": False,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    },
}

# Complexity indicators — task keywords suggesting heavier models
COMPLEXITY_KEYWORDS = {
    "high": [
        "reasoning", "analyze", "analysis", "complex", "deep", "summarize",
        "compare", "contrast", "evaluate", "critique", "synthesize",
        "research", "investigate", "diagnose", "debug", "architect",
        "prove", "derive", "theorem", "algorithm", "optimization",
        "mathematical", "equation", "proof", "scientific", "hypothesis",
    ],
    "medium": [
        "explain", "describe", "classify", "translate", "rewrite",
        "convert", "extract", "generate", "create", "write", "draft",
        "search", "find", "list", "compare", "build", "design",
    ],
    "low": [
        "hello", "hi", "what", "simple", "quick", "yes", "no",
        "ok", "thanks", "short", "brief", "one word", "trivial",
    ],
}

# Thai detection
THAI_RE = re.compile(r"[\u0E00-\u0E7F]")
TOKEN_RE = re.compile(r"[a-zA-Z0-9_\u0E00-\u0E7F]+", re.UNICODE)


# --- Model pool management ---
def load_models() -> dict:
    """Load model pool from JSON, falling back to defaults."""
    if MODELS_FILE.exists():
        with open(MODELS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return DEFAULT_MODELS.copy()


def save_models(models: dict):
    """Save model pool to JSON atomically."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = MODELS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(models, f, indent=2, ensure_ascii=False)
    tmp.rename(MODELS_FILE)


# --- Statistics ---
def load_stats() -> list[dict]:
    """Load outcome records from JSONL."""
    if not STATS_FILE.exists():
        return []
    records = []
    with open(STATS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return records


def record_outcome(model_id: str, success: int, skill: str = None,
                   latency_ms: float = None, cost_usd: float = None,
                   task: str = None):
    """Record a model outcome."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rec = {
        "model": model_id,
        "success": int(success),
        "skill": skill or "_none",
        "latency_ms": latency_ms,
        "cost_usd": cost_usd,
        "task": task or "",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    with open(STATS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")


def compute_model_stats(records: list[dict]) -> dict[str, dict]:
    """Aggregate stats per model."""
    now = datetime.now(timezone.utc)
    stats = defaultdict(lambda: {
        "successes": 0, "failures": 0, "total": 0,
        "total_latency": 0.0, "total_cost": 0.0,
        "latency_count": 0,
    })
    for rec in records:
        m = rec["model"]
        s = stats[m]
        s["total"] += 1
        if rec["success"]:
            s["successes"] += 1
        else:
            s["failures"] += 1
        if rec.get("latency_ms") is not None:
            s["total_latency"] += rec["latency_ms"]
            s["latency_count"] += 1
        if rec.get("cost_usd") is not None:
            s["total_cost"] += rec["cost_usd"]
    result = {}
    for m, s in stats.items():
        avg_lat = s["total_latency"] / s["latency_count"] if s["latency_count"] > 0 else None
        result[m] = {
            "successes": s["successes"],
            "failures": s["failures"],
            "total": s["total"],
            "success_rate": s["successes"] / s["total"] if s["total"] > 0 else 0.0,
            "avg_latency_ms": round(avg_lat, 1) if avg_lat else None,
            "total_cost_usd": round(s["total_cost"], 6),
        }
    return result


# --- Task analysis ---
def detect_thai(text: str) -> bool:
    """Check if text contains Thai characters."""
    return bool(THAI_RE.search(text))


def tokenize_task(text: str) -> list[str]:
    """Tokenize task for analysis."""
    return [t.lower() for t in TOKEN_RE.findall(text)]


def complexity_score(tokens: list[str]) -> float:
    """Score task complexity 0-1 based on keywords."""
    text_lower = " ".join(tokens)
    high_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["high"] if kw in text_lower)
    med_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["medium"] if kw in text_lower)
    low_hits = sum(1 for kw in COMPLEXITY_KEYWORDS["low"] if kw in text_lower)

    if high_hits > 0:
        return min(0.5 + 0.15 * high_hits + 0.05 * med_hits, 1.0)
    elif med_hits > 0:
        return min(0.3 + 0.1 * med_hits - 0.05 * low_hits, 0.6)
    elif low_hits > 0:
        return max(0.1, 0.2 - 0.05 * low_hits)
    return 0.3  # default medium-low


# --- ATSM Integration ---
def load_atsm_skill_models() -> dict[str, dict]:
    """Load ATSM data to find which models work best per skill.

    Reads atsm_priors.json (if it has model info) and atsm_db.jsonl.
    Returns skill -> {model: success_rate}.
    """
    skill_models = defaultdict(lambda: defaultdict(lambda: {"successes": 0, "total": 0}))

    # Check if ATSM DB has model-router records
    if ATSM_DB_FILE.exists():
        with open(ATSM_DB_FILE, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    skill = rec.get("skill", "")
                    model = rec.get("model", "")
                    if model and skill:
                        skill_models[skill][model]["total"] += 1
                        if rec.get("success"):
                            skill_models[skill][model]["successes"] += 1
                except json.JSONDecodeError:
                    continue

    # Convert to success rates
    result = {}
    for skill, models in skill_models.items():
        result[skill] = {}
        for m, counts in models.items():
            if counts["total"] > 0:
                result[skill][m] = counts["successes"] / counts["total"]
    return result


# --- Model selection ---
def pick_model(task: str, skill: str = None, verbose: bool = False) -> dict:
    """Pick the optimal model for a task.

    Selection rules (in priority order):
    1. ATSM history: if skill has a known best model, use it
    2. Thai detected → multilingual model
    3. High complexity → large cloud model
    4. Low complexity → local model
    5. Medium complexity → fast cloud model
    """
    models = load_models()
    tokens = tokenize_task(task)
    is_thai = detect_thai(task)
    complexity = complexity_score(tokens)
    atsm_skill_models = load_atsm_skill_models() if skill else {}

    # Rule 1: ATSM override
    if skill and skill in atsm_skill_models:
        best_model_id = max(atsm_skill_models[skill], key=atsm_skill_models[skill].get)
        best_rate = atsm_skill_models[skill][best_model_id]
        if best_rate >= 0.6 and best_model_id in models:
            if verbose:
                print(f"  📚 ATSM override: {skill} → {best_model_id} (success rate: {best_rate:.0%})")
            return {"model": models[best_model_id], "reason": "atsm_history", "score": best_rate}

    candidates = []

    for name, model in models.items():
        score = 0.0
        reasons = []

        # Thai matching
        if is_thai and model.get("supports_thai"):
            score += 3.0
            reasons.append("thai_support")
        elif is_thai and not model.get("supports_thai"):
            score -= 2.0  # penalty

        # Complexity matching
        params = model.get("params", 7)
        if complexity >= 0.6:
            # Prefer large models
            if params >= 50:
                score += 2.0
                reasons.append("high_params")
            elif params >= 20:
                score += 1.0
            # Reasoning strength
            if "reasoning" in model.get("strengths", []):
                score += 1.0
                reasons.append("reasoning")
        elif complexity <= 0.2:
            # Prefer fast/local models
            if model["provider"] == "local":
                score += 2.5
                reasons.append("local_fast")
            if params <= 15:
                score += 1.0
                reasons.append("lightweight")
            if "fast" in model.get("speed", ""):
                score += 0.5
        else:
            # Medium: prefer cloud free fast models
            if model["provider"] == "cloud" and model.get("cost_per_1k", 0) == 0:
                score += 1.0
                reasons.append("free_cloud")
            if "fast" in model.get("speed", ""):
                score += 0.5

        # Historical success rate for this model
        stats = compute_model_stats(load_stats())
        if name in stats and stats[name]["total"] >= 3:
            success_rate = stats[name]["success_rate"]
            # Boost models with proven track record
            score += success_rate * 1.5
            if success_rate >= 0.8:
                reasons.append(f"high_sr({success_rate:.0%})")
            elif success_rate < 0.3:
                score -= 1.5
                reasons.append(f"low_sr({success_rate:.0%})")

        # Prefer free models
        if model.get("cost_per_1k", 0) == 0:
            score += 0.3

        candidates.append((score, name, model, reasons))

    # Sort by score descending
    candidates.sort(key=lambda x: x[0], reverse=True)

    if not candidates:
        return {"model": None, "reason": "no_models", "score": 0}

    best_score, best_name, best_model, best_reasons = candidates[0]

    # Build fallback chain (ordered list of alternatives)
    fallback_chain = []
    for score, name, model, reasons in candidates[1:]:
        fallback_chain.append({
            "name": name,
            "model": model,
            "score": score,
            "reasons": reasons,
        })

    return {
        "model": best_model,
        "model_key": best_name,
        "reason": " → ".join(best_reasons) if best_reasons else "default",
        "score": best_score,
        "fallback_chain": fallback_chain,
        "complexity": complexity,
        "is_thai": is_thai,
    }


# --- CLI commands ---
def cmd_list():
    """List all models in the pool."""
    models = load_models()
    stats = compute_model_stats(load_stats())

    print(f"\n{'Key':<25} {'Model ID':<45} {'Provider':<10} {'Cost/1K':<10} {'Thai':<6}")
    print("─" * 110)

    for name, model in sorted(models.items()):
        mstat = stats.get(name, {})
        sr = f" ({mstat.get('success_rate', 0):.0%})" if mstat.get("total", 0) > 0 else ""
        print(f"{name:<25} {model['id']:<45} {model['provider']:<10} "
              f"${model.get('cost_per_1k', 0):<9.4f} {'✓' if model.get('supports_thai') else '✗':<6}{sr}")
        strengths = ", ".join(model.get("strengths", []))
        print(f"  └─ {strengths} | speed={model.get('speed', '?')} | params={model.get('params', '?')}B")

    print(f"\n  {len(models)} models in pool")


def cmd_pick(task: str, skill: str = None, verbose: bool = True):
    """Pick the best model for a task."""
    result = pick_model(task, skill=skill, verbose=verbose)

    if result["model"] is None:
        print("No models available.")
        return

    model = result["model"]

    if verbose:
        print(f"\n📝 Task: {task[:80]}")
        if skill:
            print(f"   Skill: {skill}")
        print(f"   Complexity: {result.get('complexity', '?'):.2f} | "
              f"Thai: {'✓' if result.get('is_thai') else '✗'}")
        print(f"\n🏆 Selected: {model['id']}")
        print(f"   Provider: {model['provider']} | Engine: {model.get('engine', '?')}")
        print(f"   Speed: {model.get('speed', '?')} | "
              f"Cost: ${model.get('cost_per_1k', 0):.4f}/1K tokens")
        print(f"   Reason: {result['reason']}")

        if result.get("fallback_chain"):
            print(f"\n📋 Fallback chain:")
            for i, fb in enumerate(result["fallback_chain"][:3], 1):
                print(f"   {i}. {fb['model']['id']} (score={fb['score']:.2f})")

    # Output machine-readable JSON for piping
    print(json.dumps({
        "model_id": model["id"],
        "provider": model["provider"],
        "endpoint": model.get("endpoint"),
        "reason": result["reason"],
        "score": round(result["score"], 3),
        "fallback": [fb["model"]["id"] for fb in result.get("fallback_chain", [])[:3]],
    }, indent=2))


def cmd_record(model_id: str, success: int, skill: str = None,
               latency_ms: float = None, cost_usd: float = None, task: str = None):
    """Record a model outcome."""
    if success not in (0, 1):
        print("Success must be 0 or 1")
        sys.exit(1)
    record_outcome(model_id, success, skill=skill, latency_ms=latency_ms,
                   cost_usd=cost_usd, task=task)
    print(f"Recorded: {model_id} → {'✓ success' if success else '✗ failure'}")
    if skill:
        print(f"  Skill: {skill}")


def cmd_stats():
    """Show model usage statistics."""
    records = load_stats()
    if not records:
        print("No records yet. Use 'record' to log outcomes.")
        return

    stats = compute_model_stats(records)

    print(f"\n{'Model':<30} {'S':<5} {'F':<5} {'SR':<7} {'Avg Lat':<10} {'Cost':<12} {'N':<5}")
    print("─" * 85)

    for name, s in sorted(stats.items(), key=lambda x: x[1]["successes"], reverse=True):
        lat_str = f"{s['avg_latency_ms']:.0f}ms" if s["avg_latency_ms"] else "?"
        print(f"{name:<30} {s['successes']:<5} {s['failures']:<5} "
              f"{s['success_rate']:<7.0%} {lat_str:<10} "
              f"${s['total_cost_usd']:<11.6f} {s['total']:<5}")

    # Skill breakdown
    skill_stats = defaultdict(lambda: defaultdict(int))
    for rec in records:
        if rec.get("skill") and rec["skill"] != "_none":
            skill_stats[rec["skill"]][rec["model"]] += 1

    if skill_stats:
        print(f"\n  📚 Per-skill model usage:")
        for skill, models in sorted(skill_stats.items()):
            best = max(models, key=models.get)
            print(f"    {skill}: {dict(models)} → best: {best}")

    print(f"\n  Total records: {len(records)}")
    print(f"  Tracked: {MODELS_FILE.name if MODELS_FILE.exists() else 'in-memory defaults'}")


def cmd_fallback(task: str, skill: str = None):
    """Show the full fallback chain for a task."""
    result = pick_model(task, skill=skill, verbose=False)
    if not result["model"]:
        print("No models available.")
        return

    print(f"\n📝 Task: {task[:60]}")
    print(f"   Fallback chain (priority order):")
    print(f"   0. 🏆 {result['model']['id']} ({result['reason']})")
    for i, fb in enumerate(result.get("fallback_chain", []), 1):
        print(f"   {i}. {fb['model']['id']} (score={fb['score']:.2f})")


# --- Main ---
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "list":
        cmd_list()

    elif command == "pick":
        # Parse --skill flag
        skill = None
        if "--skill" in args:
            idx = args.index("--skill")
            if idx + 1 < len(args):
                skill = args[idx + 1]
                args = args[:idx] + args[idx + 2:]
        task = " ".join(args)
        if not task:
            print("Usage: model_router.py pick [--skill NAME] <task description>")
            sys.exit(1)
        cmd_pick(task, skill=skill)

    elif command == "record":
        if len(args) < 2:
            print("Usage: model_router.py record <model> <0|1> [--skill NAME] [--latency MS] [--cost USD] [--task 'desc']")
            sys.exit(1)
        model_id = args[0]
        success = int(args[1])
        skill = None
        latency_ms = None
        cost_usd = None
        task = None
        i = 2
        while i < len(args):
            if args[i] == "--skill" and i + 1 < len(args):
                skill = args[i + 1]; i += 2
            elif args[i] == "--latency" and i + 1 < len(args):
                latency_ms = float(args[i + 1]); i += 2
            elif args[i] == "--cost" and i + 1 < len(args):
                cost_usd = float(args[i + 1]); i += 2
            elif args[i] == "--task" and i + 1 < len(args):
                task = args[i + 1]; i += 2
            else:
                i += 1
        cmd_record(model_id, success, skill=skill, latency_ms=latency_ms,
                   cost_usd=cost_usd, task=task)

    elif command == "stats":
        cmd_stats()

    elif command == "fallback":
        skill = None
        if "--skill" in args:
            idx = args.index("--skill")
            if idx + 1 < len(args):
                skill = args[idx + 1]
                args = args[:idx] + args[idx + 2:]
        task = " ".join(args)
        if not task:
            print("Usage: model_router.py fallback [--skill NAME] <task description>")
            sys.exit(1)
        cmd_fallback(task, skill=skill)

    else:
        print(f"Unknown command: {command}")
        print(f"Available: list, pick, record, stats, fallback")
        sys.exit(1)


if __name__ == "__main__":
    main()
