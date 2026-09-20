#!/usr/bin/env python3
"""ATSM Self-Learning Analysis Engine.

Reads outcome history, detects patterns, generates insights,
and auto-updates priors for the ATSM ranking engine.

Usage:
    python3 atsm_learn.py analyze  → print insights report (JSON)
    python3 atsm_learn.py priors   → auto-update atsm_priors.json
    python3 atsm_learn.py patterns → list detected patterns
"""
import json
import math
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from itertools import groupby

# --- Config ---
DATA_DIR = Path(__file__).parent.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"
PRIORS_FILE = DATA_DIR / "atsm_priors.json"

# Learning hyperparameters
DECAY_LAMBDA = 0.01        # recency decay per day
MIN_OBSERVATIONS = 3       # min observations before trusting rate
STREAK_THRESHOLD = 3       # consecutive successes/failures to flag
CONFIDENCE_THRESHOLD = 0.5 # min confidence to trust a skill's rate
RETRY_COOLDOWN_HOURS = 24  # hours before a failed skill is "due for retry"
TOP_K_SKILLS = 5           # top co-failing skills to consider


# --- Data Loading ---
def load_outcomes() -> list[dict]:
    """Load all outcome records from the JSONL database."""
    if not DB_FILE.exists():
        return []
    records = []
    with open(DB_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rec = json.loads(line)
                    rec["_ts"] = datetime.fromisoformat(rec["timestamp"])
                    records.append(rec)
                except (json.JSONDecodeError, KeyError):
                    continue
    return records


def load_priors() -> dict:
    """Load existing priors file, or return empty dict."""
    if PRIORS_FILE.exists():
        with open(PRIORS_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_priors(priors: dict):
    """Save priors to JSON file atomically."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tmp = PRIORS_FILE.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(priors, f, indent=2, ensure_ascii=False)
    tmp.rename(PRIORS_FILE)


# --- Pattern Detection ---
def detect_co_failures(records: list[dict]) -> list[dict]:
    """Identify skills that tend to fail together within a time window.

    Groups records into sessions (within 5min of each other) and finds
    skill pairs that co-fail more often than expected.
    """
    if not records:
        return []

    # Sort by timestamp
    sorted_recs = sorted(records, key=lambda r: r["_ts"])

    # Group into sessions: records within 5 minutes of each other
    sessions = []
    current_session = [sorted_recs[0]]
    for rec in sorted_recs[1:]:
        gap = (rec["_ts"] - current_session[-1]["_ts"]).total_seconds()
        if gap <= 300:  # 5-minute session window
            current_session.append(rec)
        else:
            sessions.append(current_session)
            current_session = [rec]
    sessions.append(current_session)

    # For each session, find which skills failed together
    pair_counts = Counter()
    pair_co_fail = Counter()
    skill_fail_counts = Counter()
    skill_total_counts = Counter()

    for session in sessions:
        failed_skills = set()
        for rec in session:
            skill = rec["skill"]
            skill_total_counts[skill] += 1
            if rec["success"] == 0:
                skill_fail_counts[skill] += 1
                failed_skills.add(skill)

        # Count all pairs that failed in the same session
        failed_list = sorted(failed_skills)
        for i in range(len(failed_list)):
            for j in range(i + 1, len(failed_list)):
                pair = (failed_list[i], failed_list[j])
                pair_co_fail[pair] += 1

    # Compute lift: P(A and B fail) / (P(A fails) * P(B fails))
    patterns = []
    for (s1, s2), co_fail_count in pair_co_fail.items():
        p_s1_fail = skill_fail_counts[s1] / max(skill_total_counts[s1], 1)
        p_s2_fail = skill_fail_counts[s2] / max(skill_total_counts[s2], 1)
        expected = p_s1_fail * p_s2_fail * len(sessions)
        if expected > 0:
            lift = co_fail_count / expected
            if lift > 1.0 and co_fail_count >= 2:
                patterns.append({
                    "type": "co_failure",
                    "skills": [s1, s2],
                    "co_fail_count": co_fail_count,
                    "lift": round(lift, 3),
                    "description": f"'{s1}' and '{s2}' fail together {co_fail_count}× (lift={lift:.2f})"
                })

    patterns.sort(key=lambda p: p["lift"], reverse=True)
    return patterns


def detect_time_of_day_effects(records: list[dict]) -> list[dict]:
    """Detect time-of-day effects on skill success rates.

    Buckets records by UTC hour and computes success rate per bucket.
    Flags hours with significantly deviating rates.
    """
    if not records:
        return []

    hourly = defaultdict(lambda: {"success": 0, "failure": 0})
    for rec in records:
        hour = rec["_ts"].hour
        if rec["success"]:
            hourly[hour]["success"] += 1
        else:
            hourly[hour]["failure"] += 1

    patterns = []
    global_success = sum(1 for r in records if r["success"])
    global_rate = global_success / max(len(records), 1)

    for hour in sorted(hourly.keys()):
        bucket = hourly[hour]
        total = bucket["success"] + bucket["failure"]
        if total < 2:
            continue
        rate = bucket["success"] / total
        deviation = rate - global_rate
        if abs(deviation) > 0.2:  # 20% deviation threshold
            patterns.append({
                "type": "time_of_day",
                "hour": hour,
                "success_rate": round(rate, 3),
                "global_rate": round(global_rate, 3),
                "deviation": round(deviation, 3),
                "sample_size": total,
                "description": f"Hour {hour:02d}:00 UTC — {rate:.0%} success (vs {global_rate:.0%} global)"
            })

    return patterns


def detect_streaks(records: list[dict]) -> list[dict]:
    """Detect success/failure streaks per skill.

    A streak is STREAK_THRESHOLD or more consecutive identical outcomes.
    """
    if not records:
        return []

    sorted_recs = sorted(records, key=lambda r: r["_ts"])
    by_skill = defaultdict(list)
    for rec in sorted_recs:
        by_skill[rec["skill"]].append(rec)

    patterns = []
    for skill, skill_recs in by_skill.items():
        # Find runs using groupby on consecutive outcomes
        outcomes = [r["success"] for r in skill_recs]
        for outcome, group in groupby(outcomes):
            streak_len = sum(1 for _ in group)
            if streak_len >= STREAK_THRESHOLD:
                patterns.append({
                    "type": "streak",
                    "skill": skill,
                    "outcome": "success" if outcome else "failure",
                    "streak_length": streak_len,
                    "description": f"'{skill}' has a {streak_len}-run of consecutive {'successes' if outcome else 'failures'}"
                })

    patterns.sort(key=lambda p: p["streak_length"], reverse=True)
    return patterns


def detect_recency_trends(records: list[dict]) -> list[dict]:
    """Detect whether a skill's success rate is trending up or down over time.

    Compares success rate in first half vs second half of observations.
    """
    if not records:
        return []

    sorted_recs = sorted(records, key=lambda r: r["_ts"])
    by_skill = defaultdict(list)
    for rec in sorted_recs:
        by_skill[rec["skill"]].append(rec)

    patterns = []
    for skill, skill_recs in by_skill.items():
        if len(skill_recs) < 4:
            continue
        mid = len(skill_recs) // 2
        first_half = skill_recs[:mid]
        second_half = skill_recs[mid:]

        rate_first = sum(1 for r in first_half if r["success"]) / len(first_half)
        rate_second = sum(1 for r in second_half if r["success"]) / len(second_half)
        delta = rate_second - rate_first

        if abs(delta) > 0.15:  # 15% swing threshold
            direction = "improving" if delta > 0 else "declining"
            patterns.append({
                "type": "recency_trend",
                "skill": skill,
                "direction": direction,
                "first_half_rate": round(rate_first, 3),
                "second_half_rate": round(rate_second, 3),
                "delta": round(delta, 3),
                "description": f"'{skill}' is {direction}: {rate_first:.0%} → {rate_second:.0%}"
            })

    patterns.sort(key=lambda p: abs(p["delta"]), reverse=True)
    return patterns


def detect_due_for_retry(records: list[dict]) -> list[dict]:
    """Identify skills that recently failed but historically succeed.

    A skill is "due for retry" if its last outcome was failure,
    historical success rate > 50%, and last failure was within cooldown.
    """
    if not records:
        return []

    now = datetime.now(timezone.utc)
    sorted_recs = sorted(records, key=lambda r: r["_ts"])
    by_skill = defaultdict(list)
    for rec in sorted_recs:
        by_skill[rec["skill"]].append(rec)

    recommendations = []
    for skill, skill_recs in by_skill.items():
        last = skill_recs[-1]
        if last["success"]:
            continue  # last was success, not due for retry

        total = len(skill_recs)
        successes = sum(1 for r in skill_recs if r["success"])
        rate = successes / total

        if rate < 0.3:
            continue  # historically bad, don't recommend retry

        hours_since = (now - last["_ts"]).total_seconds() / 3600
        if hours_since > RETRY_COOLDOWN_HOURS:
            continue  # too long ago

        recommendations.append({
            "type": "retry_candidate",
            "skill": skill,
            "historical_rate": round(rate, 3),
            "total_attempts": total,
            "hours_since_failure": round(hours_since, 1),
            "description": f"'{skill}' historically succeeds {rate:.0%} but just failed — retry suggested"
        })

    recommendations.sort(key=lambda r: r["historical_rate"], reverse=True)
    return recommendations


# --- Recommendation Generation ---
def generate_recommendations(patterns: list[dict], records: list[dict]) -> list[dict]:
    """Generate actionable 'Try X when Y fails' recommendations from patterns."""
    recs = []

    # From co-failure patterns: when A fails, try B
    co_fail = [p for p in patterns if p["type"] == "co_failure"]
    for pat in co_fail[:TOP_K_SKILLS]:
        s1, s2 = pat["skills"]
        recs.append({
            "rule": "fallback_suggestion",
            "trigger": f"'{s1}' fails",
            "action": f"Consider '{s2}' as alternative (co-failure lift={pat['lift']})",
            "confidence": min(pat["lift"] / 3.0, 1.0),
        })

    # From time-of-day patterns: avoid certain skills at certain hours
    tod = [p for p in patterns if p["type"] == "time_of_day"]
    for pat in tod:
        if pat["deviation"] < 0:
            recs.append({
                "rule": "time_avoidance",
                "trigger": f"Task at hour {pat['hour']:02d}:00 UTC",
                "action": f"Avoid low-success skills (rate drops to {pat['success_rate']:.0%})",
                "confidence": min(abs(pat["deviation"]) / 0.5, 1.0),
            })

    # From retry candidates
    retries = [p for p in patterns if p["type"] == "retry_candidate"]
    for pat in retries:
        recs.append({
            "rule": "scheduled_retry",
            "trigger": f"'{pat['skill']}' failed {pat['hours_since_failure']:.0f}h ago",
            "action": f"Retry now (historical success rate: {pat['historical_rate']:.0%})",
            "confidence": pat["historical_rate"],
        })

    return recs


# --- Prior Update ---
def compute_updated_priors(records: list[dict], existing_priors: dict) -> dict:
    """Compute and return updated Beta priors based on outcome history.

    For each skill, posterior alpha = prior_alpha + weighted_successes,
    posterior beta = prior_beta + weighted_failures.
    """
    now = datetime.now(timezone.utc)
    prior_alpha = existing_priors.get("_global", {}).get("alpha", 1.0)
    prior_beta = existing_priors.get("_global", {}).get("beta", 1.0)

    by_skill = defaultdict(lambda: {"successes": 0.0, "failures": 0.0, "total": 0})
    for rec in records:
        skill = rec["skill"]
        ts = rec["_ts"]
        age_days = max((now - ts).total_seconds() / 86400, 0)
        weight = math.exp(-DECAY_LAMBDA * age_days)

        by_skill[skill]["total"] += 1
        if rec["success"]:
            by_skill[skill]["successes"] += weight
        else:
            by_skill[skill]["failures"] += weight

    priors = {
        "_meta": {
            "updated_at": now.isoformat(),
            "decay_lambda": DECAY_LAMBDA,
            "prior_alpha": prior_alpha,
            "prior_beta": prior_beta,
            "total_records": len(records),
            "skills_tracked": len(by_skill),
        },
        "_global": {
            "alpha": prior_alpha,
            "beta": prior_beta,
        },
    }

    for skill, counts in sorted(by_skill.items()):
        alpha = prior_alpha + counts["successes"]
        beta = prior_beta + counts["failures"]
        expected = alpha / (alpha + beta)
        priors[skill] = {
            "alpha": round(alpha, 4),
            "beta": round(beta, 4),
            "expected_success": round(expected, 4),
            "observations": counts["total"],
        }

    return priors


# --- Analysis Report ---
def analyze() -> dict:
    """Run full analysis and return insights report."""
    records = load_outcomes()

    if not records:
        return {
            "status": "no_data",
            "message": "No outcome records found. Use 'atsm.py record' to log outcomes.",
            "patterns": [],
            "recommendations": [],
            "summary": {
                "total_records": 0,
                "unique_skills": 0,
                "overall_success_rate": None,
            }
        }

    # Run all pattern detectors
    patterns = []
    patterns.extend(detect_co_failures(records))
    patterns.extend(detect_time_of_day_effects(records))
    patterns.extend(detect_streaks(records))
    patterns.extend(detect_recency_trends(records))
    patterns.extend(detect_due_for_retry(records))

    # Generate recommendations
    recommendations = generate_recommendations(patterns, records)

    # Summary stats
    unique_skills = len(set(r["skill"] for r in records))
    total = len(records)
    successes = sum(1 for r in records if r["success"])
    overall_rate = successes / total if total else 0

    report = {
        "status": "ok",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": {
            "total_records": total,
            "unique_skills": unique_skills,
            "overall_success_rate": round(overall_rate, 4),
            "patterns_detected": len(patterns),
            "recommendations_generated": len(recommendations),
        },
        "patterns": patterns,
        "recommendations": recommendations,
        "skill_breakdown": _skill_breakdown(records),
    }

    return report


def _skill_breakdown(records: list[dict]) -> dict:
    """Per-skill summary for the report."""
    by_skill = defaultdict(list)
    for rec in records:
        by_skill[rec["skill"]].append(rec)

    breakdown = {}
    for skill, recs in sorted(by_skill.items()):
        total = len(recs)
        successes = sum(1 for r in recs if r["success"])
        last = max(r["_ts"] for r in recs)
        breakdown[skill] = {
            "total": total,
            "successes": successes,
            "failures": total - successes,
            "success_rate": round(successes / total, 4) if total else 0,
            "last_attempt": last.isoformat(),
        }
    return breakdown


# --- CLI ---
def cmd_analyze():
    """Print full insights report as JSON."""
    report = analyze()
    print(json.dumps(report, indent=2, ensure_ascii=False))


def cmd_patterns():
    """List detected patterns."""
    records = load_outcomes()
    if not records:
        print("No outcome records found.")
        return

    patterns = []
    patterns.extend(detect_co_failures(records))
    patterns.extend(detect_time_of_day_effects(records))
    patterns.extend(detect_streaks(records))
    patterns.extend(detect_recency_trends(records))
    patterns.extend(detect_due_for_retry(records))

    if not patterns:
        print("No significant patterns detected with current data.")
        print(f"  (Need more records — currently have {len(records)})")
        return

    print(f"\n{'Type':<18} {'Details'}")
    print("─" * 80)
    for p in patterns:
        print(f"{p['type']:<18} {p['description']}")

    print(f"\n  Total patterns detected: {len(patterns)}")


def cmd_priors():
    """Auto-update atsm_priors.json based on detected patterns."""
    records = load_outcomes()
    existing = load_priors()

    if not records:
        print("No outcome records — priors unchanged.")
        return

    updated = compute_updated_priors(records, existing)
    save_priors(updated)

    meta = updated["_meta"]
    print(f"\n{'Skill':<30} {'α':<10} {'β':<10} {'E[success]':<12} {'N'}")
    print("─" * 75)
    for key, val in sorted(updated.items()):
        if key.startswith("_"):
            continue
        print(f"{key:<30} {val['alpha']:<10} {val['beta']:<10} "
              f"{val['expected_success']:<12} {val['observations']}")

    print(f"\n  Updated at {meta['updated_at']}")
    print(f"  {meta['total_records']} records → {meta['skills_tracked']} skills tracked")
    print(f"  File: {PRIORS_FILE}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    commands = {
        "analyze": cmd_analyze,
        "patterns": cmd_patterns,
        "priors": cmd_priors,
    }

    handler = commands.get(command)
    if handler:
        handler()
    else:
        print(f"Unknown command: {command}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
