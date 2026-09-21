#!/usr/bin/env python3
"""Causal Inference engine for ATSM outcome history.

Analyzes causal relationships between skills based on historical outcome data.

Methods:
    - Granger causality: does skill A success predict skill B success?
    - Conditional probability: P(B|A) vs P(B|¬A)
    - Counterfactual: what if skill A had failed?

Usage:
    python3 causal_infer.py analyze
    python3 causal_infer.py causes <skill>
    python3 causal_infer.py effects <skill>
    python3 causal_infer.py counterfactual <skill> <0|1>
"""

import json
import math
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
DATA_DIR = Path(__file__).parent.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"

# Minimum observations required for statistical tests
MIN_OBS = 3
# Significance level for chi-squared test
ALPHA = 0.05


# --- Data Loading ---
def load_records() -> list[dict]:
    """Load outcome records from the ATSM database."""
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


def build_sequences(records: list[dict]) -> dict[str, list[tuple[str, int]]]:
    """Build per-skill sequences of (timestamp, success) sorted by time."""
    sequences = defaultdict(list)
    for rec in records:
        skill = rec["skill"]
        success = rec["success"]
        ts = rec["timestamp"]
        sequences[skill].append((ts, success))
    # Sort each sequence by timestamp
    for skill in sequences:
        sequences[skill].sort(key=lambda x: x[0])
    return dict(sequences)


def build_trial_matrix(sequences: dict[str, list[tuple[str, int]]]) -> dict[str, list[int]]:
    """Convert sequences to simple binary lists (chronological order)."""
    return {skill: [s for _, s in seq] for skill, seq in sequences.items()}


# --- Statistical Utilities ---
def chi_squared_test_2x2(a_and_b, a_and_not_b, not_a_and_b, not_a_and_not_b):
    """Perform chi-squared test on a 2x2 contingency table.

    Returns (chi2_stat, p_value, significant).
    Uses Yates' correction for continuity.
    """
    n = a_and_b + a_and_not_b + not_a_and_b + not_a_and_not_b
    if n == 0:
        return 0.0, 1.0, False

    # Expected frequencies
    row1 = a_and_b + a_and_not_b
    row2 = not_a_and_b + not_a_and_not_b
    col1 = a_and_b + not_a_and_b
    col2 = a_and_not_b + not_a_and_not_b

    if row1 == 0 or row2 == 0 or col1 == 0 or col2 == 0:
        return 0.0, 1.0, False

    e11 = row1 * col1 / n
    e12 = row1 * col2 / n
    e21 = row2 * col1 / n
    e22 = row2 * col2 / n

    # Chi-squared with Yates' correction
    def cell(o, e):
        return (abs(o - e) - 0.5) ** 2 / e if e > 0 else 0

    chi2 = cell(a_and_b, e11) + cell(a_and_not_b, e12) + cell(not_a_and_b, e21) + cell(not_a_and_not_b, e22)

    # p-value from chi-squared distribution with 1 df
    p_value = 1 - _chi2_cdf(chi2, 1)
    return chi2, p_value, p_value < ALPHA


def _chi2_cdf(x, df):
    """Approximate chi-squared CDF using the regularized incomplete gamma function."""
    if x <= 0:
        return 0.0
    return _regularized_gamma_q(df / 2, x / 2)


def _regularized_gamma_q(a, x):
    """Regularized incomplete gamma function Q(a, x) = 1 - P(a, x).

    Uses series expansion for small x and continued fraction for large x.
    """
    if x < 0:
        return 1.0
    if x == 0:
        return 1.0
    if x < a + 1:
        return 1 - _gamma_series(a, x)
    else:
        return _gamma_cf(a, x)


def _gamma_series(a, x):
    """Lower incomplete gamma via series expansion."""
    if x <= 0:
        return 0.0
    ap = a
    s = 1.0 / a
    ds = s
    for _ in range(200):
        ap += 1
        ds *= x / ap
        s += ds
        if abs(ds) < abs(s) * 1e-12:
            break
    return s * math.exp(-x + a * math.log(x) - math.lgamma(a))


def _gamma_cf(a, x):
    """Upper incomplete gamma via continued fraction (Lentz's method)."""
    fpmin = 1e-300
    b_cf = x + 1 - a
    c = 1 / fpmin
    d = 1 / b_cf
    h = d
    for i in range(1, 300):
        an = -i * (i - a)
        b_cf += 2
        d = an * d + b_cf
        if abs(d) < fpmin:
            d = fpmin
        c = b_cf + an / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 1e-12:
            break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h


def fisher_exact_one_sided(a, b, c, d):
    """One-sided Fisher's exact test (testing if a/b > c/d).

    Returns p-value. More appropriate for small samples.
    """
    n = a + b + c + d
    if n == 0:
        return 1.0
    # Use hypergeometric probability
    # P(X >= a) under null
    p = 0.0
    for x in range(a, min(a + b, a + c) + 1):
        p += _hypergeometric_pmf(x, a + b, a + c, n)
    return min(p, 1.0)


def _hypergeometric_pmf(k, n1, n2, N):
    """Hypergeometric PMF: P(X=k) drawing N from population with n1 type1, n2 type2."""
    if k < max(0, n1 + n2 - N) or k > min(n1, n2):
        return 0.0
    return (math.comb(n1, k) * math.comb(N - n1, n2 - k)) / math.comb(N, n2)


# --- Granger Causality ---
def granger_causality(trials: dict[str, list[int]], skill_a: str, skill_b: str, max_lag: int = 1):
    """Test if skill A Granger-causes skill B.

    For binary time series, we test: does A's past predict B's current value
    beyond B's own past?

    Simplified approach: test if P(B_t=1 | A_{t-1}=1) != P(B_t=1 | A_{t-1}=0)
    using chi-squared test on the 2x2 contingency table.

    Returns dict with test results.
    """
    if skill_a not in trials or skill_b not in trials:
        return None

    seq_a = trials[skill_a]
    seq_b = trials[skill_b]

    if len(seq_a) < MIN_OBS or len(seq_b) < MIN_OBS:
        return None

    # Build aligned pairs: for each trial index, pair A's outcome with B's next outcome
    # Since sequences may differ in length, we use index-aligned approach
    # (assuming trials are roughly simultaneous or ordered by time)
    min_len = min(len(seq_a), len(seq_b))

    # Contingency table: A_{t} vs B_{t+1} (A leads B by 1)
    a_and_b = 0    # A=1, B_next=1
    a_and_not_b = 0  # A=1, B_next=0
    not_a_and_b = 0  # A=0, B_next=1
    not_a_and_not_b = 0  # A=0, B_next=0

    for i in range(min_len - 1):
        a_val = seq_a[i]
        b_next = seq_b[i + 1]
        if a_val == 1 and b_next == 1:
            a_and_b += 1
        elif a_val == 1 and b_next == 0:
            a_and_not_b += 1
        elif a_val == 0 and b_next == 1:
            not_a_and_b += 1
        else:
            not_a_and_not_b += 1

    total = a_and_b + a_and_not_b + not_a_and_b + not_a_and_not_b
    if total < MIN_OBS:
        return None

    chi2, p_value, significant = chi_squared_test_2x2(a_and_b, a_and_not_b, not_a_and_b, not_a_and_not_b)

    # Effect size: difference in conditional probabilities
    p_b_given_a = a_and_b / (a_and_b + a_and_not_b) if (a_and_b + a_and_not_b) > 0 else 0
    p_b_given_not_a = not_a_and_b / (not_a_and_b + not_a_and_not_b) if (not_a_and_b + not_a_and_not_b) > 0 else 0
    effect_size = p_b_given_a - p_b_given_not_a

    return {
        "cause": skill_a,
        "effect": skill_b,
        "chi2": round(chi2, 4),
        "p_value": round(p_value, 4),
        "significant": significant,
        "effect_size": round(effect_size, 4),
        "p_b_given_a": round(p_b_given_a, 4),
        "p_b_given_not_a": round(p_b_given_not_a, 4),
        "n": total,
    }


# --- Conditional Probability ---
def conditional_probability(trials: dict[str, list[int]], skill_a: str, skill_b: str):
    """Compute P(B|A) and P(B|¬A) and their difference.

    Returns dict with conditional probabilities and lift.
    """
    if skill_a not in trials or skill_b not in trials:
        return None

    seq_a = trials[skill_a]
    seq_b = trials[skill_b]
    min_len = min(len(seq_a), len(seq_b))

    if min_len < MIN_OBS:
        return None

    # Count co-occurrences at same time index
    a_and_b = sum(1 for i in range(min_len) if seq_a[i] == 1 and seq_b[i] == 1)
    a_and_not_b = sum(1 for i in range(min_len) if seq_a[i] == 1 and seq_b[i] == 0)
    not_a_and_b = sum(1 for i in range(min_len) if seq_a[i] == 0 and seq_b[i] == 1)
    not_a_and_not_b = sum(1 for i in range(min_len) if seq_a[i] == 0 and seq_b[i] == 0)

    total = a_and_b + a_and_not_b + not_a_and_b + not_a_and_not_b
    a_total = a_and_b + a_and_not_b
    not_a_total = not_a_and_b + not_a_and_not_b
    b_total = a_and_b + not_a_and_b

    p_b_given_a = a_and_b / a_total if a_total > 0 else 0
    p_b_given_not_a = not_a_and_b / not_a_total if not_a_total > 0 else 0
    p_b = b_total / total if total > 0 else 0

    # Lift: P(B|A) / P(B)
    lift = p_b_given_a / p_b if p_b > 0 else float('inf')

    # Confidence interval for the difference (Wald interval)
    var_diff = (p_b_given_a * (1 - p_b_given_a) / a_total if a_total > 0 else 0) + \
               (p_b_given_not_a * (1 - p_b_given_not_a) / not_a_total if not_a_total > 0 else 0)
    se_diff = math.sqrt(var_diff) if var_diff > 0 else 0
    diff = p_b_given_a - p_b_given_not_a
    ci_low = diff - 1.96 * se_diff
    ci_high = diff + 1.96 * se_diff

    return {
        "skill_a": skill_a,
        "skill_b": skill_b,
        "p_b_given_a": round(p_b_given_a, 4),
        "p_b_given_not_a": round(p_b_given_not_a, 4),
        "p_b": round(p_b, 4),
        "difference": round(diff, 4),
        "lift": round(lift, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
        "n_a": a_total,
        "n_not_a": not_a_total,
    }


# --- Counterfactual ---
def counterfactual(trials: dict[str, list[int]], sequences: dict[str, list[tuple[str, int]]],
                   skill_a: str, hypothetical_outcome: int):
    """Estimate what would happen to other skills if skill A had a different outcome.

    Uses conditional probability: P(B|A=hypothetical) vs observed P(B).

    Returns dict mapping each other skill to its counterfactual prediction.
    """
    if skill_a not in trials:
        return None

    seq_a = trials[skill_a]
    n = len(seq_a)

    if n < MIN_OBS:
        return None

    results = {}
    for skill_b, seq_b in trials.items():
        if skill_b == skill_a:
            continue
        min_len = min(len(seq_a), len(seq_b))
        if min_len < MIN_OBS:
            continue

        # Compute P(B|A=hypothetical)
        if hypothetical_outcome == 1:
            # P(B|A=1)
            b_and_a = sum(1 for i in range(min_len) if seq_b[i] == 1 and seq_a[i] == 1)
            a_count = sum(1 for i in range(min_len) if seq_a[i] == 1)
            p_b_counter = b_and_a / a_count if a_count > 0 else 0
        else:
            # P(B|A=0)
            b_and_not_a = sum(1 for i in range(min_len) if seq_b[i] == 1 and seq_a[i] == 0)
            not_a_count = sum(1 for i in range(min_len) if seq_a[i] == 0)
            p_b_counter = b_and_not_a / not_a_count if not_a_count > 0 else 0

        # Observed P(B)
        p_b_observed = sum(seq_b[:min_len]) / min_len

        # Predicted change
        delta = p_b_counter - p_b_observed

        results[skill_b] = {
            "p_b_observed": round(p_b_observed, 4),
            "p_b_counterfactual": round(p_b_counter, 4),
            "delta": round(delta, 4),
            "direction": "increase" if delta > 0.05 else ("decrease" if delta < -0.05 else "unchanged"),
        }

    return results


# --- Full Analysis ---
def analyze_all(records: list[dict]):
    """Run full causal analysis on all skill pairs."""
    sequences = build_sequences(records)
    trials = build_trial_matrix(sequences)
    skills = sorted(trials.keys())

    print("=" * 70)
    print("CAUSAL INFERENCE ANALYSIS")
    print("=" * 70)
    print(f"\nSkills tracked: {len(skills)}")
    print(f"Total records: {len(records)}")
    print(f"Skills: {', '.join(skills)}")

    # --- Granger Causality ---
    print("\n" + "-" * 70)
    print("GRANGER CAUSALITY TESTS (A → B)")
    print("-" * 70)
    print(f"{'Cause':<15} {'Effect':<15} {'Effect Size':<12} {'p-value':<10} {'Sig':<5} {'N':<5}")
    print("-" * 70)

    granger_results = []
    for a in skills:
        for b in skills:
            if a == b:
                continue
            result = granger_causality(trials, a, b)
            if result and result["n"] >= MIN_OBS:
                granger_results.append(result)

    # Sort by effect size (absolute), then by p-value
    granger_results.sort(key=lambda x: (-abs(x["effect_size"]), x["p_value"]))

    for r in granger_results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"{r['cause']:<15} {r['effect']:<15} {r['effect_size']:<12} {r['p_value']:<10} {sig:<5} {r['n']:<5}")

    if not granger_results:
        print("  (insufficient data for Granger causality tests)")

    # --- Conditional Probabilities ---
    print("\n" + "-" * 70)
    print("CONDITIONAL PROBABILITY ANALYSIS")
    print("-" * 70)
    print(f"{'A':<15} {'B':<15} {'P(B|A)':<10} {'P(B|¬A)':<10} {'Diff':<10} {'Lift':<8}")
    print("-" * 70)

    cp_results = []
    for a in skills:
        for b in skills:
            if a == b:
                continue
            result = conditional_probability(trials, a, b)
            if result and result["n_a"] >= 2 and result["n_not_a"] >= 2:
                cp_results.append(result)

    cp_results.sort(key=lambda x: -abs(x["difference"]))

    for r in cp_results:
        print(f"{r['skill_a']:<15} {r['skill_b']:<15} {r['p_b_given_a']:<10} {r['p_b_given_not_a']:<10} {r['difference']:<10} {r['lift']:<8}")

    if not cp_results:
        print("  (insufficient data for conditional probability analysis)")

    # --- Causal Graph Summary ---
    print("\n" + "-" * 70)
    print("CAUSAL GRAPH (significant edges)")
    print("-" * 70)

    significant_edges = [r for r in granger_results if r["significant"]]
    if significant_edges:
        for r in significant_edges:
            direction = "+" if r["effect_size"] > 0 else "-"
            print(f"  {r['cause']} --({direction}{abs(r['effect_size']):.3f})--> {r['effect']}  [p={r['p_value']}]")
    else:
        print("  (no statistically significant causal edges found)")

    # --- Summary Stats ---
    print("\n" + "-" * 70)
    print("SKILL SUMMARY")
    print("-" * 70)
    print(f"{'Skill':<20} {'Trials':<8} {'Success Rate':<15} {'Causes':<20} {'Effects':<20}")
    print("-" * 70)

    for skill in skills:
        seq = trials[skill]
        n = len(seq)
        success_rate = sum(seq) / n if n > 0 else 0
        causes = [f"{r['effect']}({r['effect_size']:+.2f})" for r in granger_results if r["cause"] == skill and r["significant"]]
        effects = [f"{r['cause']}({r['effect_size']:+.2f})" for r in granger_results if r["effect"] == skill and r["significant"]]
        print(f"{skill:<20} {n:<8} {success_rate:<15.3f} {', '.join(causes) or '-':<20} {', '.join(effects) or '-':<20}")

    print("\n" + "=" * 70)


# --- CLI Commands ---
def cmd_analyze():
    """Run full causal analysis."""
    records = load_records()
    if not records:
        print("No records found. Record some outcomes first with atsm.py record.")
        sys.exit(1)
    analyze_all(records)


def cmd_causes(skill: str):
    """Show what skills are caused by the given skill."""
    records = load_records()
    if not records:
        print("No records found.")
        sys.exit(1)

    sequences = build_sequences(records)
    trials = build_trial_matrix(sequences)

    if skill not in trials:
        print(f"Skill '{skill}' not found in database.")
        sys.exit(1)

    print(f"\nCausal effects OF '{skill}' (what {skill} success predicts)")
    print("-" * 60)

    results = []
    for other_skill in sorted(trials.keys()):
        if other_skill == skill:
            continue
        r = granger_causality(trials, skill, other_skill)
        if r:
            results.append(r)

    results.sort(key=lambda x: -abs(x["effect_size"]))

    if not results:
        print("  No causal relationships found (insufficient data).")
        return

    print(f"{'Effect':<20} {'Effect Size':<15} {'P(B|A)':<10} {'P(B|¬A)':<10} {'p-value':<10} {'Sig'}")
    print("-" * 60)
    for r in results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"{r['effect']:<20} {r['effect_size']:<15} {r['p_b_given_a']:<10} {r['p_b_given_not_a']:<10} {r['p_value']:<10} {sig}")

    # Also show conditional probability
    print(f"\nConditional probability P(X | {skill}):")
    print("-" * 60)
    cp_results = []
    for other_skill in sorted(trials.keys()):
        if other_skill == skill:
            continue
        r = conditional_probability(trials, skill, other_skill)
        if r and r["n_a"] >= 2 and r["n_not_a"] >= 2:
            cp_results.append(r)

    cp_results.sort(key=lambda x: -abs(x["difference"]))

    if cp_results:
        print(f"{'Skill':<20} {'P(X|A)':<10} {'P(X|¬A)':<10} {'Diff':<10} {'Lift':<8}")
        print("-" * 60)
        for r in cp_results:
            print(f"{r['skill_b']:<20} {r['p_b_given_a']:<10} {r['p_b_given_not_a']:<10} {r['difference']:<10} {r['lift']:<8}")


def cmd_effects(skill: str):
    """Show what skills cause the given skill's success."""
    records = load_records()
    if not records:
        print("No records found.")
        sys.exit(1)

    sequences = build_sequences(records)
    trials = build_trial_matrix(sequences)

    if skill not in trials:
        print(f"Skill '{skill}' not found in database.")
        sys.exit(1)

    print(f"\nCauses OF '{skill}' (what predicts {skill} success)")
    print("-" * 60)

    results = []
    for other_skill in sorted(trials.keys()):
        if other_skill == skill:
            continue
        r = granger_causality(trials, other_skill, skill)
        if r:
            results.append(r)

    results.sort(key=lambda x: -abs(x["effect_size"]))

    if not results:
        print("  No causal relationships found (insufficient data).")
        return

    print(f"{'Cause':<20} {'Effect Size':<15} {'P(B|A)':<10} {'P(B|¬A)':<10} {'p-value':<10} {'Sig'}")
    print("-" * 60)
    for r in results:
        sig = "***" if r["p_value"] < 0.001 else "**" if r["p_value"] < 0.01 else "*" if r["p_value"] < 0.05 else ""
        print(f"{r['cause']:<20} {r['effect_size']:<15} {r['p_b_given_a']:<10} {r['p_b_given_not_a']:<10} {r['p_value']:<10} {sig}")

    # Also show conditional probability
    print(f"\nConditional probability P({skill} | X):")
    print("-" * 60)
    cp_results = []
    for other_skill in sorted(trials.keys()):
        if other_skill == skill:
            continue
        r = conditional_probability(trials, other_skill, skill)
        if r and r["n_a"] >= 2 and r["n_not_a"] >= 2:
            cp_results.append(r)

    cp_results.sort(key=lambda x: -abs(x["difference"]))

    if cp_results:
        print(f"{'Skill':<20} {'P(B|A)':<10} {'P(B|¬A)':<10} {'Diff':<10} {'Lift':<8}")
        print("-" * 60)
        for r in cp_results:
            print(f"{r['skill_a']:<20} {r['p_b_given_a']:<10} {r['p_b_given_not_a']:<10} {r['difference']:<10} {r['lift']:<8}")


def cmd_counterfactual(skill: str, outcome: int):
    """Show counterfactual: what if skill had a different outcome."""
    records = load_records()
    if not records:
        print("No records found.")
        sys.exit(1)

    sequences = build_sequences(records)
    trials = build_trial_matrix(sequences)

    if skill not in trials:
        print(f"Skill '{skill}' not found in database.")
        sys.exit(1)

    outcome_label = "success" if outcome == 1 else "failure"
    print(f"\nCOUNTERFACTUAL: What if '{skill}' had been a {outcome_label}?")
    print("=" * 60)

    results = counterfactual(trials, sequences, skill, outcome)
    if not results:
        print("  Insufficient data for counterfactual analysis.")
        return

    print(f"\n{'Skill':<20} {'P(obs)':<10} {'P(counter)':<12} {'Delta':<10} {'Direction':<12}")
    print("-" * 60)

    # Sort by absolute delta
    sorted_results = sorted(results.items(), key=lambda x: -abs(x[1]["delta"]))

    for other_skill, r in sorted_results:
        print(f"{other_skill:<20} {r['p_b_observed']:<10} {r['p_b_counterfactual']:<12} {r['delta']:<10} {r['direction']:<12}")

    print(f"\nInterpretation: If '{skill}' had been a {outcome_label}:")
    increases = [s for s, r in sorted_results if r["direction"] == "increase"]
    decreases = [s for s, r in sorted_results if r["direction"] == "decrease"]
    if increases:
        print(f"  → These skills would likely succeed MORE: {', '.join(increases)}")
    if decreases:
        print(f"  → These skills would likely succeed LESS: {', '.join(decreases)}")
    if not increases and not decreases:
        print("  → No significant change predicted for other skills.")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "analyze":
        cmd_analyze()
    elif command == "causes":
        if len(sys.argv) < 3:
            print("Usage: causal_infer.py causes <skill>")
            sys.exit(1)
        cmd_causes(sys.argv[2])
    elif command == "effects":
        if len(sys.argv) < 3:
            print("Usage: causal_infer.py effects <skill>")
            sys.exit(1)
        cmd_effects(sys.argv[2])
    elif command == "counterfactual":
        if len(sys.argv) < 4:
            print("Usage: causal_infer.py counterfactual <skill> <0|1>")
            sys.exit(1)
        skill = sys.argv[2]
        try:
            outcome = int(sys.argv[3])
            if outcome not in (0, 1):
                raise ValueError
        except ValueError:
            print("Outcome must be 0 (failure) or 1 (success)")
            sys.exit(1)
        cmd_counterfactual(skill, outcome)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
