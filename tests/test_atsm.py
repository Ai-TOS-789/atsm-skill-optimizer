#!/usr/bin/env python3
"""ATSM Test Suite — Tests every ATSM module.

Usage:
    python3 tests/test_atsm.py
    python3 tests/test_atsm.py --verbose
    python3 tests/test_atsm.py --module atsm
    python3 tests/test_atsm.py --module embed_rank
    python3 tests/test_atsm.py --module knowledge_graph
    python3 tests/test_atsm.py --module causal_infer
    python3 tests/test_atsm.py --module goal_planner
    python3 tests/test_atsm.py --module swarm
    python3 tests/test_atsm.py --module execution
    python3 tests/test_atsm.py --module bridge
    python3 tests/test_atsm.py --report
"""

import importlib
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path

# --- Config ---
SKILL_DIR = Path(__file__).parent.parent
SCRIPTS_DIR = SKILL_DIR / "scripts"
DATA_DIR = SKILL_DIR / "data"

# Ensure we can import from scripts/
sys.path.insert(0, str(SCRIPTS_DIR))


# ============================================================
# Module Discovery
# ============================================================

def discover_modules():
    """Auto-discover all Python scripts in the scripts/ directory."""
    modules = {}
    for py_file in sorted(SCRIPTS_DIR.glob("*.py")):
        name = py_file.stem
        modules[name] = str(py_file)
    return modules


def load_module(name, path):
    """Load a module from file path."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ============================================================
# Test Registry
# ============================================================

class TestResult:
    """Single test result."""
    def __init__(self, name, module, passed, duration, error=None):
        self.name = name
        self.module = module
        self.passed = passed
        self.duration = duration
        self.error = error

    def __repr__(self):
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.module}.{self.name} ({self.duration*1000:.1f}ms)"


class TestReport:
    """Collects test results and generates a report."""
    def __init__(self):
        self.results = []

    def add(self, result):
        self.results.append(result)

    @property
    def total(self):
        return len(self.results)

    @property
    def passed(self):
        return sum(1 for r in self.results if r.passed)

    @property
    def failed(self):
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self):
        return self.passed / self.total * 100 if self.total > 0 else 0

    def summary(self):
        lines = []
        lines.append("=" * 70)
        lines.append("ATSM TEST SUITE — REPORT")
        lines.append("=" * 70)
        lines.append(f"\nTotal:  {self.total}")
        lines.append(f"Passed: {self.passed}")
        lines.append(f"Failed: {self.failed}")
        lines.append(f"Rate:   {self.pass_rate:.1f}%")
        lines.append(f"\n{'Module':<25} {'Test':<35} {'Status':<8} {'Time':<10}")
        lines.append("-" * 70)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            lines.append(f"{r.module:<25} {r.name:<35} {status:<8} {r.duration*1000:.1f}ms")
            if r.error:
                lines.append(f"  └─ Error: {r.error[:80]}")

        lines.append("=" * 70)
        return "\n".join(lines)


# Global report
report = TestReport()
VERBOSE = False


def run_test(name, module_name):
    """Decorator to register and run a test."""
    def decorator(func):
        def wrapper():
            if VERBOSE:
                print(f"  Running {module_name}.{name}...", end=" ", flush=True)
            start = time.time()
            try:
                func()
                duration = time.time() - start
                result = TestResult(name, module_name, True, duration)
                report.add(result)
                if VERBOSE:
                    print(f"PASS ({duration*1000:.1f}ms)")
            except Exception as e:
                duration = time.time() - start
                error_msg = f"{type(e).__name__}: {e}"
                result = TestResult(name, module_name, False, duration, error_msg)
                report.add(result)
                if VERBOSE:
                    print(f"FAIL ({duration*1000:.1f}ms)")
                    print(f"    {error_msg}")
        wrapper.test_name = name
        wrapper.module_name = module_name
        return wrapper
    return decorator


# ============================================================
# Helper: seed test data
# ============================================================

def seed_test_db():
    """Create a small test DB with outcomes for multiple skills."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    db_file = DATA_DIR / "atsm_db.jsonl"

    # Only seed if empty or doesn't exist
    if db_file.exists():
        content = db_file.read_text().strip()
        if content:
            return  # Already has data

    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()

    records = [
        {"skill": "atsm-skill-optimizer", "success": 1, "agent": "default", "timestamp": now},
        {"skill": "atsm-skill-optimizer", "success": 1, "agent": "default", "timestamp": now},
        {"skill": "atsm-skill-optimizer", "success": 0, "agent": "default", "timestamp": now},
        {"skill": "atsm", "success": 1, "agent": "default", "timestamp": now},
        {"skill": "embed_rank", "success": 1, "agent": "default", "timestamp": now},
        {"skill": "knowledge_graph", "success": 1, "agent": "default", "timestamp": now},
        {"skill": "causal_infer", "success": 1, "agent": "default", "timestamp": now},
        {"skill": "goal_planner", "success": 0, "agent": "default", "timestamp": now},
        {"skill": "swarm", "success": 1, "agent": "agent_1", "timestamp": now},
        {"skill": "swarm", "success": 1, "agent": "agent_2", "timestamp": now},
        {"skill": "execution_engine", "success": 1, "agent": "default", "timestamp": now},
        {"skill": "hermes_bridge", "success": 1, "agent": "default", "timestamp": now},
    ]

    with open(db_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


# ============================================================
# TEST: atsm_rank
# ============================================================

@run_test("test_atsm_rank", "atsm")
def test_atsm_rank():
    """Test ATSM skill ranking returns results."""
    seed_test_db()
    mod = load_module("atsm", str(SCRIPTS_DIR / "atsm.py"))

    # Clear cache to force fresh data
    mod._cache.skills = None
    mod._cache.records = None

    results, elapsed = mod.rank_skills("search and summarize papers")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    # With skills available, should return results
    if len(list(Path.home().joinpath(".hermes/skills").rglob("SKILL.md"))) > 0:
        assert len(results) > 0, "Expected non-empty results when skills exist"
    # Each result should have expected keys
    for r in results:
        assert "name" in r, f"Missing 'name' key in result: {r}"
        assert "final_score" in r, f"Missing 'final_score' key in result: {r}"
        assert "relevance" in r, f"Missing 'relevance' key in result: {r}"


# ============================================================
# TEST: atsm_record
# ============================================================

@run_test("test_atsm_record", "atsm")
def test_atsm_record():
    """Test recording outcomes to ATSM database."""
    seed_test_db()
    mod = load_module("atsm", str(SCRIPTS_DIR / "atsm.py"))

    # Clear cache
    mod._cache.records = None

    # Record a success
    mod.append_record("test_skill_record", 1)

    # Verify it was written
    mod._cache.records = None  # Invalidate cache
    records = mod.load_db()
    found = [r for r in records if r["skill"] == "test_skill_record"]
    assert len(found) >= 1, f"Expected at least 1 record for test_skill_record, got {len(found)}"
    assert found[-1]["success"] == 1, f"Expected success=1, got {found[-1]['success']}"


# ============================================================
# TEST: embed_rank
# ============================================================

@run_test("test_embed_rank", "embed_rank")
def test_embed_rank():
    """Test embedding-based skill ranking."""
    seed_test_db()
    mod = load_module("embed_rank", str(SCRIPTS_DIR / "embed_rank.py"))

    results, elapsed, backend_name = mod.rank_skills("research academic papers")
    assert isinstance(results, list), f"Expected list, got {type(results)}"
    assert backend_name in ("sentence-transformers", "tfidf", "none"), f"Unexpected backend: {backend_name}"
    if results:
        r = results[0]
        assert "name" in r, f"Missing 'name' key in result"
        assert "final_score" in r, f"Missing 'final_score' key in result"


# ============================================================
# TEST: knowledge_graph
# ============================================================

@run_test("test_knowledge_graph", "knowledge_graph")
def test_knowledge_graph():
    """Test knowledge graph builds and returns structure."""
    seed_test_db()
    mod = load_module("knowledge_graph", str(SCRIPTS_DIR / "knowledge_graph.py"))

    graph = mod.build_graph()
    assert isinstance(graph, dict), f"Expected dict, got {type(graph)}"
    assert "meta" in graph, "Missing 'meta' key in graph"
    assert "nodes" in graph, "Missing 'nodes' key in graph"

    meta = graph["meta"]
    assert "num_skills" in meta, "Missing 'num_skills' in meta"
    assert isinstance(meta["num_skills"], int), f"num_skills should be int, got {type(meta['num_skills'])}"

    nodes = graph["nodes"]
    assert isinstance(nodes, dict), f"nodes should be dict, got {type(nodes)}"

    # Test loading graph
    loaded = mod.load_graph()
    assert loaded is not None, "load_graph() returned None"

    # If we have skills, test neighbors function
    if nodes:
        first_skill = list(nodes.keys())[0]
        neighbors = mod.get_neighbors(first_skill)
        assert isinstance(neighbors, list), f"get_neighbors should return list, got {type(neighbors)}"


# ============================================================
# TEST: causal_infer
# ============================================================

@run_test("test_causal_infer", "causal_infer")
def test_causal_infer():
    """Test causality analysis functions."""
    seed_test_db()
    mod = load_module("causal_infer", str(SCRIPTS_DIR / "causal_infer.py"))

    records = mod.load_records()
    # Test sequence building
    sequences = mod.build_sequences(records)
    assert isinstance(sequences, dict), f"build_sequences should return dict, got {type(sequences)}"

    trials = mod.build_trial_matrix(sequences)
    assert isinstance(trials, dict), f"build_trial_matrix should return dict, got {type(trials)}"

    # Test chi-squared test (with known values)
    chi2, p_val, sig = mod.chi_squared_test_2x2(10, 5, 5, 10)
    assert isinstance(chi2, float), f"chi2 should be float, got {type(chi2)}"
    assert isinstance(p_val, float), f"p_val should be float, got {type(p_val)}"
    assert isinstance(sig, bool), f"sig should be bool, got {type(sig)}"

    # Test conditional probability if enough data
    skill_names = list(trials.keys())
    if len(skill_names) >= 2:
        cp = mod.conditional_probability(trials, skill_names[0], skill_names[1])
        if cp is not None:
            assert "p_b_given_a" in cp, "Missing 'p_b_given_a' in conditional_probability result"
            assert "lift" in cp, "Missing 'lift' in conditional_probability result"


# ============================================================
# TEST: goal_planner
# ============================================================

@run_test("test_goal_planner", "goal_planner")
def test_goal_planner():
    """Test goal decomposition into subgoals."""
    seed_test_db()
    mod = load_module("goal_planner", str(SCRIPTS_DIR / "goal_planner.py"))

    # Test goal type detection
    goal_type = mod.detect_goal_type("research GPT papers and create presentation")
    assert isinstance(goal_type, str), f"detect_goal_type should return str, got {type(goal_type)}"
    assert len(goal_type) > 0, "detect_goal_type returned empty string"

    # Test decomposition
    subgoals = mod.decompose_goal("build a web application")
    assert isinstance(subgoals, list), f"decompose_goal should return list, got {type(subgoals)}"
    assert len(subgoals) > 0, "decompose_goal returned empty list"
    assert len(subgoals) <= mod.MAX_SUBGOALS, f"Too many subgoals: {len(subgoals)} > {mod.MAX_SUBGOALS}"

    # Test goal tree creation
    tree = mod.create_goal_tree("research AI papers")
    assert tree is not None, "create_goal_tree returned None"
    assert tree.root is not None, "Goal tree has no root"
    assert len(tree.root.subgoals) > 0, "Goal tree has no subgoals"
    assert tree.status == "pending", f"Expected status 'pending', got '{tree.status}'"

    # Test persistence (save and load)
    goals = {}
    goals[tree.id] = tree
    mod.save_goals_db(goals)

    loaded = mod.load_goals_db()
    assert tree.id in loaded, f"Goal {tree.id} not found after save/load"


# ============================================================
# TEST: swarm
# ============================================================

@run_test("test_swarm", "swarm")
def test_swarm():
    """Test multi-agent voting system."""
    seed_test_db()
    mod = load_module("swarm", str(SCRIPTS_DIR / "swarm.py"))

    # Test problem classification
    ptype = mod.classify_problem("research papers and write a report")
    assert isinstance(ptype, str), f"classify_problem should return str, got {type(ptype)}"
    assert ptype in ("research", "coding", "creative", "data", "communication", "file", "web", "system", "general")

    skills = mod.load_skills()
    if not skills:
        return  # Skip if no skills available

    records = mod.load_db()
    stats = mod.compute_stats(records)

    # Test each agent strategy
    r1 = mod.agent_atsm_rank("test task", skills, stats)
    assert "strategy" in r1, "Missing 'strategy' in agent result"
    assert "confidence" in r1, "Missing 'confidence' in agent result"
    assert r1["strategy"] == "atsm_rank"

    r2 = mod.agent_embedding_rank("test task", skills)
    assert r2["strategy"] == "embedding_rank"

    r3 = mod.agent_random_explore("test task", skills)
    assert r3["strategy"] == "random_explore"

    r4 = mod.agent_causal_inference("test task", skills, stats)
    assert r4["strategy"] == "causal_inference"

    # Test voting
    agent_weights = mod.get_agent_weights()
    assert isinstance(agent_weights, dict), f"agent_weights should be dict, got {type(agent_weights)}"
    assert len(agent_weights) == 4, f"Expected 4 agent weights, got {len(agent_weights)}"

    vote_result = mod.vote([r1, r2, r3, r4], agent_weights)
    assert "winner" in vote_result, "Missing 'winner' in vote result"
    assert "consensus_strength" in vote_result, "Missing 'consensus_strength' in vote result"
    assert 0 <= vote_result["consensus_strength"] <= 1, f"consensus_strength out of range: {vote_result['consensus_strength']}"


# ============================================================
# TEST: execution
# ============================================================

@run_test("test_execution", "execution")
def test_execution():
    """Test execution engine ranking and history logging."""
    seed_test_db()
    mod = load_module("execution_engine", str(SCRIPTS_DIR / "execution_engine.py"))

    # Test ranking (uses ATSM under the hood)
    ranked = mod._rank_via_import("test task for execution", top_k=1)
    assert isinstance(ranked, list), f"_rank_via_import should return list, got {type(ranked)}"
    if ranked:
        assert "name" in ranked[0], "Missing 'name' in ranked result"
        assert "final_score" in ranked[0], "Missing 'final_score' in ranked result"

    # Test history loading (empty initially)
    history = mod.load_history()
    assert isinstance(history, list), f"load_history should return list, got {type(history)}"

    # Test log_execution with a mock entry
    test_entry = {
        "execution_id": "test_001",
        "task": "test task",
        "skill": "test_skill",
        "skill_score": 0.5,
        "relevance": 0.3,
        "success_prob": 0.7,
        "command": "hermes chat -q test",
        "mode": "sync",
        "output": "test output",
        "stderr": "",
        "success": True,
        "return_code": 0,
        "duration": 0.1,
        "timestamp": "2025-01-01T00:00:00+00:00",
    }
    mod.log_execution(test_entry)

    # Verify it was logged
    history = mod.load_history()
    test_entries = [h for h in history if h.get("execution_id") == "test_001"]
    assert len(test_entries) >= 1, "Test entry not found in history after logging"


# ============================================================
# TEST: bridge
# ============================================================

@run_test("test_bridge", "bridge")
def test_bridge():
    """Test Hermes skill import bridge."""
    seed_test_db()
    mod = load_module("hermes_bridge", str(SCRIPTS_DIR / "hermes_bridge.py"))

    bridge = mod.HermesBridge()

    # Test skill scanning
    skills = bridge.scan_skills()
    assert isinstance(skills, dict), f"scan_skills should return dict, got {type(skills)}"

    # Test sync (dry run)
    results = bridge.sync(dry_run=True)
    assert isinstance(results, dict), f"sync should return dict, got {type(results)}"
    assert "added" in results, "Missing 'added' key in sync results"
    assert "updated" in results, "Missing 'updated' key in sync results"
    assert "retired" in results, "Missing 'retired' key in sync results"
    assert "unchanged" in results, "Missing 'unchanged' key in sync results"

    # Test validation
    validation = bridge.validate()
    assert isinstance(validation, dict), f"validate should return dict, got {type(validation)}"
    assert "errors" in validation, "Missing 'errors' in validation result"
    assert "warnings" in validation, "Missing 'warnings' in validation result"
    assert "valid" in validation, "Missing 'valid' in validation result"

    # Test list_skills
    skill_list = bridge.list_skills()
    assert isinstance(skill_list, list), f"list_skills should return list, got {type(skill_list)}"


# ============================================================
# Auto-discovery tests for ALL scripts
# ============================================================

def auto_discover_tests():
    """Auto-discover all scripts/ and add a basic import test for each."""
    modules = discover_modules()

    # Core modules that already have dedicated tests above
    covered = {"atsm", "embed_rank", "knowledge_graph", "causal_infer",
               "goal_planner", "swarm", "execution_engine", "hermes_bridge"}

    tests = []
    for name, path in sorted(modules.items()):
        if name.startswith("_") or name in covered:
            continue
        # Create a test for this module
        tests.append((name, path))
    return tests


def run_auto_discovered_tests():
    """Run auto-discovered import tests."""
    test_targets = auto_discover_tests()

    for name, path in test_targets:
        test_name = f"test_auto_{name}"

        def make_test(n, p):
            def test_func():
                try:
                    mod = load_module(n, p)
                    assert mod is not None, f"Module {n} loaded as None"
                    # Check it has some expected attributes
                    assert hasattr(mod, '__name__') or hasattr(mod, '__file__'), \
                        f"Module {n} has no expected attributes"
                except ImportError as e:
                    # Some modules may have missing dependencies — that's ok
                    if "No module named" in str(e):
                        return  # Skip, dependency missing
                    raise
            return test_func

        func = test_func = make_test(name, path)
        decorated = run_test(test_name, name)(func)
        decorated()


# ============================================================
# Main
# ============================================================

def parse_args():
    """Parse CLI arguments."""
    args = sys.argv[1:]
    module_filter = None
    verbose = False
    show_report = False

    if "--verbose" in args:
        verbose = True
        args.remove("--verbose")

    if "--report" in args:
        show_report = True
        args.remove("--report")

    if "--module" in args:
        idx = args.index("--module")
        if idx + 1 < len(args):
            module_filter = args[idx + 1]
            args = args[:idx] + args[idx + 2:]

    return module_filter, verbose, show_report


def main():
    global VERBOSE

    module_filter, verbose, show_report = parse_args()
    VERBOSE = verbose

    print("=" * 70)
    print("ATSM TEST SUITE")
    print("=" * 70)

    if module_filter:
        print(f"\nFiltering to module: {module_filter}")
    if verbose:
        print("Verbose mode: ON")
    print()

    # Collect all test functions
    all_tests = [
        test_atsm_rank,
        test_atsm_record,
        test_embed_rank,
        test_knowledge_graph,
        test_causal_infer,
        test_goal_planner,
        test_swarm,
        test_execution,
        test_bridge,
    ]

    # Filter tests if module specified
    if module_filter:
        filtered = []
        for test_func in all_tests:
            tn = test_func.test_name.lower()
            mn = test_func.module_name.lower()
            filter_lower = module_filter.lower()
            if filter_lower in tn or filter_lower in mn:
                filtered.append(test_func)
        all_tests = filtered

    # Run tests
    print(f"Running {len(all_tests)} tests...\n")
    for test_func in all_tests:
        if not module_filter:
            # Also run auto-discovered for the full suite
            pass
        test_func()

    # Auto-discovered tests (only in full mode)
    if not module_filter:
        print("\nRunning auto-discovered module tests...")
        run_auto_discovered_tests()

    # Print report
    print()
    print(report.summary())

    if show_report:
        report_path = Path(__file__).parent / "test_report.txt"
        report_path.write_text(report.summary())
        print(f"\nReport saved to: {report_path}")

    # Exit code
    return 0 if report.failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
