#!/usr/bin/env python3
"""ATSM Unified Control Center.

Single command center for all ATSM operations: ranking, execution, goal
planning, swarm solving, knowledge graph, causal analysis, system metrics,
and overall health — all from one CLI with a color-coded dashboard.

Usage:
    python3 unified_control.py rank "search papers"
    python3 unified_control.py run "search papers"
    python3 unified_control.py plan "research goal"
    python3 unified_control.py swarm "task description"
    python3 unified_control.py graph
    python3 unified_control.py causal
    python3 unified_control.py monitor
    python3 unified_control.py health
    python3 unified_control.py dashboard
    python3 unified_control.py interactive
"""

import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# --- Paths ---
SCRIPTS_DIR = Path(__file__).parent
SKILL_DIR = SCRIPTS_DIR.parent
DATA_DIR = SKILL_DIR / "data"
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))

DB_FILE = DATA_DIR / "atsm_db.jsonl"
GRAPH_FILE = DATA_DIR / "knowledge_graph.json"
GOALS_FILE = DATA_DIR / "goals_db.json"
SWARM_STATS_FILE = DATA_DIR / "swarm_stats.json"
SWARM_DB_FILE = DATA_DIR / "swarm_db.jsonl"
CHAINS_DB_FILE = DATA_DIR / "chains_db.jsonl"
METRICS_FILE = DATA_DIR / "sys_metrics.jsonl"

# --- ANSI colors ---
USE_COLOR = sys.stdout.isatty() or os.environ.get("FORCE_COLOR")

def c(text: str, code: str) -> str:
    if not USE_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"

def green(t):   return c(t, "32")
def yellow(t):  return c(t, "33")
def red(t):     return c(t, "31")
def bold(t):    return c(t, "1")
def dim(t):     return c(t, "2")
def cyan(t):    return c(t, "36")

STATUS_ICON = {
    "green": lambda t: green("● " + t),
    "yellow": lambda t: yellow("● " + t),
    "red": lambda t: red("● " + t),
}

def status_label(level: str, text: str) -> str:
    return STATUS_ICON.get(level, lambda t: t)(text)


# ============================================================
# Helpers
# ============================================================

def run_script(script: str, args: list, timeout: int = 120) -> tuple[int, str]:
    """Run a sibling ATSM script, return (exit_code, combined output)."""
    path = SCRIPTS_DIR / script
    if not path.exists():
        return 1, f"Script not found: {path}"
    try:
        proc = subprocess.run(
            [sys.executable, str(path)] + args,
            capture_output=True, text=True, timeout=timeout,
            cwd=str(SCRIPTS_DIR),
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, out.strip()
    except subprocess.TimeoutExpired:
        return 124, f"Timeout after {timeout}s running {script}"
    except Exception as e:
        return 1, f"Error running {script}: {e}"


def read_json(path: Path, default=None):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def read_jsonl(path: Path) -> list:
    records = []
    if not path.exists():
        return records
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except Exception:
        pass
    return records


def parse_age(iso_ts: str) -> float:
    """Age in hours of an ISO timestamp; large number if unparseable.

    Naive timestamps are treated as local time (sys_monitor writes local);
    tz-aware ones are compared against UTC now.
    """
    try:
        ts = datetime.fromisoformat(iso_ts)
        if ts.tzinfo is None:
            now = datetime.now()
        else:
            now = datetime.now(timezone.utc)
        return max((now - ts).total_seconds() / 3600, 0.0)
    except Exception:
        return 1e9


def bar(frac: float, width: int = 20, colorize: bool = True) -> str:
    frac = max(0.0, min(1.0, frac))
    filled = int(frac * width)
    b = "█" * filled + "░" * (width - filled)
    if colorize and USE_COLOR:
        code = "32" if frac >= 0.7 else ("33" if frac >= 0.4 else "31")
        b = f"\033[{code}m{b}\033[0m"
    return b


# ============================================================
# Module status probes (for dashboard / health)
# ============================================================

def probe_atsm() -> dict:
    records = read_jsonl(DB_FILE)
    n = len(records)
    succ = sum(1 for r in records if r.get("success") == 1)
    rate = (succ / n) if n else 0.0
    exists = (SCRIPTS_DIR / "atsm.py").exists()
    if not exists:
        level = "red"
    elif n == 0:
        level = "yellow"
    elif rate >= 0.5:
        level = "green"
    else:
        level = "yellow"
    return {
        "name": "ATSM Engine", "script": "atsm.py", "level": level,
        "detail": f"{n} records, success rate {rate:.0%}",
        "records": n, "success_rate": rate,
    }


def probe_goal_planner() -> dict:
    goals = read_json(GOALS_FILE, {}) or {}
    active = sum(1 for g in goals.values()
                 if isinstance(g, dict) and g.get("root", {}).get("status") not in ("completed", "failed"))
    exists = (SCRIPTS_DIR / "goal_planner.py").exists()
    level = "red" if not exists else ("green" if goals else "yellow")
    return {
        "name": "Goal Planner", "script": "goal_planner.py", "level": level,
        "detail": f"{len(goals)} goals tracked, {active} active",
        "goals": len(goals), "active": active,
    }


def probe_swarm() -> dict:
    stats = read_json(SWARM_STATS_FILE, {}) or {}
    solves = stats.get("total_solves", 0)
    solves += len(read_jsonl(SWARM_DB_FILE))
    exists = (SCRIPTS_DIR / "swarm.py").exists()
    level = "red" if not exists else ("green" if solves else "yellow")
    return {
        "name": "Swarm Solver", "script": "swarm.py", "level": level,
        "detail": f"{solves} solves recorded",
        "solves": solves,
    }


def probe_knowledge_graph() -> dict:
    g = read_json(GRAPH_FILE, {}) or {}
    meta = g.get("meta", {})
    nodes = meta.get("num_skills", len(g.get("nodes", {})))
    edges = (meta.get("num_cooccurrence_edges", 0) + meta.get("num_sequence_edges", 0)
             + meta.get("num_similarity_edges", 0) + meta.get("num_causal_edges", 0))
    built = meta.get("built_at", "")
    age_h = parse_age(built) if built else 1e9
    exists = (SCRIPTS_DIR / "knowledge_graph.py").exists()
    if not exists:
        level = "red"
    elif nodes == 0:
        level = "yellow"
    elif age_h > 24 * 30:
        level = "yellow"
    else:
        level = "green"
    age_str = f", built {age_h:.1f}h ago" if built else ""
    return {
        "name": "Knowledge Graph", "script": "knowledge_graph.py", "level": level,
        "detail": f"{nodes} skills, {edges} edges{age_str}",
        "nodes": nodes, "edges": edges,
    }


def probe_causal() -> dict:
    records = read_jsonl(DB_FILE)
    exists = (SCRIPTS_DIR / "causal_infer.py").exists()
    level = "red" if not exists else ("green" if len(records) >= 10 else "yellow")
    return {
        "name": "Causal Inference", "script": "causal_infer.py", "level": level,
        "detail": f"{len(records)} outcome records to analyze",
    }


def probe_monitor() -> dict:
    entries = read_jsonl(METRICS_FILE)
    last_age = parse_age(entries[-1]["timestamp"]) if entries and "timestamp" in entries[-1] else 1e9
    exists = (SCRIPTS_DIR / "sys_monitor.py").exists()
    if not exists:
        level = "red"
    elif not entries:
        level = "yellow"
    elif last_age > 24:
        level = "yellow"
    else:
        level = "green"
    age_str = f", last snapshot {last_age:.1f}h ago" if entries else ""
    return {
        "name": "System Monitor", "script": "sys_monitor.py", "level": level,
        "detail": f"{len(entries)} snapshots{age_str}",
    }


def probe_chains() -> dict:
    chains = read_jsonl(CHAINS_DB_FILE)
    exists = (SCRIPTS_DIR / "skill_chain.py").exists()
    level = "red" if not exists else ("green" if chains else "yellow")
    return {
        "name": "Skill Chains", "script": "skill_chain.py", "level": level,
        "detail": f"{len(chains)} chains executed",
    }


def probe_self_heal() -> dict:
    exists = (SCRIPTS_DIR / "self_heal.py").exists()
    level = "green" if exists else "red"
    return {
        "name": "Self-Heal", "script": "self_heal.py", "level": level,
        "detail": "available" if exists else "missing",
    }


def probe_embed_rank() -> dict:
    exists = (SCRIPTS_DIR / "embed_rank.py").exists()
    level = "green" if exists else "red"
    return {
        "name": "Embedding Ranker", "script": "embed_rank.py", "level": level,
        "detail": "TH/EN semantic ranking" if exists else "missing",
    }


def probe_model_router() -> dict:
    exists = (SCRIPTS_DIR / "model_router.py").exists()
    level = "green" if exists else "red"
    return {
        "name": "Model Router", "script": "model_router.py", "level": level,
        "detail": "multi-model routing" if exists else "missing",
    }


def probe_skills_root() -> dict:
    n = 0
    if SKILLS_ROOT.exists():
        n = sum(1 for _ in SKILLS_ROOT.rglob("SKILL.md"))
    level = "green" if n > 0 else "red"
    return {
        "name": "Skills Library", "script": str(SKILLS_ROOT), "level": level,
        "detail": f"{n} skills discovered",
    }


def all_probes() -> list:
    return [
        probe_atsm(),
        probe_skills_root(),
        probe_goal_planner(),
        probe_swarm(),
        probe_knowledge_graph(),
        probe_causal(),
        probe_chains(),
        probe_monitor(),
        probe_embed_rank(),
        probe_model_router(),
        probe_self_heal(),
    ]


# ============================================================
# Dashboard
# ============================================================

def cmd_dashboard() -> int:
    probes = all_probes()
    counts = {"green": 0, "yellow": 0, "red": 0}
    for p in probes:
        counts[p["level"]] += 1

    print()
    print(bold("╔══════════════════════════════════════════════════════════════╗"))
    print(bold("║          ⚡ ATSM UNIFIED CONTROL CENTER — DASHBOARD          ║"))
    print(bold("╚══════════════════════════════════════════════════════════════╝"))
    ok_n, warn_n, fail_n = counts["green"], counts["yellow"], counts["red"]
    summary = f"  {dim(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))}"
    summary += f"  {dim('|')}  {green(str(ok_n) + ' OK')}"
    summary += f"  {yellow(str(warn_n) + ' WARN')}"
    summary += f"  {red(str(fail_n) + ' FAIL')}"
    print(summary)
    print("  " + "─" * 62)

    for p in probes:
        icon = {"green": green("●"), "yellow": yellow("●"), "red": red("●")}[p["level"]]
        print(f"  {icon} {p['name']:<18} {dim(p['detail'])}")

    print("  " + "─" * 62)

    # Aggregate stats row
    atsm = probe_atsm()
    kg = probe_knowledge_graph()
    goals = probe_goal_planner()
    swarm = probe_swarm()
    print(f"  Skills: {kg['nodes']}   Records: {atsm['records']}   "
          f"Goals: {goals['goals']}   Solves: {swarm['solves']}")

    # Success rate bar
    if atsm["records"]:
        print(f"  Success rate: {bar(atsm['success_rate'])} {atsm['success_rate']:.0%}")
    else:
        print(f"  Success rate: {dim('no data yet — use: control run <task>')}")

    # Overall verdict
    if counts["red"] > 0:
        verdict = red(f"✗ {counts['red']} module(s) DOWN")
    elif counts["yellow"] > 0:
        verdict = yellow(f"◐ {counts['yellow']} module(s) need attention")
    else:
        verdict = green("✓ ALL SYSTEMS OPERATIONAL")
    print(f"  Overall: {verdict}")
    print()
    return 0


# ============================================================
# Commands
# ============================================================

def cmd_rank(task: str) -> int:
    print(bold(f"\n⚡ ATSM Ranking — “{task}”"))
    print("─" * 62)
    code, out = run_script("atsm.py", ["rank", task])
    print(out)
    return code


def cmd_run(task: str) -> int:
    print(bold(f"\n▶ Execute Top Skill — “{task}”"))
    print("─" * 62)
    # Rank first to show what will run
    code, out = run_script("atsm.py", ["rank", task])
    print(out)
    if code != 0:
        return code
    # Execute via skill chain (plans + runs top skill per stage)
    print(dim("\n→ dispatching to skill chain executor…"))
    code, out = run_script("skill_chain.py", ["run", task])
    print(out)
    return code


def cmd_plan(goal: str) -> int:
    print(bold(f"\n📋 Goal Planning — “{goal}”"))
    print("─" * 62)
    code, out = run_script("goal_planner.py", ["plan", goal])
    print(out)
    return code


def cmd_swarm(task: str) -> int:
    print(bold(f"\n🐝 Swarm Solve — “{task}”"))
    print("─" * 62)
    code, out = run_script("swarm.py", ["solve", task], timeout=300)
    print(out)
    return code


def cmd_graph() -> int:
    print(bold("\n🕸 Knowledge Graph Stats"))
    print("─" * 62)
    g = read_json(GRAPH_FILE, {}) or {}
    meta = g.get("meta", {})
    nodes = g.get("nodes", {})
    if not meta and not nodes:
        print(yellow("No graph built yet — building now…"))
        code, out = run_script("knowledge_graph.py", ["build"])
        print(out)
        if code != 0:
            return code
        g = read_json(GRAPH_FILE, {}) or {}
        meta = g.get("meta", {})
        nodes = g.get("nodes", {})

    print(f"  Skills (nodes):      {meta.get('num_skills', len(nodes))}")
    print(f"  Outcome records:     {meta.get('num_records', len(read_jsonl(DB_FILE)))}")
    print(f"  Similarity edges:    {meta.get('num_similarity_edges', 0)}")
    print(f"  Co-occurrence edges: {meta.get('num_cooccurrence_edges', 0)}")
    print(f"  Sequence edges:      {meta.get('num_sequence_edges', 0)}")
    print(f"  Causal edges:        {meta.get('num_causal_edges', 0)}")
    built = meta.get("built_at", "")
    if built:
        print(f"  Built:               {built} ({parse_age(built):.1f}h ago)")
    total_edges = (meta.get("num_similarity_edges", 0) + meta.get("num_cooccurrence_edges", 0)
                   + meta.get("num_sequence_edges", 0) + meta.get("num_causal_edges", 0))
    n = meta.get("num_skills", len(nodes)) or 1
    possible = n * (n - 1) / 2 if n > 1 else 1
    density = min(total_edges / possible, 1.0)
    print(f"  Total edges:         {total_edges} ({total_edges / n:.1f} per skill)")
    print(f"  Density:             {bar(density)} {density:.1%} of possible pairs")
    # Top connected skills
    if nodes:
        top = sorted(nodes.items(),
                     key=lambda kv: len(kv[1].get("neighbors", {})),
                     reverse=True)[:5]
        names = ", ".join(f"{k} ({len(v.get('neighbors', {}))})" for k, v in top)
        print(f"  Most connected:      {names}")
    print()
    return 0


def cmd_causal() -> int:
    print(bold("\n🔬 Causal Analysis"))
    print("─" * 62)
    code, out = run_script("causal_infer.py", ["analyze"])
    print(out)
    return code


def cmd_monitor() -> int:
    print(bold("\n🖥 System Metrics"))
    print("─" * 62)
    code, out = run_script("sys_monitor.py", ["snapshot"])
    print(out)
    return code


def cmd_health() -> int:
    print(bold("\n❤ ATSM Health Check"))
    print("─" * 62)

    # Static probes
    for p in all_probes():
        icon = {"green": green("●"), "yellow": yellow("●"), "red": red("●")}[p["level"]]
        print(f"  {icon} {p['name']:<18} {p['detail']}")

    # Live smoke test: does the engine actually respond?
    print("  " + "─" * 58)
    print(dim("  Live smoke test: atsm.py rank 'health check'…"))
    t0 = time.time()
    code, out = run_script("atsm.py", ["rank", "health check"], timeout=60)
    dt = (time.time() - t0) * 1000
    if code == 0 and "No skills found" not in out:
        print(f"  {green('●')} Engine responds in {dt:.0f}ms")
        engine_ok = True
    else:
        print(f"  {red('●')} Engine failed: {out.splitlines()[0] if out else 'no output'}")
        engine_ok = False

    # Self-heal check if available
    if (SCRIPTS_DIR / "self_heal.py").exists():
        print(dim("  Running self_heal.py check…"))
        code, out = run_script("self_heal.py", ["check"], timeout=120)
        ansi = re.compile(r"\033\[[0-9;]*m")
        first = ""
        for l in out.splitlines():
            plain = ansi.sub("", l).strip()
            if plain and not set(plain) <= {"=", "-", "─", " "}:
                first = plain
                break
        tag = green("●") if code == 0 else yellow("◐")
        print(f"  {tag} Self-heal: {first or ('ok' if code == 0 else 'issues found')}")

    probes = all_probes()
    reds = sum(1 for p in probes if p["level"] == "red")
    if reds == 0 and engine_ok:
        print(f"\n  {green('✓ ATSM HEALTHY')}\n")
        return 0
    print(f"\n  {red(f'✗ {reds} module(s) failing, engine ' + ('ok' if engine_ok else 'down'))}\n")
    return 1


# ============================================================
# Interactive REPL
# ============================================================

HELP_TEXT = """Commands:
  rank <task>       ATSM ranking for a task
  run <task>        Execute top skill for a task
  plan <goal>       Decompose a goal into subgoals + skills
  swarm <task>      Swarm solve (4 agents vote)
  graph             Knowledge graph stats
  causal            Causal analysis
  monitor           System metrics snapshot
  health            Overall ATSM health check
  dashboard         Full module status dashboard
  help              Show this help
  quit / exit       Leave interactive mode"""


def cmd_interactive() -> int:
    print(bold("\n⚡ ATSM Unified Control Center — Interactive Mode"))
    print("Type 'help' for commands, 'quit' to exit.\n")
    while True:
        try:
            line = input(cyan("atsm> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye 👋")
            return 0
        if not line:
            continue
        parts = line.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""
        if cmd in ("quit", "exit", "q"):
            print("bye 👋")
            return 0
        elif cmd == "help":
            print(HELP_TEXT)
        elif cmd in ("rank", "run", "plan", "swarm"):
            if not arg:
                print(yellow(f"Usage: {cmd} <{'goal' if cmd == 'plan' else 'task'} description>"))
                continue
            dispatch(cmd, arg)
        elif cmd in ("graph", "causal", "monitor", "health", "dashboard"):
            dispatch(cmd, "")
        else:
            print(yellow(f"Unknown command: {cmd} — type 'help'"))
        print()


def dispatch(cmd: str, arg: str) -> int:
    handlers = {
        "rank": lambda: cmd_rank(arg),
        "run": lambda: cmd_run(arg),
        "plan": lambda: cmd_plan(arg),
        "swarm": lambda: cmd_swarm(arg),
        "graph": cmd_graph,
        "causal": cmd_causal,
        "monitor": cmd_monitor,
        "health": cmd_health,
        "dashboard": cmd_dashboard,
        "interactive": cmd_interactive,
    }
    h = handlers.get(cmd)
    if h is None:
        print(red(f"Unknown command: {cmd}"))
        print(__doc__)
        return 1
    return h()


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    cmd = sys.argv[1].lower()
    arg = " ".join(sys.argv[2:]).strip()

    needs_arg = ("rank", "run", "plan", "swarm")
    if cmd in needs_arg and not arg:
        print(red(f"Error: '{cmd}' requires a task/goal description"))
        print(f"Example: python3 unified_control.py {cmd} \"search papers\"")
        return 1

    return dispatch(cmd, arg)


if __name__ == "__main__":
    sys.exit(main())
