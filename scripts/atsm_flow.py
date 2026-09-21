#!/usr/bin/env python3
"""ATSM Visual Flow Diagram Generator.

Generates a visual flow diagram of ATSM (Adaptive Task-Skill Matching)
operations using Graphviz DOT language.

Usage:
    python3 atsm_flow.py generate   # Generate DOT file
    python3 atsm_flow.py render     # Generate PNG via graphviz
    python3 atsm_flow.py show       # Open DOT in browser as SVG
"""

import argparse
import os
import subprocess
import sys
import tempfile
import webbrowser
from pathlib import Path

# --- Configuration ---
OUTPUT_DIR = Path(__file__).parent / "output"
DOT_FILE = OUTPUT_DIR / "atsm_flow.dot"
PNG_FILE = OUTPUT_DIR / "atsm_flow.png"
SVG_FILE = OUTPUT_DIR / "atsm_flow.svg"

# --- Color Scheme ---
COLORS = {
    "input": "#4CAF50",       # Green: data input/output
    "analysis": "#2196F3",    # Blue: analysis modules
    "execution": "#FF9800",   # Orange: execution
    "verification": "#9C27B0", # Purple: verification
}

# --- Node Definitions ---
NODES = [
    # (id, label, color, shape)
    ("input", "Task Description\n(Input)", COLORS["input"], "box"),
    ("atsm_rank", "ATSM Rank\n(Bayesian + TF-IDF)", COLORS["analysis"], "box"),
    ("embed_rank", "Embed Rank\n(Semantic Similarity)", COLORS["analysis"], "box"),
    ("causal", "Causal Inference\n(Granger Test)", COLORS["analysis"], "box"),
    ("goal_plan", "Goal Planning\n(Decomposition)", COLORS["analysis"], "box"),
    ("skill_chain", "Skill Chaining\n(Sequence)", COLORS["analysis"], "box"),
    ("execution", "Execution\n(Hermes CLI)", COLORS["execution"], "box"),
    ("verification", "Verification\n(Check Result)", COLORS["verification"], "box"),
]

# --- Edge Definitions ---
EDGES = [
    # (from, to, label)
    ("input", "atsm_rank", "task text"),
    ("input", "embed_rank", "task text"),
    ("input", "causal", "historical data"),
    ("atsm_rank", "goal_plan", "ranked skills"),
    ("embed_rank", "goal_plan", "semantic scores"),
    ("causal", "goal_plan", "causal links"),
    ("goal_plan", "skill_chain", "subgoals"),
    ("skill_chain", "execution", "skill sequence"),
    ("execution", "verification", "result"),
    ("verification", "atsm_rank", "feedback (success/fail)"),
    ("verification", "embed_rank", "feedback (embedding update)"),
]


def generate_dot() -> str:
    """Generate the DOT language representation of the ATSM flow."""
    lines = [
        "digraph ATSM_Flow {",
        '    rankdir=LR;',
        '    bgcolor="#FAFAFA";',
        '    label="ATSM: Adaptive Task-Skill Matching Flow";',
        '    labelloc=t;',
        '    fontsize=20;',
        '    fontname="Helvetica-Bold";',
        '    node [fontname="Helvetica", fontsize=11, style=filled, fillcolor=white, penwidth=2];',
        '    edge [fontname="Helvetica", fontsize=9, color="#666666"];',
        "",
        "    // --- Nodes ---",
    ]

    # Add nodes
    for node_id, label, color, shape in NODES:
        lines.append(f'    {node_id} [label="{label}", fillcolor="{color}", shape={shape}, fontcolor=white, style="filled,rounded"];')

    lines.append("")
    lines.append("    // --- Edges ---")

    # Add edges
    for from_node, to_node, label in EDGES:
        lines.append(f'    {from_node} -> {to_node} [label="{label}"];')

    # Add a feedback loop indicator
    lines.append("")
    lines.append("    // --- Feedback Loop ---")
    lines.append('    feedback [label="Feedback\nLoop", shape=plaintext, fontcolor="#666666", fontsize=10];')
    lines.append('    feedback -> verification [style=invis];')

    lines.append("}")

    return "\n".join(lines)


def cmd_generate() -> int:
    """Generate the DOT file."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dot_content = generate_dot()
    DOT_FILE.write_text(dot_content)
    print(f"Generated DOT file: {DOT_FILE}")
    return 0


def cmd_render() -> int:
    """Render the DOT file to PNG using graphviz."""
    if not DOT_FILE.exists():
        print("DOT file not found. Run 'generate' first.", file=sys.stderr)
        return 1

    # Check if graphviz is available
    try:
        subprocess.run(["dot", "-V"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Graphviz not installed. Installing...", file=sys.stderr)
        try:
            subprocess.run(["sudo", "apt-get", "install", "-y", "graphviz"], check=True)
        except subprocess.CalledProcessError:
            print("Failed to install graphviz. Please install manually.", file=sys.stderr)
            return 1

    try:
        subprocess.run(["dot", "-Tpng", str(DOT_FILE), "-o", str(PNG_FILE)], check=True)
        print(f"Generated PNG: {PNG_FILE}")
    except subprocess.CalledProcessError as e:
        print(f"Failed to render PNG: {e}", file=sys.stderr)
        return 1

    return 0


def cmd_show() -> int:
    """Show the flow diagram in browser as SVG."""
    if not DOT_FILE.exists():
        print("DOT file not found. Run 'generate' first.", file=sys.stderr)
        return 1

    # Try to generate SVG if graphviz is available
    svg_content = None
    try:
        result = subprocess.run(
            ["dot", "-Tsvg", str(DOT_FILE)],
            capture_output=True, text=True, check=True
        )
        svg_content = result.stdout
        SVG_FILE.write_text(svg_content)
        print(f"Generated SVG: {SVG_FILE}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Graphviz not available, generating fallback HTML viewer...")

    # Create an HTML viewer
    if svg_content:
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ATSM Flow Diagram</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #FAFAFA; font-family: Helvetica, Arial, sans-serif; }}
        h1 {{ text-align: center; color: #333; }}
        .container {{ max-width: 1200px; margin: 0 auto; text-align: center; }}
        svg {{ max-width: 100%; height: auto; }}
    </style>
</head>
<body>
    <h1>ATSM: Adaptive Task-Skill Matching Flow</h1>
    <div class="container">
        {svg_content}
    </div>
</body>
</html>"""
    else:
        # Fallback: render DOT as text in HTML
        dot_content = DOT_FILE.read_text()
        html_content = f"""<!DOCTYPE html>
<html>
<head>
    <title>ATSM Flow Diagram (DOT Source)</title>
    <style>
        body {{ margin: 0; padding: 20px; background: #FAFAFA; font-family: Helvetica, Arial, sans-serif; }}
        h1 {{ text-align: center; color: #333; }}
        .container {{ max-width: 900px; margin: 0 auto; }}
        pre {{ background: #263238; color: #AED581; padding: 20px; border-radius: 8px; overflow-x: auto; font-size: 13px; line-height: 1.5; }}
        .info {{ background: #E3F2FD; padding: 15px; border-radius: 8px; margin-bottom: 20px; }}
    </style>
</head>
<body>
    <h1>ATSM: Adaptive Task-Skill Matching Flow</h1>
    <div class="container">
        <div class="info">
            <strong>Graphviz not installed.</strong> Showing DOT source.
            Install graphviz (<code>sudo apt-get install graphviz</code>) to render as SVG.
        </div>
        <pre>{dot_content}</pre>
    </div>
</body>
</html>"""

    # Write to temp file and open in browser
    html_file = OUTPUT_DIR / "atsm_flow.html"
    html_file.write_text(html_content)
    print(f"Opening in browser: {html_file}")
    webbrowser.open(f"file://{html_file.absolute()}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ATSM Visual Flow Diagram Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "command",
        choices=["generate", "render", "show"],
        help="Command to execute",
    )
    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "render": cmd_render,
        "show": cmd_show,
    }

    return commands[args.command]()


if __name__ == "__main__":
    sys.exit(main())
