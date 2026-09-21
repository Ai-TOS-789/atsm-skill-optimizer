#!/usr/bin/env python3
"""Auto-Skill-Generator: Creates new skills when ATSM finds no match.

When ATSM rank returns all scores below threshold (max < 0.05), this script
triggers interactive skill generation: asks the user for details, creates a
valid SKILL.md with frontmatter + scripts/ template, validates it, and
registers the new skill in ATSM with a neutral prior P(success) = 0.5.

Usage:
    python3 auto_skill.py "description of what I need"
    python3 auto_skill.py --list
    python3 auto_skill.py --validate /path/to/skill
    python3 auto_skill.py --generate [task description]
    python3 auto_skill.py --register <skill_name>
"""

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

# --- Config ---
SCRIPT_DIR = Path(__file__).parent.resolve()
SKILLS_ROOT = Path(os.environ.get("HERMES_SKILLS_ROOT", Path.home() / ".hermes/skills"))
DATA_DIR = SCRIPT_DIR.parent / "data"
DB_FILE = DATA_DIR / "atsm_db.jsonl"
PRIORS_FILE = DATA_DIR / "atsm_priors.json"

# Threshold: if all skill scores are below this, trigger generation
SCORE_THRESHOLD = 0.05

# Default category for auto-generated skills
DEFAULT_CATEGORY = "productivity"


def rank_skills(task: str) -> list[dict]:
    """Rank skills using ATSM engine. Returns results sorted by final_score desc."""
    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        import atsm
        results, _ = atsm.rank_skills(task, top_k=100)
        return results
    except ImportError as e:
        print(f"Error: Cannot import atsm.py: {e}")
        sys.exit(1)
    finally:
        sys.path.pop(0)


def check_threshold(results: list[dict], threshold: float = SCORE_THRESHOLD) -> bool:
    """Check if all scores are below threshold (trigger condition)."""
    if not results:
        return True
    max_score = max(r["final_score"] for r in results)
    return max_score < threshold


def list_skills():
    """List all installed skills."""
    if not SKILLS_ROOT.exists():
        print("No skills directory found.")
        return
    skills = []
    for skill_file in SKILLS_ROOT.rglob("SKILL.md"):
        name = skill_file.parent.name
        category = skill_file.parent.parent.name
        try:
            content = skill_file.read_text(encoding="utf-8")
            desc = ""
            if content.startswith("---"):
                end = content.find("---", 3)
                if end != -1:
                    sys.path.insert(0, str(SCRIPT_DIR))
                    try:
                        import atsm
                        fm = atsm.parse_frontmatter(content)
                        desc = fm.get("description", "")
                    except Exception:
                        pass
                    finally:
                        sys.path.pop(0)
        except Exception:
            desc = ""
        skills.append((category, name, desc))

    if not skills:
        print("No skills found.")
        return

    print(f"\n{'Category':<20} {'Skill':<35} {'Description'}")
    print("-" * 85)
    for category, name, desc in sorted(skills):
        print(f"{category:<20} {name:<35} {desc[:35]}")
    print(f"\nTotal: {len(skills)} skills")


def validate_skill(skill_path: str) -> bool:
    """Validate a skill directory or SKILL.md file.

    Checks:
    - SKILL.md exists
    - Frontmatter starts at byte 0 with ---
    - Frontmatter closes with ---
    - Frontmatter parses as valid YAML
    - Required fields: name, description
    - Description length <= 1024 chars
    - Body exists after frontmatter
    """
    path = Path(skill_path)
    if path.is_dir():
        skill_md = path / "SKILL.md"
    else:
        skill_md = path

    if not skill_md.exists():
        print(f"Error: {skill_md} does not exist")
        return False

    content = skill_md.read_text(encoding="utf-8")
    errors = []
    warnings = []

    # Check frontmatter exists and starts at byte 0
    if not content.startswith("---"):
        errors.append("SKILL.md must start with '---' at byte 0 (no leading whitespace)")
    else:
        end_match = re.search(r'\n---\s*\n', content[3:])
        if not end_match:
            errors.append("Frontmatter must close with '---' followed by blank line")
        else:
            # Extract frontmatter text
            end_idx = content.find("\n---\n", 3)
            if end_idx == -1:
                end_idx = content.find("\n--- \n", 3)
            if end_idx == -1:
                end_idx = content.find("\n---\r\n", 3)

            fm_text = content[3:end_idx] if end_idx != -1 else content[3:]

            # Parse frontmatter
            sys.path.insert(0, str(SCRIPT_DIR))
            try:
                import atsm
                fm = atsm.parse_frontmatter(content)
            except Exception as e:
                errors.append(f"Frontmatter parse error: {e}")
                fm = {}
            finally:
                sys.path.pop(0)

            # Check required fields
            if "name" not in fm:
                errors.append("Missing required field: name")
            if "description" not in fm:
                errors.append("Missing required field: description")
            elif "description" in fm:
                desc = fm["description"]
                if len(desc) > 1024:
                    errors.append(f"Description too long ({len(desc)} chars, max 1024)")
                if len(desc) > 60:
                    warnings.append(f"Description is {len(desc)} chars (recommended ≤ 60)")

            # Check body exists
            body_start = content.find("\n---\n", 3)
            if body_start != -1:
                body_start += len("\n---\n")
            else:
                body_start = content.find("\n--- \n", 3)
                if body_start != -1:
                    body_start += len("\n--- \n")
                else:
                    body_start = len(content)

            body = content[body_start:]
            if len(body.strip()) < 10:
                errors.append("Body is too short or missing (need meaningful content)")

            # Check markdown structure
            if body.strip() and not body.strip().startswith("#"):
                warnings.append("Body should start with a heading (# Title)")

    # Check scripts directory exists
    scripts_dir = skill_md.parent / "scripts"
    if not scripts_dir.exists():
        warnings.append("No scripts/ directory (optional but recommended)")

    # Print results
    if errors:
        print(f"\n❌ Validation FAILED for {skill_md}:")
        for err in errors:
            print(f"  - {err}")
    else:
        print(f"\n✅ Validation PASSED for {skill_md}")

    if warnings:
        for warn in warnings:
            print(f"  ⚠️  {warn}")

    return len(errors) == 0


def sanitize_name(name: str) -> str:
    """Sanitize a skill name to lowercase-hyphenated format."""
    safe = re.sub(r'[^a-zA-Z0-9_\s-]', '', name.lower().strip())
    safe = re.sub(r'[\s_]+', '-', safe)
    safe = re.sub(r'-+', '-', safe).strip('-')
    return safe


def generate_skill(name: str, description: str, purpose: str, category: str = DEFAULT_CATEGORY) -> Path:
    """Generate a new skill from template.

    Creates:
    - <category>/<name>/SKILL.md with frontmatter and body
    - <category>/<name>/scripts/<name>.py with basic template
    """
    safe_name = sanitize_name(name)

    if not safe_name or len(safe_name) > 64:
        print(f"Error: Invalid skill name '{safe_name}' (must be 1-64 chars)")
        return None

    skill_dir = SKILLS_ROOT / category / safe_name
    if skill_dir.exists():
        print(f"Error: Skill directory already exists: {skill_dir}")
        return None

    # Create directories
    skill_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = skill_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)

    # Build YAML frontmatter
    fm_lines = ["---"]
    fm_lines.append(f"name: {safe_name}")

    # Quote description if it contains special YAML characters
    desc = description.strip()
    if any(c in desc for c in [':', '#', '"', "'", '\n', '{', '}', '[', ']']):
        # Escape double quotes and wrap in double quotes
        desc_escaped = desc.replace('\\', '\\\\').replace('"', '\\"')
        fm_lines.append(f'description: "{desc_escaped}"')
    else:
        fm_lines.append(f"description: {desc}")

    fm_lines.append("---")
    fm_text = "\n".join(fm_lines) + "\n\n"

    # Build body
    body = f"""# {safe_name} Skill

{purpose}

## When to Use

- When the user asks to {purpose.lower().rstrip('.')}

## How to Run

```bash
python3 scripts/{safe_name}.py [args]
```

## Procedure

1. Parse input and validate parameters
2. Execute core logic
3. Return results

## Pitfalls

- Document known limitations here

## Verification

- How to verify this skill worked correctly
"""

    skill_md = skill_dir / "SKILL.md"
    skill_md.write_text(fm_text + body, encoding="utf-8")

    # Generate a basic script template
    script_content = f"""#!/usr/bin/env python3
\"\"\"Script for {safe_name} skill.

Generated by auto_skill.py on {datetime.now(timezone.utc).strftime('%Y-%m-%d')}
\"\"\"

import sys


def main():
    \"\"\"Main entry point for {safe_name} skill.\"\"\"
    if len(sys.argv) < 2:
        print("Usage: {safe_name}.py [args]")
        sys.exit(1)

    # TODO: Implement skill logic
    print(f"Running {safe_name} skill")
    return 0


if __name__ == "__main__":
    sys.exit(main())
"""

    script_file = scripts_dir / f"{safe_name}.py"
    script_file.write_text(script_content, encoding="utf-8")
    script_file.chmod(0o755)

    return skill_dir


def register_in_atsm(skill_name: str):
    """Register a new skill in ATSM with neutral prior P(success) = 0.5.

    ATSM uses Beta(α=1, β=1) as the neutral prior, which gives E[success] = 0.5.
    We record a registration event so the skill appears in the system.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Initialize priors file if needed
    if PRIORS_FILE.exists():
        try:
            priors = json.loads(PRIORS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            priors = {}
    else:
        priors = {}

    # Set neutral prior: alpha=1, beta=1 → P(success) = 0.5
    priors[skill_name] = {
        "alpha": 1.0,
        "beta": 1.0,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "observations": 0,
        "source": "auto_skill_generator"
    }

    PRIORS_FILE.write_text(json.dumps(priors, indent=2), encoding="utf-8")

    # Record registration event in DB for tracking
    record = {
        "skill": skill_name,
        "success": 1,
        "agent": "auto-skill-generator",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": "Auto-registered with neutral prior P(success)=0.5"
    }
    with open(DB_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    print(f"  📊 Registered in ATSM with neutral prior P(success) = 0.5")


def interactive_generate(task_description: str = None):
    """Interactive skill generation wizard."""
    print("\n🔧 Auto-Skill Generator")
    print("=" * 40)

    if task_description:
        print(f"\nTask: {task_description}")

    print("\nI'll help you create a new skill. Please provide:\n")

    name = input("  Skill name (lowercase-hyphenated): ").strip()
    if not name:
        print("Cancelled.")
        return

    description = input("  Description (short, what it does): ").strip()
    if not description:
        print("Cancelled.")
        return

    purpose = input("  What should it do? (detailed purpose): ").strip()
    if not purpose:
        purpose = description

    category = input(f"  Category [{DEFAULT_CATEGORY}]: ").strip()
    if not category:
        category = DEFAULT_CATEGORY

    # Generate
    skill_dir = generate_skill(name, description, purpose, category)
    if not skill_dir:
        return

    # Validate
    print(f"\n  Validating...")
    valid = validate_skill(str(skill_dir))

    if valid:
        # Register in ATSM
        safe_name = sanitize_name(name)
        register_in_atsm(safe_name)

        print(f"\n  ✅ Skill created at: {skill_dir}")
        print(f"  📝 Next steps:")
        print(f"     1. Edit {skill_dir}/SKILL.md to refine")
        print(f"     2. Edit {skill_dir}/scripts/{safe_name}.py")
        print(f"     3. Record outcomes: python3 atsm.py record {safe_name} 1")
    else:
        print(f"\n  ⚠️  Skill created but validation failed. Please review.")


def auto_rank_and_generate(task: str):
    """Rank skills for a task; if all below threshold, offer to generate."""
    print(f"🔍 Ranking skills for: '{task}'")

    results = rank_skills(task)

    if not results:
        print("No skills found in system.")
        response = input("\n  Would you like to generate a new skill? [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            interactive_generate(task)
        return

    # Show top results
    max_score = max(r["final_score"] for r in results)
    print(f"\n{'Rank':<5} {'Score':<8} {'Rel':<6} {'P(success)':<11} {'Skill'}")
    print("-" * 60)
    for i, r in enumerate(results[:5], 1):
        print(f"{i:<5} {r['final_score']:<8} {r['relevance']:<6} {r['success_prob']:<11} {r['name']}")

    print(f"\n  Max score: {max_score:.4f} (threshold: {SCORE_THRESHOLD})")

    if check_threshold(results):
        print(f"\n  ⚠️  All scores below threshold ({SCORE_THRESHOLD})")
        print("  → No existing skill matches this task well.")

        response = input("\n  Would you like to generate a new skill? [Y/n]: ").strip().lower()
        if response in ("", "y", "yes"):
            interactive_generate(task)
        else:
            print("  Cancelled.")
    else:
        print(f"\n  ✅ Existing skill(s) match this task (max score: {max_score:.4f})")
        print(f"  → Top match: {results[0]['name']}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]

    if command == "--list":
        list_skills()

    elif command == "--validate":
        if len(sys.argv) < 3:
            print("Usage: auto_skill.py --validate /path/to/skill")
            sys.exit(1)
        success = validate_skill(sys.argv[2])
        sys.exit(0 if success else 1)

    elif command == "--generate":
        task = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else None
        interactive_generate(task)

    elif command == "--register":
        if len(sys.argv) < 3:
            print("Usage: auto_skill.py --register <skill_name>")
            sys.exit(1)
        register_in_atsm(sys.argv[2])

    elif command in ("--help", "-h"):
        print(__doc__)

    else:
        # Treat as task description - rank and auto-generate if below threshold
        task = " ".join(sys.argv[1:])
        auto_rank_and_generate(task)


if __name__ == "__main__":
    main()
