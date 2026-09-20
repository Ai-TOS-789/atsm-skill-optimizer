---
name: atsm-self-update
description: "Update ATSM skill from git and verify integrity."
---

# ATSM Self-Update

Pulls latest ATSM skill from GitHub, verifies SHA-256 integrity.

## Usage

```bash
python3 scripts/atsm_update.py
# or with custom repo
python3 scripts/atsm_update.py https://github.com/user/repo --branch main
```

## Integrity Check

After update, verifies:
- `SKILL.md` exists
- `scripts/atsm.py` has no syntax errors
- SHA-256 manifest updated on success

## Cron Setup

To auto-update daily:
```
hermes cron add --schedule "0 4 * * *" --skill atsm-self-update --message "Update ATSM skill"
```
