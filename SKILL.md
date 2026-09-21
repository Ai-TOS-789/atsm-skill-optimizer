---
name: atsm-skill-optimizer
description: "ATSM: rank skills by relevance and success history."
---

# ATSM Skill Optimizer

## Overview

ATSM ใช้ Bayesian Beta-Binomial update เพื่อจัดลำดับ skill ตามความเหมาะสมกับ task โดยดูจาก:

1. **Keyword overlap** ระหว่าง task กับ skill description
2. **Historical success rate** ของ skill
3. **Recency decay** — ผลล่าสุดหนักกว่า

## วิธีใช้

```bash
# จัดอันดับ skill สำหรับ task ใหม่
python3 scripts/atsm.py rank "search and summarize papers"

# บันทึกผลการทำงาน (1=สำเร็จ, 0=ไม่สำเร็จ)
python3 scripts/atsm.py record atsm-skill-optimizer 1

# ดูสถิติ
python3 scripts/atsm.py stats
```

## อัลกอริทึม

1. **TF-IDF-like relevance** — cosine ระหว่าง task ↔ skill description
2. **Success weighting** — E[success] จาก Beta(α+s, β+f)
3. **Final score** = relevance × P(success|context)

## อัลกอริทึม (v2.3)

1. **TF-IDF-like relevance** — cosine ระหว่าง task ↔ skill description
2. **Success weighting** — E[success] จาก Beta(α+s, β+f)
3. **Final score** = relevance × P(success|context)
4. **Cross-agent learning** — agent A สำเร็จ skill X → boost ให้ agent B ด้วย

## Subsystems

| สิ่ง | ไฟล์ | สถานะ |
|------|------|--------|
| Voice Assistant | `scripts/hermes_voice.py` | ✅ ใช้งานได้ |
| Wake Word Detection | `scripts/hermes_wake.py` | ✅ ใช้งานได้ |
| Proactive Agent Loop | `scripts/proactive_agent.py` | ✅ daemon รันอยู่ |
| Self-Learning Engine | `scripts/atsm_learn.py` | ✅ pattern detection |
| Agent Memory | `scripts/agent_memory.py` | ✅ persistent memory |
| Alert System | `scripts/agent_alerts.py` | ✅ desktop notifications |
| Skill Chain Planner | `scripts/skill_chain.py` | ✅ chain execution |
| Multi-Agent Coord | `scripts/agent_coord.py` | ✅ message bus |
| Browser Bot | `scripts/browser_bot.py` | ✅ CDP automation |
| System Monitor | `scripts/sys_monitor.py` | ✅ CPU/Mem/GPU |
| Embedding Ranker | `scripts/embed_rank.py` | ✅ TH/EN semantic |
| Self-Update | `scripts/atsm_update.py` | ✅ cron 4am |

ดูรายละเอียดแต่ละ subsystem ใน `references/`

## Files

- `scripts/atsm.py` — main engine (Bayesian ranker + multi-agent)
- `data/atsm_db.jsonl` — outcome log
- `data/atsm_priors.json` — learned priors
- `references/voice-assistant.md` — voice pipeline details
- `references/proactive-agent.md` — proactive loop details
