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

## Files

- `scripts/atsm.py` — main engine
- `data/atsm_db.jsonl` — outcome log
- `data/atsm_priors.json` — learned priors
