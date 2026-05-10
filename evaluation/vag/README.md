# VaG Evaluation

This directory contains paper reproduction code for Verifier-as-Gatekeeper
(VaG). Reusable runtime APIs live in `agentica/skills/`; this directory should
only contain runners, adapters, baselines, pilot data, and analysis scripts.

Current runnable pilot:

```bash
python evaluation/vag/runners/regression_injection.py
```

The pilot generates a deterministic 60-candidate semi-real Regression
Injection set from Agentica-style skill patterns, writes it to
`evaluation/vag/data/60_pilot_candidates.jsonl`, and reports harmful admission
plus benign retention for each gate configuration.

This is a P0 smoke/pilot runner, not the final 200-sample labelled paper
dataset. Final paper numbers should replace the generated pilot with
human-labelled candidates derived from real Agentica/Terminal-Bench traces.
