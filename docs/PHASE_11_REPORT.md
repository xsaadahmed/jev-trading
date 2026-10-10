# Phase 11 Report — Reporting

**Date:** 2026-10-08  
**Status:** Complete  
**Spec:** MASTER_SPEC §§16, 16.4, 21 Phase 11, 21A  
**Preserved:** Phases 0–10, 13A

---

## What was built

### `reporting/build_report.py`

Builds **HTML + Markdown** from an experiment **manifest only** (no recomputation). Optional sibling artifacts may be referenced by `*_path` keys.

Sections (missing optionals → placeholder, report still builds):

1. Strategy definition  
2. Dataset and splits (sealed / interim / forward / walk-forward + contamination status)  
3. Feature definitions  
4. Jev question definitions (exact text + hash)  
5. Equity and drawdown curves  
6. Monthly return table  
7. Trade distribution  
8. Baseline comparison  
9. Jev comparison  
10. Calibration / reliability  
11. Placebo histograms (observed marked)  
12. Cost sensitivity  
13. Robustness (data only)  
14. Statistical uncertainty + power statement  
15. Contamination-diagnostics summary (each labeled **warning signal, not proof**)  
16. Three-status table (`HISTORICAL_RESEARCH_PASS` / `TINY_LIVE_READY` / `SCALE_READY` + digests)  
17. Final conclusion  

### `reporting/conclusion.py`

Rule-based verdicts (exact strings when required):

| Kind | Text |
| ---- | ---- |
| `no_value` | `Jev does not demonstrate useful predictive value.` |
| `underpowered` | `Inconclusive: study underpowered.` |
| `pass` | Continues under `HISTORICAL_RESEARCH_PASS` constraints; does **not** claim an uncontaminated edge or authorize live capital alone |

Never emits pass language when any gate fails. Never implies live $10 trading is evidence of an edge.

### `reporting/statements.py` / `plots.py` / `templates/report.html.j2`

§16.4 mandatory banners in every report; Plotly HTML snippets for equity, placebo, reliability, trades.

### CLI

```bash
jev report build path/to/manifest.json
jev report build path/to/manifest.json --out-dir reports/<exp>/
```

---

## Sample output

`reports/phase11_golden/` from `tests/fixtures/phase11_manifest_full.json`.

---

## Tests

`tests/unit/test_phase11_reporting.py`:

- Golden excerpt render  
- One test per verdict path (pass / no_value / underpowered / MDE / unevaluable)  
- Mandatory §16.4 presence  
- Minimal manifest (missing optional sections) still builds  
- Safety assert rejects pass language on failure  

---

## Checks

```text
make check  →  ruff + mypy + pytest  (all passed)
```

---

## Decisions

1. Manifest is the sole input; numbers/plots are embedded or path-referenced — never recomputed.  
2. Pass verdict is a fixed paragraph that cites `HISTORICAL_RESEARCH_PASS` limits and contamination.  
3. Stop after Phase 11 — no Phase 12 sealed-test work.
