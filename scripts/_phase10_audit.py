"""Phase 10 completeness audit (read-only)."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

out = Path("reports/robustness_report")
p = json.loads((out / "placebo_results.json").read_text(encoding="utf-8"))
r = json.loads((out / "robustness_data.json").read_text(encoding="utf-8"))
print("PLACEBO keys", sorted(p.keys()))
for k in ["P1", "P2", "P3", "P4", "P5", "P6", "Model_F"]:
    if k not in p:
        print(k, "MISSING")
        continue
    d = p[k]
    if k in ("P1", "P2", "P3", "P4"):
        print(
            k,
            "n=",
            d.get("n"),
            "gains=",
            len(d.get("placebo_delta_log_loss_gains") or []),
            "p=",
            d.get("pvalue_log_loss"),
        )
    elif k == "P5":
        print("P5", {x: d.get(x) for x in ("observed_gain", "predicted_only_gain", "residual_only_gain")})
    elif k == "P6":
        print(
            "P6 status",
            d.get("status"),
            "live",
            d.get("live"),
            "n",
            d.get("n_shuffled_states"),
            "work_queue",
            d.get("work_queue_n"),
            "outcome_rel",
            d.get("outcome_relationship"),
        )
    else:
        print("F", d.get("n_columns"), (d.get("metrics") or {}).get("log_loss"))

print(
    "ROBUST sizes",
    {
        k: (len(v) if isinstance(v, list) else type(v).__name__)
        for k, v in r.items()
        if k != "meta"
    },
)
print("regimes", [x.get("regime") for x in r.get("regimes", [])])
print("phases", [x.get("phase") for x in r.get("phases", [])])
print("k", [x.get("top_k") for x in r.get("k_sweep", [])])
print("cost", [x.get("scenario") for x in r.get("cost_sweep", [])])
print("delayed", r.get("delayed_execution"))
print("reg", r.get("regularization"))
print("choice n", len(r.get("choice_order", [])))
print("paraphrase", json.dumps(r.get("paraphrase"), indent=2)[:1500])
print(
    "feat abl",
    len(r.get("feature_ablation", [])),
    "jev abl",
    len(r.get("jev_question_ablation", [])),
    "loo",
    len(r.get("leave_one_etf_out", [])),
)
print("horizon keys", [k for k in r if "horizon" in k.lower()])
print("digests", (out / "placebo_results.digest").read_text().strip(), (out / "robustness_data.digest").read_text().strip())

conn = sqlite3.connect("registry/experiments.sqlite")
rows = conn.execute(
    """
    SELECT trial_id, purpose, feature_set, model_class, experiment_id
    FROM trial_ledger
    WHERE experiment_id LIKE 'phase10%'
       OR feature_set LIKE '%placebo%'
       OR feature_set LIKE '%paraphrase%'
    ORDER BY rowid
    """
).fetchall()
print("ledger n", len(rows))
for row in rows:
    print(row)
