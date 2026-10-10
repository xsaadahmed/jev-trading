"""Column-set registry for models A-G (MASTER_SPEC §4.3 / §21A).

Asserted sizes:
  A=17, B=25 (A+8 Jev), C=8, D=17, E=25, F=25 (A+8 placebo),
  G=17 + one-hot bucket dummies (drop-first; count logged at build time).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from jevlab.features.columns import MODEL_A_COLUMNS
from jevlab.jev.parse import JEV_FEATURE_COLUMNS

ModelLineupId = Literal["A", "B", "C", "D", "E", "F", "G"]
LearnerKind = Literal["logistic", "lightgbm"]
TrialPurpose = Literal["confirmatory", "exploratory"]

# Placebo Jev slots — interface only until Phase 10 fills values.
PLACEBO_JEV_COLUMNS: tuple[str, ...] = tuple(f"placebo_{c}" for c in JEV_FEATURE_COLUMNS)

# Jev probability-like columns that receive clip+logit before scaling.
# j_downside_E is a score expectation (not a probability) and is left raw.
JEV_LOGIT_COLUMNS: tuple[str, ...] = tuple(c for c in JEV_FEATURE_COLUMNS if c != "j_downside_E")

MODEL_B_COLUMNS: tuple[str, ...] = MODEL_A_COLUMNS + JEV_FEATURE_COLUMNS
MODEL_C_COLUMNS: tuple[str, ...] = JEV_FEATURE_COLUMNS
MODEL_D_COLUMNS: tuple[str, ...] = MODEL_A_COLUMNS
MODEL_E_COLUMNS: tuple[str, ...] = MODEL_A_COLUMNS + JEV_FEATURE_COLUMNS
MODEL_F_COLUMNS: tuple[str, ...] = MODEL_A_COLUMNS + PLACEBO_JEV_COLUMNS

# Size assertions (G's one-hot width is dynamic; checked when built).
assert len(MODEL_A_COLUMNS) == 17
assert len(JEV_FEATURE_COLUMNS) == 8
assert len(MODEL_B_COLUMNS) == 25
assert len(MODEL_C_COLUMNS) == 8
assert len(MODEL_D_COLUMNS) == 17
assert len(MODEL_E_COLUMNS) == 25
assert len(MODEL_F_COLUMNS) == 25
assert len(PLACEBO_JEV_COLUMNS) == 8
assert set(MODEL_B_COLUMNS) - set(MODEL_A_COLUMNS) == set(JEV_FEATURE_COLUMNS)
assert MODEL_B_COLUMNS[:17] == MODEL_A_COLUMNS
assert MODEL_E_COLUMNS == MODEL_B_COLUMNS


@dataclass(frozen=True, slots=True)
class ColumnSetSpec:
    """One lineup entry: columns, learner, trial purpose, notes."""

    model_id: ModelLineupId
    columns: tuple[str, ...]
    learner: LearnerKind
    purpose: TrialPurpose
    n_columns: int
    logit_columns: tuple[str, ...] = ()
    interface_only: bool = False
    notes: str = ""

    def __post_init__(self) -> None:
        if self.n_columns != len(self.columns):
            raise AssertionError(
                f"{self.model_id}: n_columns={self.n_columns} != len(columns)={len(self.columns)}"
            )


def assert_ab_column_diff() -> None:
    """B must equal A plus exactly the eight Jev feature columns."""
    assert len(MODEL_A_COLUMNS) == 17
    assert len(MODEL_B_COLUMNS) == 25
    assert MODEL_B_COLUMNS[: len(MODEL_A_COLUMNS)] == MODEL_A_COLUMNS
    assert MODEL_B_COLUMNS[len(MODEL_A_COLUMNS) :] == JEV_FEATURE_COLUMNS
    extra = set(MODEL_B_COLUMNS) - set(MODEL_A_COLUMNS)
    assert extra == set(JEV_FEATURE_COLUMNS)


def column_set_registry(
    *,
    model_g_columns: tuple[str, ...] | None = None,
) -> dict[ModelLineupId, ColumnSetSpec]:
    """Return the A-G registry. ``model_g_columns`` supplied after one-hot build."""
    assert_ab_column_diff()
    reg: dict[ModelLineupId, ColumnSetSpec] = {
        "A": ColumnSetSpec(
            model_id="A",
            columns=MODEL_A_COLUMNS,
            learner="logistic",
            purpose="confirmatory",
            n_columns=17,
            notes="primary baseline",
        ),
        "B": ColumnSetSpec(
            model_id="B",
            columns=MODEL_B_COLUMNS,
            learner="logistic",
            purpose="confirmatory",
            n_columns=25,
            logit_columns=JEV_LOGIT_COLUMNS,
            notes="primary treatment; identical pipeline/grid/folds/seed as A",
        ),
        "C": ColumnSetSpec(
            model_id="C",
            columns=MODEL_C_COLUMNS,
            learner="logistic",
            purpose="exploratory",
            n_columns=8,
            logit_columns=JEV_LOGIT_COLUMNS,
            notes="Jev-only logistic",
        ),
        "D": ColumnSetSpec(
            model_id="D",
            columns=MODEL_D_COLUMNS,
            learner="lightgbm",
            purpose="exploratory",
            n_columns=17,
            notes="LightGBM on A columns",
        ),
        "E": ColumnSetSpec(
            model_id="E",
            columns=MODEL_E_COLUMNS,
            learner="lightgbm",
            purpose="exploratory",
            n_columns=25,
            logit_columns=JEV_LOGIT_COLUMNS,
            notes="LightGBM on A+Jev",
        ),
        "F": ColumnSetSpec(
            model_id="F",
            columns=MODEL_F_COLUMNS,
            learner="logistic",
            purpose="exploratory",
            n_columns=25,
            logit_columns=tuple(f"placebo_{c}" for c in JEV_LOGIT_COLUMNS),
            interface_only=True,
            notes="placebo Jev columns; values filled in Phase 10",
        ),
    }
    if model_g_columns is not None:
        n_onehot = len(model_g_columns) - len(MODEL_A_COLUMNS)
        if n_onehot < 0 or model_g_columns[:17] != MODEL_A_COLUMNS:
            raise AssertionError("G columns must start with MODEL_A_COLUMNS")
        reg["G"] = ColumnSetSpec(
            model_id="G",
            columns=model_g_columns,
            learner="logistic",
            purpose="exploratory",
            n_columns=len(model_g_columns),
            notes=f"A + drop-first bucket one-hot ({n_onehot} dummies)",
        )
    return reg


CONTAMINATION_LABEL: str = "potentially_training_contaminated"
