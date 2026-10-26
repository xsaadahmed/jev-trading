"""V2 frozen hashes and training-cutoff assertions (no sealed rows)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from jevlab.cv.purge import embargo_cutoff_date
from jevlab.utils.calendar import nyse_session_dates
from jevlab.v2.candidate import assert_v2_candidate_semantics
from jevlab.v2.dev_io import SealedPathError
from jevlab.v2.hashes import (
    FROZEN_ACCEPTANCE_GATE_HASH,
    FROZEN_CANDIDATE_HASH,
    FROZEN_RANDOM_NULL_HASH,
    FROZEN_RESEARCH_CONFIG_HASH,
    assert_frozen_research_hashes,
)
from jevlab.v2.training import (
    EXPECTED_CUTOFF,
    SEALED_T0,
    TrainingLeakError,
    assert_sealed_training_cutoff,
    filter_training_rows,
    fit_once_model_a,
)


def test_frozen_hashes_match_live_config() -> None:
    live = assert_frozen_research_hashes()
    assert live["research_config_hash"] == FROZEN_RESEARCH_CONFIG_HASH
    assert live["candidate"] == FROZEN_CANDIDATE_HASH
    assert live["random_null"] == FROZEN_RANDOM_NULL_HASH
    assert live["acceptance_gates"] == FROZEN_ACCEPTANCE_GATE_HASH


def test_candidate_semantics_match_v1() -> None:
    assert_v2_candidate_semantics()


def test_training_cutoff_is_2023_12_22() -> None:
    cutoff = assert_sealed_training_cutoff()
    assert cutoff == date(2023, 12, 22)
    assert cutoff == EXPECTED_CUTOFF
    sessions = nyse_session_dates(date(2023, 11, 1), date(2024, 1, 15))
    derived = embargo_cutoff_date(sessions, SEALED_T0, embargo_sessions=5)
    assert derived == cutoff
    assert sessions.index(SEALED_T0) - sessions.index(cutoff) == 5


def test_filter_training_rows_rejects_t0(tmp_path: object) -> None:
    frame = pd.DataFrame(
        {
            "symbol": ["SPY", "SPY"],
            "date": [date(2023, 12, 22), date(2024, 1, 2)],
            "y": [1, 0],
        }
    )
    with pytest.raises((TrainingLeakError, SealedPathError)):
        filter_training_rows(frame, cutoff=date(2023, 12, 22), t0=date(2024, 1, 2))


def test_fit_once_refuses_post_cutoff_rows() -> None:
    import numpy as np

    from jevlab.features.columns import MODEL_A_COLUMNS
    from jevlab.utils.calendar import nyse_session_dates

    sessions = nyse_session_dates(date(2021, 1, 4), date(2021, 6, 30))
    t0 = sessions[40]
    cutoff = embargo_cutoff_date(sessions, t0, embargo_sessions=5)
    assert cutoff is not None
    rng = np.random.default_rng(0)
    rows = []
    for d in sessions:
        for sym in ("AAA", "BBB", "CCC", "DDD"):
            feats = {c: float(rng.normal()) for c in MODEL_A_COLUMNS}
            rows.append(
                {
                    "symbol": sym,
                    "date": d,
                    **feats,
                    "eligible": 1,
                    "y": int(rng.integers(0, 2)),
                    "X": 0.0,
                    "R": 0.0,
                    "Rbar": 0.0,
                    "y_cost": 0,
                }
            )
    frame = pd.DataFrame(rows)
    with pytest.raises(TrainingLeakError, match="T0 or later"):
        fit_once_model_a(frame, sessions=sessions, t0=t0, cutoff=cutoff)
    train_only = frame.loc[frame["date"] <= cutoff].copy()
    fitted = fit_once_model_a(train_only, sessions=sessions, t0=t0, cutoff=cutoff)
    assert fitted.fit_count == 1
    assert fitted.cutoff == cutoff
    assert fitted.C in {0.01, 0.1, 1.0}
    assert (train_only["date"] < t0).all()
