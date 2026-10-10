"""Walk-forward cross-validation, purge/embargo, and sealed-partition guards."""

from jevlab.cv.purge import (
    assert_no_train_label_intersects_test,
    embargo_cutoff_date,
    filter_train_dates,
)
from jevlab.cv.sealed_guard import SealedAccessError, SealedDataGuard
from jevlab.cv.walkforward import Fold, make_folds, make_inner_folds

__all__ = [
    "Fold",
    "SealedAccessError",
    "SealedDataGuard",
    "assert_no_train_label_intersects_test",
    "embargo_cutoff_date",
    "filter_train_dates",
    "make_folds",
    "make_inner_folds",
]
