"""Simulated distribution shift for a corpus without timestamps.

The UCI corpus has no observation dates, so a "future" partition is built by
ranking records on an *atypicality* score and holding out the most atypical
ones. Atypicality is the mean absolute z-score over the most discriminative
features, so the held-out partition is shifted along exactly the dimensions
the detector relies on.

Two constructions are provided:

``naive_split``
    Takes the most atypical records overall. Because discriminative features
    separate the classes, this also moves the class prior - covariate shift
    and label shift are entangled. Kept only as a diagnostic.

``balanced_split``
    Applies the same ranking *within each class*, so the future partition is
    still the most atypical data but the class prior is held fixed. All
    reported drift results use this split.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from .config import (FUTURE_FRACTION, KS_ALPHA, N_DRIFT_FEATURES,
                     RANDOM_STATE, VAL_FRACTION)


@dataclass
class DriftSplit:
    name: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_future: pd.DataFrame
    y_future: pd.Series
    old_pool_index: pd.Index

    def priors(self) -> dict:
        old = pd.concat([self.y_train, self.y_val])
        return {
            "old_pool": float(old.mean()),
            "train": float(self.y_train.mean()),
            "validation": float(self.y_val.mean()),
            "future": float(self.y_future.mean()),
            "prior_shift_pts": 100 * float(self.y_future.mean() - old.mean()),
            "n_train": len(self.y_train), "n_val": len(self.y_val),
            "n_future": len(self.y_future),
        }


def discriminative_features(X: pd.DataFrame, y: pd.Series, k: int = N_DRIFT_FEATURES) -> list:
    corr = X.apply(lambda c: abs(np.corrcoef(c, y)[0, 1]) if c.std() > 0 else 0.0)
    return corr.sort_values(ascending=False).head(k).index.tolist()


def atypicality(X: pd.DataFrame, features: list) -> pd.Series:
    Z = (X[features] - X[features].mean()) / X[features].std(ddof=0).replace(0, 1)
    return Z.abs().mean(axis=1)


def _finish(name, X, y, future_idx, rs) -> DriftSplit:
    old_idx = X.index.difference(future_idx)
    X_tr, X_va, y_tr, y_va = train_test_split(
        X.loc[old_idx], y.loc[old_idx], test_size=VAL_FRACTION,
        stratify=y.loc[old_idx], random_state=rs)
    return DriftSplit(name, X_tr, y_tr, X_va, y_va,
                      X.loc[future_idx], y.loc[future_idx], old_idx)


def naive_split(X, y, features, frac=FUTURE_FRACTION, rs=RANDOM_STATE) -> DriftSplit:
    score = atypicality(X, features)
    # stable tie-breaking so the split is reproducible
    order = score.sample(frac=1, random_state=rs).sort_values(ascending=False, kind="mergesort")
    future_idx = order.index[: int(round(frac * len(X)))]
    return _finish("naive", X, y, future_idx, rs)


def balanced_split(X, y, features, frac=FUTURE_FRACTION, rs=RANDOM_STATE) -> DriftSplit:
    score = atypicality(X, features)
    future = []
    for cls in (0, 1):
        s = score[y == cls].sample(frac=1, random_state=rs).sort_values(
            ascending=False, kind="mergesort")
        future.append(s.index[: int(round(frac * len(s)))])
    future_idx = future[0].append(future[1])
    return _finish("balanced", X, y, future_idx, rs)


def ks_tests(X_old: pd.DataFrame, X_new: pd.DataFrame, alpha=KS_ALPHA) -> pd.DataFrame:
    """Two-sample Kolmogorov-Smirnov test per feature, old pool vs future."""
    rows = []
    for f in X_old.columns:
        stat, p = ks_2samp(X_old[f], X_new[f])
        rows.append({"feature": f, "D": stat, "p_value": p, "shifted": p < alpha})
    return pd.DataFrame(rows).sort_values("D", ascending=False).reset_index(drop=True)
