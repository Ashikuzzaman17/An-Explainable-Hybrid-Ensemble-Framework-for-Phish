"""Calibration and selective prediction (abstention).

The model answers only when its calibrated confidence max(p, 1-p) reaches a
threshold tau. Tau is chosen on the *validation* partition and locked before
the future partition is ever scored - tuning it on the future data would
report an optimistically biased trade-off.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss, recall_score

from .config import (CALIBRATION_CV, CALIBRATION_METHOD, MIN_VAL_COVERAGE,
                     TARGET_SELECTIVE_ACCURACY, THRESHOLD_GRID)


def fit_calibrated(base_model, X_train, y_train,
                   method=CALIBRATION_METHOD, cv=CALIBRATION_CV):
    """Cross-fitted calibration on the training partition only."""
    model = CalibratedClassifierCV(base_model, method=method, cv=cv)
    model.fit(X_train, y_train)
    return model


def confidence(prob: np.ndarray) -> np.ndarray:
    return np.maximum(prob, 1 - prob)


def expected_calibration_error(y, prob, n_bins: int = 10) -> float:
    conf = confidence(prob)
    pred = (prob >= 0.5).astype(int)
    edges = np.linspace(0.5, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi) if hi < 1 else (conf >= lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == np.asarray(y)[m]).mean() - conf[m].mean())
    return float(ece)


def calibration_metrics(y, prob) -> dict:
    return {"brier": float(brier_score_loss(y, prob)),
            "log_loss": float(log_loss(y, np.clip(prob, 1e-6, 1 - 1e-6))),
            "ece": expected_calibration_error(y, prob)}


def select_threshold(y_val, prob_val, target=TARGET_SELECTIVE_ACCURACY,
                     min_coverage=MIN_VAL_COVERAGE, grid=THRESHOLD_GRID) -> float:
    """Lowest tau whose validation selective accuracy reaches ``target`` while
    still answering at least ``min_coverage`` of cases. If no tau reaches the
    target, return the most accurate tau that respects the coverage floor."""
    y_val = np.asarray(y_val)
    conf = confidence(prob_val)
    pred = (prob_val >= 0.5).astype(int)
    best_tau, best_acc = grid[0], -1.0
    for tau in grid:
        m = conf >= tau
        cov = m.mean()
        if cov < min_coverage or not m.any():
            continue
        acc = (pred[m] == y_val[m]).mean()
        if acc >= target:
            return float(tau)
        if acc > best_acc:
            best_tau, best_acc = tau, acc
    return float(best_tau)


def selective_metrics(y, prob, tau: float) -> dict:
    y = np.asarray(y)
    pred = (prob >= 0.5).astype(int)
    answered = confidence(prob) >= tau
    out = {
        "accuracy_all": float((pred == y).mean()),
        "coverage": float(answered.mean()),
        "abstention_rate": float(1 - answered.mean()),
        "selective_accuracy": float((pred[answered] == y[answered]).mean()) if answered.any() else float("nan"),
        "recall_phishing_all": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "recall_legitimate_all": float(recall_score(y, pred, pos_label=0, zero_division=0)),
    }
    if answered.any():
        out["recall_phishing_answered"] = float(recall_score(y[answered], pred[answered], pos_label=1, zero_division=0))
        out["recall_legitimate_answered"] = float(recall_score(y[answered], pred[answered], pos_label=0, zero_division=0))
    return out


def risk_coverage(y, prob, n_points: int = 50) -> pd.DataFrame:
    """Error rate among the most-confident ``coverage`` fraction of cases."""
    y = np.asarray(y)
    order = np.argsort(-confidence(prob), kind="mergesort")
    correct = ((prob >= 0.5).astype(int) == y)[order]
    rows = []
    for c in np.linspace(0.05, 1.0, n_points):
        n = max(1, int(round(c * len(y))))
        rows.append({"coverage": n / len(y), "risk": 1 - correct[:n].mean()})
    return pd.DataFrame(rows)


def confidence_bins(y, prob, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0001)) -> pd.DataFrame:
    """Accuracy within confidence bands - is confidence still diagnostic?"""
    y = np.asarray(y)
    conf = confidence(prob)
    pred = (prob >= 0.5).astype(int)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < hi)
        rows.append({"bin": f"[{lo:.2f}, {min(hi, 1):.2f}{']' if hi > 1 else ')'}",
                     "n": int(m.sum()), "share": float(m.mean()),
                     "accuracy": float((pred[m] == y[m]).mean()) if m.any() else float("nan")})
    return pd.DataFrame(rows)
