"""SHAP attributions and explanation-stability metrics.

Stability is measured with a *fixed* model: the same fitted Random Forest is
explained on the in-distribution validation partition and on the shifted
future partition, so any change in attributions reflects the data, not
retraining.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .config import RANDOM_STATE, SHAP_SAMPLE, TOP_K


def shap_values_positive(model, X: pd.DataFrame) -> np.ndarray:
    """SHAP values for the phishing class (label 1) from a tree model."""
    import shap
    vals = shap.TreeExplainer(model).shap_values(X)
    if isinstance(vals, list):
        vals = vals[1]
    elif np.ndim(vals) == 3:
        vals = vals[:, :, 1]
    return np.asarray(vals)


def sample(X: pd.DataFrame, n: int = SHAP_SAMPLE, rs: int = RANDOM_STATE) -> pd.DataFrame:
    return X.sample(min(n, len(X)), random_state=rs)


def attribution_summary(values: np.ndarray, X: pd.DataFrame) -> pd.DataFrame:
    """Per feature: mean |SHAP| (importance) and direction.

    Direction is the correlation between the feature's value and its SHAP
    value: positive means higher values push toward phishing. (The mean
    signed SHAP is not used - on a class-balanced sample it is close to zero
    by construction, so its sign is noise.)
    """
    Xv = np.asarray(X, dtype=float)
    direction = []
    for j in range(Xv.shape[1]):
        if Xv[:, j].std() == 0 or values[:, j].std() == 0:
            direction.append(0.0)
        else:
            direction.append(float(np.corrcoef(Xv[:, j], values[:, j])[0, 1]))
    return pd.DataFrame({"mean_abs": np.abs(values).mean(axis=0),
                         "direction": direction},
                        index=list(X.columns)).sort_values("mean_abs", ascending=False)


def stability(a: pd.DataFrame, b: pd.DataFrame, k: int = TOP_K) -> dict:
    """Compare two attribution summaries (same features, any order)."""
    b = b.reindex(a.index)
    rho = float(spearmanr(a["mean_abs"], b["mean_abs"]).correlation)
    top_a = set(a["mean_abs"].nlargest(k).index)
    top_b = set(b["mean_abs"].nlargest(k).index)
    jaccard = len(top_a & top_b) / len(top_a | top_b)
    flips = [f for f in a.index
             if np.sign(a.loc[f, "direction"]) * np.sign(b.loc[f, "direction"]) < 0
             and f in top_a]
    return {"spearman_rho": rho, f"jaccard_at_{k}": float(jaccard),
            "sign_flips_in_top_k": len(flips), "flipped_features": flips}
