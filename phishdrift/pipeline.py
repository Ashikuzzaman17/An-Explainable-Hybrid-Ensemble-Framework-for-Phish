"""End-to-end experiment: static study, then the drift study.

Both the command-line script (``scripts/run_all.py``) and the Kaggle notebook
call these two functions, so the numbers in the README, the notebook and the
paper all come from the same code path.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from . import abstention as ab
from . import drift as dr
from . import explain as ex
from . import plots
from .config import RANDOM_STATE, TOP_K
from .models import run_static


def _r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# --------------------------------------------------------------------------- #
# 1. Static study
# --------------------------------------------------------------------------- #
def static_study(X, y, fig_dir="figures", res_dir="results", verbose=True):
    os.makedirs(res_dir, exist_ok=True)
    if verbose:
        print("\n== Static study: stratified 80/20 split ==")
    plots.class_balance(y, fig_dir)
    plots.label_correlation(X, y, fig_dir)

    results, fitted, split, best_params = run_static(X, y, verbose=verbose)
    results.to_csv(os.path.join(res_dir, "static_results.csv"))
    plots.model_comparison(results, fig_dir)

    best_name = results.index[0]
    Xte, yte = split["X_test"], split["y_test"]
    prob = fitted[best_name].predict_proba(Xte)[:, 1]
    plots.confusion_and_roc(best_name, yte, prob, fig_dir)
    plots.pr_curve(best_name, yte, prob, fig_dir)

    rf = fitted["Random Forest (tuned)"]
    imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
    imp.to_csv(os.path.join(res_dir, "rf_importance.csv"), header=["importance"])
    plots.importance_bar(imp, "Random Forest impurity importance", "mean decrease in impurity",
                         fig_dir, "rf_importance.png")

    perm = permutation_importance(fitted[best_name], Xte, yte, scoring="f1",
                                  n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
    perm_s = pd.Series(perm.importances_mean, index=X.columns).sort_values(ascending=False)
    perm_s.to_csv(os.path.join(res_dir, "permutation_importance.csv"), header=["f1_drop"])
    plots.importance_bar(perm_s, f"Permutation importance - {best_name}",
                         "drop in F1 when the feature is shuffled", fig_dir,
                         "permutation_importance.png", color=plots.ORANGE)

    Xs = ex.sample(Xte)
    sv = ex.shap_values_positive(rf, Xs)
    plots.shap_beeswarm(sv, Xs, fig_dir)
    shap_top = ex.attribution_summary(sv, Xs)
    shap_top.to_csv(os.path.join(res_dir, "shap_static.csv"))

    return {
        "results": results,
        "best_model": best_name,
        "best_rf_params": best_params,
        "top_features_shap": shap_top.index[:TOP_K].tolist(),
        "top_features_permutation": perm_s.index[:TOP_K].tolist(),
    }


# --------------------------------------------------------------------------- #
# 2. Drift study
# --------------------------------------------------------------------------- #
def drift_study(X, y, fig_dir="figures", res_dir="results", verbose=True):
    os.makedirs(res_dir, exist_ok=True)
    if verbose:
        print("\n== Drift study: simulated shift, class prior held fixed ==")

    feats = dr.discriminative_features(X, y)
    naive = dr.naive_split(X, y, feats)
    bal = dr.balanced_split(X, y, feats)
    pn, pb = naive.priors(), bal.priors()
    plots.drift_priors(pn, pb, fig_dir)
    if verbose:
        print(f"  naive split    phishing rate old {pn['old_pool']:.3f} -> future {pn['future']:.3f}"
              f"  (label shift {pn['prior_shift_pts']:+.1f} pts, diagnostic only)")
        print(f"  balanced split phishing rate old {pb['old_pool']:.3f} -> future {pb['future']:.3f}"
              f"  ({pb['n_train']} train / {pb['n_val']} val / {pb['n_future']} future)")

    X_old = X.loc[bal.old_pool_index]
    ks = dr.ks_tests(X_old, bal.X_future)
    ks.to_csv(os.path.join(res_dir, "ks_tests.csv"), index=False)
    plots.ks_bar(ks, fig_dir)
    n_shift = int(ks["shifted"].sum())

    # Base detector: Random Forest trained on the old pool's training part only.
    rf = RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(bal.X_train, bal.y_train)
    p_val_raw = rf.predict_proba(bal.X_val)[:, 1]
    p_fut_raw = rf.predict_proba(bal.X_future)[:, 1]

    cal = ab.fit_calibrated(
        RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        bal.X_train, bal.y_train)
    p_val = cal.predict_proba(bal.X_val)[:, 1]
    p_fut = cal.predict_proba(bal.X_future)[:, 1]

    calib = {
        "uncalibrated_validation": ab.calibration_metrics(bal.y_val, p_val_raw),
        "uncalibrated_future": ab.calibration_metrics(bal.y_future, p_fut_raw),
        "calibrated_validation": ab.calibration_metrics(bal.y_val, p_val),
        "calibrated_future": ab.calibration_metrics(bal.y_future, p_fut),
    }
    plots.reliability(bal.y_val, p_val, bal.y_future, p_fut, fig_dir)

    tau = ab.select_threshold(bal.y_val, p_val)     # locked on validation only
    sel_val = ab.selective_metrics(bal.y_val, p_val, tau)
    sel_fut = ab.selective_metrics(bal.y_future, p_fut, tau)
    rc_val, rc_fut = ab.risk_coverage(bal.y_val, p_val), ab.risk_coverage(bal.y_future, p_fut)
    pd.concat([rc_val.assign(partition="validation"), rc_fut.assign(partition="future")]) \
        .to_csv(os.path.join(res_dir, "risk_coverage.csv"), index=False)
    plots.risk_coverage(rc_val, rc_fut, sel_fut["coverage"], fig_dir)
    plots.drift_accuracy(sel_val["accuracy_all"], sel_fut["accuracy_all"],
                         sel_fut["selective_accuracy"], sel_fut["coverage"], fig_dir)
    bins = ab.confidence_bins(bal.y_future, p_fut)
    bins.to_csv(os.path.join(res_dir, "confidence_bins_future.csv"), index=False)
    plots.confidence_bins_plot(bins, fig_dir)

    # Explanation stability: same fitted model, two partitions.
    Xv_s, Xf_s = ex.sample(bal.X_val), ex.sample(bal.X_future)
    s_val = ex.attribution_summary(ex.shap_values_positive(rf, Xv_s), Xv_s)
    s_fut = ex.attribution_summary(ex.shap_values_positive(rf, Xf_s), Xf_s)
    stab = ex.stability(s_val, s_fut)
    s_val.join(s_fut, lsuffix="_val", rsuffix="_future") \
        .to_csv(os.path.join(res_dir, "shap_stability.csv"))
    plots.shap_stability(s_val, s_fut, stab, fig_dir)

    if verbose:
        print(f"  KS: {n_shift}/{len(ks)} features shifted (p < 0.05)")
        print(f"  accuracy validation {sel_val['accuracy_all']:.4f} -> future {sel_fut['accuracy_all']:.4f}")
        print(f"  tau locked on validation = {tau:.2f}; future coverage {sel_fut['coverage']:.4f}, "
              f"selective accuracy {sel_fut['selective_accuracy']:.4f}")
        print(f"  calibrated future Brier {calib['calibrated_future']['brier']:.4f}, "
              f"ECE {calib['calibrated_future']['ece']:.4f}")
        print(f"  SHAP stability Spearman {stab['spearman_rho']:.4f}, "
              f"Jaccard@{TOP_K} {stab[f'jaccard_at_{TOP_K}']:.2f}, "
              f"sign flips in top-{TOP_K}: {stab['sign_flips_in_top_k']}")

    return {
        "discriminative_features": feats,
        "naive_priors": {k: _r(v) for k, v in pn.items()},
        "balanced_priors": {k: _r(v) for k, v in pb.items()},
        "ks_shifted": n_shift, "ks_total": len(ks),
        "threshold": tau,
        "validation": {k: _r(v) for k, v in sel_val.items()},
        "future": {k: _r(v) for k, v in sel_fut.items()},
        "calibration": {k: {m: _r(v) for m, v in d.items()} for k, d in calib.items()},
        "confidence_bins_future": bins.round(4).to_dict(orient="records"),
        "shap_stability": {k: (_r(v) if isinstance(v, float) else v) for k, v in stab.items()},
    }
