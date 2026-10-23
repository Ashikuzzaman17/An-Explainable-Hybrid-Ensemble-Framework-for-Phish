"""Every figure used in the README and the paper.

Static PNGs, one value axis per chart, a fixed categorical order
(``config.SERIES``), a legend whenever there are two or more series, and
labels in ink rather than series colour.
"""

from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve

from .config import (AQUA, BLUE, CLASS_NAMES, GRID, INK, MUTED, ORANGE,
                     SEQ_CMAP, SERIES)

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 160, "font.size": 10,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "axes.titlecolor": INK,
    "axes.titleweight": "bold", "axes.titlesize": 11,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.axisbelow": True, "legend.frameon": False,
})


def _save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, name)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def class_balance(y, out_dir):
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    counts = np.bincount(np.asarray(y), minlength=2)
    bars = ax.bar(CLASS_NAMES, counts, color=[BLUE, ORANGE], width=0.55)
    for b, v in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v / counts.sum():.1%})",
                ha="center", va="bottom", color=INK, fontsize=9)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.set_ylabel("sites")
    ax.set_title("Class balance after deduplication")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "class_balance.png")


def label_correlation(X, y, out_dir, k=12):
    corr = X.apply(lambda c: np.corrcoef(c, y)[0, 1]).dropna()
    corr = corr.reindex(corr.abs().sort_values(ascending=False).index).head(k)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    ax.barh(corr.index, corr.values,
            color=[ORANGE if v > 0 else BLUE for v in corr.values], height=0.6)
    ax.axvline(0, color=MUTED, lw=0.8)
    ax.set_xlabel("correlation with phishing label  (+ pushes toward phishing)")
    ax.set_title(f"Top {k} features by correlation with the label")
    ax.grid(axis="y", visible=False)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="toward phishing"),
                       Patch(color=BLUE, label="toward legitimate")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
    return _save(fig, out_dir, "label_correlation.png")


def model_comparison(results, out_dir):
    cols = ["F1", "ROC-AUC", "CV F1"]
    df = results[cols]
    x = np.arange(len(df))
    w = 0.26
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for i, c in enumerate(cols):
        ax.bar(x + (i - 1) * w, df[c], w - 0.03, color=SERIES[i], label=c)
    ax.set_xticks(x, df.index, rotation=20, ha="right")
    ax.set_ylim(max(0, df.values.min() - 0.05), 1.0)
    ax.set_ylabel("score")
    ax.set_title("Model comparison on the conventional 80/20 split")
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    ax.set_title("Model comparison on the conventional 80/20 split", pad=28)
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "model_comparison.png")


def confusion_and_roc(name, y_true, prob, out_dir):
    pred = (prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, pred)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    ax = axes[0]
    ax.imshow(cm, cmap=SEQ_CMAP)
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else INK, fontsize=12)
    ax.set_xticks([0, 1], CLASS_NAMES); ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("predicted"); ax.set_ylabel("actual"); ax.grid(False)
    ax.set_title(f"Confusion matrix - {name}")
    fpr, tpr, _ = roc_curve(y_true, prob)
    from sklearn.metrics import roc_auc_score
    axes[1].plot(fpr, tpr, color=BLUE, lw=2, label=f"{name} (AUC {roc_auc_score(y_true, prob):.4f})")
    axes[1].plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="chance")
    axes[1].set_xlabel("false positive rate"); axes[1].set_ylabel("true positive rate")
    axes[1].set_title("ROC curve"); axes[1].legend(loc="lower right")
    return _save(fig, out_dir, "confusion_roc.png")


def pr_curve(name, y_true, prob, out_dir):
    from sklearn.metrics import average_precision_score
    p, r, _ = precision_recall_curve(y_true, prob)
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    ax.plot(r, p, color=BLUE, lw=2, label=f"{name} (AP {average_precision_score(y_true, prob):.4f})")
    ax.axhline(np.mean(y_true), ls="--", color=MUTED, lw=1, label="no-skill baseline")
    ax.set_xlabel("recall (phishing)"); ax.set_ylabel("precision (phishing)")
    ax.set_ylim(0, 1.02); ax.set_title("Precision-recall curve"); ax.legend(loc="lower left")
    return _save(fig, out_dir, "precision_recall.png")


def importance_bar(series, title, xlabel, out_dir, fname, color=BLUE, k=15):
    s = series.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(s.index, s.values, color=color, height=0.6)
    ax.set_xlabel(xlabel); ax.set_title(title); ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, fname)


def shap_beeswarm(values, X, out_dir, fname="shap_summary.png",
                  title="SHAP - impact of each feature on the phishing prediction"):
    import shap
    shap.summary_plot(values, X, max_display=15, show=False)
    fig = plt.gcf()
    fig.axes[0].set_title(title)
    return _save(fig, out_dir, fname)


def drift_priors(naive_priors, balanced_priors, out_dir):
    labels = ["old pool", "future"]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    x = np.arange(2); w = 0.34
    for i, (name, pr) in enumerate([("naive split", naive_priors),
                                    ("class-balanced split", balanced_priors)]):
        vals = [100 * pr["old_pool"], 100 * pr["future"]]
        bars = ax.bar(x + (i - 0.5) * w, vals, w - 0.04, color=SERIES[i], label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}%",
                    ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, labels); ax.set_ylim(0, 110)
    ax.set_ylabel("phishing rate (%)")
    ax.set_title("Class prior per partition: the naive split moves it")
    ax.legend(loc="upper left"); ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "drift_priors.png")


def ks_bar(ks, out_dir, k=15):
    s = ks.head(k)[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.8))
    ax.barh(s["feature"], s["D"], height=0.6,
            color=[ORANGE if sh else BLUE for sh in s["shifted"]])
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=ORANGE, label="shifted (p < 0.05)"),
                       Patch(color=BLUE, label="not significant")], loc="lower right")
    ax.set_xlabel("KS statistic D (old pool vs future)")
    ax.set_title(f"Feature shift under the balanced split - {int(ks['shifted'].sum())} of {len(ks)} shifted")
    ax.grid(axis="y", visible=False)
    return _save(fig, out_dir, "ks_shift.png")


def drift_accuracy(val_acc, fut_acc, sel_acc, coverage, out_dir):
    names = ["validation\n(in-distribution)", "future\n(shifted)",
             f"future, answered\n({coverage:.0%} coverage)"]
    vals = [100 * val_acc, 100 * fut_acc, 100 * sel_acc]
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(names, vals, color=[BLUE, ORANGE, AQUA], width=0.55)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.2f}%", ha="center", color=INK)
    ax.set_ylim(0, 110); ax.set_ylabel("accuracy (%)")
    ax.set_title("Accuracy under drift, and what abstention recovers")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "drift_accuracy.png")


def reliability(y_val, p_val, y_fut, p_fut, out_dir, n_bins=10):
    from sklearn.calibration import calibration_curve
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.plot([0, 1], [0, 1], ls="--", color=MUTED, lw=1, label="perfect calibration")
    for (y, p, name, c) in [(y_val, p_val, "validation", BLUE), (y_fut, p_fut, "future", ORANGE)]:
        frac, mean = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean, frac, marker="o", ms=5, lw=2, color=c, label=name)
    ax.set_xlabel("mean predicted P(phishing)"); ax.set_ylabel("observed phishing rate")
    ax.set_title("Reliability diagram (calibrated model)"); ax.legend(loc="upper left")
    return _save(fig, out_dir, "reliability.png")


def risk_coverage(rc_val, rc_fut, tau_cov_fut, out_dir):
    fig, ax = plt.subplots(figsize=(6, 4.2))
    ax.plot(100 * rc_val["coverage"], 100 * rc_val["risk"], color=BLUE, lw=2, label="validation")
    ax.plot(100 * rc_fut["coverage"], 100 * rc_fut["risk"], color=ORANGE, lw=2, label="future")
    ax.axvline(100 * tau_cov_fut, color=MUTED, ls=":", lw=1.2, label="locked threshold (future)")
    ax.set_xlabel("coverage (% of cases answered)"); ax.set_ylabel("error rate on answered (%)")
    ax.set_title("Risk-coverage curve"); ax.legend(loc="upper left")
    return _save(fig, out_dir, "risk_coverage.png")


def confidence_bins_plot(bins, out_dir):
    fig, ax = plt.subplots(figsize=(6.4, 4))
    bars = ax.bar(bins["bin"], 100 * bins["accuracy"].fillna(0), color=BLUE, width=0.6)
    for b, (acc, n) in zip(bars, zip(bins["accuracy"], bins["n"])):
        if n:
            ax.text(b.get_x() + b.get_width() / 2, 100 * acc + 1, f"{acc:.0%}\nn={n}",
                    ha="center", color=INK, fontsize=8)
    ax.set_ylim(0, 115); ax.set_xlabel("calibrated confidence max(p, 1-p)")
    ax.set_ylabel("accuracy on future partition (%)")
    ax.set_title("Is confidence still diagnostic under drift?")
    ax.grid(axis="x", visible=False)
    return _save(fig, out_dir, "confidence_bins.png")


def shap_stability(a, b, stab, out_dir):
    b = b.reindex(a.index)
    fig, ax = plt.subplots(figsize=(5.4, 5))
    ax.scatter(a["mean_abs"], b["mean_abs"], s=40, color=BLUE,
               edgecolor="white", linewidth=1.5, zorder=3)
    lim = 1.05 * max(a["mean_abs"].max(), b["mean_abs"].max())
    ax.plot([0, lim], [0, lim], ls="--", color=MUTED, lw=1)
    for f in a.index[:5]:
        ax.annotate(f, (a.loc[f, "mean_abs"], b.loc[f, "mean_abs"]), fontsize=8,
                    color=INK, xytext=(4, 4), textcoords="offset points")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_xlabel("mean |SHAP| - validation"); ax.set_ylabel("mean |SHAP| - future")
    k = [x for x in stab if x.startswith("jaccard")][0]
    ax.set_title(f"Explanation stability  (Spearman {stab['spearman_rho']:.3f}, "
                 f"Jaccard@{k.split('_')[-1]} {stab[k]:.2f})")
    return _save(fig, out_dir, "shap_stability.png")
