#!/usr/bin/env python
"""Reproduce every number and figure in the repository.

    python scripts/run_all.py                     # finds data/*.csv automatically
    python scripts/run_all.py --data path/to/phishing.csv

Writes results/*.csv, results/summary.json, results/RESULTS.md and
figures/*.png, then refreshes the results block in README.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import asdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from phishdrift import __version__                      # noqa: E402
from phishdrift.config import TOP_K                      # noqa: E402
from phishdrift.data import load                         # noqa: E402
from phishdrift.pipeline import drift_study, static_study  # noqa: E402

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def pct(x):
    return "n/a" if x is None else f"{100 * x:.2f}%"


def results_markdown(info, st, dr) -> str:
    res = st["results"]
    table = ["| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV F1 |",
             "|---|---|---|---|---|---|---|---|"]
    for name, r in res.iterrows():
        bold = "**" if name == st["best_model"] else ""
        table.append(f"| {bold}{name}{bold} | {r['Accuracy']:.4f} | {r['Precision']:.4f} | "
                     f"{r['Recall']:.4f} | {r['F1']:.4f} | {r['ROC-AUC']:.4f} | "
                     f"{r['PR-AUC']:.4f} | {r['CV F1']:.4f} |")
    v, f = dr["validation"], dr["future"]
    cv, cf = dr["calibration"]["calibrated_validation"], dr["calibration"]["calibrated_future"]
    uf = dr["calibration"]["uncalibrated_future"]
    s = dr["shap_stability"]
    nb, bp = dr["naive_priors"], dr["balanced_priors"]
    return "\n".join([
        f"Dataset: {info.raw_rows:,} rows, {info.duplicates_removed:,} exact duplicates removed, "
        f"{info.rows:,} unique rows and {info.n_features} features; phishing rate "
        f"{pct(info.phishing_rate)}. Label check: corr(SSLfinal_State, phishing) = "
        f"{info.ssl_label_correlation:+.3f}. Positive class = phishing throughout.",
        "",
        "**Static study** (stratified 80/20 split, 5-fold CV on the training part)",
        "",
        *table,
        "",
        f"Tuned Random Forest: `{st['best_rf_params']}`. Top SHAP features: "
        f"{', '.join(st['top_features_shap'][:5])}.",
        "",
        "**Drift study** (simulated shift, class prior held fixed)",
        "",
        "| Quantity | Value |",
        "|---|---|",
        f"| Naive split phishing rate, old pool -> future | {pct(nb['old_pool'])} -> {pct(nb['future'])} (label shift, diagnostic only) |",
        f"| Balanced split phishing rate, old pool -> future | {pct(bp['old_pool'])} -> {pct(bp['future'])} |",
        f"| Train / validation / future records | {bp['n_train']:,} / {bp['n_val']:,} / {bp['n_future']:,} |",
        f"| Features with a significant KS shift | {dr['ks_shifted']} of {dr['ks_total']} |",
        f"| Accuracy, validation -> future | {pct(v['accuracy_all'])} -> {pct(f['accuracy_all'])} |",
        f"| Phishing recall, validation -> future | {pct(v['recall_phishing_all'])} -> {pct(f['recall_phishing_all'])} |",
        f"| Abstention threshold (locked on validation) | {dr['threshold']:.2f} |",
        f"| Future coverage / selective accuracy | {pct(f['coverage'])} / {pct(f['selective_accuracy'])} |",
        f"| Brier, validation -> future (calibrated) | {cv['brier']:.4f} -> {cf['brier']:.4f} |",
        f"| ECE, validation -> future (calibrated) | {cv['ece']:.4f} -> {cf['ece']:.4f} |",
        f"| Future Brier, uncalibrated -> calibrated | {uf['brier']:.4f} -> {cf['brier']:.4f} |",
        f"| SHAP rank stability (Spearman) | {s['spearman_rho']:.4f} |",
        f"| Top-{TOP_K} feature overlap (Jaccard) | {s[f'jaccard_at_{TOP_K}']:.2f} |",
        f"| Direction flips in top-{TOP_K} | {s['sign_flips_in_top_k']} |",
    ])


def update_readme(md: str):
    path = os.path.join(ROOT, "README.md")
    if not os.path.exists(path):
        return
    text = open(path, encoding="utf-8").read()
    if START in text and END in text:
        new = re.sub(re.escape(START) + r".*?" + re.escape(END),
                     lambda _: f"{START}\n{md}\n{END}", text, flags=re.S)
        open(path, "w", encoding="utf-8").write(new)
        print("README.md results block updated")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", help="path to the UCI Phishing Websites CSV or ARFF")
    ap.add_argument("--out", default=ROOT, help="where results/ and figures/ are written")
    ap.add_argument("--no-readme", action="store_true", help="do not touch README.md")
    args = ap.parse_args()

    t0 = time.time()
    fig_dir, res_dir = os.path.join(args.out, "figures"), os.path.join(args.out, "results")
    X, y, info = load(args.data)
    st = static_study(X, y, fig_dir, res_dir)
    dr = drift_study(X, y, fig_dir, res_dir)

    summary = {"version": __version__, "data": asdict(info),
               "static": {k: v for k, v in st.items() if k != "results"},
               "static_results": st["results"].reset_index().to_dict(orient="records"),
               "drift": dr}
    with open(os.path.join(res_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2, default=str)

    md = results_markdown(info, st, dr)
    with open(os.path.join(res_dir, "RESULTS.md"), "w") as fh:
        fh.write("# Results\n\nGenerated by `scripts/run_all.py`.\n\n" + md + "\n")
    if not args.no_readme and os.path.abspath(args.out) == ROOT:
        update_readme(md)
    print(f"\nDone in {time.time() - t0:.0f}s. See results/RESULTS.md and figures/.")


if __name__ == "__main__":
    main()
