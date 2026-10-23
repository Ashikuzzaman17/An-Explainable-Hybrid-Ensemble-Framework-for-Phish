#!/usr/bin/env python
"""Generate notebooks/phishing_drift_full.ipynb from the package sources.

The notebook is self-contained: its first cells write the ``phishdrift``
package next to itself, so it runs on Kaggle (or Colab) with nothing but the
dataset attached. Re-run this script after editing anything in phishdrift/.

    python scripts/build_notebook.py
"""

import glob
import os

import nbformat as nbf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "notebooks", "phishing_drift_full.ipynb")

md, code = nbf.v4.new_markdown_cell, nbf.v4.new_code_cell
cells = [md("""# Explainable Phishing Detection Under Distribution Shift

Hybrid ensembles, calibration, and selective prediction on the UCI Phishing Websites dataset.

This notebook reproduces every number and figure in the repository. It has two parts:

1. **Static study** - seven models (Decision Tree, Random Forest, Logistic Regression, SVM, tuned Random Forest, soft voting, stacking) on a conventional stratified 80/20 split, with SHAP and permutation importance.
2. **Drift study** - a simulated distribution shift that holds the class prior fixed, KS tests, calibration, abstention with a threshold locked on validation data, and SHAP explanation stability.

**Label encoding.** In the UCI file `-1` = phishing and `1` = legitimate. The mapping is fixed in code (never inferred from class fractions) and verified with a sanity check: valid SSL (`SSLfinal_State = 1`) must be *negatively* correlated with phishing. The positive class everywhere is **phishing**.

**On Kaggle:** add the dataset (*Add Input -> search "phishing websites dataset"*, e.g. `mmelgizin/phishing-websites-dataset`) and choose *Run All*. It finds the CSV under `/kaggle/input` automatically. Runtime is roughly 5-10 minutes on CPU."""),
         code("%pip install -q shap\n"),
         md("## The `phishdrift` package\n\nThe cells below write the package source to disk so the rest of the notebook can import it. The same code lives in the GitHub repository.")]

cells.append(code("import os\nos.makedirs('phishdrift', exist_ok=True)"))
for path in sorted(glob.glob(os.path.join(ROOT, "phishdrift", "*.py"))):
    name = os.path.basename(path)
    src = open(path, encoding="utf-8").read().rstrip() + "\n"
    cells.append(code(f"%%writefile phishdrift/{name}\n{src}"))

cells += [
    md("## 1. Load and clean the data"),
    code("""import warnings, json
warnings.filterwarnings('ignore')
import pandas as pd
from IPython.display import Image, display, Markdown
from phishdrift.data import load
from phishdrift.pipeline import static_study, drift_study

X, y, info = load()          # pass a path here if auto-detection picks the wrong file
FIG, RES = 'figures', 'results'"""),
    md("## 2. Static study"),
    code("st = static_study(X, y, FIG, RES)\nst['results']"),
    code("for f in ['class_balance', 'label_correlation', 'model_comparison', 'confusion_roc',\n"
         "          'precision_recall', 'rf_importance', 'permutation_importance', 'shap_summary']:\n"
         "    display(Image(f'{FIG}/{f}.png'))"),
    md("""## 3. Drift study

The UCI corpus has no timestamps, so a "future" partition is simulated: records are ranked by atypicality (mean |z| over the 10 most discriminative features) and the most atypical 30% *of each class* are held out. Ranking within each class keeps the class prior fixed, so the shift is covariate shift rather than label shift. The naive version (top 30% overall) is shown only to demonstrate how it entangles the two.

The model is a Random Forest trained on the old pool, calibrated with 5-fold isotonic regression. The abstention threshold is chosen on validation data and locked before the future partition is scored."""),
    code("dr = drift_study(X, y, FIG, RES)"),
    code("for f in ['drift_priors', 'ks_shift', 'drift_accuracy', 'reliability',\n"
         "          'risk_coverage', 'confidence_bins', 'shap_stability']:\n"
         "    display(Image(f'{FIG}/{f}.png'))"),
    md("## 4. Summary"),
    code("""import dataclasses
v, fu, s = dr['validation'], dr['future'], dr['shap_stability']
cal = dr['calibration']
rows = {
    'Best static model': st['best_model'],
    'Best static F1 (phishing)': st['results'].iloc[0]['F1'],
    'Balanced split phishing rate old -> future': f"{dr['balanced_priors']['old_pool']:.3f} -> {dr['balanced_priors']['future']:.3f}",
    'Features shifted (KS, p<0.05)': f"{dr['ks_shifted']} / {dr['ks_total']}",
    'Accuracy validation -> future': f"{v['accuracy_all']:.4f} -> {fu['accuracy_all']:.4f}",
    'Threshold (locked on validation)': dr['threshold'],
    'Future coverage': fu['coverage'],
    'Future selective accuracy': fu['selective_accuracy'],
    'Future Brier (calibrated)': cal['calibrated_future']['brier'],
    'Future ECE (calibrated)': cal['calibrated_future']['ece'],
    'SHAP Spearman (val vs future)': s['spearman_rho'],
    'SHAP Jaccard@10': s['jaccard_at_10'],
}
summary = pd.Series(rows, name='value').to_frame()
summary"""),
    code("""with open(f'{RES}/summary.json', 'w') as fh:
    json.dump({'data': dataclasses.asdict(info), 'best_rf_params': st['best_rf_params'],
               'static_results': st['results'].reset_index().to_dict(orient='records'),
               'drift': dr}, fh, indent=2, default=str)
print('Saved results/ and figures/ - download them from the Output panel.')"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
os.makedirs(os.path.dirname(OUT), exist_ok=True)
nbf.write(nb, OUT)
print("wrote", os.path.relpath(OUT, ROOT), f"({len(cells)} cells)")
