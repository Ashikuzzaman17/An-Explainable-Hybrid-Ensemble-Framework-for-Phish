"""Base classifiers, the tuned Random Forest and the two hybrid ensembles.

Logistic Regression and the RBF SVM sit inside a ``Pipeline`` with a
``StandardScaler`` so the scaler is re-fitted inside every CV fold and never
sees held-out data. Tree models take raw features.
"""

from __future__ import annotations

import warnings

import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, StackingClassifier,
                              VotingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import (GridSearchCV, StratifiedKFold,
                                     cross_val_score, train_test_split)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .config import CV_FOLDS, RANDOM_STATE, RF_GRID, TEST_SIZE

# scikit-learn >= 1.9 deprecates SVC(probability=True); it still works and keeps
# results comparable with older versions (e.g. the Kaggle image), so silence it.
warnings.filterwarnings("ignore", message=".*probability.*deprecated", category=FutureWarning)


def scaled(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("clf", estimator)])


def make_cv(rs: int = RANDOM_STATE) -> StratifiedKFold:
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=rs)


def base_models(rs: int = RANDOM_STATE) -> dict:
    return {
        "Decision Tree": DecisionTreeClassifier(random_state=rs),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1),
        "Logistic Regression": scaled(LogisticRegression(max_iter=2000, random_state=rs)),
        "SVM (RBF)": scaled(SVC(kernel="rbf", probability=True, random_state=rs)),
    }


def ensemble_members(rs: int = RANDOM_STATE) -> list:
    return [
        ("rf", RandomForestClassifier(n_estimators=300, random_state=rs, n_jobs=-1)),
        ("lr", scaled(LogisticRegression(max_iter=2000, random_state=rs))),
        ("svm", scaled(SVC(kernel="rbf", probability=True, random_state=rs))),
    ]


def ensembles(rs: int = RANDOM_STATE) -> dict:
    return {
        "Soft-Voting Ensemble": VotingClassifier(ensemble_members(rs), voting="soft", n_jobs=-1),
        "Stacking Ensemble": StackingClassifier(
            estimators=ensemble_members(rs),
            final_estimator=LogisticRegression(max_iter=2000, random_state=rs),
            cv=CV_FOLDS, n_jobs=-1),
    }


def tune_random_forest(X, y, cv, rs: int = RANDOM_STATE) -> GridSearchCV:
    grid = GridSearchCV(RandomForestClassifier(random_state=rs, n_jobs=-1),
                        RF_GRID, cv=cv, scoring="f1", n_jobs=-1)
    grid.fit(X, y)
    return grid


def metrics(y_true, y_pred, y_prob) -> dict:
    """Positive class = phishing (label 1)."""
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
    }


def evaluate(name, model, X_train, y_train, X_test, y_test, cv):
    """Cross-validated F1 on train, then fit and score on the held-out test set."""
    cv_f1 = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1", n_jobs=-1).mean()
    model.fit(X_train, y_train)
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    row = {"Model": name, "CV F1": cv_f1, **metrics(y_test, pred, prob)}
    return row, model


def run_static(X, y, rs: int = RANDOM_STATE, verbose: bool = True):
    """The conventional experiment: 80/20 stratified split, 7 models."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=rs)
    cv = make_cv(rs)
    rows, fitted = [], {}

    def log(row):
        if verbose:
            print(f"  {row['Model']:24s} F1={row['F1']:.4f}  CV-F1={row['CV F1']:.4f}  "
                  f"AUC={row['ROC-AUC']:.4f}")

    for name, model in base_models(rs).items():
        row, fit = evaluate(name, model, X_train, y_train, X_test, y_test, cv)
        rows.append(row); fitted[name] = fit; log(row)

    grid = tune_random_forest(X_train, y_train, cv, rs)
    if verbose:
        print(f"  RF grid search best {grid.best_params_}  CV-F1 {grid.best_score_:.4f}")
    row, fit = evaluate("Random Forest (tuned)", grid.best_estimator_,
                        X_train, y_train, X_test, y_test, cv)
    rows.append(row); fitted[row["Model"]] = fit; log(row)

    for name, model in ensembles(rs).items():
        row, fit = evaluate(name, model, X_train, y_train, X_test, y_test, cv)
        rows.append(row); fitted[name] = fit; log(row)

    results = (pd.DataFrame(rows).set_index("Model")
               .sort_values("F1", ascending=False).round(4))
    split = {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}
    return results, fitted, split, grid.best_params_
