"""Loading, cleaning and label encoding for the UCI Phishing Websites corpus.

The label direction is the single easiest thing to get silently wrong in this
dataset, so it is hard-coded (see ``config.PHISHING_RAW_VALUE``) and then
checked against domain knowledge: a valid SSL certificate must push a site
*toward* legitimate. If the encoded labels disagree, loading fails loudly
instead of producing inverted results.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import PHISHING_RAW_VALUE

NON_FEATURE_COLUMNS = ["index", "Index", "id", "ID", "Unnamed: 0",
                       "FILENAME", "URL", "url", "Domain", "domain"]
LABEL_CANDIDATES = ["Result", "result", "class", "Class", "label", "Label",
                    "target", "Target", "status"]
SEARCH_ROOTS = ["/kaggle/input", "data", "../data", "."]


@dataclass
class DataInfo:
    path: str
    raw_rows: int
    label_column: str
    raw_counts: dict
    dropped_columns: list
    duplicates_removed: int
    conflicting_feature_vectors: int
    rows: int
    n_features: int
    phishing_rate: float
    ssl_label_correlation: float | None
    notes: list = field(default_factory=list)


def find_dataset(path: str | None = None) -> str:
    """Return a path to the dataset, searching the usual places if needed."""
    if path and os.path.exists(path):
        return path
    for root in SEARCH_ROOTS:
        for pattern in ("**/*.csv", "**/*.arff"):
            hits = sorted(glob.glob(os.path.join(root, pattern), recursive=True))
            hits = [h for h in hits if "results" not in h]
            if hits:
                return hits[0]
    raise FileNotFoundError(
        "Dataset not found. Download the UCI Phishing Websites dataset and put "
        "the CSV in data/ (see data/README.md), or pass --data PATH.")


def load_raw(path: str) -> pd.DataFrame:
    """Read the CSV, or the original UCI ARFF file."""
    if path.lower().endswith(".arff"):
        from scipy.io import arff
        records, _ = arff.loadarff(path)
        df = pd.DataFrame(records)
        for col in df.columns:
            if df[col].dtype == object:
                df[col] = df[col].str.decode("utf-8").astype(int)
        return df
    return pd.read_csv(path)


def _label_column(df: pd.DataFrame) -> str:
    for c in LABEL_CANDIDATES:
        if c in df.columns:
            return c
    return df.columns[-1]


def check_label_direction(X: pd.DataFrame, y: pd.Series) -> float | None:
    """Fail if phishing labels look inverted.

    ``SSLfinal_State`` = 1 means a trusted, valid certificate. It must be
    negatively correlated with the phishing label. A positive correlation
    means the labels are flipped and every downstream number (F1, precision,
    recall, SHAP directions, class priors) would be reported for the wrong
    class.
    """
    if "SSLfinal_State" not in X.columns:
        return None
    corr = float(np.corrcoef(X["SSLfinal_State"], y)[0, 1])
    if corr > 0:
        raise ValueError(
            f"Label direction check failed: corr(SSLfinal_State, phishing) = "
            f"{corr:+.3f} > 0, i.e. valid SSL would predict phishing. The "
            f"labels are inverted - check config.PHISHING_RAW_VALUE.")
    return corr


def prepare(df_raw: pd.DataFrame, path: str = "", verbose: bool = True):
    """Clean the raw frame and return ``(X, y, info)``.

    Steps: drop identifier columns, drop exact duplicate rows, encode the
    label with the fixed mapping (1 = phishing), coerce features to numeric,
    drop constant columns, then verify the label direction.
    """
    df = df_raw.copy()
    raw_rows = len(df)
    dropped = [c for c in NON_FEATURE_COLUMNS if c in df.columns]
    df = df.drop(columns=dropped)

    label_col = _label_column(df)
    raw_counts = {str(k): int(v) for k, v in df[label_col].value_counts().items()}

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    duplicates_removed = before - len(df)

    features = [c for c in df.columns if c != label_col]
    conflicting = int(df.duplicated(subset=features, keep=False).sum())

    values = set(pd.unique(df[label_col]))
    if not values <= {-1, 1, 0}:
        raise ValueError(f"Unexpected label values {sorted(values)}")
    y = (df[label_col] == PHISHING_RAW_VALUE).astype(int).rename("phishing")

    X = df[features].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median())
    constant = X.columns[X.nunique() <= 1].tolist()
    X = X.drop(columns=constant)

    corr = check_label_direction(X, y)

    info = DataInfo(
        path=path, raw_rows=raw_rows, label_column=label_col,
        raw_counts=raw_counts, dropped_columns=dropped + constant,
        duplicates_removed=duplicates_removed,
        conflicting_feature_vectors=conflicting, rows=len(X),
        n_features=X.shape[1], phishing_rate=float(y.mean()),
        ssl_label_correlation=corr)
    if conflicting:
        info.notes.append(
            f"{conflicting} rows share a feature vector with a row of the "
            f"opposite label (irreducible noise); they are kept.")

    if verbose:
        print(f"Loaded {path or 'dataframe'}: {raw_rows:,} rows")
        print(f"  label column '{label_col}', raw counts {raw_counts}")
        print(f"  encoding: {PHISHING_RAW_VALUE} -> phishing (1), otherwise legitimate (0)")
        print(f"  removed {duplicates_removed:,} duplicate rows -> {len(X):,} rows, "
              f"{X.shape[1]} features")
        print(f"  phishing rate {y.mean():.3f}")
        if corr is not None:
            print(f"  sanity check corr(SSLfinal_State, phishing) = {corr:+.3f}  (must be < 0) OK")
        for n in info.notes:
            print("  note:", n)
    return X, y, info


def load(path: str | None = None, verbose: bool = True):
    """Find, read and prepare the dataset in one call."""
    path = find_dataset(path)
    return prepare(load_raw(path), path=path, verbose=verbose)
