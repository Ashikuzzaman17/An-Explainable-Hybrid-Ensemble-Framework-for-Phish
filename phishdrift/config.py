"""Constants shared across the pipeline.

Everything that affects a reported number lives here, so a reader can see
the whole experimental setup in one place.
"""

RANDOM_STATE = 42

# ---- static study -----------------------------------------------------------
TEST_SIZE = 0.20          # held-out fraction for the conventional split
CV_FOLDS = 5              # stratified folds for CV-F1, grid search, stacking

RF_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 20, 30],
    "min_samples_leaf": [1, 2],
}

# ---- drift study ------------------------------------------------------------
FUTURE_FRACTION = 0.30    # share of each class assigned to the "future" partition
VAL_FRACTION = 0.20       # share of the remaining "old" pool used for validation
N_DRIFT_FEATURES = 10     # most discriminative features used to score atypicality
KS_ALPHA = 0.05           # significance level for per-feature KS tests

# ---- calibration / abstention ----------------------------------------------
CALIBRATION_METHOD = "isotonic"
CALIBRATION_CV = 5
TARGET_SELECTIVE_ACCURACY = 0.99   # accuracy the threshold must reach on validation
MIN_VAL_COVERAGE = 0.50            # never accept a threshold that answers < 50%
THRESHOLD_GRID = [round(0.50 + 0.01 * i, 2) for i in range(50)]  # 0.50 .. 0.99

# ---- explanation stability --------------------------------------------------
SHAP_SAMPLE = 800         # rows per partition explained with SHAP
TOP_K = 10                # Jaccard@k overlap of top-k features

# ---- labels -----------------------------------------------------------------
# UCI Phishing Websites encodes the target as  -1 = phishing, 1 = legitimate.
# Verified against the dataset's published composition (4,898 phishing /
# 6,157 legitimate of 11,055) and against the raw value counts of the file.
# The mapping is fixed here on purpose: it must NOT be inferred from class
# fractions, because deduplication changes those fractions.
PHISHING_RAW_VALUE = -1
CLASS_NAMES = ["legitimate", "phishing"]   # index 0, index 1 (positive class)

# ---- plotting ---------------------------------------------------------------
# Fixed categorical order (colorblind-validated): blue, orange, aqua, yellow.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
BLUE, ORANGE, AQUA, YELLOW = SERIES
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3de"
SEQ_CMAP = "Blues"        # single-hue sequential for confusion matrices
