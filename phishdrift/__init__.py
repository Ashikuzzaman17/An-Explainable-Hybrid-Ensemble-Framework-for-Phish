"""phishdrift: explainable hybrid-ensemble phishing detection, evaluated under
distribution shift with calibration, selective prediction and explanation
stability.

Modules
-------
config      constants shared across the pipeline
data        loading, cleaning and (verified) label encoding
models      base classifiers, tuned Random Forest, soft-voting and stacking
drift       simulated drift splits (naive and class-balanced) and KS tests
abstention  calibration, validation-locked abstention threshold, risk-coverage
explain     SHAP attributions and explanation-stability metrics
plots       every figure in the README / paper
"""

__version__ = "1.0.0"
