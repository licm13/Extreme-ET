import numpy as np
from extreme_et_detection.ridge_detection import ForcedResponseDetector

def test_ridge_forced_response():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 15))
    beta = np.linspace(0.5, 2.0, 15)
    y = X @ beta + rng.normal(scale=0.1, size=200)
    groups = np.repeat([0, 1], 100)

    det = ForcedResponseDetector()
    det.fit(X, y, groups=groups)
    pred = det.predict_forced(X)
    corr = np.corrcoef(y, pred)[0, 1]
    assert corr > 0.9
