"""Ridge-based forced response detection.

使用 Ridge 回归从单成员空间场中提取强迫响应分量，
支持 group-wise CV 以模拟 “model group as truth” 策略。
"""

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold


@dataclass
class ForcedResponseDetector:
    alpha: float = 10.0
    coef_: Optional[np.ndarray] = None
    intercept_: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: Optional[Iterable] = None,
        alphas: Iterable[float] = (0.1, 1.0, 3.0, 10.0, 30.0, 100.0),
        n_splits: int = 5,
    ) -> "ForcedResponseDetector":
        """Fit Ridge model with optional group-wise CV.

        Parameters
        ----------
        X : (n_samples, n_features)
            Flattened ET fields or anomalies.
        y : (n_samples,)
            Target (e.g. ensemble-mean forced response).
        groups : iterable
            Group labels (e.g. model families).
        """
        X = np.asarray(X)
        y = np.asarray(y)

        if groups is None:
            best_alpha, best_score = None, -np.inf
            for a in alphas:
                m = Ridge(alpha=a, fit_intercept=True)
                m.fit(X, y)
                score = m.score(X, y)
                if score > best_score:
                    best_score = score
                    best_alpha = a
            self.alpha = float(best_alpha)
            final = Ridge(alpha=self.alpha, fit_intercept=True).fit(X, y)
        else:
            groups = np.asarray(groups)
            gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups))))
            best_alpha, best_score = None, -np.inf
            for a in alphas:
                scores = []
                for tr, va in gkf.split(X, y, groups):
                    m = Ridge(alpha=a, fit_intercept=True)
                    m.fit(X[tr], y[tr])
                    scores.append(m.score(X[va], y[va]))
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = a
            self.alpha = float(best_alpha)
            final = Ridge(alpha=self.alpha, fit_intercept=True).fit(X, y)

        self.coef_ = final.coef_.copy()
        self.intercept_ = float(final.intercept_)
        return self

    def predict_forced(self, X: np.ndarray) -> np.ndarray:
        """Predict forced component for new samples (models or observations)."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted.")
        X = np.asarray(X)
        return X @ self.coef_ + self.intercept_

    def get_spatial_pattern(self, template_shape: Tuple[int, int]) -> np.ndarray:
        """Reshape coefficients back to (lat, lon) spatial pattern."""
        if self.coef_ is None:
            raise RuntimeError("Model not fitted.")
        return self.coef_.reshape(template_shape)
