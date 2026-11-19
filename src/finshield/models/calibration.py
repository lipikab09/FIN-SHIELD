from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    """
    Post-hoc probability calibration on a strictly held-out chronological calibration window.
    Supports:
      - 'isotonic': Non-parametric monotonic Isotonic Regression
      - 'platt': Logistic sigmoid (Platt) scaling on log-odds
      - 'temperature': Single-parameter temperature scaling T > 0 on logits
    """

    def __init__(self, method: str = "isotonic") -> None:
        self.method = method.lower().strip()
        if self.method not in ("isotonic", "platt", "temperature", "none"):
            raise ValueError(f"Unsupported calibration method: {method}")
        self.iso = IsotonicRegression(y_min=1e-6, y_max=1.0 - 1e-6, out_of_bounds="clip")
        self.platt = LogisticRegression(C=1e4)
        self.temperature: float = 1.0
        self._fitted = False

    @staticmethod
    def _to_logits(probs: np.ndarray) -> np.ndarray:
        p = np.clip(probs, 1e-6, 1.0 - 1e-6)
        return np.log(p / (1.0 - p))

    @staticmethod
    def _sigmoid(logits: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))

    def fit(self, raw_probs: np.ndarray, y_true: np.ndarray) -> "ProbabilityCalibrator":
        if self.method == "none":
            self._fitted = True
            return self
        if self.method == "isotonic":
            self.iso.fit(raw_probs, y_true)
        elif self.method == "platt":
            logits = self._to_logits(raw_probs).reshape(-1, 1)
            self.platt.fit(logits, y_true)
        elif self.method == "temperature":
            logits = self._to_logits(raw_probs)

            def nll(temp: float) -> float:
                p = np.clip(self._sigmoid(logits / temp), 1e-7, 1.0 - 1e-7)
                return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))

            res = minimize_scalar(nll, bounds=(0.15, 10.0), method="bounded")
            self.temperature = float(res.x)

        self._fitted = True
        return self

    def transform(self, raw_probs: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("ProbabilityCalibrator must be fitted before calling transform().")
        if self.method == "none":
            return np.clip(raw_probs, 1e-6, 1.0 - 1e-6)
        if self.method == "isotonic":
            return np.clip(self.iso.predict(raw_probs), 1e-6, 1.0 - 1e-6)
        if self.method == "platt":
            logits = self._to_logits(raw_probs).reshape(-1, 1)
            return np.clip(self.platt.predict_proba(logits)[:, 1], 1e-6, 1.0 - 1e-6)
        logits = self._to_logits(raw_probs)
        return np.clip(self._sigmoid(logits / self.temperature), 1e-6, 1.0 - 1e-6)


class ConformalRiskWrapper:
    """
    Split-Conformal Prediction & Abstention Router for binary fraud risk.
    Computes non-conformity quantile thresholds on the calibration window so that
    marginal coverage P(Y in C(X)) >= 1 - alpha holds distribution-free under exchangeability,
    and flags ambiguous/uncertain events (prediction set = {0, 1}) for human step-up review.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.q_neg: float = 0.95
        self.q_pos: float = 0.95
        self._fitted = False

    def fit(self, calib_probs: np.ndarray, y_calib: np.ndarray) -> "ConformalRiskWrapper":
        # Class-conditional Mondrian conformal non-conformity scores: s_i = 1 - P(Y = y_i | x_i)
        neg_scores = calib_probs[y_calib == 0]  # 1 - P(Y=0) = P(Y=1)
        pos_scores = 1.0 - calib_probs[y_calib == 1]  # 1 - P(Y=1)

        if len(neg_scores) > 0:
            level_0 = min(1.0, np.ceil((len(neg_scores) + 1) * (1.0 - self.alpha)) / len(neg_scores))
            self.q_neg = float(np.quantile(neg_scores, level_0))
        if len(pos_scores) > 0:
            level_1 = min(1.0, np.ceil((len(pos_scores) + 1) * (1.0 - self.alpha)) / len(pos_scores))
            self.q_pos = float(np.quantile(pos_scores, level_1))

        self._fitted = True
        return self

    def predict_sets(self, probs: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
          - lower_bound: conformal conservative lower risk bound
          - upper_bound: conformal conservative upper risk bound
          - is_uncertain: boolean mask where conformal prediction set contains both {0, 1}
        """
        if not self._fitted:
            raise RuntimeError("ConformalRiskWrapper must be fitted first.")
        # Include 0 in prediction set if P(Y=1) <= q_neg
        include_0 = probs <= self.q_neg
        # Include 1 in prediction set if (1 - P(Y=1)) <= q_pos  <=>  P(Y=1) >= 1 - q_pos
        include_1 = probs >= (1.0 - self.q_pos)
        is_uncertain = include_0 & include_1

        # Empirical conformal interval around calibrated probability
        margin = np.maximum(0.02, 0.5 * np.abs(self.q_neg - (1.0 - self.q_pos)))
        lower = np.clip(probs - margin, 0.0, 1.0)
        upper = np.clip(probs + margin, 0.0, 1.0)
        return lower, upper, is_uncertain
