from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier


class GBDTFraudModel:
    """
    Gradient Boosted Decision Tree wrapper supporting LightGBM, XGBoost, CatBoost,
    with automatic fallback to scikit-learn HistGradientBoostingClassifier when native C++
    libraries are absent in constrained runtimes.
    """

    def __init__(
        self,
        backend: str = "auto",
        random_state: int = 42,
        max_iter: int = 220,
        learning_rate: float = 0.055,
        max_depth: int = 6,
    ) -> None:
        self.requested_backend = backend
        self.random_state = random_state
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.active_backend_: str = "hist_gbdt"
        self.feature_names_: List[str] = []
        self.feature_medians_: np.ndarray | None = None
        self.feature_stds_: np.ndarray | None = None
        self.model = None

    def _build_estimator(self, pos_weight: float):
        choice = self.requested_backend.lower()
        if choice in ("auto", "lightgbm"):
            try:
                import lightgbm as lgb  # type: ignore

                self.active_backend_ = "lightgbm"
                return lgb.LGBMClassifier(
                    n_estimators=self.max_iter,
                    learning_rate=self.learning_rate,
                    num_leaves=31,
                    subsample_for_bin=50000,
                    colsample_bytree=0.85,
                    subsample=0.85,
                    scale_pos_weight=min(pos_weight, 15.0),
                    random_state=self.random_state,
                    verbose=-1,
                )
            except ImportError:
                if choice == "lightgbm":
                    pass

        if choice in ("auto", "xgboost"):
            try:
                import xgboost as xgb  # type: ignore

                self.active_backend_ = "xgboost"
                return xgb.XGBClassifier(
                    n_estimators=self.max_iter,
                    learning_rate=self.learning_rate,
                    max_depth=self.max_depth,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    scale_pos_weight=min(pos_weight, 15.0),
                    eval_metric="aucpr",
                    random_state=self.random_state,
                )
            except ImportError:
                pass

        if choice == "catboost":
            try:
                from catboost import CatBoostClassifier  # type: ignore

                self.active_backend_ = "catboost"
                return CatBoostClassifier(
                    iterations=self.max_iter,
                    learning_rate=self.learning_rate,
                    depth=self.max_depth,
                    auto_class_weights="SqrtBalanced",
                    random_seed=self.random_state,
                    verbose=0,
                )
            except ImportError:
                pass

        self.active_backend_ = "hist_gbdt"
        return HistGradientBoostingClassifier(
            max_iter=self.max_iter,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            l2_regularization=1.5,
            random_state=self.random_state,
        )

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "GBDTFraudModel":
        self.feature_names_ = list(X.columns)
        X_arr = X.to_numpy(dtype=np.float32)
        self.feature_medians_ = np.nanmedian(X_arr, axis=0)
        self.feature_stds_ = np.maximum(np.nanstd(X_arr, axis=0), 1e-5)

        pos = float(np.sum(y == 1))
        neg = float(np.sum(y == 0))
        pos_weight = neg / max(1.0, pos)

        self.model = self._build_estimator(pos_weight)
        if self.active_backend_ == "hist_gbdt":
            sample_weight = np.where(y == 1, min(np.sqrt(pos_weight), 6.0), 1.0)
            self.model.fit(X_arr, y, sample_weight=sample_weight)
        else:
            self.model.fit(X_arr, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X[self.feature_names_].to_numpy(dtype=np.float32)
        return self.model.predict_proba(X_arr)[:, 1]

    def predict_logits(self, X: pd.DataFrame) -> np.ndarray:
        probs = np.clip(self.predict_proba(X), 1e-6, 1.0 - 1e-6)
        return np.log(probs / (1.0 - probs))

    def explain_local(self, x_row: pd.DataFrame, top_k: int = 5) -> List[Dict[str, object]]:
        """
        Fast directional feature attribution using baseline-perturbation marginal sensitivity
        weighted by standardized feature deviation from training median.
        """
        if self.feature_medians_ is None or self.feature_stds_ is None:
            return []
        row_arr = x_row[self.feature_names_].iloc[0].to_numpy(dtype=np.float32)
        base_prob = float(self.predict_proba(x_row)[0])
        z_scores = (row_arr - self.feature_medians_) / self.feature_stds_

        # Evaluate top 14 deviated features by marginal replacement with training median
        candidate_idx = np.argsort(np.abs(z_scores))[::-1][:14]
        attributions = []
        for idx in candidate_idx:
            perturbed = row_arr.copy()
            perturbed[idx] = self.feature_medians_[idx]
            pert_df = pd.DataFrame([perturbed], columns=self.feature_names_)
            pert_prob = float(self.predict_proba(pert_df)[0])
            delta = base_prob - pert_prob
            attributions.append(
                {
                    "feature": self.feature_names_[idx],
                    "value": round(float(row_arr[idx]), 4),
                    "baseline_median": round(float(self.feature_medians_[idx]), 4),
                    "z_score": round(float(z_scores[idx]), 2),
                    "risk_contribution": round(delta, 5),
                    "direction": "INCREASES_RISK" if delta >= 0 else "REDUCES_RISK",
                }
            )
        attributions.sort(key=lambda d: abs(float(d["risk_contribution"])), reverse=True)
        return attributions[:top_k]
