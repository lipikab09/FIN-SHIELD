from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp


@dataclass
class FeatureDriftMetric:
    feature: str
    psi: float
    ks_stat: float
    ks_pvalue: float
    ref_mean: float
    cur_mean: float
    shift_pct: float
    status: str  # OK | WARN | CRITICAL


@dataclass
class DriftReport:
    overall_status: str
    max_psi: float
    mean_psi: float
    drifted_feature_count: int
    score_psi: float
    feature_metrics: List[Dict[str, object]]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


class DriftMonitor:
    """
    Population Stability Index (PSI) + Kolmogorov-Smirnov (KS) covariate & prediction drift detector.
    Tracks feature distribution shift across chronological windows and flags OOD degradation.
    """

    def __init__(
        self,
        psi_warn_threshold: float = 0.10,
        psi_critical_threshold: float = 0.25,
        ks_alpha: float = 0.01,
        n_bins: int = 10,
    ) -> None:
        self.psi_warn_threshold = psi_warn_threshold
        self.psi_critical_threshold = psi_critical_threshold
        self.ks_alpha = ks_alpha
        self.n_bins = n_bins
        self.reference_bins_: Dict[str, np.ndarray] = {}
        self.reference_props_: Dict[str, np.ndarray] = {}
        self.reference_samples_: Dict[str, np.ndarray] = {}
        self.ref_score_bins_: np.ndarray | None = None
        self.ref_score_props_: np.ndarray | None = None

    def fit(self, ref_df: pd.DataFrame, feature_cols: List[str], ref_scores: np.ndarray | None = None) -> "DriftMonitor":
        self.reference_bins_.clear()
        self.reference_props_.clear()
        self.reference_samples_.clear()

        for col in feature_cols:
            if col.startswith("cat_"):
                continue
            vals = ref_df[col].to_numpy(dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            if len(vals) < 10:
                continue
            quantiles = np.linspace(0.0, 1.0, self.n_bins + 1)
            edges = np.unique(np.quantile(vals, quantiles))
            if len(edges) < 3:
                continue
            edges[0] = -np.inf
            edges[-1] = np.inf
            counts, _ = np.histogram(vals, bins=edges)
            props = np.maximum(counts / max(1, len(vals)), 1e-5)
            self.reference_bins_[col] = edges
            self.reference_props_[col] = props
            self.reference_samples_[col] = vals[: min(5000, len(vals))]

        if ref_scores is not None and len(ref_scores) >= 10:
            edges = np.unique(np.quantile(ref_scores, np.linspace(0.0, 1.0, self.n_bins + 1)))
            if len(edges) >= 3:
                edges[0] = -np.inf
                edges[-1] = np.inf
                counts, _ = np.histogram(ref_scores, bins=edges)
                self.ref_score_bins_ = edges
                self.ref_score_props_ = np.maximum(counts / len(ref_scores), 1e-5)
        return self

    def compute_psi_for_series(self, col: str, cur_vals: np.ndarray) -> float:
        if col not in self.reference_bins_:
            return 0.0
        edges = self.reference_bins_[col]
        ref_p = self.reference_props_[col]
        counts, _ = np.histogram(cur_vals, bins=edges)
        cur_p = np.maximum(counts / max(1, len(cur_vals)), 1e-5)
        return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))

    def evaluate(self, cur_df: pd.DataFrame, cur_scores: np.ndarray | None = None) -> DriftReport:
        metrics: List[FeatureDriftMetric] = []
        for col, ref_sample in self.reference_samples_.items():
            if col not in cur_df.columns:
                continue
            cur_vals = cur_df[col].to_numpy(dtype=np.float64)
            cur_vals = cur_vals[np.isfinite(cur_vals)]
            if len(cur_vals) < 5:
                continue

            psi = self.compute_psi_for_series(col, cur_vals)
            ks_res = ks_2samp(ref_sample, cur_vals)
            ref_mean = float(np.mean(ref_sample))
            cur_mean = float(np.mean(cur_vals))
            shift_pct = ((cur_mean - ref_mean) / (abs(ref_mean) + 1e-5)) * 100.0

            if psi >= self.psi_critical_threshold:
                status = "CRITICAL"
            elif psi >= self.psi_warn_threshold or ks_res.pvalue < self.ks_alpha:
                status = "WARN"
            else:
                status = "OK"

            metrics.append(
                FeatureDriftMetric(
                    feature=col,
                    psi=round(psi, 4),
                    ks_stat=round(float(ks_res.statistic), 4),
                    ks_pvalue=round(float(ks_res.pvalue), 6),
                    ref_mean=round(ref_mean, 4),
                    cur_mean=round(cur_mean, 4),
                    shift_pct=round(shift_pct, 2),
                    status=status,
                )
            )

        metrics.sort(key=lambda m: m.psi, reverse=True)
        score_psi = 0.0
        if cur_scores is not None and self.ref_score_bins_ is not None and self.ref_score_props_ is not None:
            counts, _ = np.histogram(cur_scores, bins=self.ref_score_bins_)
            cur_p = np.maximum(counts / max(1, len(cur_scores)), 1e-5)
            score_psi = float(np.sum((cur_p - self.ref_score_props_) * np.log(cur_p / self.ref_score_props_)))

        max_psi = max((m.psi for m in metrics), default=0.0)
        mean_psi = float(np.mean([m.psi for m in metrics])) if metrics else 0.0
        drifted_count = sum(1 for m in metrics if m.status in ("WARN", "CRITICAL"))

        if max_psi >= self.psi_critical_threshold or score_psi >= self.psi_critical_threshold:
            overall = "CRITICAL_SHIFT"
        elif max_psi >= self.psi_warn_threshold or score_psi >= self.psi_warn_threshold:
            overall = "MODERATE_SHIFT"
        else:
            overall = "STABLE"

        return DriftReport(
            overall_status=overall,
            max_psi=round(max_psi, 4),
            mean_psi=round(mean_psi, 4),
            drifted_feature_count=drifted_count,
            score_psi=round(score_psi, 4),
            feature_metrics=[asdict(m) for m in metrics],
        )
