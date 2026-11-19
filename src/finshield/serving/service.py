from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd

from finshield.config import Settings, load_settings
from finshield.serving.schemas import (
    BatchScoreResponse,
    HealthResponse,
    ReasonCodeItem,
    ScoreResponse,
    TransactionEventInput,
)
from finshield.training.pipeline import run_training_and_experiments

logger = logging.getLogger("finshield.serving")


class FraudInferenceService:
    """Production-grade stateful inference engine for single and batch transaction risk scoring."""

    def __init__(self, settings: Settings | None = None, auto_train_if_missing: bool = True) -> None:
        self.settings = settings or load_settings()
        self.bundle_path = Path(self.settings.artifacts_dir) / "finshield_bundle.joblib"
        self.bundle: Dict[str, Any] | None = None
        self.total_scored_events: int = 0
        self.cumulative_latency_ms: float = 0.0

        if self.bundle_path.exists():
            self.load_bundle()
        elif auto_train_if_missing:
            logger.info("Artifact bundle not found at %s; running initial training pipeline...", self.bundle_path)
            run_training_and_experiments(self.settings)
            self.load_bundle()

    def load_bundle(self) -> None:
        self.bundle = joblib.load(self.bundle_path)
        logger.info("Loaded FIN-SHIELD ML artifact bundle version=%s", self.bundle.get("version"))

    def _events_to_canonical_df(self, events: List[TransactionEventInput]) -> pd.DataFrame:
        rows = []
        for ev in events:
            d = ev.model_dump()
            for col in ["device_id", "ip_subnet", "email_domain", "addr_region", "dist_km"]:
                d[f"missing_{col}"] = 1 if d.get(col) is None else 0
            for col in ["device_id", "ip_subnet", "email_domain", "addr_region"]:
                if d.get(col) is None:
                    d[col] = "MISSING"
            if d.get("dist_km") is None:
                d["dist_km"] = -1.0
            d["is_fraud"] = 0
            rows.append(d)
        return pd.DataFrame(rows)

    def _detect_row_ood(self, feat_row: pd.Series, drift_monitor) -> List[str]:
        flagged: List[str] = []
        for col, edges in drift_monitor.reference_bins_.items():
            if col not in feat_row.index:
                continue
            val = float(feat_row[col])
            # Flag extreme tail values outside [1st finite edge, last finite edge]
            if len(edges) >= 4 and (val < edges[1] or val > edges[-2]):
                flagged.append(col)
        return flagged[:5]

    def score_batch(self, events: List[TransactionEventInput]) -> BatchScoreResponse:
        if self.bundle is None:
            raise RuntimeError("Model bundle is not loaded.")

        t0 = time.perf_counter()
        pipe = self.bundle["feature_pipeline"]
        gbdt = self.bundle["gbdt_model"]
        calibrator = self.bundle["calibrator"]
        conformal = self.bundle["conformal"]
        policy_engine = self.bundle["policy_engine"]
        thresholds = self.bundle["thresholds"]
        drift_monitor = self.bundle["drift_monitor"]
        history_df = self.bundle.get("reference_history_df")

        query_df = self._events_to_canonical_df(events)
        X_feat = pipe.transform_single_or_batch(query_df, history_df=history_df)

        raw_probs = gbdt.predict_proba(X_feat)
        cal_probs = calibrator.transform(raw_probs)
        lower_b, upper_b, is_uncertain = conformal.predict_sets(cal_probs)

        total_ms = (time.perf_counter() - t0) * 1000.0
        per_item_ms = total_ms / max(1, len(events))

        results: List[ScoreResponse] = []
        action_counts = {"APPROVE": 0, "STEPUP_CHALLENGE": 0, "BLOCK": 0}

        for idx, ev in enumerate(events):
            p_raw = float(raw_probs[idx])
            p_cal = float(cal_probs[idx])
            unc = bool(is_uncertain[idx])
            pol = policy_engine.decide_single(
                prob=p_cal,
                amount=float(ev.amount),
                t_stepup=float(thresholds["stepup_threshold"]),
                t_block=float(thresholds["block_threshold"]),
                is_conformal_uncertain=unc,
            )
            action = str(pol["action"])
            action_counts[action] = action_counts.get(action, 0) + 1

            ood_cols = self._detect_row_ood(X_feat.iloc[idx], drift_monitor)
            reasons_raw = gbdt.explain_local(X_feat.iloc[[idx]], top_k=5)
            reasons = [ReasonCodeItem(**r) for r in reasons_raw]

            results.append(
                ScoreResponse(
                    transaction_id=ev.transaction_id,
                    raw_fraud_score=round(p_raw, 6),
                    calibrated_fraud_probability=round(p_cal, 6),
                    conformal_lower_bound=round(float(lower_b[idx]), 6),
                    conformal_upper_bound=round(float(upper_b[idx]), 6),
                    is_conformal_uncertain=unc,
                    decision_action=action,
                    expected_loss_approve=float(pol["expected_loss_approve"]),
                    expected_loss_stepup=float(pol["expected_loss_stepup"]),
                    expected_loss_block=float(pol["expected_loss_block"]),
                    potential_chargeback_loss=float(pol["potential_chargeback_loss"]),
                    ood_shift_alert=len(ood_cols) >= 2,
                    ood_flagged_features=ood_cols,
                    reason_codes=reasons,
                    latency_ms=round(per_item_ms, 3),
                    model_version=str(self.bundle.get("version", "0.1.0")),
                )
            )

        self.total_scored_events += len(events)
        self.cumulative_latency_ms += total_ms

        return BatchScoreResponse(
            results=results,
            batch_size=len(events),
            total_latency_ms=round(total_ms, 3),
            mean_latency_ms=round(per_item_ms, 3),
            action_summary=action_counts,
            model_version=str(self.bundle.get("version", "0.1.0")),
        )

    def score_single(self, event: TransactionEventInput) -> ScoreResponse:
        return self.score_batch([event]).results[0]

    def health(self) -> HealthResponse:
        avg_lat = (
            self.cumulative_latency_ms / self.total_scored_events
            if self.total_scored_events > 0
            else 0.0
        )
        return HealthResponse(
            status="healthy" if self.bundle is not None else "degraded",
            model_loaded=self.bundle is not None,
            model_name=str(self.bundle.get("model_name", "unknown")) if self.bundle else "none",
            gbdt_backend=str(self.bundle.get("gbdt_backend", "unknown")) if self.bundle else "none",
            model_version=str(self.bundle.get("version", "0.1.0")) if self.bundle else "none",
            total_scored_events=self.total_scored_events,
            avg_latency_ms=round(avg_lat, 3),
            disclaimer="RESEARCH & PORTFOLIO DEMONSTRATION ONLY — Not certified for live payment authorization.",
        )
