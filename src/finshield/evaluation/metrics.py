from __future__ import annotations

from typing import Dict, List

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve

from finshield.decision.cost_policy import CostAwareDecisionEngine, PolicyThresholds


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1] if i < n_bins - 1 else y_prob <= bin_edges[i + 1])
        if np.any(mask):
            bin_acc = float(np.mean(y_true[mask]))
            bin_conf = float(np.mean(y_prob[mask]))
            ece += (np.sum(mask) / n) * abs(bin_acc - bin_conf)
    return float(ece)


def reliability_diagram_bins(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> List[Dict[str, float]]:
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows: List[Dict[str, float]] = []
    for i in range(n_bins):
        mask = (y_prob >= bin_edges[i]) & (y_prob < bin_edges[i + 1] if i < n_bins - 1 else y_prob <= bin_edges[i + 1])
        count = int(np.sum(mask))
        rows.append(
            {
                "bin_lower": round(float(bin_edges[i]), 2),
                "bin_upper": round(float(bin_edges[i + 1]), 2),
                "count": count,
                "mean_predicted_prob": round(float(np.mean(y_prob[mask])), 5) if count > 0 else round(float((bin_edges[i] + bin_edges[i + 1]) / 2), 5),
                "observed_fraud_rate": round(float(np.mean(y_true[mask])), 5) if count > 0 else 0.0,
            }
        )
    return rows


def recall_at_fpr(y_true: np.ndarray, y_prob: np.ndarray, target_fpr: float) -> float:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    valid = np.where(fpr <= target_fpr)[0]
    if len(valid) == 0:
        return 0.0
    return float(np.max(tpr[valid]))


def compute_fraud_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    policy_engine: CostAwareDecisionEngine,
    thresholds: PolicyThresholds,
    conformal_uncertain: np.ndarray | None = None,
) -> Dict[str, object]:
    pr_auc = float(average_precision_score(y_true, y_prob))
    roc_auc = float(roc_auc_score(y_true, y_prob))
    brier = float(brier_score_loss(y_true, y_prob))
    ece = expected_calibration_error(y_true, y_prob, n_bins=10)

    r_01 = recall_at_fpr(y_true, y_prob, 0.001)
    r_05 = recall_at_fpr(y_true, y_prob, 0.005)
    r_10 = recall_at_fpr(y_true, y_prob, 0.010)

    policy_res = policy_engine.evaluate_policy(
        probs=y_prob,
        amounts=amounts,
        y_true=y_true,
        t_stepup=thresholds.stepup_threshold,
        t_block=thresholds.block_threshold,
        conformal_uncertain=conformal_uncertain,
    )

    return {
        "pr_auc": round(pr_auc, 5),
        "roc_auc": round(roc_auc, 5),
        "recall_at_fpr_0_1pct": round(r_01, 5),
        "recall_at_fpr_0_5pct": round(r_05, 5),
        "recall_at_fpr_1_0pct": round(r_10, 5),
        "ece": round(ece, 5),
        "brier_score": round(brier, 5),
        "expected_net_savings": policy_res.expected_net_savings,
        "fraud_prevented_value": policy_res.fraud_prevented_value,
        "chargeback_loss_incurred": policy_res.chargeback_loss_incurred,
        "review_cost_incurred": policy_res.review_cost_incurred,
        "false_decline_loss_incurred": policy_res.false_decline_loss_incurred,
        "approve_rate": policy_res.approve_rate,
        "stepup_rate": policy_res.stepup_rate,
        "block_rate": policy_res.block_rate,
    }


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    amounts: np.ndarray,
    policy_engine: CostAwareDecisionEngine,
    thresholds: PolicyThresholds,
    n_bootstrap: int = 250,
    seed: int = 42,
) -> Dict[str, Dict[str, float]]:
    rng = np.random.default_rng(seed)
    n = len(y_true)
    pr_aucs, roc_aucs, eces, savings = [], [], [], []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt = y_true[idx]
        if yt.sum() == 0 or yt.sum() == n:
            continue
        yp = y_prob[idx]
        am = amounts[idx]
        pr_aucs.append(float(average_precision_score(yt, yp)))
        roc_aucs.append(float(roc_auc_score(yt, yp)))
        eces.append(expected_calibration_error(yt, yp, n_bins=10))
        pol = policy_engine.evaluate_policy(yp, am, yt, thresholds.stepup_threshold, thresholds.block_threshold)
        savings.append(pol.expected_net_savings)

    def summarize(arr: List[float]) -> Dict[str, float]:
        a = np.array(arr, dtype=float)
        return {
            "mean": round(float(np.mean(a)), 5),
            "ci_lower_95": round(float(np.quantile(a, 0.025)), 5),
            "ci_upper_95": round(float(np.quantile(a, 0.975)), 5),
        }

    return {
        "pr_auc": summarize(pr_aucs),
        "roc_auc": summarize(roc_aucs),
        "ece": summarize(eces),
        "expected_net_savings": summarize(savings),
    }
