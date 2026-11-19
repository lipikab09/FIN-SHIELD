from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List

import numpy as np


@dataclass
class PolicyThresholds:
    stepup_threshold: float
    block_threshold: float
    expected_net_savings: float
    fraud_prevented_value: float
    chargeback_loss_incurred: float
    review_cost_incurred: float
    false_decline_loss_incurred: float
    approve_rate: float
    stepup_rate: float
    block_rate: float

    def to_dict(self) -> Dict[str, float]:
        return asdict(self)


class CostAwareDecisionEngine:
    """
    Bayes-optimal & capacity-constrained 3-way action policy engine:
      - APPROVE: p < t_stepup and not conformal_abstain
      - STEPUP_CHALLENGE: t_stepup <= p < t_block (or conformal uncertainty routing)
      - BLOCK: p >= t_block

    Explicit financial cost model per transaction i with calibrated fraud probability p_i and amount A_i:
      - Chargeback loss if fraud approved: L_cb(A_i) = multiplier * A_i + fixed_chargeback_fee
      - Expected loss under APPROVE: p_i * L_cb(A_i)
      - Expected loss under STEPUP_CHALLENGE: c_review + p_i * (1 - eta_intercept) * L_cb(A_i)
      - Expected loss under BLOCK: (1 - p_i) * c_false_decline
    """

    def __init__(
        self,
        fixed_chargeback_fee: float = 25.0,
        chargeback_amount_multiplier: float = 1.0,
        stepup_review_cost: float = 4.50,
        stepup_intercept_rate: float = 0.82,
        false_decline_penalty: float = 12.0,
        max_stepup_rate: float = 0.06,
        max_block_rate: float = 0.025,
    ) -> None:
        self.fixed_chargeback_fee = fixed_chargeback_fee
        self.chargeback_amount_multiplier = chargeback_amount_multiplier
        self.stepup_review_cost = stepup_review_cost
        self.stepup_intercept_rate = stepup_intercept_rate
        self.false_decline_penalty = false_decline_penalty
        self.max_stepup_rate = max_stepup_rate
        self.max_block_rate = max_block_rate

    def chargeback_loss(self, amounts: np.ndarray) -> np.ndarray:
        return self.chargeback_amount_multiplier * amounts + self.fixed_chargeback_fee

    def evaluate_policy(
        self,
        probs: np.ndarray,
        amounts: np.ndarray,
        y_true: np.ndarray,
        t_stepup: float,
        t_block: float,
        conformal_uncertain: np.ndarray | None = None,
    ) -> PolicyThresholds:
        cb_loss_vec = self.chargeback_loss(amounts)
        baseline_unmitigated_loss = float(np.sum(cb_loss_vec[y_true == 1]))

        actions = self.decide_batch(
            probs=probs,
            amounts=amounts,
            t_stepup=t_stepup,
            t_block=t_block,
            conformal_uncertain=conformal_uncertain,
        )

        is_approve = actions == "APPROVE"
        is_stepup = actions == "STEPUP_CHALLENGE"
        is_block = actions == "BLOCK"
        is_fraud = y_true == 1
        is_legit = y_true == 0

        # Fraud losses incurred
        loss_approved_fraud = float(np.sum(cb_loss_vec[is_approve & is_fraud]))
        loss_missed_stepup_fraud = float(
            np.sum(cb_loss_vec[is_stepup & is_fraud]) * (1.0 - self.stepup_intercept_rate)
        )
        total_cb_loss = loss_approved_fraud + loss_missed_stepup_fraud

        # Operational & customer friction costs
        review_cost = float(np.sum(is_stepup) * self.stepup_review_cost)
        false_decline_loss = float(np.sum(is_block & is_legit) * self.false_decline_penalty)

        total_cost = total_cb_loss + review_cost + false_decline_loss
        net_savings = baseline_unmitigated_loss - total_cost
        fraud_prevented = baseline_unmitigated_loss - total_cb_loss

        return PolicyThresholds(
            stepup_threshold=round(float(t_stepup), 5),
            block_threshold=round(float(t_block), 5),
            expected_net_savings=round(net_savings, 2),
            fraud_prevented_value=round(fraud_prevented, 2),
            chargeback_loss_incurred=round(total_cb_loss, 2),
            review_cost_incurred=round(review_cost, 2),
            false_decline_loss_incurred=round(false_decline_loss, 2),
            approve_rate=round(float(np.mean(is_approve)), 4),
            stepup_rate=round(float(np.mean(is_stepup)), 4),
            block_rate=round(float(np.mean(is_block)), 4),
        )

    def optimize_thresholds(
        self,
        calib_probs: np.ndarray,
        calib_amounts: np.ndarray,
        y_calib: np.ndarray,
    ) -> PolicyThresholds:
        """Grid-searches (t_stepup, t_block) on the calibration window subject to operational review/block caps."""
        best: PolicyThresholds | None = None
        stepup_grid = np.linspace(0.03, 0.35, 33)
        block_grid = np.linspace(0.15, 0.85, 36)

        for t_s in stepup_grid:
            for t_b in block_grid:
                if t_b <= t_s + 0.03:
                    continue
                res = self.evaluate_policy(calib_probs, calib_amounts, y_calib, float(t_s), float(t_b))
                if res.stepup_rate > self.max_stepup_rate or res.block_rate > self.max_block_rate:
                    continue
                if best is None or res.expected_net_savings > best.expected_net_savings:
                    best = res

        if best is None:
            best = self.evaluate_policy(calib_probs, calib_amounts, y_calib, 0.12, 0.45)
        return best

    def decide_batch(
        self,
        probs: np.ndarray,
        amounts: np.ndarray,
        t_stepup: float,
        t_block: float,
        conformal_uncertain: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Combines probability thresholds with transaction-amount-adjusted Bayes risk
        and conformal abstention routing.
        """
        n = len(probs)
        actions = np.full(n, "APPROVE", dtype=object)

        # High-value transactions lower the effective block threshold slightly under Bayes expected loss
        cb_loss = self.chargeback_loss(amounts)
        bayes_block_p = self.false_decline_penalty / np.maximum(1.0, cb_loss + self.false_decline_penalty)
        effective_block_t = np.minimum(t_block, np.maximum(t_stepup + 0.05, bayes_block_p * 1.35))

        stepup_mask = probs >= t_stepup
        if conformal_uncertain is not None:
            # Conformal ambiguity near decision boundary routes to step-up review rather than blind auto-approve
            near_boundary = (probs >= (t_stepup * 0.65)) & conformal_uncertain
            stepup_mask = stepup_mask | near_boundary

        block_mask = probs >= effective_block_t
        actions[stepup_mask] = "STEPUP_CHALLENGE"
        actions[block_mask] = "BLOCK"
        return actions

    def decide_single(
        self,
        prob: float,
        amount: float,
        t_stepup: float,
        t_block: float,
        is_conformal_uncertain: bool = False,
    ) -> Dict[str, object]:
        action = self.decide_batch(
            probs=np.array([prob]),
            amounts=np.array([amount]),
            t_stepup=t_stepup,
            t_block=t_block,
            conformal_uncertain=np.array([is_conformal_uncertain]),
        )[0]
        cb_loss = float(self.chargeback_loss(np.array([amount]))[0])
        exp_loss_approve = prob * cb_loss
        exp_loss_stepup = self.stepup_review_cost + prob * (1.0 - self.stepup_intercept_rate) * cb_loss
        exp_loss_block = (1.0 - prob) * self.false_decline_penalty
        return {
            "action": str(action),
            "expected_loss_approve": round(exp_loss_approve, 2),
            "expected_loss_stepup": round(exp_loss_stepup, 2),
            "expected_loss_block": round(exp_loss_block, 2),
            "potential_chargeback_loss": round(cb_loss, 2),
        }
