from __future__ import annotations

import numpy as np

from finshield.decision.cost_policy import CostAwareDecisionEngine
from finshield.evaluation.metrics import expected_calibration_error
from finshield.models.calibration import ConformalRiskWrapper, ProbabilityCalibrator


def test_calibration_and_conformal_coverage():
    rng = np.random.default_rng(42)
    y = (rng.random(1500) < 0.08).astype(int)
    raw_scores = np.clip(0.03 + 0.55 * y + rng.normal(0, 0.12, size=1500), 0.001, 0.999)

    cal = ProbabilityCalibrator(method="isotonic").fit(raw_scores[:800], y[:800])
    cal_p = cal.transform(raw_scores[800:])
    ece = expected_calibration_error(y[800:], cal_p, n_bins=10)
    assert ece < 0.08

    conf = ConformalRiskWrapper(alpha=0.10).fit(cal.transform(raw_scores[:800]), y[:800])
    lo, hi, unc = conf.predict_sets(cal_p)
    assert np.all(lo <= hi)
    assert unc.shape == cal_p.shape


def test_cost_aware_policy_ordering():
    engine = CostAwareDecisionEngine(fixed_chargeback_fee=25.0, stepup_review_cost=4.5, false_decline_penalty=12.0)
    low_risk = engine.decide_single(prob=0.005, amount=50.0, t_stepup=0.10, t_block=0.45)
    high_risk = engine.decide_single(prob=0.88, amount=600.0, t_stepup=0.10, t_block=0.45)
    assert low_risk["action"] == "APPROVE"
    assert high_risk["action"] == "BLOCK"
