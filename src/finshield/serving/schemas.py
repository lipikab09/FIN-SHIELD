from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionEventInput(BaseModel):
    transaction_id: str = Field(..., examples=["TXN_9000123"])
    timestamp: float = Field(..., ge=0.0, description="Unix or relative event timestamp in seconds", examples=[3628900.0])
    amount: float = Field(..., gt=0.0, le=1_000_000.0, examples=[245.50])
    card_id: str = Field(..., examples=["CARD_00412"])
    device_id: Optional[str] = Field(default=None, examples=["DEV_00105"])
    ip_subnet: Optional[str] = Field(default=None, examples=["10.0.14.0/24"])
    merchant_id: str = Field(..., examples=["MERCH_0042"])
    email_domain: Optional[str] = Field(default="gmail.com", examples=["proton.me"])
    product_code: str = Field(default="W", examples=["W"])
    card_network: str = Field(default="visa", examples=["visa"])
    card_type: str = Field(default="credit", examples=["credit"])
    addr_region: Optional[str] = Field(default="REG_12", examples=["REG_12"])
    dist_km: Optional[float] = Field(default=15.0, examples=[1420.0])
    is_international: int = Field(default=0, ge=0, le=1, examples=[1])


class ReasonCodeItem(BaseModel):
    feature: str
    value: float
    baseline_median: float
    z_score: float
    risk_contribution: float
    direction: str


class ScoreResponse(BaseModel):
    transaction_id: str
    raw_fraud_score: float
    calibrated_fraud_probability: float
    conformal_lower_bound: float
    conformal_upper_bound: float
    is_conformal_uncertain: bool
    decision_action: str
    expected_loss_approve: float
    expected_loss_stepup: float
    expected_loss_block: float
    potential_chargeback_loss: float
    ood_shift_alert: bool
    ood_flagged_features: List[str]
    reason_codes: List[ReasonCodeItem]
    latency_ms: float
    model_version: str


class BatchScoreRequest(BaseModel):
    events: List[TransactionEventInput] = Field(..., min_length=1, max_length=2000)


class BatchScoreResponse(BaseModel):
    results: List[ScoreResponse]
    batch_size: int
    total_latency_ms: float
    mean_latency_ms: float
    action_summary: Dict[str, int]
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_name: str
    gbdt_backend: str
    model_version: str
    total_scored_events: int
    avg_latency_ms: float
    disclaimer: str
