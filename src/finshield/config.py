from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import yaml
from pydantic import BaseModel, Field


class SplitConfig(BaseModel):
    train_ratio: float = 0.60
    calib_ratio: float = 0.15
    test_ratio: float = 0.25


class DataConfig(BaseModel):
    adapter: str = "synthetic_stream"
    data_dir: str = "data"
    ieee_cis_dir: str = "data/ieee-cis"
    credit_card_path: str = "data/creditcard.csv"
    n_synthetic_events: int = 24000
    label_delay_hours: int = 72
    splits: SplitConfig = Field(default_factory=SplitConfig)


class FeatureConfig(BaseModel):
    windows_seconds: List[int] = Field(default_factory=lambda: [3600, 86400, 604800])
    entity_columns: List[str] = Field(
        default_factory=lambda: ["card_id", "device_id", "ip_subnet", "merchant_id", "email_domain"]
    )
    graph_lookback_seconds: int = 604800


class NeuralConfig(BaseModel):
    hidden_dims: List[int] = Field(default_factory=lambda: [128, 64])
    dropout: float = 0.25
    epochs: int = 14
    batch_size: int = 256
    lr: float = 0.0015
    focal_gamma: float = 2.0


class ModelConfig(BaseModel):
    active_model: str = "graph_augmented_gbdt"
    calibration_method: str = "isotonic"
    conformal_alpha: float = 0.05
    neural: NeuralConfig = Field(default_factory=NeuralConfig)


class PolicyConfig(BaseModel):
    fixed_chargeback_fee: float = 25.0
    chargeback_amount_multiplier: float = 1.0
    stepup_review_cost: float = 4.50
    stepup_intercept_rate: float = 0.82
    false_decline_penalty: float = 12.0
    max_stepup_rate: float = 0.06
    max_block_rate: float = 0.025


class DriftConfig(BaseModel):
    psi_warn_threshold: float = 0.10
    psi_critical_threshold: float = 0.25
    ks_alpha: float = 0.01
    n_bins: int = 10


class ProjectConfig(BaseModel):
    name: str = "FIN-SHIELD ML"
    version: str = "0.1.0"
    seed: int = 42


class Settings(BaseModel):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    drift: DriftConfig = Field(default_factory=DriftConfig)
    artifacts_dir: str = "artifacts"


def load_settings(config_path: str | Path | None = None) -> Settings:
    raw: Dict[str, Any] = {}
    default_path = Path(config_path) if config_path else Path(os.getenv("FINSHIELD_CONFIG", "configs/default.yaml"))
    if default_path.exists():
        with open(default_path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

    cfg = Settings(**raw)
    # Apply environment variable overrides safely without hardcoded local paths
    if os.getenv("FINSHIELD_DATA_DIR"):
        cfg.data.data_dir = os.environ["FINSHIELD_DATA_DIR"]
    if os.getenv("FINSHIELD_ARTIFACTS_DIR"):
        cfg.artifacts_dir = os.environ["FINSHIELD_ARTIFACTS_DIR"]
    if os.getenv("FINSHIELD_DATASET_ADAPTER"):
        cfg.data.adapter = os.environ["FINSHIELD_DATASET_ADAPTER"]
    if os.getenv("FINSHIELD_IEEE_CIS_DIR"):
        cfg.data.ieee_cis_dir = os.environ["FINSHIELD_IEEE_CIS_DIR"]
    if os.getenv("FINSHIELD_MODEL_NAME"):
        cfg.models.active_model = os.environ["FINSHIELD_MODEL_NAME"]
    if os.getenv("FINSHIELD_CALIBRATION_METHOD"):
        cfg.models.calibration_method = os.environ["FINSHIELD_CALIBRATION_METHOD"]
    if os.getenv("FINSHIELD_LABEL_DELAY_HOURS"):
        cfg.data.label_delay_hours = int(os.environ["FINSHIELD_LABEL_DELAY_HOURS"])
    if os.getenv("FINSHIELD_CONFORMAL_ALPHA"):
        cfg.models.conformal_alpha = float(os.environ["FINSHIELD_CONFORMAL_ALPHA"])
    return cfg
