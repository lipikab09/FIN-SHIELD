from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd

from finshield.config import Settings, load_settings
from finshield.data.adapters import get_dataset_adapter
from finshield.data.splits import ChronologicalSplitter
from finshield.decision.cost_policy import CostAwareDecisionEngine
from finshield.evaluation.metrics import (
    bootstrap_confidence_intervals,
    compute_fraud_metrics,
    reliability_diagram_bins,
)
from finshield.features.graph_features import CausalEntityGraphExtractor
from finshield.features.pipeline import FraudFeaturePipeline
from finshield.models.calibration import ConformalRiskWrapper, ProbabilityCalibrator
from finshield.models.gbdt_models import GBDTFraudModel
from finshield.models.neural_resnet import TabularEntityResNet
from finshield.monitoring.drift import DriftMonitor


def run_training_and_experiments(settings: Settings | None = None) -> Dict[str, Any]:
    """
    Executes the full FIN-SHIELD ML research benchmark and persists production artifacts:
      1. Canonical dataset loading + strict chronological OOT split with label maturity embargo
      2. Causal tabular-only vs. causal graph-augmented vs. leaky ablation feature extraction
      3. Multi-paradigm model training (Tabular GBDT, Tabular Entity ResNet, Graph-Augmented GBDT, Graph-Augmented Neural)
      4. Calibration comparison (Uncalibrated, Platt, Temperature Scaling, Isotonic) + Split-Conformal intervals
      5. Cost-aware policy optimization + drift slice monitoring + rolling retraining adaptation experiment
    """
    t0 = time.perf_counter()
    cfg = settings or load_settings()
    artifacts_dir = Path(cfg.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    adapter = get_dataset_adapter(
        adapter_name=cfg.data.adapter,
        ieee_cis_dir=cfg.data.ieee_cis_dir,
        credit_card_path=cfg.data.credit_card_path,
        n_synthetic_events=cfg.data.n_synthetic_events,
        seed=cfg.project.seed,
    )
    df, dataset_meta = adapter.load()

    splitter = ChronologicalSplitter(
        train_ratio=cfg.data.splits.train_ratio,
        calib_ratio=cfg.data.splits.calib_ratio,
        test_ratio=cfg.data.splits.test_ratio,
        label_delay_hours=cfg.data.label_delay_hours,
    )
    split_res = splitter.split(df, time_col="timestamp")
    train_df, calib_df, test_df = split_res.train, split_res.calib, split_res.test

    y_train = train_df["is_fraud"].to_numpy(dtype=int)
    y_calib = calib_df["is_fraud"].to_numpy(dtype=int)
    y_test = test_df["is_fraud"].to_numpy(dtype=int)
    amt_calib = calib_df["amount"].to_numpy(dtype=float)
    amt_test = test_df["amount"].to_numpy(dtype=float)

    policy_engine = CostAwareDecisionEngine(
        fixed_chargeback_fee=cfg.policy.fixed_chargeback_fee,
        chargeback_amount_multiplier=cfg.policy.chargeback_amount_multiplier,
        stepup_review_cost=cfg.policy.stepup_review_cost,
        stepup_intercept_rate=cfg.policy.stepup_intercept_rate,
        false_decline_penalty=cfg.policy.false_decline_penalty,
        max_stepup_rate=cfg.policy.max_stepup_rate,
        max_block_rate=cfg.policy.max_block_rate,
    )

    # 1. Feature pipelines: Tabular-only (no graph), Graph-augmented (production), and Leaky Ablation
    tab_pipe = FraudFeaturePipeline(
        windows_seconds=cfg.features.windows_seconds,
        graph_lookback_seconds=cfg.features.graph_lookback_seconds,
        label_delay_hours=cfg.data.label_delay_hours,
        use_graph_features=False,
        leaky_mode=False,
    )
    X_tr_tab, X_cal_tab, X_te_tab = tab_pipe.fit_transform_splits(train_df, calib_df, test_df)

    graph_pipe = FraudFeaturePipeline(
        windows_seconds=cfg.features.windows_seconds,
        graph_lookback_seconds=cfg.features.graph_lookback_seconds,
        label_delay_hours=cfg.data.label_delay_hours,
        use_graph_features=True,
        leaky_mode=False,
    )
    X_tr_grp, X_cal_grp, X_te_grp = graph_pipe.fit_transform_splits(train_df, calib_df, test_df)

    leaky_pipe = FraudFeaturePipeline(
        windows_seconds=cfg.features.windows_seconds,
        graph_lookback_seconds=cfg.features.graph_lookback_seconds,
        label_delay_hours=cfg.data.label_delay_hours,
        use_graph_features=True,
        leaky_mode=True,
    )
    X_tr_lky, X_cal_lky, X_te_lky = leaky_pipe.fit_transform_splits(train_df, calib_df, test_df)

    # 2. Train candidate models
    tab_gbdt = GBDTFraudModel(backend="auto", random_state=cfg.project.seed).fit(X_tr_tab, y_train)
    tab_resnet = TabularEntityResNet(
        cat_cardinalities=tab_pipe.cat_cardinalities_,
        hidden_dim=cfg.models.neural.hidden_dims[0],
        dropout=cfg.models.neural.dropout,
        epochs=cfg.models.neural.epochs,
        batch_size=cfg.models.neural.batch_size,
        lr=cfg.models.neural.lr,
        focal_gamma=cfg.models.neural.focal_gamma,
        random_state=cfg.project.seed,
    ).fit(X_tr_tab, y_train)

    graph_gbdt = GBDTFraudModel(backend="auto", random_state=cfg.project.seed).fit(X_tr_grp, y_train)
    graph_resnet = TabularEntityResNet(
        cat_cardinalities=graph_pipe.cat_cardinalities_,
        hidden_dim=cfg.models.neural.hidden_dims[0],
        dropout=cfg.models.neural.dropout,
        epochs=cfg.models.neural.epochs,
        batch_size=cfg.models.neural.batch_size,
        lr=cfg.models.neural.lr,
        focal_gamma=cfg.models.neural.focal_gamma,
        random_state=cfg.project.seed,
    ).fit(X_tr_grp, y_train)

    leaky_gbdt = GBDTFraudModel(backend="auto", random_state=cfg.project.seed).fit(X_tr_lky, y_train)

    # Hybrid Graph-GBDT + Graph-ResNet Ensemble
    def ensemble_raw(X_df: pd.DataFrame) -> np.ndarray:
        return 0.72 * graph_gbdt.predict_proba(X_df) + 0.28 * graph_resnet.predict_proba(X_df)

    model_candidates = {
        "tabular_gbdt_baseline": (
            tab_gbdt.predict_proba(X_cal_tab),
            tab_gbdt.predict_proba(X_te_tab),
            "Causal Tabular GBDT (No Graph)",
        ),
        "tabular_entity_resnet": (
            tab_resnet.predict_proba(X_cal_tab),
            tab_resnet.predict_proba(X_te_tab),
            "Causal Tabular Entity-ResNet (Focal Loss)",
        ),
        "graph_augmented_resnet": (
            graph_resnet.predict_proba(X_cal_grp),
            graph_resnet.predict_proba(X_te_grp),
            "Graph-Augmented Entity-ResNet",
        ),
        "graph_augmented_gbdt": (
            graph_gbdt.predict_proba(X_cal_grp),
            graph_gbdt.predict_proba(X_te_grp),
            "Graph-Augmented GBDT (Causal Bipartite + Delayed-Label Risk)",
        ),
        "hybrid_graph_ensemble": (
            ensemble_raw(X_cal_grp),
            ensemble_raw(X_te_grp),
            "Hybrid Graph-GBDT + Graph-ResNet Ensemble",
        ),
        "leaky_future_ablation_gbdt": (
            leaky_gbdt.predict_proba(X_cal_lky),
            leaky_gbdt.predict_proba(X_te_lky),
            "Ablation: Non-Causal Future-Leaky Features (Invalid for Prod)",
        ),
    }

    model_benchmarks: Dict[str, Any] = {}
    calibrated_test_probs: Dict[str, np.ndarray] = {}
    calibrated_calib_probs: Dict[str, np.ndarray] = {}
    best_thresholds_by_model: Dict[str, Any] = {}

    for name, (cal_raw, te_raw, desc) in model_candidates.items():
        calibrator = ProbabilityCalibrator(method=cfg.models.calibration_method).fit(cal_raw, y_calib)
        cal_p = calibrator.transform(cal_raw)
        te_p = calibrator.transform(te_raw)

        conformal = ConformalRiskWrapper(alpha=cfg.models.conformal_alpha).fit(cal_p, y_calib)
        _, _, te_uncertain = conformal.predict_sets(te_p)

        opt_thresh = policy_engine.optimize_thresholds(cal_p, amt_calib, y_calib)
        metrics = compute_fraud_metrics(
            y_true=y_test,
            y_prob=te_p,
            amounts=amt_test,
            policy_engine=policy_engine,
            thresholds=opt_thresh,
            conformal_uncertain=te_uncertain,
        )
        metrics["description"] = desc
        metrics["stepup_threshold"] = opt_thresh.stepup_threshold
        metrics["block_threshold"] = opt_thresh.block_threshold
        metrics["conformal_abstention_rate"] = round(float(np.mean(te_uncertain)), 4)

        if name in ("tabular_gbdt_baseline", "graph_augmented_gbdt", "hybrid_graph_ensemble"):
            metrics["bootstrap_95_ci"] = bootstrap_confidence_intervals(
                y_true=y_test,
                y_prob=te_p,
                amounts=amt_test,
                policy_engine=policy_engine,
                thresholds=opt_thresh,
                n_bootstrap=200,
                seed=cfg.project.seed,
            )

        model_benchmarks[name] = metrics
        calibrated_test_probs[name] = te_p
        calibrated_calib_probs[name] = cal_p
        best_thresholds_by_model[name] = opt_thresh

    # 3. Calibration method ablation on Graph-Augmented GBDT
    raw_cal_g = graph_gbdt.predict_proba(X_cal_grp)
    raw_te_g = graph_gbdt.predict_proba(X_te_grp)
    calibration_comparison: Dict[str, Any] = {}
    reliability_curves: Dict[str, Any] = {}

    for cal_method in ["none", "platt", "temperature", "isotonic"]:
        cal_obj = ProbabilityCalibrator(method=cal_method).fit(raw_cal_g, y_calib)
        cp = cal_obj.transform(raw_cal_g)
        tp = cal_obj.transform(raw_te_g)
        th = policy_engine.optimize_thresholds(cp, amt_calib, y_calib)
        calibration_comparison[cal_method] = compute_fraud_metrics(y_test, tp, amt_test, policy_engine, th)
        reliability_curves[cal_method] = reliability_diagram_bins(y_test, tp, n_bins=10)

    # 4. Drift monitoring across chronological test slices (Early Test vs Late Drift Regime)
    prod_calibrator = ProbabilityCalibrator(method=cfg.models.calibration_method).fit(raw_cal_g, y_calib)
    prod_cal_probs = prod_calibrator.transform(raw_cal_g)
    prod_te_probs = prod_calibrator.transform(raw_te_g)
    prod_conformal = ConformalRiskWrapper(alpha=cfg.models.conformal_alpha).fit(prod_cal_probs, y_calib)
    prod_thresholds = policy_engine.optimize_thresholds(prod_cal_probs, amt_calib, y_calib)

    drift_monitor = DriftMonitor(
        psi_warn_threshold=cfg.drift.psi_warn_threshold,
        psi_critical_threshold=cfg.drift.psi_critical_threshold,
        ks_alpha=cfg.drift.ks_alpha,
        n_bins=cfg.drift.n_bins,
    ).fit(X_tr_grp, graph_pipe.feature_names_, ref_scores=prod_cal_probs)

    mid_test = len(X_te_grp) // 2
    early_drift = drift_monitor.evaluate(X_te_grp.iloc[:mid_test], cur_scores=prod_te_probs[:mid_test])
    late_drift = drift_monitor.evaluate(X_te_grp.iloc[mid_test:], cur_scores=prod_te_probs[mid_test:])

    # 5. Concept Drift Adaptation Experiment: Static Model vs Rolling Window Retraining on Late Test Window
    late_y = y_test[mid_test:]
    late_amt = amt_test[mid_test:]
    late_static_probs = prod_te_probs[mid_test:]
    static_late_metrics = compute_fraud_metrics(late_y, late_static_probs, late_amt, policy_engine, prod_thresholds)

    # Retrain on recent sliding window (second half of train + calib + early test)
    adapt_X = pd.concat([X_tr_grp.iloc[len(X_tr_grp) // 2 :], X_cal_grp, X_te_grp.iloc[:mid_test]], axis=0)
    adapt_y = np.concatenate([y_train[len(y_train) // 2 :], y_calib, y_test[:mid_test]])
    adapted_gbdt = GBDTFraudModel(backend="auto", random_state=cfg.project.seed + 7).fit(adapt_X, adapt_y)
    adapt_raw_cal = adapted_gbdt.predict_proba(X_te_grp.iloc[:mid_test])
    adapt_calibrator = ProbabilityCalibrator(method=cfg.models.calibration_method).fit(adapt_raw_cal, y_test[:mid_test])
    adapt_late_probs = adapt_calibrator.transform(adapted_gbdt.predict_proba(X_te_grp.iloc[mid_test:]))
    adapted_late_metrics = compute_fraud_metrics(late_y, adapt_late_probs, late_amt, policy_engine, prod_thresholds)

    # 6. Subgraph summary for Network Explorer
    graph_summary = CausalEntityGraphExtractor.build_subgraph_summary(test_df, max_edges=400)
    graph_summary_serializable = {k: v for k, v in graph_summary.items() if k != "graph"}

    # 7. Persist production bundle & experiment report
    bundle = {
        "version": cfg.project.version,
        "model_name": "graph_augmented_gbdt",
        "gbdt_backend": graph_gbdt.active_backend_,
        "feature_pipeline": graph_pipe,
        "gbdt_model": graph_gbdt,
        "resnet_model": graph_resnet,
        "calibrator": prod_calibrator,
        "conformal": prod_conformal,
        "policy_engine": policy_engine,
        "thresholds": prod_thresholds.to_dict(),
        "drift_monitor": drift_monitor,
        "reference_history_df": pd.concat([calib_df.tail(1500), test_df.tail(1500)], axis=0).reset_index(drop=True),
    }
    joblib.dump(bundle, artifacts_dir / "finshield_bundle.joblib")

    # Save sample test stream for dashboard interactive replay & batch scoring
    sample_test_export = test_df.tail(500).copy()
    sample_test_export["predicted_fraud_prob"] = np.round(prod_te_probs[-500:], 5)
    sample_test_export.to_csv(artifacts_dir / "test_scored_sample.csv", index=False)

    elapsed = round(time.perf_counter() - t0, 2)
    report = {
        "project": cfg.project.name,
        "version": cfg.project.version,
        "gbdt_backend": graph_gbdt.active_backend_,
        "runtime_seconds": elapsed,
        "dataset_metadata": dataset_meta,
        "split_metadata": split_res.metadata,
        "active_policy_thresholds": prod_thresholds.to_dict(),
        "model_benchmarks": model_benchmarks,
        "calibration_comparison": calibration_comparison,
        "reliability_curves": reliability_curves,
        "drift_monitoring": {
            "early_test_window": early_drift.to_dict(),
            "late_test_window": late_drift.to_dict(),
        },
        "drift_adaptation_experiment": {
            "static_model_on_late_drift": static_late_metrics,
            "rolling_retrained_on_late_drift": adapted_late_metrics,
        },
        "entity_graph_summary": graph_summary_serializable,
    }

    with open(artifacts_dir / "experiment_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    return report


if __name__ == "__main__":
    res = run_training_and_experiments()
    print(json.dumps(res["model_benchmarks"], indent=2))
