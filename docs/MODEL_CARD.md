# Model Card: FIN-SHIELD ML Risk & Fraud Scoring Engine

## Model Details
- **Architecture**: Causal Bipartite Entity Graph + Streaming Temporal Window Feature Pipeline paired with Gradient Boosted Decision Trees (`LightGBM` / `XGBoost` / `HistGBDT`), Focal-Loss `TabularEntityResNet`, post-hoc probability calibration (`Isotonic` / `Platt` / `Temperature`), Mondrian Split-Conformal intervals, and Bayes Cost-Aware 3-Way Decisioning.
- **Version**: 0.1.0
- **Intended Use**: Research demonstration, ML engineering portfolio showcase, and offline evaluation of non-stationary payment fraud pipelines.

## Out-of-Scope & Limitations
- Not certified for live payment authorization, PCI-DSS production workloads, or autonomous credit/account termination without human compliance review.
- Synthetic streaming defaults simulate covariate and concept drift regimes; production deployments require domain-specific chargeback cost calibration and fairness/disparate-impact auditing across cardholder segments.
