# FIN-SHIELD ML: Drift-Resilient Transaction Fraud & Relational Risk Engine

[![CI](https://img.shields.io/badge/CI-pytest%20verified-14B8A6)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-0A1628)]()
[![FastAPI](https://img.shields.io/badge/Serving-FastAPI-2DD4BF)]()
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-38BDF8)]()
[![License](https://img.shields.io/badge/Research-Portfolio%20Engine-1E3A5F)]()

**FIN-SHIELD ML** is an end-to-end, research-oriented and production-engineered transaction risk and relational fraud detection engine[cite: 2, 31]. Unlike static tabular toy classifiers, FIN-SHIELD ML is built around real-world payment engineering constraints: **strict out-of-time (OOT) chronology**[cite: 7], **chargeback label-maturity delays**[cite: 7, 10], **causal streaming velocity**[cite: 11], **bipartite entity-graph aggregations**[cite: 10], **post-hoc probability calibration**[cite: 13], **Mondrian split-conformal abstention**[cite: 13], **covariate/concept drift monitoring**[cite: 19], and **Bayes cost-aware decisioning**[cite: 21].

> **Research & Engineering Demonstration Disclaimer**: Built for ML systems research, technical interviews, and portfolio demonstration[cite: 31, 33]. Not certified for live payment authorization or PCI-DSS production workloads without formal compliance and bank risk governance review[cite: 26, 29, 33].

---

## Table of Contents
1. [Why FIN-SHIELD ML? (Problem & Engineering Motivation)](#1-why-fin-shield-ml-problem--engineering-motivation)
2. [System Architecture](#2-system-architecture)
3. [Repository & Folder Structure](#3-repository--folder-structure)
4. [Empirical Out-of-Time (OOT) Benchmark Results](#4-empirical-out-of-time-oot-benchmark-results)
5. [Mathematical Formulation: Cost-Aware 3-Way Policy & Conformal Risk](#5-mathematical-formulation-cost-aware-3-way-policy--conformal-risk)
6. [Quickstart & Local Setup](#6-quickstart--local-setup)
7. [Dataset Adapters (IEEE-CIS, Credit Card CSV, Streaming)](#7-dataset-adapters-ieee-cis-credit-card-csv-streaming)
8. [FastAPI Scoring Service & API Contract](#8-fastapi-scoring-service--api-contract)
9. [Interactive 7-Page Streamlit Dashboard](#9-interactive-7-page-streamlit-dashboard)
10. [Testing, Leakage Verification & Docker Deployment](#10-testing-leakage-verification--docker-deployment)

---

## 1. Why FIN-SHIELD ML? (Problem & Engineering Motivation)

Standard fraud detection pipelines frequently suffer from four structural flaws that invalidate offline metrics in production:

| Flaw in Standard Pipelines | How FIN-SHIELD ML Solves It |
| :--- | :--- |
| **Random K-Fold Splitting**: Mixes future and past transactions from the same card/device. | **Strict Out-of-Time (OOT) Splitting**: Enforces $t_{\text{train}} < t_{\text{calib}} < t_{\text{test}}$ with zero temporal overlap[cite: 7, 29]. |
| **Future Feature Leakage**: Global `groupby('device_id').nunique()` counts cards from *next week* to score *today's* event[cite: 11, 29]. | **Causal Sliding Windows & Graph Deques**: Every rolling window (`1h`, `24h`, `7d`) and graph edge strictly conditions on $t' < t$, verified by automated prefix-invariance tests (`test_causal_leakage.py`)[cite: 3, 10, 11, 36]. |
| **Instantaneous Label Assumption**: Treats fraud chargebacks as known immediately at transaction time. | **Label Maturity Embargo**: Holds back events within `label_delay_hours` (default `72h`) of split boundaries and delays neighbor fraud propagation until chargebacks mature[cite: 3, 7, 10]. |
| **Raw Score Thresholding**: Uses uncalibrated scores and ignores asymmetric dollar costs. | **Calibrated Probabilities + Bayes Policy**: Calibrates scores on a dedicated OOT calibration slice, computes Mondrian conformal intervals, and minimizes expected dollar loss across `APPROVE`, `STEPUP_CHALLENGE`, and `BLOCK`. |

---

## 2. System Architecture

```text
Incoming Payment Events (Streaming Simulator / IEEE-CIS / Credit Card CSV)
       │
       ▼
[1. Canonical Schema Adapter & Missingness Indicators] ──► SHA-256 Dataset Fingerprint
       │
       ▼
[2. Strict OOT Chronological Splitter + 72h Label-Delay Embargo]
       │
       ├──► [3a. Causal Sliding-Window Extractor (1h / 24h / 7d, strictly t' < t)]
       └──► [3b. Causal Bipartite Entity Graph (1-hop/2-hop, Jaccard, Triads, Matured Risk)]
                    │
                    ▼
[4. Multi-Paradigm Scoring: Graph-GBDT + Focal-Loss Tabular Entity-ResNet + Hybrid Ensemble]
                    │
                    ▼
[5. Held-Out Calibration (Isotonic / Platt / Temperature) + Mondrian Conformal Bounds]
                    │
                    ▼
[6. Bayes Cost-Optimal 3-Way Action Policy] ──► APPROVE | STEPUP_CHALLENGE | BLOCK
                    │
                    ├──► FastAPI REST Service (/health, /v1/score, /v1/score/batch)
                    └──► 7-Page Streamlit Analytics, Drift & Cost Simulation Dashboard