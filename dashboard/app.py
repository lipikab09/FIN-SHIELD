from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Ensure src is importable when launched via `streamlit run dashboard/app.py`
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR / "src") not in sys.path:
    sys.path.insert(0, str(ROOT_DIR / "src"))

from finshield.config import load_settings
from finshield.decision.cost_policy import CostAwareDecisionEngine
from finshield.features.graph_features import CausalEntityGraphExtractor
from finshield.serving.schemas import TransactionEventInput
from finshield.serving.service import FraudInferenceService
from finshield.training.pipeline import run_training_and_experiments

st.set_page_config(
    page_title="FIN-SHIELD ML | Relational Fraud & Drift Risk Engine",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark Navy (#0A1628 / #112240), Crisp White (#F8FAFC), Subtle Teal (#2DD4BF / #14B8A6) Theme
CUSTOM_CSS = """
<style>
    .stApp {
        background-color: #0A1628;
        color: #F8FAFC;
    }
    section[data-testid="stSidebar"] {
        background-color: #07101E;
        border-right: 1px solid #1E3A5F;
    }
    .metric-card {
        background: linear-gradient(145deg, #112240 0%, #0D1B32 100%);
        border: 1px solid #1E3A5F;
        border-left: 4px solid #2DD4BF;
        border-radius: 10px;
        padding: 16px 18px;
        margin-bottom: 12px;
    }
    .metric-label {
        font-size: 0.80rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #94A3B8;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.65rem;
        font-weight: 700;
        color: #F8FAFC;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #2DD4BF;
        margin-top: 2px;
    }
    .disclaimer-banner {
        background-color: #112240;
        border: 1px solid #2DD4BF55;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 0.82rem;
        color: #CBD5E1;
        margin-bottom: 16px;
    }
    h1, h2, h3 {
        color: #F8FAFC !important;
    }
    .teal-accent {
        color: #2DD4BF;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(17, 34, 64, 0.0)",
    plot_bgcolor="rgba(17, 34, 64, 0.65)",
    font=dict(color="#F8FAFC", family="Inter, sans-serif"),
    xaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    yaxis=dict(gridcolor="#1E3A5F", zerolinecolor="#1E3A5F"),
    margin=dict(l=30, r=20, t=42, b=30),
)


@st.cache_resource
def get_inference_service() -> FraudInferenceService:
    cfg = load_settings()
    return FraudInferenceService(settings=cfg, auto_train_if_missing=True)


def load_experiment_report() -> dict:
    cfg = load_settings()
    report_path = Path(cfg.artifacts_dir) / "experiment_report.json"
    if not report_path.exists():
        return run_training_and_experiments(cfg)
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def render_metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{sub}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# Sidebar Navigation
with st.sidebar:
    st.markdown("## 🛡️ FIN-SHIELD ML")
    st.caption("Drift-Resilient Transaction Fraud & Relational Risk Engine")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [
            "1. Executive Architecture & Story",
            "2. Live & Batch Transaction Scorer",
            "3. Risk Calibration & Conformal Bounds",
            "4. Entity Network & Ring Explorer",
            "5. Concept Drift & Shift Monitor",
            "6. Cost-Aware Threshold Simulator",
            "7. Benchmark & Ablation Lab",
        ],
    )
    st.markdown("---")
    if st.button("🔄 Re-Run Full Training & Benchmarks", use_container_width=True):
        with st.spinner("Running chronological OOT training, graph extraction, calibration & bootstrap evaluation..."):
            cfg = load_settings()
            run_training_and_experiments(cfg)
            st.cache_resource.clear()
        st.success("Artifacts & benchmark report refreshed.")

    st.markdown(
        """
        <div style="font-size:0.75rem; color:#94A3B8; margin-top:16px;">
        <strong>Research & Demonstration Notice:</strong><br/>
        Built for ML systems research, architecture walkthroughs, and portfolio evaluation.
        Not certified for live production payment processing without PCI-DSS & bank compliance review.
        </div>
        """,
        unsafe_allow_html=True,
    )

service = get_inference_service()
report = load_experiment_report()

st.markdown(
    """
    <div class="disclaimer-banner">
        <strong>FIN-SHIELD ML v0.1.0</strong> — Research & Engineering Demonstration Platform |
        Strict Out-of-Time (OOT) Chronological Validation · Causal Entity Graph Features ·
        Mondrian Conformal Intervals · Bayes Cost-Optimal Action Policy
    </div>
    """,
    unsafe_allow_html=True,
)

# ==============================================================================
# PAGE 1: EXECUTIVE ARCHITECTURE & INTERACTIVE PROJECT WALKTHROUGH
# ==============================================================================
if page.startswith("1."):
    st.title("FIN-SHIELD ML: System Architecture & Research Walkthrough")
    st.markdown(
        "An end-to-end payment risk engine engineered to eliminate **temporal leakage**, capture **multi-entity fraud rings**, "
        "quantify **predictive uncertainty**, and optimize **dollar-denominated net fraud savings** under concept drift."
    )

    active_bench = report["model_benchmarks"]["graph_augmented_gbdt"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("OOT Test PR-AUC", f"{active_bench['pr_auc']:.4f}", f"ROC-AUC: {active_bench['roc_auc']:.4f}")
    with c2:
        render_metric_card("Net Financial Savings", f"${active_bench['expected_net_savings']:,.2f}", "On OOT Test Horizon")
    with c3:
        render_metric_card("Calibration ECE", f"{active_bench['ece']:.4f}", f"Brier: {active_bench['brier_score']:.4f}")
    with c4:
        render_metric_card(
            "Recall @ 1.0% FPR",
            f"{active_bench['recall_at_fpr_1_0pct'] * 100:.1f}%",
            f"Conformal Abstain: {active_bench['conformal_abstention_rate'] * 100:.1f}%",
        )

    st.subheader("End-to-End Causal Decision Pipeline")
    tab_arch, tab_leak, tab_math = st.tabs(
        ["Pipeline Architecture", "Why Causal + Label-Delay Matters", "Cost-Aware Decision Formulation"]
    )
    with tab_arch:
        st.markdown(
            """
            1. **Canonical Adapter & Schema Layer**: Normalizes heterogeneous payment logs (`IEEE-CIS`, `ULB Credit Card`, or non-stationary streaming benchmarks) into a validated event schema with explicit missingness indicators (`missing_device_id`, `missing_dist_km`).
            2. **Strict OOT Chronological Splitter + Label Maturity Embargo**: Enforces $t_{\text{train}} < t_{\text{calib}} < t_{\text{test}}$ and holds back events within `label_delay_hours` (72h) of split boundaries so unverified chargebacks never leak into training labels.
            3. **Causal Streaming Velocity & Bipartite Entity Graph**: Maintains sliding deques over `[1h, 24h, 7d]` across `(Card, Device, IP Subnet, Merchant, Email Domain)`. Computes 1-hop degrees, 2-hop card reach, Jaccard neighborhood overlap, closed triads, and Laplace-smoothed matured neighbor fraud rates strictly for $t' < t$.
            4. **Calibrated Scoring & Mondrian Conformal Prediction**: Evaluates GBDTs (`LightGBM`/`XGBoost`/`HistGBDT`) and Focal-Loss `TabularEntityResNet`, then calibrates scores on the held-out calibration slice and constructs distribution-free prediction intervals $[L(x), U(x)]$.
            5. **Bayes Cost-Optimal 3-Way Policy**: Routes transactions into `APPROVE`, `STEPUP_CHALLENGE` (3DS / OTP / analyst review), or `BLOCK` subject to operational review capacity constraints.
            """
        )
    with tab_leak:
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("#### Standard Toy Pipelines (Flawed)")
            st.markdown(
                """
                - Random K-Fold cross-validation mixes future and past transactions of the same card/device.
                - Global `groupby('device_id').nunique()` counts cards from *next week* to score *today's* event.
                - Instantaneous label availability ignores the 30–90 day chargeback reporting lag.
                - Raw uncalibrated classifier scores treated directly as probabilities in dollar expected-value formulas.
                """
            )
        with col_b:
            st.markdown("#### FIN-SHIELD ML Guarantees")
            st.markdown(
                """
                - **Zero Future Lookahead**: Verified by automated unit tests (`test_no_future_leakage`) asserting prefix invariance when future rows are mutated.
                - **Label Maturity Buffer**: Neighbor fraud propagation only aggregates chargebacks whose verification window elapsed prior to $t$.
                - **Drift Resilience**: Tracks PSI and KS-statistics across time slices and validates rolling-window retraining recovery.
                """
            )
    with tab_math:
        st.latex(
            r"""
            egin{aligned}
            \mathcal{L}_{	ext{Approve}}(x) &= \hat{p}(x) \cdot \left(m \cdot A + c_{	ext{cb}}ight) \
            \mathcal{L}_{	ext{StepUp}}(x) &= c_{	ext{review}} + \hat{p}(x)(1 - \eta_{	ext{int}})\left(m \cdot A + c_{	ext{cb}}ight) \
            \mathcal{L}_{	ext{Block}}(x) &= (1 - \hat{p}(x)) \cdot c_{	ext{false\_decline}}
            \end{aligned}
            """
        )
        st.caption(
            "Where A is transaction amount, m is chargeback multiplier, c_cb is fixed chargeback fee, "
            "c_review is step-up friction/analyst cost, eta_int is step-up interception efficiency, and c_false_decline is customer churn penalty."
        )

# ==============================================================================
# PAGE 2: LIVE & BATCH TRANSACTION SCORER
# ==============================================================================
elif page.startswith("2."):
    st.title("Live & Batch Transaction Risk Scorer")
    mode = st.radio("Scoring Mode", ["Single Transaction Simulator", "Batch CSV / Sample Replay"], horizontal=True)

    if mode == "Single Transaction Simulator":
        with st.form("single_txn_form"):
            c1, c2, c3 = st.columns(3)
            with c1:
                txn_id = st.text_input("Transaction ID", value="TXN_DEMO_9901")
                amount = st.number_input("Transaction Amount ($)", min_value=0.5, max_value=50000.0, value=485.0, step=10.0)
                card_id = st.text_input("Card ID", value="CARD_00120")
                device_id = st.text_input("Device ID (leave blank for missing)", value="DEV_00014")
            with c2:
                ip_subnet = st.text_input("IP Subnet", value="10.0.12.0/24")
                merchant_id = st.text_input("Merchant ID", value="MERCH_0008")
                email_domain = st.selectbox(
                    "Email Domain",
                    ["gmail.com", "yahoo.com", "outlook.com", "icloud.com", "proton.me", "tempmail.io"],
                    index=4,
                )
                product_code = st.selectbox("Product Code", ["W", "C", "H", "R", "S"], index=1)
            with c3:
                card_network = st.selectbox("Card Network", ["visa", "mastercard", "amex", "discover"])
                card_type = st.selectbox("Card Type", ["credit", "debit"])
                dist_km = st.number_input("Billing-to-IP Distance (km)", min_value=0.0, max_value=15000.0, value=1850.0)
                is_intl = st.selectbox("International Transaction", [0, 1], index=1)

            submitted = st.form_submit_button("⚡ Score Transaction Event", use_container_width=True)

        if submitted:
            hist_df = service.bundle.get("reference_history_df")
            next_ts = float(hist_df["timestamp"].max() + 15.0) if hist_df is not None and len(hist_df) > 0 else 3650000.0
            ev = TransactionEventInput(
                transaction_id=txn_id,
                timestamp=next_ts,
                amount=amount,
                card_id=card_id,
                device_id=device_id.strip() or None,
                ip_subnet=ip_subnet.strip() or None,
                merchant_id=merchant_id,
                email_domain=email_domain,
                product_code=product_code,
                card_network=card_network,
                card_type=card_type,
                dist_km=dist_km,
                is_international=is_intl,
            )
            res = service.score_single(ev)

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                render_metric_card("Decision Action", res.decision_action, f"Latency: {res.latency_ms:.2f} ms")
            with m2:
                render_metric_card(
                    "Calibrated Fraud Prob",
                    f"{res.calibrated_fraud_probability * 100:.2f}%",
                    f"Raw Score: {res.raw_fraud_score * 100:.2f}%",
                )
            with m3:
                render_metric_card(
                    "95% Conformal Interval",
                    f"[{res.conformal_lower_bound * 100:.1f}%, {res.conformal_upper_bound * 100:.1f}%]",
                    f"Uncertain Set: {res.is_conformal_uncertain}",
                )
            with m4:
                render_metric_card(
                    "OOD Shift Alert",
                    "ALERT" if res.ood_shift_alert else "NORMAL",
                    ", ".join(res.ood_flagged_features) if res.ood_flagged_features else "In-distribution",
                )

            col_l, col_r = st.columns([1.3, 1.0])
            with col_l:
                st.subheader("Top Directional Reason Codes")
                rc_df = pd.DataFrame([r.model_dump() for r in res.reason_codes])
                st.dataframe(rc_df, use_container_width=True, hide_index=True)
            with col_r:
                st.subheader("Expected Dollar Loss by Action")
                loss_df = pd.DataFrame(
                    {
                        "Action": ["APPROVE", "STEPUP_CHALLENGE", "BLOCK"],
                        "Expected Loss ($)": [
                            res.expected_loss_approve,
                            res.expected_loss_stepup,
                            res.expected_loss_block,
                        ],
                    }
                )
                fig = px.bar(
                    loss_df,
                    x="Action",
                    y="Expected Loss ($)",
                    color="Action",
                    color_discrete_map={
                        "APPROVE": "#38BDF8",
                        "STEPUP_CHALLENGE": "#2DD4BF",
                        "BLOCK": "#F87171",
                    },
                )
                fig.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=280)
                st.plotly_chart(fig, use_container_width=True)

    else:
        cfg = load_settings()
        sample_path = Path(cfg.artifacts_dir) / "test_scored_sample.csv"
        uploaded = st.file_uploader("Upload CSV with canonical transaction columns (optional)", type=["csv"])
        if uploaded is not None:
            batch_df = pd.read_csv(uploaded).head(150)
        elif sample_path.exists():
            batch_df = pd.read_csv(sample_path).head(60)
        else:
            batch_df = pd.DataFrame()

        if len(batch_df) > 0 and st.button("Run Batch Scoring on Sample", use_container_width=True):
            events = []
            for _, row in batch_df.iterrows():
                events.append(
                    TransactionEventInput(
                        transaction_id=str(row["transaction_id"]),
                        timestamp=float(row["timestamp"]),
                        amount=float(row["amount"]),
                        card_id=str(row["card_id"]),
                        device_id=None if str(row.get("device_id", "MISSING")) == "MISSING" else str(row["device_id"]),
                        ip_subnet=None if str(row.get("ip_subnet", "MISSING")) == "MISSING" else str(row["ip_subnet"]),
                        merchant_id=str(row["merchant_id"]),
                        email_domain=str(row.get("email_domain", "gmail.com")),
                        product_code=str(row.get("product_code", "W")),
                        card_network=str(row.get("card_network", "visa")),
                        card_type=str(row.get("card_type", "credit")),
                        addr_region=str(row.get("addr_region", "REG_00")),
                        dist_km=float(row.get("dist_km", 10.0)),
                        is_international=int(row.get("is_international", 0)),
                    )
                )
            batch_out = service.score_batch(events)
            st.success(
                f"Scored {batch_out.batch_size} events in {batch_out.total_latency_ms:.2f} ms "
                f"({batch_out.mean_latency_ms:.3f} ms/event) | Actions: {batch_out.action_summary}"
            )
            res_table = pd.DataFrame(
                [
                    {
                        "transaction_id": r.transaction_id,
                        "calibrated_prob": r.calibrated_fraud_probability,
                        "conformal_interval": f"[{r.conformal_lower_bound:.3f}, {r.conformal_upper_bound:.3f}]",
                        "action": r.decision_action,
                        "ood_alert": r.ood_shift_alert,
                        "top_reason": r.reason_codes[0].feature if r.reason_codes else "",
                    }
                    for r in batch_out.results
                ]
            )
            st.dataframe(res_table, use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 3: RISK CALIBRATION & CONFORMAL BOUNDS
# ==============================================================================
elif page.startswith("3."):
    st.title("Probability Calibration & Split-Conformal Uncertainty")
    cal_comp = report["calibration_comparison"]
    rel_curves = report["reliability_curves"]

    cols = st.columns(4)
    for idx, (m_name, m_stats) in enumerate(cal_comp.items()):
        with cols[idx]:
            render_metric_card(
                f"{m_name.upper()} Calibration",
                f"ECE: {m_stats['ece']:.4f}",
                f"Brier: {m_stats['brier_score']:.4f} | Savings: ${m_stats['expected_net_savings']:,.0f}",
            )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Perfect Calibration",
            line=dict(dash="dash", color="#94A3B8"),
        )
    )
    palette = {"none": "#F87171", "platt": "#38BDF8", "temperature": "#FBBF24", "isotonic": "#2DD4BF"}
    for method, bins in rel_curves.items():
        bdf = pd.DataFrame(bins)
        bdf = bdf[bdf["count"] > 0]
        fig.add_trace(
            go.Scatter(
                x=bdf["mean_predicted_prob"],
                y=bdf["observed_fraud_rate"],
                mode="lines+markers",
                name=f"{method.capitalize()} (ECE={cal_comp[method]['ece']:.4f})",
                line=dict(color=palette.get(method, "#2DD4BF"), width=2.5),
            )
        )
    fig.update_layout(
        **PLOTLY_LAYOUT,
        title="Out-of-Time Reliability Diagram (Predicted Fraud Probability vs. Empirical Fraud Rate)",
        xaxis_title="Mean Predicted Probability",
        yaxis_title="Observed Fraud Frequency",
        height=440,
    )
    st.plotly_chart(fig, use_container_width=True)

# ==============================================================================
# PAGE 4: ENTITY NETWORK & RELATIONAL RISK EXPLORER
# ==============================================================================
elif page.startswith("4."):
    st.title("Entity Network & Relational Risk Explorer")
    g_sum = report["entity_graph_summary"]
    c1, c2, c3 = st.columns(3)
    with c1:
        render_metric_card("Subgraph Nodes", str(g_sum["nodes"]), "Cards, Devices, Subnets, Merchants")
    with c2:
        render_metric_card("Interaction Edges", str(g_sum["edges"]), "Causal Multi-Entity Links")
    with c3:
        render_metric_card("Connected Components", str(g_sum["connected_components"]), "Detected Entity Sub-Communities")

    st.subheader("Top High-Density Entity Clusters (Potential Fraud Rings)")
    clusters_df = pd.DataFrame(g_sum["top_clusters"])
    st.dataframe(clusters_df, use_container_width=True, hide_index=True)

    cfg = load_settings()
    sample_path = Path(cfg.artifacts_dir) / "test_scored_sample.csv"
    if sample_path.exists():
        sdf = pd.read_csv(sample_path).tail(140)
        net_data = CausalEntityGraphExtractor.build_subgraph_summary(sdf, max_edges=140)
        G = net_data["graph"]
        import networkx as nx

        pos = nx.spring_layout(G, seed=42, k=0.45)
        edge_x, edge_y = [], []
        for u, v in G.edges():
            x0, y0 = pos[u]
            x1, y1 = pos[v]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        node_x, node_y, node_text, node_color = [], [], [], []
        kind_colors = {"card": "#38BDF8", "device": "#2DD4BF", "subnet": "#FBBF24", "merchant": "#A78BFA"}
        for node, attr in G.nodes(data=True):
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            k = attr.get("kind", "card")
            node_text.append(f"{node} ({k}) | deg={G.degree(node)}")
            node_color.append(kind_colors.get(k, "#F8FAFC"))

        g_fig = go.Figure()
        g_fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=0.8, color="#334155"), hoverinfo="none")
        )
        g_fig.add_trace(
            go.Scatter(
                x=node_x,
                y=node_y,
                mode="markers",
                text=node_text,
                hoverinfo="text",
                marker=dict(size=9, color=node_color, line=dict(width=1, color="#0A1628")),
            )
        )
        g_fig.update_layout(
            **PLOTLY_LAYOUT,
            title="Bipartite Transaction Entity Topology (Blue=Card, Teal=Device, Gold=IP Subnet, Violet=Merchant)",
            showlegend=False,
            height=480,
        )
        g_fig.update_xaxes(showticklabels=False)
        g_fig.update_yaxes(showticklabels=False)
        st.plotly_chart(g_fig, use_container_width=True)

# ==============================================================================
# PAGE 5: CONCEPT DRIFT & COVARIATE SHIFT MONITOR
# ==============================================================================
elif page.startswith("5."):
    st.title("Concept Drift & Covariate Shift Monitor")
    early_d = report["drift_monitoring"]["early_test_window"]
    late_d = report["drift_monitoring"]["late_test_window"]
    adapt_exp = report["drift_adaptation_experiment"]

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_metric_card("Early Test Window", early_d["overall_status"], f"Max PSI: {early_d['max_psi']:.4f}")
    with c2:
        render_metric_card("Late Test Window", late_d["overall_status"], f"Max PSI: {late_d['max_psi']:.4f}")
    with c3:
        render_metric_card(
            "Static PR-AUC (Late Drift)",
            f"{adapt_exp['static_model_on_late_drift']['pr_auc']:.4f}",
            f"Savings: ${adapt_exp['static_model_on_late_drift']['expected_net_savings']:,.0f}",
        )
    with c4:
        render_metric_card(
            "Rolling-Retrained PR-AUC",
            f"{adapt_exp['rolling_retrained_on_late_drift']['pr_auc']:.4f}",
            f"Savings: ${adapt_exp['rolling_retrained_on_late_drift']['expected_net_savings']:,.0f}",
        )

    late_feats = pd.DataFrame(late_d["feature_metrics"]).head(15)
    fig = px.bar(
        late_feats,
        x="psi",
        y="feature",
        orientation="h",
        color="status",
        color_discrete_map={"OK": "#2DD4BF", "WARN": "#FBBF24", "CRITICAL": "#F87171"},
        title="Top Feature Population Stability Index (PSI) in Late Out-of-Time Window",
    )
    fig.update_layout(**PLOTLY_LAYOUT, yaxis=dict(autorange="reversed", gridcolor="#1E3A5F"), height=460)
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(late_feats, use_container_width=True, hide_index=True)

# ==============================================================================
# PAGE 6: COST-AWARE THRESHOLD & POLICY SIMULATOR
# ==============================================================================
elif page.startswith("6."):
    st.title("Cost-Aware Threshold & Action Policy Simulator")
    cfg = load_settings()
    sample_path = Path(cfg.artifacts_dir) / "test_scored_sample.csv"
    sdf = pd.read_csv(sample_path)

    col_ctrl, col_viz = st.columns([1.0, 1.5])
    with col_ctrl:
        st.subheader("Unit Economics & Threshold Controls")
        cb_fee = st.slider("Fixed Chargeback Fee ($)", 0.0, 100.0, float(cfg.policy.fixed_chargeback_fee), 2.5)
        review_cost = st.slider("Step-Up Challenge / Review Cost ($)", 0.5, 25.0, float(cfg.policy.stepup_review_cost), 0.5)
        fd_penalty = st.slider("False-Decline Customer Friction Cost ($)", 1.0, 60.0, float(cfg.policy.false_decline_penalty), 1.0)
        t_stepup = st.slider("Step-Up Challenge Threshold", 0.02, 0.50, float(report["active_policy_thresholds"]["stepup_threshold"]), 0.01)
        t_block = st.slider("Auto-Block Threshold", 0.10, 0.95, float(report["active_policy_thresholds"]["block_threshold"]), 0.01)

    sim_engine = CostAwareDecisionEngine(
        fixed_chargeback_fee=cb_fee,
        stepup_review_cost=review_cost,
        false_decline_penalty=fd_penalty,
    )
    probs = sdf["predicted_fraud_prob"].to_numpy(dtype=float)
    amts = sdf["amount"].to_numpy(dtype=float)
    y_true = sdf["is_fraud"].to_numpy(dtype=int)

    sim_res = sim_engine.evaluate_policy(probs, amts, y_true, t_stepup=t_stepup, t_block=max(t_stepup + 0.02, t_block))

    with col_viz:
        m1, m2, m3 = st.columns(3)
        with m1:
            render_metric_card("Simulated Net Savings", f"${sim_res.expected_net_savings:,.2f}", "vs. No-Model Baseline")
        with m2:
            render_metric_card("Fraud Prevented", f"${sim_res.fraud_prevented_value:,.2f}", f"CB Loss: ${sim_res.chargeback_loss_incurred:,.2f}")
        with m3:
            render_metric_card(
                "Action Mix (Appr / Step / Blk)",
                f"{sim_res.approve_rate*100:.1f}% / {sim_res.stepup_rate*100:.1f}% / {sim_res.block_rate*100:.1f}%",
                f"Review + Friction: ${sim_res.review_cost_incurred + sim_res.false_decline_loss_incurred:,.2f}",
            )

        sweep_ts = np.linspace(0.05, 0.85, 35)
        sweep_savings = [
            sim_engine.evaluate_policy(probs, amts, y_true, t_stepup=min(t_stepup, tb - 0.02), t_block=tb).expected_net_savings
            for tb in sweep_ts
        ]
        s_fig = px.line(
            x=sweep_ts,
            y=sweep_savings,
            labels={"x": "Auto-Block Threshold", "y": "Net Financial Savings ($)"},
            title="Net Financial Savings Curve Across Auto-Block Thresholds",
        )
        s_fig.update_traces(line=dict(color="#2DD4BF", width=3))
        s_fig.update_layout(**PLOTLY_LAYOUT, height=340)
        st.plotly_chart(s_fig, use_container_width=True)

# ==============================================================================
# PAGE 7: BENCHMARK & ABLATION LAB
# ==============================================================================
else:
    st.title("Model Benchmarks & Causal Ablation Lab")
    rows = []
    for model_key, stats in report["model_benchmarks"].items():
        ci = stats.get("bootstrap_95_ci", {})
        pr_ci = (
            f"[{ci['pr_auc']['ci_lower_95']:.4f}, {ci['pr_auc']['ci_upper_95']:.4f}]"
            if "pr_auc" in ci
            else "—"
        )
        rows.append(
            {
                "Model / Experiment": model_key,
                "PR-AUC": stats["pr_auc"],
                "PR-AUC 95% CI": pr_ci,
                "ROC-AUC": stats["roc_auc"],
                "Recall@0.5%FPR": stats["recall_at_fpr_0_5pct"],
                "Recall@1.0%FPR": stats["recall_at_fpr_1_0pct"],
                "ECE": stats["ece"],
                "Brier": stats["brier_score"],
                "Net Savings ($)": stats["expected_net_savings"],
            }
        )
    bench_df = pd.DataFrame(rows)
    st.dataframe(bench_df, use_container_width=True, hide_index=True)

    fig = px.bar(
        bench_df,
        x="Model / Experiment",
        y="PR-AUC",
        color="PR-AUC",
        color_continuous_scale=["#1E3A5F", "#14B8A6", "#2DD4BF"],
        title="Out-of-Time PR-AUC Comparison Across Paradigms & Leakage Ablation",
    )
    fig.update_layout(**PLOTLY_LAYOUT, height=400)
    st.plotly_chart(fig, use_container_width=True)
