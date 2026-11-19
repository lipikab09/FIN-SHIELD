from __future__ import annotations

import pandas as pd

from finshield.data.adapters import SyntheticStreamingFraudAdapter
from finshield.data.splits import ChronologicalSplitter
from finshield.features.causal_features import CausalFeatureExtractor
from finshield.features.graph_features import CausalEntityGraphExtractor


def test_strict_chronological_split_and_embargo():
    df, meta = SyntheticStreamingFraudAdapter(n_events=2000, seed=7).load()
    assert meta["n_rows"] == 2000
    splitter = ChronologicalSplitter(train_ratio=0.6, calib_ratio=0.15, test_ratio=0.25, label_delay_hours=24)
    res = splitter.split(df)

    assert res.train["timestamp"].max() < res.calib["timestamp"].min()
    assert res.calib["timestamp"].max() < res.test["timestamp"].min()
    assert (res.calib["timestamp"].min() - res.train["timestamp"].max()) >= 24 * 3600


def test_causal_features_prefix_invariance_no_future_leakage():
    df, _ = SyntheticStreamingFraudAdapter(n_events=600, seed=11).load()
    cutoff = 300

    extractor = CausalFeatureExtractor(windows_seconds=[3600, 86400, 604800])
    feats_full = extractor.transform(df, leaky_mode=False).iloc[:cutoff]

    df_mutated = df.copy()
    df_mutated.loc[cutoff:, "amount"] = 9999.99
    df_mutated.loc[cutoff:, "device_id"] = "DEV_00001"
    df_mutated.loc[cutoff:, "ip_subnet"] = "10.0.1.0/24"
    feats_mutated = extractor.transform(df_mutated, leaky_mode=False).iloc[:cutoff]

    pd.testing.assert_frame_equal(feats_full, feats_mutated, check_exact=False, atol=1e-6)


def test_graph_features_prefix_invariance_and_label_delay():
    df, _ = SyntheticStreamingFraudAdapter(n_events=600, seed=19).load()
    cutoff = 300

    g_ext = CausalEntityGraphExtractor(lookback_seconds=604800, label_delay_seconds=86400)
    g_full = g_ext.transform(df, include_labels=True).iloc[:cutoff]

    df_mutated = df.copy()
    df_mutated.loc[cutoff:, "is_fraud"] = 1
    df_mutated.loc[cutoff:, "device_id"] = "DEV_00001"
    g_mutated = g_ext.transform(df_mutated, include_labels=True).iloc[:cutoff]

    pd.testing.assert_frame_equal(g_full, g_mutated, check_exact=False, atol=1e-6)
