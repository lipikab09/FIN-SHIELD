from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from finshield.features.causal_features import CausalFeatureExtractor
from finshield.features.graph_features import CausalEntityGraphExtractor

CATEGORICAL_COLS = ["product_code", "card_network", "card_type", "addr_region", "email_domain"]


class FraudFeaturePipeline:
    """
    End-to-end feature assembly pipeline combining:
    1. Causal temporal velocity & entity reuse features
    2. Causal bipartite entity graph & neighborhood risk aggregations
    3. Frequency-encoded high-cardinality entity representations (fit strictly on train split)
    4. Ordinal-encoded categorical features for neural entity embeddings & GBDTs
    """

    def __init__(
        self,
        windows_seconds: List[int] | None = None,
        graph_lookback_seconds: int = 604800,
        label_delay_hours: int = 72,
        use_graph_features: bool = True,
        leaky_mode: bool = False,
    ) -> None:
        self.causal_extractor = CausalFeatureExtractor(windows_seconds=windows_seconds)
        self.graph_extractor = CausalEntityGraphExtractor(
            lookback_seconds=graph_lookback_seconds,
            label_delay_seconds=int(label_delay_hours * 3600),
        )
        self.use_graph_features = use_graph_features
        self.leaky_mode = leaky_mode
        self.encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        self.scaler = StandardScaler()
        self.freq_maps: dict[str, dict[str, float]] = {}
        self.feature_names_: List[str] = []
        self.cat_cardinalities_: List[int] = []
        self._fitted = False

    def _compute_raw_feature_frame(self, df: pd.DataFrame, include_labels: bool = True) -> pd.DataFrame:
        ordered = df.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
        causal_df = self.causal_extractor.transform(ordered, leaky_mode=self.leaky_mode)
        if self.use_graph_features:
            graph_df = self.graph_extractor.transform(ordered, include_labels=include_labels)
            combined = pd.concat([causal_df, graph_df], axis=1)
        else:
            combined = causal_df
        return combined

    def fit_transform_splits(
        self,
        train_df: pd.DataFrame,
        calib_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Extracts causal features across the continuous stream so historical windows carry across
        boundaries without future lookahead, while fitting scalers/encoders strictly on `train_df`.
        """
        full_df = pd.concat(
            [
                train_df.assign(_split="train"),
                calib_df.assign(_split="calib"),
                test_df.assign(_split="test"),
            ],
            axis=0,
        ).sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)

        # Mask test labels during causal feature extraction so test targets never leak
        masked_df = full_df.copy()
        masked_df.loc[masked_df["_split"] == "test", "is_fraud"] = 0

        raw_feats = self._compute_raw_feature_frame(masked_df, include_labels=True)

        train_mask = (full_df["_split"] == "train").to_numpy()
        calib_mask = (full_df["_split"] == "calib").to_numpy()
        test_mask = (full_df["_split"] == "test").to_numpy()

        # Fit frequency encodings strictly on train split
        self.freq_maps = {}
        n_train = max(1, int(train_mask.sum()))
        for col in ["card_id", "device_id", "ip_subnet", "merchant_id", "email_domain"]:
            vc = full_df.loc[train_mask, col].astype(str).value_counts()
            self.freq_maps[col] = (vc / n_train).to_dict()

        for col, fmap in self.freq_maps.items():
            raw_feats[f"train_freq_{col}"] = full_df[col].astype(str).map(fmap).fillna(0.0).astype(np.float32)

        # Fit categorical ordinal encoder strictly on train split
        train_cats = full_df.loc[train_mask, CATEGORICAL_COLS].astype(str)
        self.encoder.fit(train_cats)
        all_cats_encoded = self.encoder.transform(full_df[CATEGORICAL_COLS].astype(str))
        # Shift by +1 so unknown (-1) becomes index 0 for nn.Embedding
        all_cats_shifted = (all_cats_encoded + 1).astype(np.int64)
        self.cat_cardinalities_ = [int(len(cats) + 2) for cats in self.encoder.categories_]

        for idx, col in enumerate(CATEGORICAL_COLS):
            raw_feats[f"cat_{col}"] = all_cats_shifted[:, idx]

        self.feature_names_ = list(raw_feats.columns)
        num_cols = [c for c in self.feature_names_ if not c.startswith("cat_")]
        self.scaler.fit(raw_feats.loc[train_mask, num_cols])
        self._fitted = True

        X_train = raw_feats.loc[train_mask].reset_index(drop=True)
        X_calib = raw_feats.loc[calib_mask].reset_index(drop=True)
        X_test = raw_feats.loc[test_mask].reset_index(drop=True)
        return X_train, X_calib, X_test

    def transform_single_or_batch(self, df: pd.DataFrame, history_df: pd.DataFrame | None = None) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("FraudFeaturePipeline must be fitted before inference.")
        incoming = df.copy().reset_index(drop=True)
        incoming["_is_query"] = 1

        if history_df is not None and len(history_df) > 0:
            hist = history_df.copy()
            hist["_is_query"] = 0
            combined = pd.concat([hist, incoming], axis=0).sort_values(
                ["timestamp", "transaction_id"], kind="mergesort"
            ).reset_index(drop=True)
        else:
            combined = incoming

        raw_feats = self._compute_raw_feature_frame(combined, include_labels=False)
        for col, fmap in self.freq_maps.items():
            raw_feats[f"train_freq_{col}"] = combined[col].astype(str).map(fmap).fillna(0.0).astype(np.float32)

        cats_encoded = self.encoder.transform(combined[CATEGORICAL_COLS].astype(str))
        cats_shifted = (cats_encoded + 1).astype(np.int64)
        for idx, col in enumerate(CATEGORICAL_COLS):
            raw_feats[f"cat_{col}"] = cats_shifted[:, idx]

        # Ensure exact column alignment
        for col in self.feature_names_:
            if col not in raw_feats.columns:
                raw_feats[col] = 0.0
        raw_feats = raw_feats[self.feature_names_]

        query_mask = (combined["_is_query"] == 1).to_numpy()
        return raw_feats.loc[query_mask].reset_index(drop=True)
