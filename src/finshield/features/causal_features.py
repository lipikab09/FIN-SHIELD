from __future__ import annotations

from collections import deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd


class CausalFeatureExtractor:
    """
    Strictly causal streaming time-window feature extractor.
    For any event at timestamp t, ONLY events with timestamp < t (strictly prior in sort order)
    are visible. Zero future information can enter any aggregation.
    """

    def __init__(self, windows_seconds: List[int] | None = None) -> None:
        self.windows = sorted(windows_seconds or [3600, 86400, 604800])
        self.max_window = max(self.windows)

    def transform(self, df: pd.DataFrame, leaky_mode: bool = False) -> pd.DataFrame:
        """
        Compute causal rolling velocity, spend statistics, and cross-entity reuse counts.
        If `leaky_mode=True` (used ONLY for controlled leakage ablation experiments),
        includes full-series future entity statistics to quantify data-leakage inflation.
        """
        ordered = df.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
        n = len(ordered)

        ts_arr = ordered["timestamp"].to_numpy(dtype=np.float64)
        amt_arr = ordered["amount"].to_numpy(dtype=np.float64)
        card_arr = ordered["card_id"].astype(str).to_numpy()
        dev_arr = ordered["device_id"].astype(str).to_numpy()
        sub_arr = ordered["ip_subnet"].astype(str).to_numpy()
        merch_arr = ordered["merchant_id"].astype(str).to_numpy()

        # Preallocate output arrays
        card_cnt_1h = np.zeros(n, dtype=np.float32)
        card_cnt_24h = np.zeros(n, dtype=np.float32)
        card_cnt_7d = np.zeros(n, dtype=np.float32)
        card_amt_sum_24h = np.zeros(n, dtype=np.float32)
        card_amt_mean_7d = np.zeros(n, dtype=np.float32)
        card_amt_ratio_7d = np.ones(n, dtype=np.float32)
        card_time_since_last = np.full(n, 604800.0, dtype=np.float32)

        dev_cnt_1h = np.zeros(n, dtype=np.float32)
        dev_cnt_24h = np.zeros(n, dtype=np.float32)
        dev_unique_cards_7d = np.zeros(n, dtype=np.float32)

        sub_cnt_1h = np.zeros(n, dtype=np.float32)
        sub_cnt_24h = np.zeros(n, dtype=np.float32)
        sub_unique_cards_7d = np.zeros(n, dtype=np.float32)

        merch_cnt_1h = np.zeros(n, dtype=np.float32)
        merch_amt_ratio_7d = np.ones(n, dtype=np.float32)

        # Sliding event histories per entity: deque of (timestamp, amount, auxiliary_entity)
        card_hist: Dict[str, Deque[Tuple[float, float]]] = {}
        dev_hist: Dict[str, Deque[Tuple[float, str]]] = {}
        sub_hist: Dict[str, Deque[Tuple[float, str]]] = {}
        merch_hist: Dict[str, Deque[Tuple[float, float]]] = {}

        w1, w24, w7 = self.windows[0], self.windows[1], self.windows[2]

        for i in range(n):
            t = ts_arr[i]
            amt = amt_arr[i]
            c_id = card_arr[i]
            d_id = dev_arr[i]
            s_id = sub_arr[i]
            m_id = merch_arr[i]

            # 1. Card history (strictly t_prev < t)
            dq_c = card_hist.get(c_id)
            if dq_c:
                while dq_c and (t - dq_c[0][0]) > w7:
                    dq_c.popleft()
                if dq_c:
                    card_time_since_last[i] = float(max(0.0, t - dq_c[-1][0]))
                    c1 = c24 = c7 = 0
                    s24 = s7 = 0.0
                    for prev_t, prev_amt in dq_c:
                        dt = t - prev_t
                        c7 += 1
                        s7 += prev_amt
                        if dt <= w24:
                            c24 += 1
                            s24 += prev_amt
                            if dt <= w1:
                                c1 += 1
                    card_cnt_1h[i] = c1
                    card_cnt_24h[i] = c24
                    card_cnt_7d[i] = c7
                    card_amt_sum_24h[i] = s24
                    mean7 = s7 / max(1, c7)
                    card_amt_mean_7d[i] = mean7
                    card_amt_ratio_7d[i] = amt / max(1.0, mean7)

            # 2. Device history (skip MISSING bucket for shared fan-out)
            if d_id != "MISSING":
                dq_d = dev_hist.get(d_id)
                if dq_d:
                    while dq_d and (t - dq_d[0][0]) > w7:
                        dq_d.popleft()
                    if dq_d:
                        d1 = d24 = 0
                        uniq_cards = set()
                        for prev_t, prev_c in dq_d:
                            dt = t - prev_t
                            uniq_cards.add(prev_c)
                            if dt <= w24:
                                d24 += 1
                                if dt <= w1:
                                    d1 += 1
                        dev_cnt_1h[i] = d1
                        dev_cnt_24h[i] = d24
                        dev_unique_cards_7d[i] = len(uniq_cards)

            # 3. IP Subnet history
            if s_id != "MISSING":
                dq_s = sub_hist.get(s_id)
                if dq_s:
                    while dq_s and (t - dq_s[0][0]) > w7:
                        dq_s.popleft()
                    if dq_s:
                        s1 = s24 = 0
                        uniq_cards = set()
                        for prev_t, prev_c in dq_s:
                            dt = t - prev_t
                            uniq_cards.add(prev_c)
                            if dt <= w24:
                                s24 += 1
                                if dt <= w1:
                                    s1 += 1
                        sub_cnt_1h[i] = s1
                        sub_cnt_24h[i] = s24
                        sub_unique_cards_7d[i] = len(uniq_cards)

            # 4. Merchant history
            dq_m = merch_hist.get(m_id)
            if dq_m:
                while dq_m and (t - dq_m[0][0]) > w7:
                    dq_m.popleft()
                if dq_m:
                    m1 = m7 = 0
                    ms7 = 0.0
                    for prev_t, prev_amt in dq_m:
                        dt = t - prev_t
                        m7 += 1
                        ms7 += prev_amt
                        if dt <= w1:
                            m1 += 1
                    merch_cnt_1h[i] = m1
                    merch_amt_ratio_7d[i] = amt / max(1.0, ms7 / max(1, m7))

            # Update causal state AFTER extracting features for event i
            card_hist.setdefault(c_id, deque()).append((t, amt))
            if d_id != "MISSING":
                dev_hist.setdefault(d_id, deque()).append((t, c_id))
            if s_id != "MISSING":
                sub_hist.setdefault(s_id, deque()).append((t, c_id))
            merch_hist.setdefault(m_id, deque()).append((t, amt))

        hours = (ts_arr / 3600.0) % 24.0
        feats = pd.DataFrame(
            {
                "log_amount": np.log1p(amt_arr).astype(np.float32),
                "amount_cents": (amt_arr - np.floor(amt_arr)).astype(np.float32),
                "is_round_amount": (np.isclose(amt_arr % 10.0, 0.0)).astype(np.float32),
                "hour_sin": np.sin(2.0 * np.pi * hours / 24.0).astype(np.float32),
                "hour_cos": np.cos(2.0 * np.pi * hours / 24.0).astype(np.float32),
                "is_night_hours": ((hours >= 1.0) & (hours <= 5.0)).astype(np.float32),
                "log_dist_km": np.log1p(np.maximum(0.0, ordered["dist_km"].to_numpy(dtype=np.float32))),
                "is_international": ordered["is_international"].to_numpy(dtype=np.float32),
                "card_cnt_1h": card_cnt_1h,
                "card_cnt_24h": card_cnt_24h,
                "card_cnt_7d": card_cnt_7d,
                "card_amt_sum_24h": card_amt_sum_24h,
                "card_amt_mean_7d": card_amt_mean_7d,
                "card_amt_ratio_7d": np.clip(card_amt_ratio_7d, 0.0, 50.0),
                "log_card_time_since_last": np.log1p(card_time_since_last),
                "dev_cnt_1h": dev_cnt_1h,
                "dev_cnt_24h": dev_cnt_24h,
                "dev_unique_cards_7d": dev_unique_cards_7d,
                "sub_cnt_1h": sub_cnt_1h,
                "sub_cnt_24h": sub_cnt_24h,
                "sub_unique_cards_7d": sub_unique_cards_7d,
                "merch_cnt_1h": merch_cnt_1h,
                "merch_amt_ratio_7d": np.clip(merch_amt_ratio_7d, 0.0, 50.0),
            }
        )

        # Include missing indicators if present
        for col in ordered.columns:
            if col.startswith("missing_"):
                feats[col] = ordered[col].to_numpy(dtype=np.float32)

        if leaky_mode:
            # Intentional future leakage baseline for ablation comparison ONLY
            feats["leaky_dev_total_cards"] = ordered.groupby("device_id")["card_id"].transform("nunique").astype(np.float32)
            feats["leaky_sub_total_cards"] = ordered.groupby("ip_subnet")["card_id"].transform("nunique").astype(np.float32)
            feats["leaky_card_future_mean_amt"] = ordered.groupby("card_id")["amount"].transform("mean").astype(np.float32)

        return feats
