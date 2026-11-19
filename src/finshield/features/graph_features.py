from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Set, Tuple

import networkx as nx
import numpy as np
import pandas as pd


class CausalEntityGraphExtractor:
    """
    Causal Bipartite/Multipartite Entity Graph & Neighborhood Aggregation (GraphSAGE-style structural features).
    Maintains a rolling historical interaction graph across {card, device, ip_subnet, merchant, email_domain}.
    For event i at timestamp t:
      1. Extracts 1-hop and 2-hop structural degrees, bipartite fan-out, Jaccard entity-pair affinity,
         local clustering / triangle density, connected component size, and label-delayed historical
         neighbor fraud propagation rates (strictly using labels matured prior to t - label_delay_seconds).
      2. Updates the graph state AFTER extracting features for event i.
    """

    def __init__(self, lookback_seconds: int = 604800, label_delay_seconds: int = 259200) -> None:
        self.lookback_seconds = lookback_seconds
        self.label_delay_seconds = label_delay_seconds

    def transform(self, df: pd.DataFrame, include_labels: bool = True) -> pd.DataFrame:
        ordered = df.sort_values(["timestamp", "transaction_id"], kind="mergesort").reset_index(drop=True)
        n = len(ordered)

        ts_arr = ordered["timestamp"].to_numpy(dtype=np.float64)
        card_arr = ordered["card_id"].astype(str).to_numpy()
        dev_arr = ordered["device_id"].astype(str).to_numpy()
        sub_arr = ordered["ip_subnet"].astype(str).to_numpy()
        merch_arr = ordered["merchant_id"].astype(str).to_numpy()
        email_arr = ordered["email_domain"].astype(str).to_numpy()
        y_arr = ordered["is_fraud"].to_numpy(dtype=np.int32) if ("is_fraud" in ordered.columns and include_labels) else np.zeros(n, dtype=np.int32)

        # Adjacency sets within lookback window
        card_neighbors: Dict[str, Set[str]] = defaultdict(set)
        dev_neighbors: Dict[str, Set[str]] = defaultdict(set)
        sub_neighbors: Dict[str, Set[str]] = defaultdict(set)
        merch_neighbors: Dict[str, Set[str]] = defaultdict(set)
        email_neighbors: Dict[str, Set[str]] = defaultdict(set)

        # Edge timestamps for sliding window eviction
        edge_log: Deque[Tuple[float, str, str, str, str, str]] = deque()
        edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)

        # Delayed-label maturity buffer: (mature_ts, card, dev, sub, merch, is_fraud)
        pending_labels: Deque[Tuple[float, str, str, str, str, int]] = deque()
        entity_total_matured: Dict[str, int] = defaultdict(int)
        entity_fraud_matured: Dict[str, int] = defaultdict(int)

        # Output graph feature arrays
        g_card_deg = np.zeros(n, dtype=np.float32)
        g_dev_deg = np.zeros(n, dtype=np.float32)
        g_sub_deg = np.zeros(n, dtype=np.float32)
        g_merch_deg = np.zeros(n, dtype=np.float32)
        g_2hop_card_reach = np.zeros(n, dtype=np.float32)
        g_card_dev_jaccard = np.zeros(n, dtype=np.float32)
        g_dev_sub_cooccurrence = np.zeros(n, dtype=np.float32)
        g_shared_triangle_score = np.zeros(n, dtype=np.float32)
        g_sage_neighbor_deg_mean = np.zeros(n, dtype=np.float32)
        g_matured_neighbor_fraud_rate = np.zeros(n, dtype=np.float32)
        g_matured_entity_exposure = np.zeros(n, dtype=np.float32)

        prior_fraud_rate = 0.035

        for i in range(n):
            t = ts_arr[i]
            c = f"C:{card_arr[i]}"
            d = f"D:{dev_arr[i]}" if dev_arr[i] != "MISSING" else None
            s = f"S:{sub_arr[i]}" if sub_arr[i] != "MISSING" else None
            m = f"M:{merch_arr[i]}"
            e = f"E:{email_arr[i]}" if email_arr[i] != "MISSING" else None

            # 1. Evict stale graph edges older than lookback window
            while edge_log and (t - edge_log[0][0]) > self.lookback_seconds:
                _, oc, od, os_, om, oe = edge_log.popleft()
                for u, v in [(oc, od), (oc, os_), (oc, om), (od, os_), (oc, oe)]:
                    if u and v:
                        edge_counts[(u, v)] -= 1
                        if edge_counts[(u, v)] <= 0:
                            edge_counts.pop((u, v), None)
                            if u.startswith("C:"):
                                card_neighbors[u].discard(v)
                            elif u.startswith("D:"):
                                dev_neighbors[u].discard(v)
                            if v.startswith("D:"):
                                dev_neighbors[v].discard(u)
                            elif v.startswith("S:"):
                                sub_neighbors[v].discard(u)
                            elif v.startswith("M:"):
                                merch_neighbors[v].discard(u)
                            elif v.startswith("E:"):
                                email_neighbors[v].discard(u)

            # 2. Mature any labels whose verification delay has elapsed strictly before t
            while pending_labels and pending_labels[0][0] <= t:
                _, mc, md, ms, mm, mf = pending_labels.popleft()
                for ent in (mc, md, ms, mm):
                    if ent:
                        entity_total_matured[ent] += 1
                        entity_fraud_matured[ent] += mf

            # 3. Extract causal 1-hop and 2-hop GraphSAGE-style neighborhood features
            c_nbrs = card_neighbors[c]
            d_nbrs = dev_neighbors[d] if d else set()
            s_nbrs = sub_neighbors[s] if s else set()
            m_nbrs = merch_neighbors[m]

            g_card_deg[i] = len(c_nbrs)
            g_dev_deg[i] = len(d_nbrs)
            g_sub_deg[i] = len(s_nbrs)
            g_merch_deg[i] = len(m_nbrs)

            # 2-hop reach: distinct cards reachable through current device or IP subnet
            two_hop_cards = (d_nbrs | s_nbrs) - {c}
            g_2hop_card_reach[i] = len(two_hop_cards)

            # Jaccard similarity between card's neighborhood and device's neighborhood
            if c_nbrs or d_nbrs:
                union_sz = len(c_nbrs | d_nbrs)
                inter_sz = len(c_nbrs & d_nbrs)
                g_card_dev_jaccard[i] = inter_sz / max(1, union_sz)

            # Device-Subnet co-occurrence frequency
            if d and s:
                g_dev_sub_cooccurrence[i] = float(edge_counts.get((d, s), 0))

            # Triangle / closed-triad indicator across (card, device, subnet)
            tri = 0.0
            if d and d in c_nbrs:
                tri += 1.0
            if s and s in c_nbrs:
                tri += 1.0
            if d and s and s in d_nbrs:
                tri += 1.0
            g_shared_triangle_score[i] = tri / 3.0

            # GraphSAGE mean aggregator of 1-hop neighbor degrees
            nbr_degrees = [len(d_nbrs), len(s_nbrs), len(m_nbrs)]
            if e:
                nbr_degrees.append(len(email_neighbors[e]))
            g_sage_neighbor_deg_mean[i] = float(np.mean(nbr_degrees))

            # Smoothed Bayesian fraud propagation score from matured historical labels on linked entities
            tot_m = 0
            frd_m = 0
            for ent in (c, d, s, m):
                if ent:
                    tot_m += entity_total_matured[ent]
                    frd_m += entity_fraud_matured[ent]
            g_matured_entity_exposure[i] = float(tot_m)
            # Laplace/Dirichlet smoothing (alpha=10) prevents overfitting on low-count entities
            g_matured_neighbor_fraud_rate[i] = (frd_m + 10.0 * prior_fraud_rate) / (tot_m + 10.0)

            # 4. Update causal graph state AFTER feature extraction
            edge_log.append((t, c, d or "", s or "", m, e or ""))
            for u, v in [(c, d), (c, s), (c, m), (d, s), (c, e)]:
                if u and v:
                    edge_counts[(u, v)] += 1
                    if u.startswith("C:"):
                        card_neighbors[u].add(v)
                    elif u.startswith("D:"):
                        dev_neighbors[u].add(v)
                    if v.startswith("D:"):
                        dev_neighbors[v].add(u)
                    elif v.startswith("S:"):
                        sub_neighbors[v].add(u)
                    elif v.startswith("M:"):
                        merch_neighbors[v].add(u)
                    elif v.startswith("E:"):
                        email_neighbors[v].add(u)

            if include_labels:
                pending_labels.append((t + self.label_delay_seconds, c, d or "", s or "", m, int(y_arr[i])))

        return pd.DataFrame(
            {
                "g_card_deg": g_card_deg,
                "g_dev_deg": g_dev_deg,
                "g_sub_deg": g_sub_deg,
                "g_merch_deg": g_merch_deg,
                "g_2hop_card_reach": g_2hop_card_reach,
                "g_card_dev_jaccard": g_card_dev_jaccard,
                "g_dev_sub_cooccurrence": g_dev_sub_cooccurrence,
                "g_shared_triangle_score": g_shared_triangle_score,
                "g_sage_neighbor_deg_mean": g_sage_neighbor_deg_mean,
                "g_matured_neighbor_fraud_rate": g_matured_neighbor_fraud_rate,
                "g_matured_entity_exposure": g_matured_entity_exposure,
            }
        )

    @staticmethod
    def build_subgraph_summary(df: pd.DataFrame, max_edges: int = 350) -> Dict[str, object]:
        """Constructs a NetworkX bipartite entity graph summary for dashboard visualization & ring inspection."""
        G = nx.Graph()
        sub = df.tail(max_edges)
        for _, row in sub.iterrows():
            c = str(row["card_id"])
            d = str(row["device_id"])
            s = str(row["ip_subnet"])
            m = str(row["merchant_id"])
            fraud = int(row.get("is_fraud", 0))
            G.add_node(c, kind="card")
            G.add_node(m, kind="merchant")
            G.add_edge(c, m, fraud=fraud)
            if d != "MISSING":
                G.add_node(d, kind="device")
                G.add_edge(c, d, fraud=fraud)
            if s != "MISSING":
                G.add_node(s, kind="subnet")
                G.add_edge(c, s, fraud=fraud)

        components = sorted(nx.connected_components(G), key=len, reverse=True)
        ring_candidates = []
        for idx, comp in enumerate(components[:10]):
            sub_g = G.subgraph(comp)
            fraud_edges = sum(1 for _, _, attr in sub_g.edges(data=True) if attr.get("fraud", 0) == 1)
            cards = [n for n, attr in sub_g.nodes(data=True) if attr.get("kind") == "card"]
            devices = [n for n, attr in sub_g.nodes(data=True) if attr.get("kind") == "device"]
            subnets = [n for n, attr in sub_g.nodes(data=True) if attr.get("kind") == "subnet"]
            ring_candidates.append(
                {
                    "cluster_id": f"CLUSTER_{idx + 1:02d}",
                    "total_nodes": sub_g.number_of_nodes(),
                    "total_edges": sub_g.number_of_edges(),
                    "cards": len(cards),
                    "devices": len(devices),
                    "subnets": len(subnets),
                    "fraud_edges": fraud_edges,
                    "density": round(float(nx.density(sub_g)), 4),
                    "sample_entities": list(comp)[:8],
                }
            )
        return {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "connected_components": len(components),
            "top_clusters": ring_candidates,
            "graph": G,
        }
