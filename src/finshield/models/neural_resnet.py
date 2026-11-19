from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


class FocalLoss(nn.Module):
    """Binary Focal Loss for severe class imbalance in fraud detection."""

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1.0 - targets) * (1.0 - probs)
        alpha_t = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)
        loss = alpha_t * ((1.0 - pt) ** self.gamma) * bce
        return loss.mean()


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.block(x))


class TabularResNetModule(nn.Module):
    """Tabular ResNet with categorical entity embeddings and skip-connected residual blocks."""

    def __init__(self, num_features: int, cat_cardinalities: List[int], hidden_dim: int = 96, dropout: float = 0.25) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(num_embeddings=card, embedding_dim=min(12, max(4, (card + 1) // 2))) for card in cat_cardinalities]
        )
        emb_total_dim = sum(e.embedding_dim for e in self.embeddings)
        in_dim = num_features + emb_total_dim
        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )
        self.res1 = ResidualBlock(hidden_dim, dropout=dropout)
        self.res2 = ResidualBlock(hidden_dim, dropout=dropout)
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x_num: torch.Tensor, x_cat: torch.Tensor) -> torch.Tensor:
        if self.embeddings:
            embs = [emb(x_cat[:, idx]) for idx, emb in enumerate(self.embeddings)]
            x = torch.cat([x_num] + embs, dim=-1)
        else:
            x = x_num
        h = self.input_proj(x)
        h = self.res1(h)
        h = self.res2(h)
        return self.head(h).squeeze(-1)


class TabularEntityResNet:
    """Scikit-learn style wrapper around TabularResNetModule trained with Focal Loss."""

    def __init__(
        self,
        cat_cardinalities: List[int],
        hidden_dim: int = 96,
        dropout: float = 0.25,
        epochs: int = 12,
        batch_size: int = 256,
        lr: float = 0.0015,
        focal_gamma: float = 2.0,
        random_state: int = 42,
    ) -> None:
        self.cat_cardinalities = cat_cardinalities
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.focal_gamma = focal_gamma
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.num_cols_: List[str] = []
        self.cat_cols_: List[str] = []
        self.net: TabularResNetModule | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "TabularEntityResNet":
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self.cat_cols_ = [c for c in X.columns if c.startswith("cat_")]
        self.num_cols_ = [c for c in X.columns if not c.startswith("cat_")]

        X_num = self.scaler.fit_transform(X[self.num_cols_].to_numpy(dtype=np.float32))
        X_cat = np.clip(X[self.cat_cols_].to_numpy(dtype=np.int64), 0, None)
        for idx, card in enumerate(self.cat_cardinalities):
            X_cat[:, idx] = np.clip(X_cat[:, idx], 0, card - 1)

        self.net = TabularResNetModule(
            num_features=len(self.num_cols_),
            cat_cardinalities=self.cat_cardinalities,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        )

        dataset = TensorDataset(
            torch.tensor(X_num, dtype=torch.float32),
            torch.tensor(X_cat, dtype=torch.long),
            torch.tensor(y.astype(np.float32), dtype=torch.float32),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        criterion = FocalLoss(alpha=0.78, gamma=self.focal_gamma)
        optimizer = torch.optim.AdamW(self.net.parameters(), lr=self.lr, weight_decay=1e-4)

        self.net.train()
        for _ in range(self.epochs):
            for bx_num, bx_cat, by in loader:
                optimizer.zero_grad()
                logits = self.net(bx_num, bx_cat)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()
        return self

    def predict_logits(self, X: pd.DataFrame) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("TabularEntityResNet has not been fitted.")
        self.net.eval()
        X_num = self.scaler.transform(X[self.num_cols_].to_numpy(dtype=np.float32))
        X_cat = np.clip(X[self.cat_cols_].to_numpy(dtype=np.int64), 0, None)
        for idx, card in enumerate(self.cat_cardinalities):
            X_cat[:, idx] = np.clip(X_cat[:, idx], 0, card - 1)

        with torch.no_grad():
            logits = self.net(
                torch.tensor(X_num, dtype=torch.float32),
                torch.tensor(X_cat, dtype=torch.long),
            ).numpy()
        return logits

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        logits = self.predict_logits(X)
        return 1.0 / (1.0 + np.exp(-np.clip(logits, -30.0, 30.0)))
