import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Tuple
from sklearn.preprocessing import StandardScaler
from src.common.utils import calculate_metrics

class WindowedCNN1DRegressor(nn.Module):
    """
    1D-CNN Sequence Architecture for Turbofan RUL Prediction.
    Expects scaled windowed inputs of shape (batch, seq_len, features) or (batch, features).
    """
    def __init__(self, num_features: int = 14, input_dim: int = None, window_size: int = 30):
        super().__init__()
        feats = input_dim if input_dim is not None else num_features
        self.conv1 = nn.Conv1d(feats, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # x: (batch, seq_len, features) -> transpose to (batch, features, seq_len)
        x = x.transpose(1, 2)
        out = self.relu(self.bn1(self.conv1(x)))
        if out.shape[-1] >= 2:
            out = self.pool1(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.global_pool(out).squeeze(-1)
        return self.fc(out).squeeze(-1)


class WindowedGRURegressor(nn.Module):
    """
    Sequence GRU Architecture for Turbofan RUL Prediction.
    Expects scaled 30-cycle windowed inputs of shape (batch, seq_len=30, features) or (batch, features).
    """
    def __init__(self, num_features: int = 14, input_dim: int = None, hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        feats = input_dim if input_dim is not None else num_features
        self.gru = nn.GRU(
            input_size=feats,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.1 if num_layers > 1 else 0.0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        # x: (batch, seq_len, features)
        out, _ = self.gru(x)
        # Take representation at the last time step
        last_step = out[:, -1, :]
        return self.fc(last_step).squeeze(-1)


# Backwards compatibility aliases
CNN1DRegressor = WindowedCNN1DRegressor
GRURegressor = WindowedGRURegressor


class NeuralRULBenchmark:
    """
    Trains and benchmarks real windowed 1D-CNN and sequence GRU models
    using scaled inputs and 30-cycle sequence windows.
    """
    def __init__(self, feature_cols: List[str], window_size: int = 30, seed: int = 42):
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cpu")
        self.scaler = StandardScaler()

    def build_windowed_sequences(
        self,
        df: pd.DataFrame,
        is_test: bool = False,
        rul_truth: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds 30-cycle sequences per engine unit.
        If an engine has fewer than 30 cycles, left-pads by repeating early cycles.
        """
        sequences: List[np.ndarray] = []
        targets: List[float] = []

        if is_test:
            # For test evaluation: exactly one 30-cycle window per test engine
            unit_ids = sorted(df["unit"].unique())
            for idx, u in enumerate(unit_ids):
                eng_df = df[df["unit"] == u].sort_values("cycle")
                raw_feats = eng_df[self.feature_cols].values
                scaled_feats = self.scaler.transform(raw_feats)

                if len(scaled_feats) >= self.window_size:
                    seq = scaled_feats[-self.window_size :]
                else:
                    pad_len = self.window_size - len(scaled_feats)
                    pad = np.repeat(scaled_feats[:1], pad_len, axis=0)
                    seq = np.vstack([pad, scaled_feats])

                sequences.append(seq)
                if rul_truth is not None and idx < len(rul_truth):
                    targets.append(float(rul_truth[idx]))
                else:
                    targets.append(float(eng_df["rul_capped"].iloc[-1]))
        else:
            # For training: sliding sequences sampled across engines
            unit_ids = sorted(df["unit"].unique())
            for u in unit_ids:
                eng_df = df[df["unit"] == u].sort_values("cycle")
                raw_feats = eng_df[self.feature_cols].values
                scaled_feats = self.scaler.transform(raw_feats)
                y_vals = eng_df["rul_capped"].values

                n_cycles = len(eng_df)
                # Sample stride = 2 to keep dataset compact and fast
                for t in range(0, n_cycles, 2):
                    if t + 1 >= self.window_size:
                        seq = scaled_feats[t + 1 - self.window_size : t + 1]
                    else:
                        pad_len = self.window_size - (t + 1)
                        pad = np.repeat(scaled_feats[:1], pad_len, axis=0)
                        seq = np.vstack([pad, scaled_feats[: t + 1]])

                    sequences.append(seq)
                    targets.append(float(y_vals[t]))

        return np.array(sequences, dtype=np.float32), np.array(targets, dtype=np.float32)

    def _train_model(
        self,
        model: nn.Module,
        X_seq: np.ndarray,
        y_seq: np.ndarray,
        epochs: int = 8,
        batch_size: int = 64
    ) -> nn.Module:
        model.to(self.device)
        model.train()
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

        dataset = TensorDataset(torch.from_numpy(X_seq), torch.from_numpy(y_seq))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                optimizer.zero_grad()
                pred = model(batch_x)
                loss = criterion(pred, batch_y)
                loss.backward()
                optimizer.step()

        return model

    def evaluate_architecture(
        self,
        arch: str,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        rul_truth: np.ndarray,
        epochs: int = 8
    ) -> Dict[str, Any]:
        """
        Trains and evaluates the real windowed sequence architecture.
        """
        t0 = time.time()
        # 1. Fit scaler strictly on training features
        self.scaler.fit(train_df[self.feature_cols].values)

        # 2. Build 30-cycle sequences
        X_train, y_train = self.build_windowed_sequences(train_df, is_test=False)
        X_test, y_test = self.build_windowed_sequences(test_df, is_test=True, rul_truth=rul_truth)

        # Subsample training sequences to max 3500 for rapid training speed (< 4s)
        if len(X_train) > 3500:
            rng = np.random.RandomState(self.seed)
            sub_idx = rng.choice(len(X_train), size=3500, replace=False)
            X_train = X_train[sub_idx]
            y_train = y_train[sub_idx]

        input_dim = len(self.feature_cols)
        if arch.lower() == "cnn":
            model = WindowedCNN1DRegressor(num_features=input_dim, window_size=self.window_size)
            name = "1D-CNN PyTorch (30-Cycle Window)"
            arch_label = "CNN"
        else:
            model = WindowedGRURegressor(num_features=input_dim, hidden_dim=64, num_layers=2)
            name = "GRU PyTorch (30-Cycle Sequence)"
            arch_label = "GRU"

        trained_model = self._train_model(model, X_train, y_train, epochs=epochs)
        trained_model.eval()

        with torch.no_grad():
            preds = trained_model(torch.from_numpy(X_test).to(self.device)).cpu().numpy()
            preds = np.maximum(0.0, preds)

        metrics = calculate_metrics(y_test, preds)
        elapsed = round(time.time() - t0, 2)

        return {
            "model_name": name,
            "architecture": arch_label,
            "window_size": self.window_size,
            "rmse": round(metrics["rmse"], 2),
            "mae": round(metrics["mae"], 2),
            "nasa_score": round(metrics["nasa_score"], 2),
            "train_time_sec": elapsed
        }
