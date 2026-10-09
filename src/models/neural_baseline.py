import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Dict, Any, List, Tuple
from src.common.utils import calculate_metrics

class CNN1DRegressor(nn.Module):
    """1D-CNN Architecture for Turbofan RUL Prediction."""
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # x: (batch_size, input_dim) -> (batch_size, 1, input_dim)
        x = x.unsqueeze(1)
        return self.net(x).squeeze(-1)

class GRURegressor(nn.Module):
    """GRU Recurrent Network for Turbofan RUL Prediction."""
    def __init__(self, input_dim: int, hidden_dim: int = 48):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 24),
            nn.ReLU(),
            nn.Linear(24, 1)
        )

    def forward(self, x):
        # x: (batch_size, input_dim)
        proj = torch.relu(self.input_proj(x)).unsqueeze(1) # (batch, 1, hidden)
        out, _ = self.gru(proj)
        out = out[:, -1, :]
        return self.fc(out).squeeze(-1)

class NeuralRULBenchmark:
    """Trains and compares 1D-CNN and GRU deep models against GBDT."""
    def __init__(self, feature_cols: List[str], seed: int = 42):
        self.feature_cols = feature_cols
        self.seed = seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.device = torch.device("cpu")

    def _train_model(self, model: nn.Module, X_train: np.ndarray, y_train: np.ndarray, epochs: int = 12) -> nn.Module:
        model.to(self.device)
        model.train()
        criterion = nn.SmoothL1Loss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=1e-4)

        dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
        loader = DataLoader(dataset, batch_size=128, shuffle=True)

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
        self, arch: str, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, Any]:
        input_dim = X_train.shape[1]
        t0 = time.time()
        if arch.lower() == "cnn":
            model = CNN1DRegressor(input_dim)
            name = "1D-CNN PyTorch"
        else:
            model = GRURegressor(input_dim)
            name = "GRU PyTorch"

        sample_size = min(6000, len(X_train))
        rng = np.random.RandomState(self.seed)
        sub_idx = rng.choice(len(X_train), size=sample_size, replace=False)
        X_sub = X_train[sub_idx]
        y_sub = y_train[sub_idx]

        model = self._train_model(model, X_sub, y_sub, epochs=4)
        train_sec = round(time.time() - t0, 2)

        model.eval()
        with torch.no_grad():
            preds = model(torch.tensor(X_test, dtype=torch.float32).to(self.device)).cpu().numpy()
            preds = np.maximum(0.0, preds)

        metrics = calculate_metrics(y_test, preds)
        return {
            "model_name": name,
            "architecture": arch.upper(),
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "nasa_score": metrics["nasa_score"],
            "train_time_sec": train_sec
        }
