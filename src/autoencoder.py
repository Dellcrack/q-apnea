"""
Autoencoder Representation Learning Module
Implements the 10D -> 8D -> 4D -> 8D -> 10D PyTorch Autoencoder,
latent space extraction, latent normalization, and collapse checking.

Source of truth: SIH 2026 Sleep Apnea Best Combined Pipeline
"""

import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Any, List, Optional, Callable

AE_INPUT_DIM = 10
AE_HIDDEN_DIM = 8
AE_LATENT_DIM = 4


class HRVAutoencoder(nn.Module):
    """
    Symmetric multi-layer perceptron Autoencoder:
    Encoder: Input (10) -> Linear(10, 8) -> ReLU -> Linear(8, 4) -> Latent z
    Decoder: Latent (4) -> Linear(4, 8) -> ReLU -> Linear(8, 10) -> Reconstructed x_hat
    """
    def __init__(self, input_dim: int = 10, hidden_dim: int = 8, latent_dim: int = 4):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed, z


def train_autoencoder(
    X_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    epochs: int = 200,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 20,
    seed: int = 42,
    progress_callback: Optional[Callable[[int, int, float, Optional[float]], None]] = None
) -> Tuple[HRVAutoencoder, List[float], Dict[str, Any]]:
    """
    Train HRV Autoencoder on X_train (TRAIN ONLY).
    Supports early stopping and progress updates.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = X_train.shape[1]
    model = HRVAutoencoder(
        input_dim=input_dim,
        hidden_dim=AE_HIDDEN_DIM,
        latent_dim=AE_LATENT_DIM
    ).to(device)

    train_tensor = torch.tensor(X_train, dtype=torch.float32)
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    train_loader = DataLoader(
        TensorDataset(train_tensor),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_loss = np.inf
    best_state = None
    patience_counter = 0
    loss_history = []
    val_loss_history = []

    start_time = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        n_seen = 0

        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            reconstructed, _ = model(batch_x)
            loss = criterion(reconstructed, batch_x)
            loss.backward()
            optimizer.step()

            bsize = batch_x.size(0)
            epoch_loss += loss.item() * bsize
            n_seen += bsize

        epoch_loss /= max(1, n_seen)
        loss_history.append(epoch_loss)

        # Validation MSE if available
        val_mse = None
        if X_val is not None:
            model.eval()
            with torch.no_grad():
                val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
                rec_val, _ = model(val_tensor)
                val_mse = criterion(rec_val, val_tensor).item()
                val_loss_history.append(val_mse)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if progress_callback:
            progress_callback(epoch + 1, epochs, epoch_loss, val_mse)

        if patience_counter >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    elapsed = time.perf_counter() - start_time

    stats = {
        "device": str(device),
        "best_train_mse": float(best_loss),
        "val_mse": float(val_loss_history[-1]) if val_loss_history else None,
        "epochs_completed": len(loss_history),
        "training_time_sec": float(elapsed),
    }

    return model, loss_history, stats


def extract_latent_representations(
    model: HRVAutoencoder,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler, Dict[str, Any]]:
    """
    Extract 4D latent vectors from trained Autoencoder encoder and scale them (TRAIN ONLY).
    Checks for latent collapse (standard deviation < 1e-6).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    def encode(arr):
        tensor = torch.tensor(arr, dtype=torch.float32, device=device)
        with torch.no_grad():
            z = model.encoder(tensor)
        return z.detach().cpu().numpy().astype(np.float32)

    X_train_ae = encode(X_train)
    X_val_ae = encode(X_val)
    X_test_ae = encode(X_test)

    latent_std = X_train_ae.std(axis=0)
    has_collapse = bool(np.any(latent_std < 1e-6))

    # Scaler fitted ONLY on train latent
    latent_scaler = StandardScaler()
    X_train_ae_scaled = latent_scaler.fit_transform(X_train_ae).astype(np.float32)
    X_val_ae_scaled = latent_scaler.transform(X_val_ae).astype(np.float32)
    X_test_ae_scaled = latent_scaler.transform(X_test_ae).astype(np.float32)

    stats = {
        "latent_dim": AE_LATENT_DIM,
        "latent_stds": latent_std.tolist(),
        "has_collapse": has_collapse,
        "train_scaled_mean": X_train_ae_scaled.mean(axis=0).tolist(),
        "train_scaled_std": X_train_ae_scaled.std(axis=0).tolist(),
    }

    return X_train_ae_scaled, X_val_ae_scaled, X_test_ae_scaled, latent_scaler, stats
