#!/usr/bin/env python3
"""
Train two separate MLPs to predict EnKF analysis increments:

  - NN_full   : using full observation EnKF data only
  - NN_sparse : using sparse observation EnKF data only

Key Improvements over user draft:
1. FIXED DATA LEAKAGE: Normalization stats are computed strictly on training split.
2. METRICS: Logs Correlation Coefficient to verify physical consistency.
3. ARCHITECTURE: Uses ReLU + Dropout for better generalization.
"""

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# ----------------------------------------------------------------------
# Paths & basic config
# ----------------------------------------------------------------------
PROJECT_ROOT = "/Users/huanghongxiao/Desktop/Uchicago/Inverse Problem/HW/acm270_projects-main/Project 4"
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
ENKF_DIR = os.path.join(DATA_DIR, "EnKF")
NN_DIR = os.path.join(DATA_DIR, "NN")
os.makedirs(NN_DIR, exist_ok=True)

SEED = 2025
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ----------------------------------------------------------------------
# Model definition (Improved)
# ----------------------------------------------------------------------
class IncrementNet(nn.Module):
    """
    Simple MLP: R^40 (forecast) -> R^40 (increment)
    Uses ReLU and Dropout for better convergence and regularization.
    """

    def __init__(self, dim_in=40, dim_hidden=128, dim_out=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(dim_hidden, dim_out),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# Generic training function
# ----------------------------------------------------------------------
def train_one_case(X, Y, out_prefix, num_epochs=150, batch_size=64, lr=1e-3):
    """
    X: (N, 40) forecast states
    Y: (N, 40) analysis increments
    out_prefix: 'full' or 'sparse'
    """
    N, dim = X.shape
    print(f"\n========== Training NN_{out_prefix.upper()} ==========")
    print(f"Dataset size: {N}, dim: {dim}")

    # 1. Split Indices FIRST (Fixes Data Leakage)
    perm = np.random.permutation(N)
    split = int(0.8 * N)
    idx_tr, idx_val = perm[:split], perm[split:]

    X_tr_raw, Y_tr_raw = X[idx_tr], Y[idx_tr]
    X_val_raw, Y_val_raw = X[idx_val], Y[idx_val]

    # 2. Compute Stats ONLY on Training Data
    X_mean = X_tr_raw.mean(axis=0)
    X_std = X_tr_raw.std(axis=0) + 1e-8
    Y_mean = Y_tr_raw.mean(axis=0)
    Y_std = Y_tr_raw.std(axis=0) + 1e-8

    # 3. Apply Normalization
    X_tr = (X_tr_raw - X_mean) / X_std
    X_val = (X_val_raw - X_mean) / X_std

    Y_tr = (Y_tr_raw - Y_mean) / Y_std
    Y_val = (Y_val_raw - Y_mean) / Y_std

    # Loaders
    train_ds = TensorDataset(torch.from_numpy(X_tr).float(),
                             torch.from_numpy(Y_tr).float())
    val_ds = TensorDataset(torch.from_numpy(X_val).float(),
                           torch.from_numpy(Y_val).float())

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model Setup
    model = IncrementNet(dim_in=dim, dim_hidden=128, dim_out=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val_loss = np.inf
    best_state = None

    # Training Loop
    for epoch in range(1, num_epochs + 1):
        # -- Train --
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            tr_loss += loss.item() * xb.size(0)
        tr_loss /= len(train_ds)

        # -- Validation --
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                val_loss += loss.item() * xb.size(0)

                # Collect for correlation calc
                val_preds.append(pred.cpu().numpy())
                val_targets.append(yb.cpu().numpy())
        val_loss /= len(val_ds)

        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()

        # Log
        if epoch == 1 or epoch % 10 == 0:
            # Calc correlation to see if directions are correct
            vp = np.concatenate(val_preds, axis=0).flatten()
            vt = np.concatenate(val_targets, axis=0).flatten()
            corr = np.corrcoef(vp, vt)[0, 1]
            print(f"[{out_prefix}] Epoch {epoch:3d} | MSE: {tr_loss:.4e} / {val_loss:.4e} | Val Corr: {corr:.4f}")

    print(f"[{out_prefix}] Best validation MSE = {best_val_loss:.4e}")

    # ----- Save -----
    model_file = os.path.join(NN_DIR, f"increment_{out_prefix}_mlp.pth")
    norm_file = os.path.join(NN_DIR, f"normalization_{out_prefix}.npz")

    torch.save(best_state, model_file)
    np.savez(norm_file,
             X_mean=X_mean, X_std=X_std,
             Y_mean=Y_mean, Y_std=Y_std)

    print(f"[{out_prefix}] Saved model -> {os.path.basename(model_file)}")
    print(f"[{out_prefix}] Saved stats -> {os.path.basename(norm_file)}")


# ----------------------------------------------------------------------
# 1. Load EnKF data
# ----------------------------------------------------------------------
print(">>> Loading EnKF data from:", ENKF_DIR)

# Ensure files exist
try:
    x_f_full = np.load(os.path.join(ENKF_DIR, "x_f_full.npy"))
    x_a_full = np.load(os.path.join(ENKF_DIR, "x_a_full.npy"))
    x_f_sparse = np.load(os.path.join(ENKF_DIR, "x_f_sparse.npy"))
    x_a_sparse = np.load(os.path.join(ENKF_DIR, "x_a_sparse.npy"))
except FileNotFoundError as e:
    print(f"\nError: Could not load data files. {e}")
    exit(1)

# Full case
X_full = x_f_full.T
Y_full = (x_a_full - x_f_full).T

# Sparse case
X_sparse = x_f_sparse.T
Y_sparse = (x_a_sparse - x_f_sparse).T

# ----------------------------------------------------------------------
# 2. Train NN_full
# ----------------------------------------------------------------------
train_one_case(X_full, Y_full, out_prefix="full", num_epochs=150)

# ----------------------------------------------------------------------
# 3. Train NN_sparse
# ----------------------------------------------------------------------
train_one_case(X_sparse, Y_sparse, out_prefix="sparse", num_epochs=150)

print("\nAll training done.")