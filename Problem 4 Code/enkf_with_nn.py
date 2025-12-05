#!/usr/bin/env python3
"""
EnKF with NN-corrected imperfect model (single-scale Lorenz96).

UPDATES:
1. Added 'run_case' function to easily run Baseline (Alpha=0) vs NN.
2. Increased Default INFLATION (1.01 is too low for imperfect models, try 1.05+).
3. Added Debug prints to monitor NN correction magnitude.
"""

import os
import numpy as np
import torch
import torch.nn as nn

# Assuming these exist in your folder
from utils import lorenz96
from EnKF import enkf_analysis_step

# ----------------------------------------------------------------------
# Paths & config
# ----------------------------------------------------------------------
ROOT = "/Users/huanghongxiao/Desktop/Uchicago/Inverse Problem/HW/acm270_projects-main/Project 4"
DATA_DIR = os.path.join(ROOT, "Data")
SOL_DIR = os.path.join(DATA_DIR, "sol")
OBS_DIR = os.path.join(DATA_DIR, "Observation")
NN_DIR = os.path.join(DATA_DIR, "NN")
OUT_DIR = os.path.join(DATA_DIR, "NN_Correct_1")

os.makedirs(OUT_DIR, exist_ok=True)

dt = 0.005
assim_step = 20
F = 8.0

# --- TUNING PARAMETERS ---
# For imperfect models, inflation usually needs to be higher (1.05 - 1.10)
# to prevent filter divergence.
# PREVIOUS RUN: 1.08 -> RMSE ~3.02 (Still improving, but high).
# ACTION: Aggressive increase to find the stability "sweet spot".
INFLATION_BASE = 1.15
INFLATION_NN = 1.15

# Start with a conservative Alpha
# PREVIOUS: 0.05 -> Sparse RMSE dropped to 2.72 (Good result).
# ACTION: Keep at 0.05. The NN norms are high (~2.3), so a low Alpha is safe.
ALPHA_NN = 0.05

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ----------------------------------------------------------------------
# Data Loading
# ----------------------------------------------------------------------
x_truth = np.load(os.path.join(SOL_DIR, "x_truth.npy"))
state_dim, T = x_truth.shape

y_full = np.load(os.path.join(OBS_DIR, "y_full.npy"))
y_sparse = np.load(os.path.join(OBS_DIR, "y_sparse.npy"))
H_full = np.load(os.path.join(OBS_DIR, "H_full.npy"))
H_sparse = np.load(os.path.join(OBS_DIR, "H_sparse.npy"))
R_full = np.load(os.path.join(OBS_DIR, "R_full.npy"))
R_sparse = np.load(os.path.join(OBS_DIR, "R_sparse.npy"))
assim_indices = np.load(os.path.join(OBS_DIR, "assim_indices.npy"))
X0 = np.load(os.path.join(OBS_DIR, "X0.npy"))

state_dim_from_X0, N_ens = X0.shape
N_assim = assim_indices.shape[0]


# ----------------------------------------------------------------------
# NN Model (Must match training architecture)
# ----------------------------------------------------------------------
class IncrementNet(nn.Module):
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


def load_nn_increment(prefix: str):
    """Returns a function: nn_increment(x) -> dx"""
    model_path = os.path.join(NN_DIR, f"increment_{prefix}_mlp.pth")
    norm_path = os.path.join(NN_DIR, f"normalization_{prefix}.npz")

    if not os.path.exists(model_path):
        print(f"Warning: {model_path} not found. NN will be disabled.")
        return None

    norm = np.load(norm_path)
    X_mean = torch.from_numpy(norm["X_mean"]).float().to(device)
    X_std = torch.from_numpy(norm["X_std"]).float().to(device)
    Y_mean = norm["Y_mean"]
    Y_std = norm["Y_std"]

    dim = X_mean.shape[0]
    model = IncrementNet(dim_in=dim, dim_hidden=128, dim_out=dim).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    def nn_increment(x_in: np.ndarray) -> np.ndarray:
        x_tensor = torch.from_numpy(x_in).float().to(device)
        x_norm = (x_tensor - X_mean) / X_std
        x_norm = x_norm.unsqueeze(0)
        with torch.no_grad():
            dx_norm = model(x_norm)
        dx_norm_np = dx_norm.cpu().numpy().flatten()
        dx = dx_norm_np * Y_std + Y_mean
        return dx

    return nn_increment


print("\nLoading Neural Networks...")
nn_increment_full = load_nn_increment("full")
nn_increment_sparse = load_nn_increment("sparse")


# ----------------------------------------------------------------------
# Forecast & EnKF
# ----------------------------------------------------------------------
def rk4_step_physical(x, dt=0.005, F=8.0):
    k1 = lorenz96(0.0, x, F=F)
    k2 = lorenz96(0.0, x + 0.5 * dt * k1, F=F)
    k3 = lorenz96(0.0, x + 0.5 * dt * k2, F=F)
    k4 = lorenz96(0.0, x + dt * k3, F=F)
    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def forecast_ensemble_nn(X, nn_increment, alpha, dt, assim_step, F):
    X_new = X.copy()

    # 1. Physics Forecast
    for _ in range(assim_step):
        for j in range(N_ens):
            X_new[:, j] = rk4_step_physical(X_new[:, j], dt=dt, F=F)

    # 2. NN Correction (if enabled)
    if alpha > 0.0 and nn_increment is not None:
        corrections = []
        for j in range(N_ens):
            x_f = X_new[:, j]
            dx = nn_increment(x_f)
            X_new[:, j] = x_f + alpha * dx
            if j == 0: corrections.append(np.linalg.norm(dx))  # Track magnitude

        # Debug: Print correction magnitude occasionally
        if np.random.rand() < 0.001:
            print(f"  [DEBUG] Mean NN Correction Norm: {np.mean(corrections):.4f}")

    return X_new


def run_case(case_name, y_obs, H, R, nn_inc, alpha, inflation):
    """
    Generic runner for Baseline, Full, Sparse cases
    """
    print(f"\n>>> Running Case: {case_name}")
    print(f"    Alpha={alpha}, Inflation={inflation}")

    X_ens = X0.copy()
    x_a_means = np.zeros((state_dim, N_assim))

    for k, ti in enumerate(assim_indices):
        # Forecast
        X_f = forecast_ensemble_nn(
            X_ens, nn_inc, alpha, dt=dt, assim_step=assim_step, F=F
        )

        # Analysis
        y_k = y_obs[:, ti]
        X_a, _, x_a_mean = enkf_analysis_step(
            X_f, y_k, H, R, inflation=inflation, stochastic=False
        )

        x_a_means[:, k] = x_a_mean
        X_ens = X_a

        if (k + 1) % 200 == 0:
            print(f"    Step {k + 1}/{N_assim} done.")

    # Compute RMSE
    truth_assim = x_truth[:, assim_indices]
    rmse = np.sqrt(np.mean((x_a_means - truth_assim) ** 2, axis=0)).mean()
    print(f"    FINISHED. Time-Averaged RMSE: {rmse:.4f}")

    return x_a_means, rmse


# ----------------------------------------------------------------------
# Execution
# ----------------------------------------------------------------------

# 1. Baseline (No NN) - To check if Inflation is good enough
# We use Full Obs for baseline check
print("=" * 40)
print("PHASE 1: BASELINE CHECK (No NN)")
print("=" * 40)
x_a_base, rmse_base = run_case(
    "Baseline (Full Obs, No NN)",
    y_full, H_full, R_full,
    None, alpha=0.0, inflation=INFLATION_BASE
)

# 2. NN - Full
print("\n" + "=" * 40)
print("PHASE 2: NN CORRECTED (Full)")
print("=" * 40)
x_a_full_nn, rmse_full = run_case(
    "NN Full",
    y_full, H_full, R_full,
    nn_increment_full, alpha=ALPHA_NN, inflation=INFLATION_NN
)

# 3. NN - Sparse
print("\n" + "=" * 40)
print("PHASE 3: NN CORRECTED (Sparse)")
print("=" * 40)
x_a_sparse_nn, rmse_sparse = run_case(
    "NN Sparse",
    y_sparse, H_sparse, R_sparse,
    nn_increment_sparse, alpha=ALPHA_NN, inflation=INFLATION_NN
)

# ----------------------------------------------------------------------
# Save
# ----------------------------------------------------------------------
np.save(os.path.join(OUT_DIR, "x_a_full_nn.npy"), x_a_full_nn)
np.save(os.path.join(OUT_DIR, "x_a_sparse_nn.npy"), x_a_sparse_nn)
# Save baseline too for comparison
np.save(os.path.join(OUT_DIR, "x_a_baseline.npy"), x_a_base)

print("\nSUMMARY:")
print(f"  Baseline RMSE (Infl={INFLATION_BASE}): {rmse_base:.4f}")
print(f"  NN Full  RMSE (Alpha={ALPHA_NN}):      {rmse_full:.4f}")
print(f"  NN Sparse RMSE (Alpha={ALPHA_NN}):     {rmse_sparse:.4f}")