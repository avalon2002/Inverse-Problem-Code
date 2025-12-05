#!/usr/bin/env python3
"""
Unified script to generate all observation / EnKF init files:

  Data/Observation/H_full.npy
  Data/Observation/H_sparse.npy
  Data/Observation/R_full.npy
  Data/Observation/R_sparse.npy
  Data/Observation/X0.npy
  Data/Observation/assim_indices.npy
  Data/Observation/obs_indices_sparse.npy
  Data/Observation/obs_metadata.npz
"""

import os
import numpy as np

# ----------------------------------------------------------------------
# Paths and basic settings
# ----------------------------------------------------------------------
ROOT = "/Users/huanghongxiao/Desktop/Uchicago/Inverse Problem/HW/acm270_projects-main/Project 4"
DATA_DIR = os.path.join(ROOT, "Data")
SOL_DIR  = os.path.join(DATA_DIR, "sol")
OBS_DIR  = os.path.join(DATA_DIR, "Observation")

os.makedirs(OBS_DIR, exist_ok=True)

state_dim = 40          # number of slow variables
dt = 0.005
assim_step = 20
N_ens = 50
init_sigma = 1.0        # initial ensemble spread
obs_noise_std_full = 1.0
obs_noise_std_sparse = 1.0

rng = np.random.default_rng(123)

# ----------------------------------------------------------------------
# 1. Load truth to get x0 and T
# ----------------------------------------------------------------------
x_truth_path = os.path.join(SOL_DIR, "x_truth.npy")
x_truth = np.load(x_truth_path)
K, T = x_truth.shape
assert K == state_dim, f"state_dim mismatch: {K} vs {state_dim}"
print(f"Loaded x_truth: shape = {x_truth.shape}")

x0_truth = x_truth[:, 0]  # (40,)

# ----------------------------------------------------------------------
# 2. H_full, H_sparse, R_full, R_sparse
# ----------------------------------------------------------------------
# Full observation: observe all 40 components
H_full = np.eye(state_dim)
m_full = state_dim
R_full = (obs_noise_std_full ** 2) * np.eye(m_full)

# Sparse observation: observe every second variable: indices [0, 2, ..., 38]
obs_indices_sparse = np.arange(0, state_dim, 2)
m_sparse = len(obs_indices_sparse)

H_sparse = np.zeros((m_sparse, state_dim))
H_sparse[np.arange(m_sparse), obs_indices_sparse] = 1.0
R_sparse = (obs_noise_std_sparse ** 2) * np.eye(m_sparse)

print("H_full shape   =", H_full.shape)
print("H_sparse shape =", H_sparse.shape)
print("R_full shape   =", R_full.shape)
print("R_sparse shape =", R_sparse.shape)

# ----------------------------------------------------------------------
# 3. Initial ensemble X0 (same for full & sparse)
#    X0_j = x0_truth + N(0, init_sigma^2 I)
# ----------------------------------------------------------------------
X0 = x0_truth[:, None] + init_sigma * rng.normal(size=(state_dim, N_ens))
print("X0 shape =", X0.shape)

# ----------------------------------------------------------------------
# 4. Assimilation indices: 0, 20, 40, ..., < T
# ----------------------------------------------------------------------
assim_indices = np.arange(0, T, assim_step)
print("assim_indices length =", len(assim_indices))
print("assim_indices[0:10]  =", assim_indices[:10])

# ----------------------------------------------------------------------
# 5. Save everything
# ----------------------------------------------------------------------
np.save(os.path.join(OBS_DIR, "H_full.npy"),   H_full)
np.save(os.path.join(OBS_DIR, "H_sparse.npy"), H_sparse)
np.save(os.path.join(OBS_DIR, "R_full.npy"),   R_full)
np.save(os.path.join(OBS_DIR, "R_sparse.npy"), R_sparse)
np.save(os.path.join(OBS_DIR, "X0.npy"),       X0)
np.save(os.path.join(OBS_DIR, "assim_indices.npy"), assim_indices)
np.save(os.path.join(OBS_DIR, "obs_indices_sparse.npy"), obs_indices_sparse)

# Optional metadata for convenience
meta_path = os.path.join(OBS_DIR, "obs_metadata.npz")
np.savez(
    meta_path,
    dt=dt,
    assim_step=assim_step,
    N_ens=N_ens,
    init_sigma=init_sigma,
    obs_noise_std_full=obs_noise_std_full,
    obs_noise_std_sparse=obs_noise_std_sparse,
    obs_indices_sparse=obs_indices_sparse,
)
print("\nSaved observation pack to", OBS_DIR)
print("  - H_full.npy")
print("  - H_sparse.npy")
print("  - R_full.npy")
print("  - R_sparse.npy")
print("  - X0.npy")
print("  - assim_indices.npy")
print("  - obs_indices_sparse.npy")
print("  - obs_metadata.npz")
