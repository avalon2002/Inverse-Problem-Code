"""
Part 1: Generate data for EnKF experiments on the QG model
This script generates the truth trajectory and observations, then saves them to disk.

Author: Hongxiao Huang
Course: ACM 270 - Inverse Problems
Project 3: Ensemble Kalman Filter with ML Surrogate
"""

import os
import numpy as np
from physical_model import drymodel

# ============================================================================
# Configuration
# ============================================================================

# Data output directory
DATA_DIR = "/Data"

# Simulation parameters
T = 50  # Number of assimilation time steps
N_TRANSIENT = 100  # Transient steps to spin up the model
SIGMA_OBS = 0.5  # Observation noise standard deviation
STRIDE = 8  # Observation stride (spatial subsampling)
SEED = 10  # Random seed for reproducibility


# ============================================================================
# Helper Functions
# ============================================================================

def flatten_state(psi):
    """Flatten the 3D state (nx, ny, n_layers) to 1D vector."""
    return psi.ravel()


def unflatten_state(vec, nx, ny, n_layers=2):
    """Unflatten 1D vector back to 3D state."""
    return vec.reshape(nx, ny, n_layers)


def build_observation_indices(nx, ny, n_layers=2, stride=8, layer=0):
    """
    Build indices for observing a subset of grid points.
    Observations are taken from a single layer with spatial subsampling.
    """
    idxs = []
    for i in range(0, nx, stride):
        for j in range(0, ny, stride):
            idx = ((i * ny) + j) * n_layers + layer
            idxs.append(idx)
    return np.array(idxs, dtype=np.int64)


def generate_truth_and_observations(T, n_transient=100, sigma_obs=0.5, stride=8, seed=0):
    """
    Generate truth trajectory using the physical QG model and
    synthetic observations with Gaussian noise.

    Parameters
    ----------
    T : int
        Number of assimilation time steps
    n_transient : int
        Number of transient steps to spin up the model
    sigma_obs : float
        Observation noise standard deviation
    stride : int
        Spatial subsampling stride for observations
    seed : int
        Random seed for reproducibility

    Returns
    -------
    psi_truth : list of ndarray
        Truth trajectory, list of (nx, ny, n_layers) arrays, length T+1
    y_obs : list of ndarray
        Observations at each time step, length T
    obs_idx : ndarray
        Indices of observed grid points
    R : ndarray
        Observation error covariance matrix
    """
    rng = np.random.default_rng(seed)

    # Initial condition
    psi0 = rng.standard_normal((96, 192, 2))

    # Spin up the model
    print(f"Spinning up the model for {n_transient} transient steps...")
    psi = psi0.copy()
    for step in range(n_transient):
        psi = drymodel(psi)
        if (step + 1) % 20 == 0:
            print(f"  Transient step {step + 1}/{n_transient}")

    # Generate truth trajectory
    print(f"\nGenerating truth trajectory for {T} time steps...")
    psi_truth = [psi.copy()]
    for step in range(T):
        psi = drymodel(psi)
        psi_truth.append(psi.copy())
        if (step + 1) % 10 == 0:
            print(f"  Truth step {step + 1}/{T}")

    nx, ny, n_layers = psi_truth[0].shape

    # Build observation operator
    obs_idx = build_observation_indices(nx, ny, n_layers, stride=stride, layer=0)
    m = len(obs_idx)

    # Observation error covariance
    R = (sigma_obs ** 2) * np.eye(m)

    # Generate observations with noise
    print(f"\nGenerating observations with noise (sigma = {sigma_obs})...")
    y_obs = []
    for t in range(1, T + 1):
        x_vec = flatten_state(psi_truth[t])
        y_clean = x_vec[obs_idx]
        noise = rng.normal(0.0, sigma_obs, size=m)
        y_obs.append(y_clean + noise)

    return psi_truth, y_obs, obs_idx, R


def idx_to_coord(idx, nx, ny, n_layers):
    """Convert flat index to (i, j, layer) coordinates."""
    layer = idx % n_layers
    tmp = idx // n_layers
    j = tmp % ny
    i = tmp // ny
    return i, j, layer


def build_localization_xy(nx, ny, n_layers, obs_idx, ell, periodic=True, dtype=np.float32):
    """
    Build a state-observation localization matrix using Gaussian correlation.

    Parameters
    ----------
    nx, ny : int
        Grid dimensions
    n_layers : int
        Number of vertical layers
    obs_idx : ndarray
        Indices of observed grid points
    ell : float
        Localization length scale (squared distance in exponent)
    periodic : bool
        Whether to use periodic boundary conditions
    dtype : numpy dtype
        Output data type

    Returns
    -------
    L_xy : ndarray
        Localization matrix of shape (d, m) where d = nx*ny*n_layers and m = len(obs_idx)
    """
    d = nx * ny * n_layers
    m = len(obs_idx)

    # Build state coordinates
    state_coords = np.zeros((d, 2), dtype=np.int32)
    for i in range(nx):
        for j in range(ny):
            for l in range(n_layers):
                idx = ((i * ny) + j) * n_layers + l
                state_coords[idx, 0] = i
                state_coords[idx, 1] = j

    # Build observation coordinates
    obs_coords = np.zeros((m, 2), dtype=np.int32)
    for k, idx in enumerate(obs_idx):
        i, j, layer = idx_to_coord(idx, nx, ny, n_layers)
        obs_coords[k, 0] = i
        obs_coords[k, 1] = j

    # Compute distances
    si = state_coords[:, 0][:, None]  # (d, 1)
    sj = state_coords[:, 1][:, None]  # (d, 1)
    oi = obs_coords[:, 0][None, :]  # (1, m)
    oj = obs_coords[:, 1][None, :]  # (1, m)

    dx = si - oi  # (d, m)
    dy = sj - oj  # (d, m)

    if periodic:
        dx = np.minimum(np.abs(dx), nx - np.abs(dx))
        dy = np.minimum(np.abs(dy), ny - np.abs(dy))
    else:
        dx = np.abs(dx)
        dy = np.abs(dy)

    D2 = dx.astype(np.float64) ** 2 + dy.astype(np.float64) ** 2

    # Gaussian localization
    L_xy = np.exp(-D2 / ell, dtype=np.float64)
    L_xy = L_xy.astype(dtype)

    print(f"Localization matrix shape: {L_xy.shape}")
    print(f"L_xy min: {float(np.min(L_xy)):.6f}, max: {float(np.max(L_xy)):.6f}")
    print(f"Any non-finite values: {not np.all(np.isfinite(L_xy))}")

    return L_xy


def save_data(data_dir, psi_truth, y_obs, obs_idx, R, L_xy=None):
    """
    Save all generated data to disk.

    Parameters
    ----------
    data_dir : str
        Directory to save data
    psi_truth : list of ndarray
        Truth trajectory
    y_obs : list of ndarray
        Observations
    obs_idx : ndarray
        Observation indices
    R : ndarray
        Observation error covariance
    L_xy : ndarray, optional
        Localization matrix
    """
    # Create directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    # Save truth trajectory as single array
    psi_truth_array = np.stack(psi_truth, axis=0)
    np.save(os.path.join(data_dir, "psi_truth.npy"), psi_truth_array)
    print(f"Saved psi_truth.npy with shape {psi_truth_array.shape}")

    # Save observations as single array
    y_obs_array = np.stack(y_obs, axis=0)
    np.save(os.path.join(data_dir, "y_obs.npy"), y_obs_array)
    print(f"Saved y_obs.npy with shape {y_obs_array.shape}")

    # Save observation indices
    np.save(os.path.join(data_dir, "obs_idx.npy"), obs_idx)
    print(f"Saved obs_idx.npy with shape {obs_idx.shape}")

    # Save observation error covariance
    np.save(os.path.join(data_dir, "R.npy"), R)
    print(f"Saved R.npy with shape {R.shape}")

    # Save localization matrix if provided
    if L_xy is not None:
        np.save(os.path.join(data_dir, "L_xy.npy"), L_xy)
        print(f"Saved L_xy.npy with shape {L_xy.shape}")

    # Save metadata
    metadata = {
        "T": len(y_obs),
        "nx": psi_truth[0].shape[0],
        "ny": psi_truth[0].shape[1],
        "n_layers": psi_truth[0].shape[2],
        "n_obs": len(obs_idx),
        "sigma_obs": SIGMA_OBS,
        "stride": STRIDE,
        "seed": SEED,
        "n_transient": N_TRANSIENT,
    }
    np.savez(os.path.join(data_dir, "metadata.npz"), **metadata)
    print(f"Saved metadata.npz")

    print(f"\nAll data saved to: {data_dir}")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Part 1: Generate Data for EnKF Experiments")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  T (time steps)     : {T}")
    print(f"  Transient steps    : {N_TRANSIENT}")
    print(f"  Observation noise  : {SIGMA_OBS}")
    print(f"  Observation stride : {STRIDE}")
    print(f"  Random seed        : {SEED}")
    print(f"  Output directory   : {DATA_DIR}")
    print()

    # Generate truth and observations
    print("-" * 70)
    print("Step 1: Generate truth trajectory and observations")
    print("-" * 70)
    psi_truth, y_obs, obs_idx, R = generate_truth_and_observations(
        T=T,
        n_transient=N_TRANSIENT,
        sigma_obs=SIGMA_OBS,
        stride=STRIDE,
        seed=SEED,
    )

    nx, ny, n_layers = psi_truth[0].shape
    print(f"\nGeneration complete:")
    print(f"  Number of time steps (truth): {len(psi_truth) - 1}")
    print(f"  State shape: {psi_truth[0].shape}")
    print(f"  State dimension d = {nx * ny * n_layers}")
    print(f"  Observation dimension m = {len(obs_idx)}")

    # Build localization matrix
    print()
    print("-" * 70)
    print("Step 2: Build localization matrix")
    print("-" * 70)
    ell = 100.0  # Localization length scale parameter
    L_xy = build_localization_xy(
        nx=nx,
        ny=ny,
        n_layers=n_layers,
        obs_idx=obs_idx,
        ell=ell,
        periodic=True,
    )

    # Save all data
    print()
    print("-" * 70)
    print("Step 3: Save data to disk")
    print("-" * 70)
    save_data(DATA_DIR, psi_truth, y_obs, obs_idx, R, L_xy)

    print()
    print("=" * 70)
    print("Data generation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
