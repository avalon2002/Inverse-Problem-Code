"""
Part 2: Run EnKF experiments on the QG model
This script loads pre-generated data and runs EnKF with physical and ML forecasts.

Author: Hongxiao Huang
Course: ACM 270 - Inverse Problems
Project 3: Ensemble Kalman Filter with ML Surrogate
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from physical_model import drymodel
from ml_model import model

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = "/Data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")

# ============================================================================
# Helper Functions
# ============================================================================

def flatten_state(psi):
    return psi.ravel()

def unflatten_state(vec, nx, ny, n_layers=2):
    return vec.reshape(nx, ny, n_layers)

def crps_ensemble(samples, y):
    x = np.asarray(samples).ravel()
    term1 = np.mean(np.abs(x - y))
    diff = np.abs(x[:, None] - x[None, :])
    term2 = 0.5 * np.mean(diff)
    return term1 - term2

def step_physical_ensemble(ensemble):
    Ne = ensemble.shape[0]
    out = [drymodel(ensemble[i]) for i in range(Ne)]
    return np.stack(out, axis=0)

def step_ml_ensemble(ensemble):
    Ne, nx, ny, n_layers = ensemble.shape
    batch = ensemble.transpose(0, 2, 1, 3)
    out = model.predict(batch, verbose=0)
    return out.transpose(0, 2, 1, 3)

def enkf_analysis(X_f, y_obs_t, obs_idx, R, inflation=None, rng=None, L_xy=None):
    if rng is None:
        rng = np.random.default_rng()

    d, Ne = X_f.shape
    m = len(obs_idx)

    x_bar = X_f.mean(axis=1, keepdims=True)
    A = X_f - x_bar

    if inflation is not None:
        A *= inflation
        X_f = x_bar + A

    Y_f = X_f[obs_idx, :]
    y_bar = Y_f.mean(axis=1, keepdims=True)
    Ay = Y_f - y_bar

    factor = 1.0 / (Ne - 1)
    P_xy = factor * (A @ Ay.T)
    P_yy = factor * (Ay @ Ay.T) + R

    K = np.linalg.solve(P_yy.T, P_xy.T).T

    if L_xy is not None:
        K = K * L_xy

    eps = rng.multivariate_normal(mean=np.zeros(m), cov=R, size=Ne).T
    Y_tilde = y_obs_t[:, None] + eps
    innovation = Y_tilde - Y_f
    X_a = X_f + K @ innovation

    return X_a

def run_enkf(psi_truth, y_obs, obs_idx, R, forecast_step, Ne=20, sigma_init=1.0,
             inflation=None, seed=0, compute_crps=False, L_xy=None, verbose=True):
    rng = np.random.default_rng(seed)

    T = len(y_obs)
    nx, ny, n_layers = psi_truth[0].shape
    d = nx * ny * n_layers

    truth_vecs = np.array([flatten_state(psi_truth[t]) for t in range(T + 1)])

    x0_truth = truth_vecs[0]
    X_a = x0_truth[:, None] + sigma_init * rng.standard_normal((d, Ne))

    X_means = np.zeros((T + 1, d))
    X_means[0, :] = X_a.mean(axis=1)

    spreads = np.zeros(T + 1)
    spreads[0] = np.std(X_a)

    if compute_crps:
        crps = np.zeros(T + 1)
        crps[0] = np.nan

    for t in range(T):
        if verbose and (t + 1) % 10 == 0:
            print(f"    Step {t + 1}/{T}")

        ensemble_fields = np.stack(
            [unflatten_state(X_a[:, i], nx, ny, n_layers) for i in range(Ne)], axis=0
        )
        ensemble_forecast = forecast_step(ensemble_fields)
        X_f = np.stack([flatten_state(ensemble_forecast[i]) for i in range(Ne)], axis=1)

        X_a = enkf_analysis(X_f, y_obs_t=y_obs[t], obs_idx=obs_idx, R=R,
                           inflation=inflation, rng=rng, L_xy=L_xy)

        X_means[t + 1, :] = X_a.mean(axis=1)
        spreads[t + 1] = np.std(X_a)

        if compute_crps:
            truth_obs = truth_vecs[t + 1][obs_idx]
            ens_obs = X_a[obs_idx, :].T
            m = truth_obs.size
            crps_t = sum(crps_ensemble(ens_obs[:, j], truth_obs[j]) for j in range(m))
            crps[t + 1] = crps_t / m

    rmse = np.sqrt(np.mean((X_means - truth_vecs) ** 2, axis=1))

    if compute_crps:
        return rmse, X_means, crps, spreads
    return rmse, X_means, spreads

def load_data(data_dir):
    print(f"Loading data from: {data_dir}")

    psi_truth_array = np.load(os.path.join(data_dir, "psi_truth.npy"))
    y_obs_array = np.load(os.path.join(data_dir, "y_obs.npy"))
    obs_idx = np.load(os.path.join(data_dir, "obs_idx.npy"))
    R = np.load(os.path.join(data_dir, "R.npy"))
    L_xy = np.load(os.path.join(data_dir, "L_xy.npy"))
    metadata = dict(np.load(os.path.join(data_dir, "metadata.npz")))

    psi_truth = [psi_truth_array[t] for t in range(psi_truth_array.shape[0])]
    y_obs = [y_obs_array[t] for t in range(y_obs_array.shape[0])]

    print(f"  psi_truth: {len(psi_truth)} time steps, shape {psi_truth[0].shape}")
    print(f"  y_obs: {len(y_obs)} observations, shape {y_obs[0].shape}")

    return psi_truth, y_obs, obs_idx, R, L_xy, metadata

# ============================================================================
# Experiments
# ============================================================================

def experiment_1_basic_comparison(psi_truth, y_obs, obs_idx, R, L_xy, Ne=20, inflation=1.02, seed=20):
    """
    Experiment 1: Basic comparison (same ensemble size).
    Figures: fig1_rmse_time_series.png, fig2_crps_time_series.png
    """
    print("\n" + "=" * 70)
    print("Experiment 1: Physical vs ML Forecast EnKF (Same Ensemble Size)")
    print("=" * 70)

    T = len(y_obs)
    time_steps = np.arange(T + 1)

    print(f"\nRunning Physical forecast EnKF (Ne = {Ne})...")
    rmse_phys, _, crps_phys, spread_phys = run_enkf(
        psi_truth, y_obs, obs_idx, R, step_physical_ensemble,
        Ne=Ne, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy
    )

    print(f"\nRunning ML-surrogate forecast EnKF (Ne = {Ne})...")
    rmse_ml, _, crps_ml, spread_ml = run_enkf(
        psi_truth, y_obs, obs_idx, R, step_ml_ensemble,
        Ne=Ne, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy
    )

    # Figure 1: RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, rmse_phys, 'b-o', markersize=4, label=f"Physical model (Ne={Ne})")
    plt.plot(time_steps, rmse_ml, 'r-s', markersize=4, label=f"ML surrogate (Ne={Ne})")
    plt.xlabel("Assimilation step", fontsize=12)
    plt.ylabel("RMSE of ensemble mean", fontsize=12)
    plt.title("EnKF Performance: RMSE over Time", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig1_rmse_time_series.png"), dpi=150)
    plt.close()

    # Figure 2: CRPS
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, crps_phys, 'b-o', markersize=4, label=f"Physical model (Ne={Ne})")
    plt.plot(time_steps, crps_ml, 'r-s', markersize=4, label=f"ML surrogate (Ne={Ne})")
    plt.xlabel("Assimilation step", fontsize=12)
    plt.ylabel("CRPS (observation space)", fontsize=12)
    plt.title("EnKF Performance: CRPS over Time", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig2_crps_time_series.png"), dpi=150)
    plt.close()

    print(f"  Saved: fig1_rmse_time_series.png, fig2_crps_time_series.png")
    print(f"\n  Summary (t=1 to T):")
    print(f"    Physical - Mean RMSE: {rmse_phys[1:].mean():.4f}, Mean CRPS: {crps_phys[1:].mean():.4f}")
    print(f"    ML       - Mean RMSE: {rmse_ml[1:].mean():.4f}, Mean CRPS: {crps_ml[1:].mean():.4f}")

    return {'rmse_phys': rmse_phys, 'rmse_ml': rmse_ml, 'crps_phys': crps_phys, 'crps_ml': crps_ml}


def experiment_2_ensemble_size(psi_truth, y_obs, obs_idx, R, L_xy,
                                Ne_list_phys=[10, 20, 40],
                                Ne_list_ml=[10, 20, 40, 80, 160],
                                inflation=1.02, seed=20):
    """
    Experiment 2: Effect of ensemble size.
    Key: ML can use much larger ensembles.
    Figures: fig3_rmse_vs_ensemble_size.png, fig4_crps_vs_ensemble_size.png
    """
    print("\n" + "=" * 70)
    print("Experiment 2: Effect of Ensemble Size")
    print(f"  Physical: Ne = {Ne_list_phys}")
    print(f"  ML:       Ne = {Ne_list_ml}")
    print("=" * 70)

    results_phys = {'Ne': [], 'rmse': [], 'crps': []}
    results_ml = {'Ne': [], 'rmse': [], 'crps': []}

    print("\n--- Physical Model ---")
    for Ne in Ne_list_phys:
        print(f"  Ne = {Ne}...", end=" ")
        rmse, _, crps, _ = run_enkf(
            psi_truth, y_obs, obs_idx, R, step_physical_ensemble,
            Ne=Ne, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy, verbose=False
        )
        results_phys['Ne'].append(Ne)
        results_phys['rmse'].append(rmse[1:].mean())
        results_phys['crps'].append(crps[1:].mean())
        print(f"RMSE: {rmse[1:].mean():.4f}, CRPS: {crps[1:].mean():.4f}")

    print("\n--- ML Surrogate ---")
    for Ne in Ne_list_ml:
        print(f"  Ne = {Ne}...", end=" ")
        rmse, _, crps, _ = run_enkf(
            psi_truth, y_obs, obs_idx, R, step_ml_ensemble,
            Ne=Ne, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy, verbose=False
        )
        results_ml['Ne'].append(Ne)
        results_ml['rmse'].append(rmse[1:].mean())
        results_ml['crps'].append(crps[1:].mean())
        print(f"RMSE: {rmse[1:].mean():.4f}, CRPS: {crps[1:].mean():.4f}")

    # Figure 3: RMSE vs Ne
    plt.figure(figsize=(10, 6))
    plt.plot(results_phys['Ne'], results_phys['rmse'], 'b-o', markersize=8, linewidth=2, label="Physical model")
    plt.plot(results_ml['Ne'], results_ml['rmse'], 'r-s', markersize=8, linewidth=2, label="ML surrogate")
    plt.xlabel("Ensemble size (Ne)", fontsize=12)
    plt.ylabel("Time-mean RMSE", fontsize=12)
    plt.title("Effect of Ensemble Size on RMSE", fontsize=14)
    plt.xscale('log')
    plt.xticks(Ne_list_ml, [str(n) for n in Ne_list_ml])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig3_rmse_vs_ensemble_size.png"), dpi=150)
    plt.close()

    # Figure 4: CRPS vs Ne
    plt.figure(figsize=(10, 6))
    plt.plot(results_phys['Ne'], results_phys['crps'], 'b-o', markersize=8, linewidth=2, label="Physical model")
    plt.plot(results_ml['Ne'], results_ml['crps'], 'r-s', markersize=8, linewidth=2, label="ML surrogate")
    plt.xlabel("Ensemble size (Ne)", fontsize=12)
    plt.ylabel("Time-mean CRPS", fontsize=12)
    plt.title("Effect of Ensemble Size on CRPS", fontsize=14)
    plt.xscale('log')
    plt.xticks(Ne_list_ml, [str(n) for n in Ne_list_ml])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig4_crps_vs_ensemble_size.png"), dpi=150)
    plt.close()

    print(f"\n  Saved: fig3_rmse_vs_ensemble_size.png, fig4_crps_vs_ensemble_size.png")

    return results_phys, results_ml


def experiment_3_large_ml_vs_small_phys(psi_truth, y_obs, obs_idx, R, L_xy,
                                         Ne_phys=20, Ne_ml=100, inflation=1.02, seed=20):
    """
    Experiment 3: Large ML ensemble vs small physical ensemble.
    Figures: fig5_large_ml_vs_small_phys_rmse.png, fig6_large_ml_vs_small_phys_crps.png
    """
    print("\n" + "=" * 70)
    print(f"Experiment 3: Physical (Ne={Ne_phys}) vs ML (Ne={Ne_ml})")
    print("=" * 70)

    T = len(y_obs)
    time_steps = np.arange(T + 1)

    print(f"\nRunning Physical (Ne={Ne_phys})...")
    rmse_phys, _, crps_phys, _ = run_enkf(
        psi_truth, y_obs, obs_idx, R, step_physical_ensemble,
        Ne=Ne_phys, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy
    )

    print(f"\nRunning ML (Ne={Ne_ml})...")
    rmse_ml, _, crps_ml, _ = run_enkf(
        psi_truth, y_obs, obs_idx, R, step_ml_ensemble,
        Ne=Ne_ml, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy
    )

    # Figure 5: RMSE
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, rmse_phys, 'b-o', markersize=4, label=f"Physical (Ne={Ne_phys})")
    plt.plot(time_steps, rmse_ml, 'r-s', markersize=4, label=f"ML (Ne={Ne_ml})")
    plt.xlabel("Assimilation step", fontsize=12)
    plt.ylabel("RMSE", fontsize=12)
    plt.title("Large ML Ensemble vs Small Physical Ensemble: RMSE", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig5_large_ml_vs_small_phys_rmse.png"), dpi=150)
    plt.close()

    # Figure 6: CRPS
    plt.figure(figsize=(10, 5))
    plt.plot(time_steps, crps_phys, 'b-o', markersize=4, label=f"Physical (Ne={Ne_phys})")
    plt.plot(time_steps, crps_ml, 'r-s', markersize=4, label=f"ML (Ne={Ne_ml})")
    plt.xlabel("Assimilation step", fontsize=12)
    plt.ylabel("CRPS", fontsize=12)
    plt.title("Large ML Ensemble vs Small Physical Ensemble: CRPS", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig6_large_ml_vs_small_phys_crps.png"), dpi=150)
    plt.close()

    print(f"  Saved: fig5, fig6")
    print(f"\n  Summary:")
    print(f"    Physical (Ne={Ne_phys}) - Mean RMSE: {rmse_phys[1:].mean():.4f}, CRPS: {crps_phys[1:].mean():.4f}")
    print(f"    ML (Ne={Ne_ml})       - Mean RMSE: {rmse_ml[1:].mean():.4f}, CRPS: {crps_ml[1:].mean():.4f}")

    return {'rmse_phys': rmse_phys, 'rmse_ml': rmse_ml, 'crps_phys': crps_phys, 'crps_ml': crps_ml}


def experiment_4_spread_skill(psi_truth, y_obs, obs_idx, R, L_xy, Ne=20, inflation=1.02, seed=20):
    """
    Experiment 4: Spread-skill relationship.
    Figure: fig7_spread_skill.png
    """
    print("\n" + "=" * 70)
    print("Experiment 4: Spread-Skill Relationship")
    print("=" * 70)

    T = len(y_obs)
    time_steps = np.arange(T + 1)

    print(f"\nRunning Physical (Ne={Ne})...")
    rmse_phys, _, crps_phys, spread_phys = run_enkf(
        psi_truth, y_obs, obs_idx, R, step_physical_ensemble,
        Ne=Ne, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy
    )

    print(f"\nRunning ML (Ne={Ne})...")
    rmse_ml, _, crps_ml, spread_ml = run_enkf(
        psi_truth, y_obs, obs_idx, R, step_ml_ensemble,
        Ne=Ne, inflation=inflation, seed=seed, compute_crps=True, L_xy=L_xy
    )

    # Figure 7
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.plot(time_steps[1:], rmse_phys[1:], 'b-', linewidth=2, label='RMSE')
    ax.plot(time_steps[1:], spread_phys[1:], 'b--', linewidth=2, label='Spread')
    ax.set_xlabel("Assimilation step", fontsize=12)
    ax.set_ylabel("RMSE / Spread", fontsize=12)
    ax.set_title(f"Physical Model (Ne={Ne})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(time_steps[1:], rmse_ml[1:], 'r-', linewidth=2, label='RMSE')
    ax.plot(time_steps[1:], spread_ml[1:], 'r--', linewidth=2, label='Spread')
    ax.set_xlabel("Assimilation step", fontsize=12)
    ax.set_ylabel("RMSE / Spread", fontsize=12)
    ax.set_title(f"ML Surrogate (Ne={Ne})", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("Spread-Skill Relationship", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig7_spread_skill.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: fig7_spread_skill.png")

    return {'rmse_phys': rmse_phys, 'spread_phys': spread_phys, 'rmse_ml': rmse_ml, 'spread_ml': spread_ml}


def experiment_5_summary_bar(results_phys, results_ml):
    """
    Experiment 5: Summary bar chart.
    Figure: fig8_summary_comparison.png
    """
    print("\n" + "=" * 70)
    print("Experiment 5: Summary Comparison")
    print("=" * 70)

    from matplotlib.patches import Patch

    labels, rmse_vals, crps_vals, colors = [], [], [], []

    for i, Ne in enumerate(results_phys['Ne']):
        labels.append(f"Phys\nNe={Ne}")
        rmse_vals.append(results_phys['rmse'][i])
        crps_vals.append(results_phys['crps'][i])
        colors.append('steelblue')

    for i, Ne in enumerate(results_ml['Ne']):
        labels.append(f"ML\nNe={Ne}")
        rmse_vals.append(results_ml['rmse'][i])
        crps_vals.append(results_ml['crps'][i])
        colors.append('coral')

    x = np.arange(len(labels))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].bar(x, rmse_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("Time-mean RMSE", fontsize=12)
    axes[0].set_title("RMSE Comparison", fontsize=13)
    axes[0].grid(True, alpha=0.3, axis='y')

    axes[1].bar(x, crps_vals, color=colors, edgecolor='black', linewidth=0.5)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Time-mean CRPS", fontsize=12)
    axes[1].set_title("CRPS Comparison", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis='y')

    legend_elements = [
        Patch(facecolor='steelblue', edgecolor='black', label='Physical Model'),
        Patch(facecolor='coral', edgecolor='black', label='ML Surrogate'),
    ]
    fig.legend(handles=legend_elements, loc='upper center', ncol=2, fontsize=11, bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout()
    plt.savefig(os.path.join(IMAGE_DIR, "fig8_summary_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved: fig8_summary_comparison.png")


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("Part 2: Run EnKF Experiments")
    print("=" * 70)

    os.makedirs(IMAGE_DIR, exist_ok=True)
    print(f"\nImages will be saved to: {IMAGE_DIR}")

    print("\n" + "-" * 70)
    print("Loading data...")
    print("-" * 70)
    psi_truth, y_obs, obs_idx, R, L_xy, metadata = load_data(DATA_DIR)

    print(f"\nMetadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    # Run experiments
    exp1 = experiment_1_basic_comparison(psi_truth, y_obs, obs_idx, R, L_xy, Ne=20, inflation=1.02, seed=20)

    results_phys, results_ml = experiment_2_ensemble_size(
        psi_truth, y_obs, obs_idx, R, L_xy,
        Ne_list_phys=[10, 20, 40],
        Ne_list_ml=[10, 20, 40, 80, 160],
        inflation=1.02, seed=20
    )

    exp3 = experiment_3_large_ml_vs_small_phys(psi_truth, y_obs, obs_idx, R, L_xy, Ne_phys=20, Ne_ml=100, inflation=1.02, seed=20)

    exp4 = experiment_4_spread_skill(psi_truth, y_obs, obs_idx, R, L_xy, Ne=20, inflation=1.02, seed=20)

    experiment_5_summary_bar(results_phys, results_ml)

    print("\n" + "=" * 70)
    print("All experiments complete!")
    print("=" * 70)
    print(f"\nGenerated 8 figures:")
    print(f"  fig1_rmse_time_series.png")
    print(f"  fig2_crps_time_series.png")
    print(f"  fig3_rmse_vs_ensemble_size.png")
    print(f"  fig4_crps_vs_ensemble_size.png")
    print(f"  fig5_large_ml_vs_small_phys_rmse.png")
    print(f"  fig6_large_ml_vs_small_phys_crps.png")
    print(f"  fig7_spread_skill.png")
    print(f"  fig8_summary_comparison.png")
    print(f"\nAll saved to: {IMAGE_DIR}")


if __name__ == "__main__":
    main()