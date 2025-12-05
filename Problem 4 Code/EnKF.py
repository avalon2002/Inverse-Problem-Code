import os
import numpy as np
from utils import lorenz96   # <- 用你的 single-scale Lorenz96


# ============================================================
# 1. 利用 utils.lorenz96 做 ensemble 积分
# ============================================================

def rk4_step_ensemble(X, dt, F=8.0):
    """
    对 ensemble 做一步 RK4:
    X: (K, N_ens)
    这里假设 utils.lorenz96(t, x, F=8.0, ...) 接受 x.shape = (K,)
    """
    K, N_ens = X.shape
    X_new = np.empty_like(X)

    for j in range(N_ens):
        x = X[:, j]

        k1 = lorenz96(0.0, x, F=F)
        k2 = lorenz96(0.0, x + 0.5 * dt * k1, F=F)
        k3 = lorenz96(0.0, x + 0.5 * dt * k2, F=F)
        k4 = lorenz96(0.0, x + dt * k3, F=F)

        X_new[:, j] = x + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)

    return X_new


def forecast_ensemble(X0, dt=0.005, n_steps=20, F=8.0):
    """
    从当前 ensemble X0 积分 n_steps 步，返回 forecast 的 ensemble。
    X0: (K, N_ens)
    """
    X = X0.copy()
    for _ in range(n_steps):
        X = rk4_step_ensemble(X, dt, F)
    return X


# ============================================================
# 2. EnKF 更新
# ============================================================

def enkf_analysis_step(
    X_f,        # forecast ensemble, (K, N_ens)
    y_k,        # observation at this time, (m,)
    H,          # obs operator, (m, K)
    R,          # obs covariance, (m, m)
    rng=None,
    inflation=1.02,
    stochastic=True,
):
    """
    标准 EnKF 分析步 (perturbed observations 版本)
    """
    if rng is None:
        rng = np.random.default_rng()

    K, N_ens = X_f.shape
    m = H.shape[0]

    # 1) forecast mean & perturbation
    x_f_mean = np.mean(X_f, axis=1, keepdims=True)  # (K, 1)
    X_f_pert = X_f - x_f_mean                       # (K, N_ens)

    # 2) 观测空间中的 forecast
    Y_f = H @ X_f                                   # (m, N_ens)
    y_f_mean = np.mean(Y_f, axis=1, keepdims=True) # (m, 1)
    Y_f_pert = Y_f - y_f_mean                       # (m, N_ens)

    # 3) 协方差 & Kalman 增益
    Pf_ht = (X_f_pert @ Y_f_pert.T) / (N_ens - 1)   # (K, m)
    S = (Y_f_pert @ Y_f_pert.T) / (N_ens - 1) + R   # (m, m)
    K_gain = Pf_ht @ np.linalg.inv(S)               # (K, m)

    # 4) 创新项
    y_k = y_k.reshape(-1)                           # (m,)

    if stochastic:
        obs_pert = rng.multivariate_normal(
            mean=y_k,
            cov=R,
            size=N_ens
        ).T                                         # (m, N_ens)
        innovation = obs_pert - Y_f                 # (m, N_ens)
    else:
        innovation = (y_k[:, None] - Y_f)           # (m, N_ens)

    # 5) 更新
    X_a = X_f + K_gain @ innovation                 # (K, N_ens)

    # 6) inflation
    x_a_mean = np.mean(X_a, axis=1, keepdims=True)
    X_a_pert = X_a - x_a_mean
    X_a = x_a_mean + inflation * X_a_pert

    return X_a, x_f_mean[:, 0], x_a_mean[:, 0]


def enkf_run(
    y,            # obs (m, T)
    H, R,         # obs model
    X0,           # initial ensemble (40, N_ens)
    dt=0.005,
    assim_step=20,
    F=8.0,
    inflation=1.02,
    rng=None,
    stochastic=True,
):
    """
    完整的 EnKF 循环。
    返回:
      x_f_means: (K, N_assim)
      x_a_means: (K, N_assim)
    """
    if rng is None:
        rng = np.random.default_rng()

    m, T = y.shape
    K, N_ens = X0.shape

    assim_indices = np.arange(0, T, assim_step)
    N_assim = len(assim_indices)

    x_f_means = np.zeros((K, N_assim))
    x_a_means = np.zeros((K, N_assim))

    X_ens = X0.copy()

    for k, t_idx in enumerate(assim_indices):
        # forecast
        if k == 0:
            X_f = X_ens
        else:
            X_f = forecast_ensemble(
                X_ens,
                dt=dt,
                n_steps=assim_step,
                F=F
            )

        # analysis
        y_k = y[:, t_idx]
        X_a, x_f_mean_k, x_a_mean_k = enkf_analysis_step(
            X_f, y_k, H, R,
            rng=rng,
            inflation=inflation,
            stochastic=stochastic,
        )

        x_f_means[:, k] = x_f_mean_k
        x_a_means[:, k] = x_a_mean_k

        X_ens = X_a

    return x_f_means, x_a_means, assim_indices


# ============================================================
# 3. 主函数：加载数据 + 跑 full / sparse EnKF
# ============================================================

def main():
    # -------- 固定绝对路径 --------
    project_root = "/Users/huanghongxiao/Desktop/Uchicago/Inverse Problem/HW/acm270_projects-main/Project 4"
    data_dir = os.path.join(project_root, "Data")
    sol_dir = os.path.join(data_dir, "sol")
    obs_dir = os.path.join(data_dir, "Observation")
    enkf_dir = os.path.join(data_dir, "EnKF")

    os.makedirs(enkf_dir, exist_ok=True)

    # -------- 真值 --------
    x_truth_path = os.path.join(sol_dir, "x_truth.npy")
    x_truth = np.load(x_truth_path)  # (40, T)
    K, T = x_truth.shape
    print(f"Loaded x_truth: {x_truth.shape}")

    # -------- 观测 --------
    y_full = np.load(os.path.join(obs_dir, "y_full.npy"))       # (40, T)
    y_sparse = np.load(os.path.join(obs_dir, "y_sparse.npy"))   # (m, T)
    print(f"Loaded y_full:   {y_full.shape}")
    print(f"Loaded y_sparse: {y_sparse.shape}")

    # sparse 观测索引
    idx_path = os.path.join(obs_dir, "obs_indices_sparse.npy")
    if os.path.exists(idx_path):
        obs_indices_sparse = np.load(idx_path)
    else:
        obs_indices_sparse = np.arange(0, 40, 2)
    print(f"obs_indices_sparse = {obs_indices_sparse}")

    # -------- 观测算子 H & 噪声协方差 R --------
    # full
    H_full = np.eye(40)
    obs_noise_std_full = 1.0
    R_full = (obs_noise_std_full ** 2) * np.eye(40)

    # sparse
    m_sparse = obs_indices_sparse.size
    H_sparse = np.zeros((m_sparse, 40))
    H_sparse[np.arange(m_sparse), obs_indices_sparse] = 1.0
    obs_noise_std_sparse = 1.0
    R_sparse = (obs_noise_std_sparse ** 2) * np.eye(m_sparse)

    # -------- 初始 ensemble --------
    rng = np.random.default_rng(123)
    N_ens = 50

    x0_truth = x_truth[:, 0]
    X0_full = x0_truth[:, None] + rng.normal(
        loc=0.0, scale=1.0, size=(40, N_ens)
    )
    X0_sparse = X0_full.copy()

    # -------- EnKF 参数 --------
    dt = 0.005
    assim_step = 20
    F = 8.0
    inflation = 1.02

    # =======================================================
    # 3.1 Full observation EnKF
    # =======================================================
    print("\n=== Running EnKF (FULL observation) ===")
    x_f_full, x_a_full, assim_indices = enkf_run(
        y_full,
        H_full, R_full,
        X0_full,
        dt=dt,
        assim_step=assim_step,
        F=F,
        inflation=1.15,
        rng=rng,
        stochastic=True,
    )

    # =======================================================
    # 3.2 Sparse observation EnKF
    # =======================================================
    print("\n=== Running EnKF (SPARSE observation) ===")
    x_f_sparse, x_a_sparse, _ = enkf_run(
        y_sparse,
        H_sparse, R_sparse,
        X0_sparse,
        dt=dt,
        assim_step=assim_step,
        F=F,
        inflation=1.15,
        rng=rng,
        stochastic=True,
    )

    # -------- 把所有 EnKF 相关数据保存到 Data/EnKF --------
    np.save(os.path.join(enkf_dir, "x_f_full.npy"),   x_f_full)
    np.save(os.path.join(enkf_dir, "x_a_full.npy"),   x_a_full)
    np.save(os.path.join(enkf_dir, "x_f_sparse.npy"), x_f_sparse)
    np.save(os.path.join(enkf_dir, "x_a_sparse.npy"), x_a_sparse)
    np.save(os.path.join(enkf_dir, "assim_indices.npy"), assim_indices)

    # 也存一个 meta 信息，方便之后用
    np.savez(
        os.path.join(enkf_dir, "enkf_metadata.npz"),
        dt=dt,
        assim_step=assim_step,
        F=F,
        inflation=inflation,
        N_ens=N_ens,
        obs_noise_std_full=obs_noise_std_full,
        obs_noise_std_sparse=obs_noise_std_sparse,
        obs_indices_sparse=obs_indices_sparse,
    )

    print("\nSaved EnKF results to:")
    print(f"  {os.path.join(enkf_dir, 'x_f_full.npy')}")
    print(f"  {os.path.join(enkf_dir, 'x_a_full.npy')}")
    print(f"  {os.path.join(enkf_dir, 'x_f_sparse.npy')}")
    print(f"  {os.path.join(enkf_dir, 'x_a_sparse.npy')}")
    print(f"  {os.path.join(enkf_dir, 'assim_indices.npy')}")
    print(f"  {os.path.join(enkf_dir, 'enkf_metadata.npz')}")


if __name__ == "__main__":
    main()
