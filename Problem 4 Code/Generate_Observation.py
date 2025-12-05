import os
import numpy as np


def main():
    # ---------------------------------------------------------
    # 1. 路径设置（假设在 Project 4 根目录下运行这个脚本）
    # ---------------------------------------------------------
    project_root = os.getcwd()  # 当前目录就是 Project 4
    data_dir = os.path.join(project_root, "Data")
    sol_dir = os.path.join(data_dir, "sol")
    obs_dir = os.path.join(data_dir, "Observation")

    os.makedirs(obs_dir, exist_ok=True)

    # x_truth 存放路径
    truth_path = os.path.join(sol_dir, "x_truth.npy")

    # ---------------------------------------------------------
    # 2. 加载 x_truth, 形状应该是 (40, T)
    # ---------------------------------------------------------
    x_truth = np.load(truth_path)  # (40, T)

    if x_truth.ndim != 2:
        raise ValueError(f"x_truth 维度不是 2D, 实际 shape = {x_truth.shape}")

    n_states, T = x_truth.shape
    if n_states != 40:
        raise ValueError(
            f"x_truth 第一维不是 40, 而是 {n_states}, 请确认数据是否正确。"
        )

    print(f"Loaded x_truth from {truth_path}")
    print(f"x_truth shape = {x_truth.shape} (states={n_states}, T={T})")

    # ---------------------------------------------------------
    # 3. 观测噪声设置（高斯噪声）
    # ---------------------------------------------------------
    rng = np.random.default_rng(2025)  # 固定随机种子，保证可复现

    obs_noise_std_full = 1.0    # full 观测噪声标准差
    obs_noise_std_sparse = 1.0  # sparse 观测噪声标准差

    # ---------------------------------------------------------
    # 4. Full observation: 所有 40 个变量都被观测
    #    y_full 形状: (40, T)
    # ---------------------------------------------------------
    noise_full = rng.normal(
        loc=0.0,
        scale=obs_noise_std_full,
        size=x_truth.shape
    )
    y_full = x_truth + noise_full

    # ---------------------------------------------------------
    # 5. Sparse observation: 只观测部分变量
    #    这里示例：观测索引 [0, 2, 4, ..., 38] 共 20 个维度
    # ---------------------------------------------------------
    obs_indices_sparse = np.arange(0, 40, 2)  # (20,)
    x_truth_sparse = x_truth[obs_indices_sparse, :]  # (20, T)

    noise_sparse = rng.normal(
        loc=0.0,
        scale=obs_noise_std_sparse,
        size=x_truth_sparse.shape
    )
    y_sparse = x_truth_sparse + noise_sparse

    # ---------------------------------------------------------
    # 6. 保存结果到 Data/Observation/
    # ---------------------------------------------------------
    y_full_path = os.path.join(obs_dir, "y_full.npy")
    y_sparse_path = os.path.join(obs_dir, "y_sparse.npy")
    idx_sparse_path = os.path.join(obs_dir, "obs_indices_sparse.npy")
    meta_path = os.path.join(obs_dir, "obs_metadata.npz")

    np.save(y_full_path, y_full)
    np.save(y_sparse_path, y_sparse)
    np.save(idx_sparse_path, obs_indices_sparse)

    # 可选：把噪声信息一起存起来，之后构造 R 用
    np.savez(
        meta_path,
        obs_noise_std_full=obs_noise_std_full,
        obs_noise_std_sparse=obs_noise_std_sparse,
        obs_indices_sparse=obs_indices_sparse,
    )

    print("\nSaved observations to:")
    print(f"  {y_full_path}")
    print(f"  {y_sparse_path}")
    print(f"  {idx_sparse_path}")
    print(f"  {meta_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
