#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate and Save All Project Plots
此脚本用于生成 Project 4 所需的所有分析图表，并保存到指定目录。

包含以下图表：
1. Trajectory Comparison.png (自由运行 vs 真值)
2. Global Error Growth.png (自由运行误差增长)
3. RMSE comparison.png (Baseline EnKF vs Free Run)
4. RSME(NN-Corrected).png (NN-EnKF Full vs Sparse)
5. State Trajectory Comparison.png (NN-EnKF 轨迹对比)
6. RMSE AD_EnKF.png (Offline NN vs AD-EnKF RMSE 对比)
7. Stat Trajectory AD_EnKF.png (AD-EnKF 轨迹对比)
"""

import os
import numpy as np
import pickle
import matplotlib.pyplot as plt

# =======================================================
# 1. 路径配置 (Configuration)
# =======================================================
# 项目根目录
PROJECT_ROOT = "/Users/huanghongxiao/Desktop/Uchicago/Inverse Problem/HW/acm270_projects-main/Project 4"

# 数据子目录
DATA_DIR = os.path.join(PROJECT_ROOT, "Data")
SOL_DIR = os.path.join(DATA_DIR, "sol")
ENKF_DIR = os.path.join(DATA_DIR, "EnKF")
NN_DIR = os.path.join(DATA_DIR, "NN_Correct_1")
AD_DIR = os.path.join(DATA_DIR, "EnKF_jax")
OBS_DIR = os.path.join(DATA_DIR, "Observation")

# 图片保存路径 (你指定的目标路径)
SAVE_DIR = os.path.join(DATA_DIR, "Image")
os.makedirs(SAVE_DIR, exist_ok=True)

print(f">>> 图片保存路径: {SAVE_DIR}")

# =======================================================
# 2. 数据加载 (Data Loading)
# =======================================================
print(">>> 正在加载数据...")

# 2.1 加载真值 (Truth)
try:
    x_truth = np.load(os.path.join(SOL_DIR, "x_truth.npy"))  # Shape: (40, 20001)
    print("  [Loaded] x_truth")
except Exception as e:
    print(f"  [Error] 无法加载 x_truth: {e}")
    exit()

# 2.2 加载自由运行误差 (Free Run Error)
try:
    with open(os.path.join(SOL_DIR, "sol_error.pkl"), "rb") as f:
        x_error = pickle.load(f)  # Shape: (40, 20001) or similar
    print("  [Loaded] sol_error.pkl")

    # 计算 Free Run 的 RMSE (Global RMSE over time)
    # RMSE = sqrt( mean(error^2) ) over state dimension
    rmse_free = np.sqrt(np.mean(x_error ** 2, axis=0))

    # 重构自由运行状态 (用于画轨迹)
    # x_free = x_truth + x_error
    x_free = x_truth + x_error
except Exception as e:
    print(f"  [Warning] 无法加载 sol_error.pkl: {e}")
    rmse_free = None
    x_free = None

# 2.3 加载同化时间索引 (Assim Indices)
try:
    assim_indices = np.load(os.path.join(OBS_DIR, "assim_indices.npy"))
    # 提取同化时刻的真值
    x_truth_assim = x_truth[:, assim_indices]  # (40, N_assim)
    N_state, N_assim = x_truth_assim.shape

    # 提取同化时刻的 Free Run RMSE (为了对比)
    if rmse_free is not None:
        rmse_free_assim = rmse_free[assim_indices]
    else:
        rmse_free_assim = None

    print(f"  [Info] Assimilation Steps: {N_assim}")
except Exception as e:
    print(f"  [Error] 无法加载 assim_indices: {e}")
    exit()

# 2.4 加载 Baseline EnKF (Full & Sparse)
x_a_base_full = None
x_a_base_sparse = None
try:
    # 尝试加载 Baseline 结果 (如果在 EnKF 文件夹下)
    # 注意：文件名取决于你之前的保存方式，这里假设是标准的 x_a_full.npy
    if os.path.exists(os.path.join(ENKF_DIR, "x_a_full.npy")):
        x_a_base_full = np.load(os.path.join(ENKF_DIR, "x_a_full.npy"))
        # 确保形状是 (40, N_assim)
        if x_a_base_full.shape[0] != 40: x_a_base_full = x_a_base_full.T

    if os.path.exists(os.path.join(ENKF_DIR, "x_a_sparse.npy")):
        x_a_base_sparse = np.load(os.path.join(ENKF_DIR, "x_a_sparse.npy"))
        if x_a_base_sparse.shape[0] != 40: x_a_base_sparse = x_a_base_sparse.T

    print("  [Loaded] Baseline EnKF data")
except Exception as e:
    print(f"  [Warning] Baseline EnKF 数据加载失败: {e}")

# 2.5 加载 Offline NN-EnKF (Full & Sparse)
x_a_nn_full = None
x_a_nn_sparse = None
try:
    x_a_nn_full = np.load(os.path.join(NN_DIR, "x_a_full_nn.npy"))
    x_a_nn_sparse = np.load(os.path.join(NN_DIR, "x_a_sparse_nn.npy"))
    if x_a_nn_full.shape[0] != 40: x_a_nn_full = x_a_nn_full.T
    if x_a_nn_sparse.shape[0] != 40: x_a_nn_sparse = x_a_nn_sparse.T
    print("  [Loaded] Offline NN-EnKF data")
except Exception as e:
    print(f"  [Warning] Offline NN-EnKF 数据加载失败: {e}")

# 2.6 加载 AD-EnKF (End-to-End)
x_a_ad = None
try:
    x_a_ad = np.load(os.path.join(AD_DIR, "x_a_ad_enkf.npy"))
    if x_a_ad.shape[0] != 40: x_a_ad = x_a_ad.T
    print("  [Loaded] AD-EnKF data")
except Exception as e:
    print(f"  [Warning] AD-EnKF 数据加载失败: {e}")


# =======================================================
# 3. 绘图函数定义 (Plotting Functions)
# =======================================================
def save_figure(filename, dpi=300):
    path = os.path.join(SAVE_DIR, filename)
    plt.savefig(path, dpi=dpi, bbox_inches='tight')
    print(f"  [Saved] {filename}")
    plt.close()


# -------------------------------------------------------
# Plot 1: Trajectory Comparison (Truth vs Free Run)
# -------------------------------------------------------
print(">>> 生成图表 1: Trajectory Comparison.png")
plt.figure(figsize=(12, 5))
time_axis = np.arange(20001) * 0.005  # DT=0.005
plot_len = 1000  # 展示前1000步

plt.plot(time_axis[:plot_len], x_truth[0, :plot_len], 'k-', lw=2, label='Truth')
if x_free is not None:
    plt.plot(time_axis[:plot_len], x_free[0, :plot_len], 'r--', lw=1.5, label='Free Run')

plt.title("Trajectory Comparison (First Coordinate)")
plt.xlabel("Time (MTU)")
plt.ylabel("$x_0$")
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
save_figure("Trajectory Comparison.png")

# -------------------------------------------------------
# Plot 2: Global Error Growth
# -------------------------------------------------------
print(">>> 生成图表 2: Global Error Growth.png")
if rmse_free is not None:
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, rmse_free, 'k-', lw=1)
    plt.title("Global Error Growth (Free Run RMSE)")
    plt.xlabel("Time (MTU)")
    plt.ylabel("Spatial RMSE")
    plt.grid(True, linestyle=':', alpha=0.6)
    save_figure("Global Error Growth.png")
else:
    print("  [Skip] 缺少 Free Run Error 数据")

# -------------------------------------------------------
# Plot 3: RMSE Comparison (Baseline vs Free Run)
# -------------------------------------------------------
print(">>> 生成图表 3: RMSE comparison.png")
plt.figure(figsize=(12, 5))
steps = np.arange(N_assim)

# Free Run (Assim Steps)
if rmse_free_assim is not None:
    plt.plot(steps, rmse_free_assim, 'k-', alpha=0.5, label=f'Free run (Mean={np.mean(rmse_free_assim):.2f})')

# Baseline EnKF Full
if x_a_base_full is not None:
    rmse_base_full = np.sqrt(np.mean((x_a_base_full - x_truth_assim) ** 2, axis=0))
    plt.plot(steps, rmse_base_full, 'b-', lw=1, label=f'EnKF Full (Mean={np.mean(rmse_base_full):.2f})')

# Baseline EnKF Sparse
if x_a_base_sparse is not None:
    rmse_base_sparse = np.sqrt(np.mean((x_a_base_sparse - x_truth_assim) ** 2, axis=0))
    plt.plot(steps, rmse_base_sparse, 'm-', lw=1, alpha=0.8,
             label=f'EnKF Sparse (Mean={np.mean(rmse_base_sparse):.2f})')

plt.title("RMSE Comparison: Baseline EnKF vs Free Run")
plt.xlabel("Assimilation Step Index")
plt.ylabel("Spatial RMSE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
save_figure("RMSE comparison.png")

# -------------------------------------------------------
# Plot 4: NN-EnKF RMSE (Full vs Sparse)
# -------------------------------------------------------
print(">>> 生成图表 4: RSME(NN-Corrected).png")
plt.figure(figsize=(12, 5))

if x_a_nn_full is not None and x_a_nn_sparse is not None:
    rmse_nn_full = np.sqrt(np.mean((x_a_nn_full - x_truth_assim) ** 2, axis=0))
    rmse_nn_sparse = np.sqrt(np.mean((x_a_nn_sparse - x_truth_assim) ** 2, axis=0))

    plt.plot(steps, rmse_nn_full, 'b-', alpha=0.8, lw=1, label=f'Full Obs (Mean={np.mean(rmse_nn_full):.2f})')
    plt.plot(steps, rmse_nn_sparse, 'r-', alpha=0.8, lw=1, label=f'Sparse Obs (Mean={np.mean(rmse_nn_sparse):.2f})')

    plt.title("RMSE over Assimilation Steps (NN-Corrected)")
    plt.xlabel("Assimilation Step Index")
    plt.ylabel("Spatial RMSE")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_figure("RSME(NN-Corrected).png")
else:
    print("  [Skip] 缺少 NN-EnKF 数据")

# -------------------------------------------------------
# Plot 5: State Trajectory Comparison (NN vs Truth)
# -------------------------------------------------------
print(">>> 生成图表 5: State Trajectory Comparison.png")
plt.figure(figsize=(12, 6))
plot_steps = 200  # 只画前200步

plt.plot(steps[:plot_steps], x_truth_assim[0, :plot_steps], 'k-', lw=2, label='Truth')

if x_a_nn_full is not None:
    plt.plot(steps[:plot_steps], x_a_nn_full[0, :plot_steps], 'b--', label='Analysis (Full NN)')

if x_a_nn_sparse is not None:
    plt.plot(steps[:plot_steps], x_a_nn_sparse[0, :plot_steps], 'r:', lw=2, label='Analysis (Sparse NN)')

plt.title("State Trajectory Comparison (Variable x_0) - First 200 Steps")
plt.xlabel("Assimilation Step Index")
plt.ylabel("State Value (x_0)")
plt.legend()
plt.grid(True, alpha=0.5)
save_figure("State Trajectory Comparison.png")

# -------------------------------------------------------
# Plot 6: RMSE AD-EnKF vs Offline NN
# -------------------------------------------------------
print(">>> 生成图表 6: RMSE AD_EnKF.png")
plt.figure(figsize=(12, 5))

# Plot Offline NN (as Baseline here)
if x_a_nn_full is not None:
    rmse_offline = np.sqrt(np.mean((x_a_nn_full - x_truth_assim) ** 2, axis=0))
    plt.plot(steps, rmse_offline, color='grey', linestyle='--', alpha=0.6, lw=1,
             label=f'Offline NN (Mean={np.mean(rmse_offline):.2f})')

# Plot AD-EnKF
if x_a_ad is not None:
    rmse_ad = np.sqrt(np.mean((x_a_ad - x_truth_assim) ** 2, axis=0))
    plt.plot(steps, rmse_ad, 'b-', lw=1.5, label=f'AD-EnKF (Mean={np.mean(rmse_ad):.2f})')

plt.title("RMSE Comparison: Offline Learning vs End-to-End AD-EnKF")
plt.xlabel("Assimilation Step Index")
plt.ylabel("Spatial RMSE")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
save_figure("RMSE AD_EnKF.png")

# -------------------------------------------------------
# Plot 7: Trajectory AD-EnKF
# -------------------------------------------------------
print(">>> 生成图表 7: Stat Trajectory AD_EnKF.png")
plt.figure(figsize=(12, 6))
plot_steps_traj = 300

plt.plot(steps[:plot_steps_traj], x_truth_assim[0, :plot_steps_traj], 'k-', lw=2.5, alpha=0.8, label='Truth')

if x_a_nn_full is not None:
    plt.plot(steps[:plot_steps_traj], x_a_nn_full[0, :plot_steps_traj], 'g--', lw=1.5, label='Offline NN')

if x_a_ad is not None:
    plt.plot(steps[:plot_steps_traj], x_a_ad[0, :plot_steps_traj], 'r:', lw=2.0, label='AD-EnKF')

plt.title("State Trajectory Comparison (Variable x_0) - First 300 Steps")
plt.xlabel("Assimilation Step")
plt.ylabel("State Value (x_0)")
plt.legend()
plt.grid(True, alpha=0.5)
save_figure("Stat Trajectory AD_EnKF.png")

print("\n>>> 所有图表生成完毕！")