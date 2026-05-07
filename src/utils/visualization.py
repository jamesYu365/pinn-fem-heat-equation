import os
import logging
import numpy as np
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def plot_comparison_2x3(nodes, u_pred, u_exact, t_val, filename=None, method="PINN",
                        collocation_points=None, ts_data=None, cs_data=None):
    """绘制 2×3 对比图。

    Row 1: 解析解 | 数值解 | 误差分布
    Row 2 (可选): 时间曲线 | x=0.5 切面 | 误差切面

    参数:
        collocation_points: (N,2) 域内配点坐标，叠加在数值解子图上
        ts_data: dict, keys: 'times', 'locations', 'u_pred', 'u_exact'
        cs_data: dict, keys: 'y', 'u_pred', 'u_exact'
    """
    has_row2 = ts_data is not None and cs_data is not None
    nrows = 2 if has_row2 else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # Row 1: 空间场
    sc0 = axes[0, 0].tricontourf(nodes[:, 0], nodes[:, 1], u_exact, levels=50, cmap="hot")
    axes[0, 0].set_title(f"解析解 (t={t_val})")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(sc0, ax=axes[0, 0], label="u")

    sc1 = axes[0, 1].tricontourf(nodes[:, 0], nodes[:, 1], u_pred, levels=50, cmap="hot")
    axes[0, 1].set_title(f"{method} 解 (t={t_val})")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(sc1, ax=axes[0, 1], label="u")

    if collocation_points is not None:
        axes[0, 1].scatter(collocation_points[:, 0], collocation_points[:, 1],
                           s=2, c="cyan", alpha=0.4)

    error = np.abs(u_pred - u_exact)
    sc2 = axes[0, 2].tricontourf(nodes[:, 0], nodes[:, 1], error, levels=50, cmap="viridis")
    axes[0, 2].set_title(f"误差分布 (t={t_val})")
    axes[0, 2].set_xlabel("x")
    axes[0, 2].set_ylabel("y")
    axes[0, 2].set_aspect("equal")
    fig.colorbar(sc2, ax=axes[0, 2], label="|u_pred - u_exact|")

    # Row 2: 时间曲线 + 切面
    if has_row2:
        colors = ["#1f77b4", "#2ca02c", "#d62728"]
        times = ts_data["times"]
        for i, (loc, u_p, u_e) in enumerate(
            zip(ts_data["locations"], ts_data["u_pred"], ts_data["u_exact"])
        ):
            tag = f"({loc[0]:.2f}, {loc[1]:.2f})"
            axes[1, 0].plot(times, u_e, color=colors[i], linestyle="-", alpha=0.8,
                            label=f"解析 {tag}")
            axes[1, 0].plot(times, u_p, color=colors[i], linestyle="--", alpha=0.8,
                            label=f"{method} {tag}")
        axes[1, 0].set_xlabel("t")
        axes[1, 0].set_ylabel("u")
        axes[1, 0].set_title("时间维度变化")
        axes[1, 0].legend(fontsize=6, ncol=2)
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(cs_data["y"], cs_data["u_exact"], "b-", label="解析解")
        axes[1, 1].plot(cs_data["y"], cs_data["u_pred"], "r--", label=f"{method} 解")
        axes[1, 1].set_xlabel("y")
        axes[1, 1].set_ylabel("u")
        axes[1, 1].set_title(f"x=0.5 切面 (t={t_val})")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        cs_error = np.abs(cs_data["u_pred"] - cs_data["u_exact"])
        axes[1, 2].plot(cs_data["y"], cs_error, "k-")
        axes[1, 2].set_xlabel("y")
        axes[1, 2].set_ylabel("|u_pred - u_exact|")
        axes[1, 2].set_title(f"x=0.5 误差切面 (t={t_val})")
        axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_loss_with_components(loss_history, components_history, filename=None,
                             val_history=None):
    """绘制总损失和各分量损失曲线。"""
    fig, ax = plt.subplots(figsize=(8, 5))
    epochs = range(1, len(loss_history) + 1)

    pde_losses = [c["pde"] for c in components_history]
    ic_losses = [c["ic"] for c in components_history]
    bc_losses = [c["bc"] for c in components_history]

    ax.semilogy(epochs, loss_history, label="训练损失", color="black", linewidth=1.5)
    if val_history and len(val_history) == len(loss_history):
        ax.semilogy(epochs, val_history, label="验证损失", color="red",
                    linewidth=1.5, linestyle="--")
    ax.semilogy(epochs, pde_losses, label="PDE 残差", alpha=0.8)
    ax.semilogy(epochs, ic_losses, label="初始条件", alpha=0.8)
    ax.semilogy(epochs, bc_losses, label="边界条件", alpha=0.8)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN 训练损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_error_over_time(times, l2_errors, max_errors, filename=None):
    """绘制误差随时间变化曲线。"""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(times, l2_errors, label="相对 L2 误差")
    ax.semilogy(times, max_errors, label="最大绝对误差")
    ax.set_xlabel("t")
    ax.set_ylabel("误差")
    ax.set_title("FEM 误差随时间变化")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()
