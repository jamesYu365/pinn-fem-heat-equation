import os
import logging
import numpy as np
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def _is_structured_grid(nodes, side):
    """验证节点是否构成 side×side 的规则网格。"""
    x = nodes[:, 0]
    y = nodes[:, 1]
    x_sorted = np.sort(np.unique(x))
    y_sorted = np.sort(np.unique(y))
    if len(x_sorted) != side or len(y_sorted) != side:
        return False
    x_uniform = np.allclose(np.diff(x_sorted), np.diff(x_sorted)[0])
    y_uniform = np.allclose(np.diff(y_sorted), np.diff(y_sorted)[0])
    return x_uniform and y_uniform


def _fill_field(ax, nodes, values, levels=50, **contourf_kwargs):
    """根据数据类型选择 contourf 或 tricontourf 绘制标量场。

    当节点构成规则网格时使用 contourf（避免 Delaunay 三角化伪影），
    否则回退到 tricontourf。
    """
    n = len(values)
    side = int(np.sqrt(n))
    if side * side == n and _is_structured_grid(nodes, side):
        x2d = nodes[:, 0].reshape(side, side)
        y2d = nodes[:, 1].reshape(side, side)
        v2d = values.reshape(side, side)
        return ax.contourf(x2d, y2d, v2d, levels=levels, **contourf_kwargs)
    else:
        return ax.tricontourf(nodes[:, 0], nodes[:, 1], values, levels=levels,
                              **contourf_kwargs)


def plot_comparison_2x3(nodes, u_pred, u_exact, t_val, filename=None, method="PINN",
                        ts_data=None, cs_data=None, T_train=None):
    """绘制 2×3 对比图。

    Row 1: 解析解 | 数值解 | 误差分布
    Row 2 (可选): 时间曲线 | x=0.5 切面 | 误差切面

    参数:
        nodes: (N, 2) 节点坐标，支持规则网格和非结构网格
        ts_data: dict, keys: 'times', 'locations', 'u_pred', 'u_exact'
        cs_data: dict, keys: 'y', 'u_pred', 'u_exact'
        T_train: 训练时间上限，在时间曲线图中画竖线区分训练/外推域
    """
    has_row2 = ts_data is not None and cs_data is not None
    nrows = 2 if has_row2 else 1
    fig, axes = plt.subplots(nrows, 3, figsize=(16, 4.5 * nrows))
    if nrows == 1:
        axes = axes.reshape(1, -1)

    # 统一 colorbar 范围
    u_min = min(u_exact.min(), u_pred.min())
    u_max = max(u_exact.max(), u_pred.max())

    # Row 1: 空间场
    sc0 = _fill_field(axes[0, 0], nodes, u_exact, levels=50,
                      cmap="hot", vmin=u_min, vmax=u_max)
    axes[0, 0].set_title(f"解析解 (t={t_val})")
    axes[0, 0].set_xlabel("x")
    axes[0, 0].set_ylabel("y")
    axes[0, 0].set_aspect("equal")
    fig.colorbar(sc0, ax=axes[0, 0], label="u")

    sc1 = _fill_field(axes[0, 1], nodes, u_pred, levels=50,
                      cmap="hot", vmin=u_min, vmax=u_max)
    axes[0, 1].set_title(f"{method} 解 (t={t_val})")
    axes[0, 1].set_xlabel("x")
    axes[0, 1].set_ylabel("y")
    axes[0, 1].set_aspect("equal")
    fig.colorbar(sc1, ax=axes[0, 1], label="u")

    error = np.abs(u_pred - u_exact)
    sc2 = _fill_field(axes[0, 2], nodes, error, levels=50, cmap="viridis")
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
            # 标记监测点位置
            axes[0, 0].plot(loc[0], loc[1], "x", color=colors[i], markersize=8,
                            markeredgewidth=2)
        # 训练/外推分界线
        if T_train is not None:
            axes[1, 0].axvline(T_train, color="gray", linestyle=":", linewidth=1.5,
                                label=f"T_train={T_train:.2f}")
            ylim = axes[1, 0].get_ylim()
            ymid = (ylim[0] + ylim[1]) / 2
            axes[1, 0].text(T_train, ymid, " 训练|外推 ", fontsize=7,
                            color="gray", va="center", ha="center",
                            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))
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


def plot_alpha_learning(alpha_history, true_alpha, filename=None):
    """绘制 α 学习曲线。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = range(1, len(alpha_history) + 1)
    ax.plot(epochs, alpha_history, label="学习值 α", linewidth=1.5)
    ax.axhline(true_alpha, color="red", linestyle="--", linewidth=1.5,
               label=f"真实值 α={true_alpha}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("α")
    ax.set_title("α 学习曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()
