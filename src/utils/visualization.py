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
    # contourf 不支持 vmin/vmax，需要转为显式 levels
    vmin = contourf_kwargs.pop("vmin", None)
    vmax = contourf_kwargs.pop("vmax", None)
    if vmin is not None and vmax is not None:
        levels_arr = np.linspace(vmin, vmax, levels + 1)
    else:
        levels_arr = levels
    if side * side == n and _is_structured_grid(nodes, side):
        x2d = nodes[:, 0].reshape(side, side)
        y2d = nodes[:, 1].reshape(side, side)
        v2d = values.reshape(side, side)
        return ax.contourf(x2d, y2d, v2d, levels=levels_arr, **contourf_kwargs)
    else:
        return ax.tricontourf(nodes[:, 0], nodes[:, 1], values, levels=levels,
                              vmin=vmin, vmax=vmax, **contourf_kwargs)


def _add_train_val_split(ax, T_train):
    """在时间轴子图上画训练/验证分界线。"""
    ax.axvline(T_train, color="gray", linestyle=":", linewidth=1.5,
               label=f"T_train={T_train:.2f}")
    ylim = ax.get_ylim()
    ymid = (ylim[0] + ylim[1]) / 2
    ax.text(T_train, ymid, " 训练|验证 ", fontsize=7,
            color="gray", va="center", ha="center",
            bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))


def plot_comparison_2x3(nodes, u_pred, u_exact, t_val, filename=None, method="PINN",
                        ts_data=None, cs_data=None, T_train=None, obs_data=None):
    """绘制对比图。

    正问题: 2×3 — Row 1: 空间场 | Row 2: 时间曲线+切面
    反问题: 3×3 — Row 3: 训练观测 | 验证观测 | 观测vs预测

    参数:
        nodes: (N, 2) 节点坐标
        ts_data: dict, keys: 'times', 'locations', 'u_pred', 'u_exact'
        cs_data: dict, keys: 'y', 'u_pred', 'u_exact'
        T_train: 训练时间上限，画竖线区分训练/验证域
        obs_data: dict, keys: 't_train', 'u_train' (5个代表点),
                  't_val', 'u_val' (5个代表点, 可选),
                  'ts_u_exact_train', 'ts_u_pred_train' (时间序列曲线),
                  'ts_u_exact_val', 'ts_u_pred_val',
                  'T_end', 'u_obs_all', 'u_pred_all' (完整数据, ObsVSPred用),
                  'u_obs_val_all', 'u_pred_val_all' (可选)
    """
    has_row2 = ts_data is not None and cs_data is not None
    has_row3 = obs_data is not None
    nrows = 3 if has_row3 else (2 if has_row2 else 1)
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
            _add_train_val_split(axes[1, 0], T_train)
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

    # Row 3: 观测数据（仅反问题）
    if has_row3:
        n_ts = 100
        ts_t = np.linspace(0, obs_data["T_end"], n_ts)
        obs_colors = ["#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
        n_train_show = len(obs_data["ts_u_exact_train"])
        n_val_show = len(obs_data.get("ts_u_exact_val") or [])
        has_val_obs = n_val_show > 0

        # 训练观测拟合
        ax = axes[2, 0]
        for i in range(n_train_show):
            c = obs_colors[i % len(obs_colors)]
            ax.plot(ts_t, obs_data["ts_u_exact_train"][i], color=c, linestyle="-",
                    alpha=0.7, linewidth=0.8)
            ax.plot(ts_t, obs_data["ts_u_pred_train"][i], color=c, linestyle="--",
                    alpha=0.7, linewidth=0.8)
            ot, ou = obs_data["obs_t_train"][i], obs_data["obs_u_train"][i]
            if len(ot) > 0:
                ax.plot(ot, ou, "o", color=c, markersize=5, markeredgewidth=1.5,
                        markerfacecolor="none")
        if T_train is not None:
            _add_train_val_split(ax, T_train)
        ax.set_xlabel("t")
        ax.set_ylabel("u")
        ax.set_title(f"训练观测拟合 ({n_train_show}传感器)")
        ax.grid(True, alpha=0.3)

        # 验证观测拟合
        ax = axes[2, 1]
        if has_val_obs:
            n_val_with_data = sum(1 for ot in obs_data["obs_t_val"] if len(ot) > 0)
            for i in range(n_val_show):
                c = obs_colors[i % len(obs_colors)]
                ax.plot(ts_t, obs_data["ts_u_exact_val"][i], color=c, linestyle="-",
                        alpha=0.7, linewidth=0.8)
                ax.plot(ts_t, obs_data["ts_u_pred_val"][i], color=c, linestyle="--",
                        alpha=0.7, linewidth=0.8)
                ot, ou = obs_data["obs_t_val"][i], obs_data["obs_u_val"][i]
                if len(ot) > 0:
                    ax.plot(ot, ou, "o", color=c, markersize=5, markeredgewidth=1.5,
                            markerfacecolor="none")
        else:
            n_val_with_data = 0
            ax.text(0.5, 0.5, "无验证观测数据", transform=ax.transAxes,
                    ha="center", va="center", fontsize=10, color="gray")
        if T_train is not None:
            _add_train_val_split(ax, T_train)
        ax.set_xlabel("t")
        ax.set_ylabel("u")
        ax.set_title(f"验证观测拟合 ({n_val_with_data}传感器)")
        ax.grid(True, alpha=0.3)

        # 观测 vs 预测散点图
        ax = axes[2, 2]
        ax.scatter(obs_data["u_obs_all"], obs_data["u_pred_all"],
                   s=10, alpha=0.5, c="darkorange", label="训练")
        if obs_data.get("u_obs_val_all") is not None:
            ax.scatter(obs_data["u_obs_val_all"], obs_data["u_pred_val_all"],
                       s=10, alpha=0.5, c="purple", marker="x", label="验证")
        all_u = np.concatenate([obs_data["u_obs_all"], obs_data["u_pred_all"]])
        if obs_data.get("u_obs_val_all") is not None:
            all_u = np.concatenate([all_u, obs_data["u_obs_val_all"],
                                    obs_data["u_pred_val_all"]])
        umin, umax = all_u.min(), all_u.max()
        margin = (umax - umin) * 0.05
        ax.plot([umin - margin, umax + margin], [umin - margin, umax + margin],
                "k--", linewidth=1, label="y=x")
        ax.set_xlabel("u_obs")
        ax.set_ylabel("u_pred")
        ax.set_title("观测 vs 预测")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
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
        fig.savefig(filename, dpi=150, bbox_inches="tight")
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
        fig.savefig(filename, dpi=150, bbox_inches="tight")
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
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def _draw_scatter(ax, x_r, y_r, x_bc, y_bc, x_ic, y_ic, x_obs, y_obs, title):
    """在一个子图上绘制采样点（配点+BC+IC+观测）。"""
    if x_r is not None and y_r is not None:
        ax.scatter(np.asarray(x_r).flatten(), np.asarray(y_r).flatten(),
                   s=3, alpha=0.4, c="royalblue", label="域内配点")
    if x_bc is not None and y_bc is not None:
        ax.scatter(np.asarray(x_bc).flatten(), np.asarray(y_bc).flatten(),
                   s=8, alpha=0.6, c="green", label="BC 配点", marker="s")
    if x_ic is not None and y_ic is not None:
        ax.scatter(np.asarray(x_ic).flatten(), np.asarray(y_ic).flatten(),
                   s=3, alpha=0.4, c="darkorange", label="IC 配点")
    if x_obs is not None and y_obs is not None:
        ax.scatter(np.asarray(x_obs).flatten(), np.asarray(y_obs).flatten(),
                   s=20, alpha=0.8, c="red", label="观测数据", marker="*")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.legend(fontsize=10, bbox_to_anchor=(1.01, 1), loc="upper left",
               borderaxespad=0)
    ax.set_aspect("equal")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _setup_sampling_ax(ax, title, xlim_margin=0.03):
    """配置采样点子图的通用样式。"""
    ax.set_title(title, loc="left", fontsize=11)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_xlim(-xlim_margin, 1 + xlim_margin)
    ax.set_ylim(-xlim_margin, 1 + xlim_margin)


def plot_sampling_points(train_data=None, val_data=None,
                         x_obs=None, y_obs=None,
                         x_obs_val=None, y_obs_val=None,
                         filename=None, title_prefix=""):
    """绘制采样点分布图。

    正问题：1×3 — PDE | IC | BC
    反问题：1×4 — PDE | IC | BC | 观测
    训练点用实心标记，验证点用 × 标记（不同颜色）。
    """
    has_val = val_data is not None
    has_obs = x_obs is not None and y_obs is not None
    n_plots = 4 if has_obs else 3
    fig_w = 5.5 * n_plots
    fig, axes = plt.subplots(1, n_plots, figsize=(fig_w, 5.5))
    leg_kwargs = dict(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, 1.08),
                      ncol=2, framealpha=0.8)

    # PDE 配点
    ax = axes[0]
    if train_data.get("x_r") is not None:
        ax.scatter(np.asarray(train_data["x_r"]).flatten(),
                   np.asarray(train_data["y_r"]).flatten(),
                   s=4, alpha=0.5, c="darkorange", label="训练")
    if has_val and val_data.get("x_r") is not None:
        ax.scatter(np.asarray(val_data["x_r"]).flatten(),
                   np.asarray(val_data["y_r"]).flatten(),
                   s=8, alpha=0.6, c="purple", label="验证", marker="x")
    _setup_sampling_ax(ax, f"{title_prefix}PDE残差配点")
    ax.legend(**leg_kwargs)

    # IC 配点
    ax = axes[1]
    if train_data.get("x_ic") is not None:
        ax.scatter(np.asarray(train_data["x_ic"]).flatten(),
                   np.asarray(train_data["y_ic"]).flatten(),
                   s=6, alpha=0.5, c="darkorange", label="训练")
    if has_val and val_data.get("x_ic") is not None:
        ax.scatter(np.asarray(val_data["x_ic"]).flatten(),
                   np.asarray(val_data["y_ic"]).flatten(),
                   s=8, alpha=0.6, c="purple", label="验证", marker="x")
    _setup_sampling_ax(ax, f"{title_prefix}IC 配点")
    ax.legend(**leg_kwargs)

    # BC 配点
    ax = axes[2]
    if train_data.get("x_bc") is not None:
        ax.scatter(np.asarray(train_data["x_bc"]).flatten(),
                   np.asarray(train_data["y_bc"]).flatten(),
                   s=10, alpha=0.6, c="darkorange", label="训练", marker="s")
    if has_val and val_data.get("x_bc") is not None:
        ax.scatter(np.asarray(val_data["x_bc"]).flatten(),
                   np.asarray(val_data["y_bc"]).flatten(),
                   s=10, alpha=0.6, c="purple", label="验证", marker="x")
    _setup_sampling_ax(ax, f"{title_prefix}BC 配点")
    ax.legend(**leg_kwargs)

    # 观测数据（仅反问题）
    if has_obs:
        ax = axes[3]
        ax.scatter(np.asarray(x_obs).flatten(), np.asarray(y_obs).flatten(),
                   s=30, alpha=0.8, c="darkorange", label="训练", marker="*")
        if x_obs_val is not None and y_obs_val is not None:
            ax.scatter(np.asarray(x_obs_val).flatten(),
                       np.asarray(y_obs_val).flatten(),
                       s=15, alpha=0.6, c="purple", label="验证", marker="x")
        _setup_sampling_ax(ax, f"{title_prefix}观测数据")
        ax.legend(**leg_kwargs)

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def plot_inverse_training(loss_history, components_history, alpha_history,
                          true_alpha, filename=None):
    """绘制反问题训练过程（2×1）：loss 曲线 + α 学习曲线。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

    epochs = range(1, len(loss_history) + 1)

    # 上图：loss
    pde_losses = [c["pde"] for c in components_history]
    ic_losses = [c["ic"] for c in components_history]
    bc_losses = [c["bc"] for c in components_history]
    data_losses = [c.get("data", 0) for c in components_history]

    ax1.semilogy(epochs, loss_history, label="总损失", color="black", linewidth=1.5)
    ax1.semilogy(epochs, pde_losses, label="PDE", alpha=0.8)
    ax1.semilogy(epochs, ic_losses, label="IC", alpha=0.8)
    ax1.semilogy(epochs, bc_losses, label="BC", alpha=0.8)
    if any(d > 0 for d in data_losses):
        ax1.semilogy(epochs, data_losses, label="Data", alpha=0.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("PINN 反问题训练损失")
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)

    # 下图：α
    ax2.plot(epochs, alpha_history, label="学习值 α", linewidth=1.5)
    ax2.axhline(true_alpha, color="red", linestyle="--", linewidth=1.5,
                label=f"真实值 α={true_alpha}")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("α")
    ax2.set_title("α 学习曲线")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
