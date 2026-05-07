import os
import logging
import numpy as np
import matplotlib

matplotlib.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["axes.unicode_minus"] = False

import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)


def plot_temperature_field(nodes, U, title, filename=None):
    """绘制温度场 2D 色彩图。"""
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.tricontourf(nodes[:, 0], nodes[:, 1], U, levels=50, cmap="hot")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="u")
    plt.tight_layout()
    if filename:
        os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
        fig.savefig(filename, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_error_field(nodes, u_h, u_exact, title, filename=None):
    """绘制误差分布图。"""
    error = np.abs(u_h - u_exact)
    fig, ax = plt.subplots(figsize=(6, 5))
    sc = ax.tricontourf(nodes[:, 0], nodes[:, 1], error, levels=50, cmap="viridis")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="|u_h - u_exact|")
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
