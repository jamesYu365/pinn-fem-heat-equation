"""PINN 入口脚本（Case 1：无源项齐次 Dirichlet，正问题）。"""

import argparse
import yaml
import json
import shutil
import numpy as np
import torch
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pinn.model import PINN
from src.pinn.sampling import sample_collocation, sample_initial, sample_boundary
from src.pinn.train import train
from src.utils.exact_solution import case1_exact
from src.utils.metrics import relative_l2_error, max_absolute_error
from src.utils.visualization import plot_temperature_field, plot_error_field


def main():
    parser = argparse.ArgumentParser(description="PINN 求解二维热传导方程")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--name", type=str, default=None, help="自定义实验目录名")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 读取配置
    T_end = config["fem"]["T_end"]
    alpha = config["physics"]["alpha"]
    case = config["case"]
    pinn_cfg = config["pinn"]

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PINN 求解器 (Case {case}, 正问题) ===")
    print(f"设备: {device}")

    # 实验目录
    layers = pinn_cfg["layers"]
    arch = f"{len(layers)-2}x{layers[1]}"
    if args.name:
        run_dir = os.path.join("results", f"case{case}", "pinn", args.name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", f"case{case}", "pinn", f"{timestamp}_{arch}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"实验目录: {run_dir}")

    # 1. 采样配点
    n_col = pinn_cfg["num_collocation"]
    n_ic = pinn_cfg["num_ic"]
    n_bc = pinn_cfg["num_bc"]

    x_r, y_r, t_r = sample_collocation(n_col, T_end, device)
    x_ic, y_ic = sample_initial(n_ic, device)
    x_bc, y_bc, t_bc = sample_boundary(n_bc, T_end, device)

    print(f"配点: 域内 {n_col}, IC {n_ic}, BC {n_bc}")

    # 2. 构建模型
    model = PINN(layers=layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"网络参数量: {total_params}")
    print(f"网络结构: {layers}")

    # 3. 训练
    loss_weights = {
        "lambda_r": pinn_cfg["lambda_r"],
        "lambda_ic": pinn_cfg["lambda_ic"],
        "lambda_bc": pinn_cfg["lambda_bc"],
    }
    alpha_t = torch.tensor(alpha, device=device)

    loss_history, components_history = train(
        model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc,
        alpha_t, pinn_cfg["epochs"], pinn_cfg["lr"],
        loss_weights, log_every=pinn_cfg["log_every"], device=device
    )

    # 4. 在规则网格上推理
    nx_eval = 50
    x_lin = np.linspace(0, 1, nx_eval)
    y_lin = np.linspace(0, 1, nx_eval)
    xx, yy = np.meshgrid(x_lin, y_lin)
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    nodes = np.column_stack([x_flat, y_flat])

    # 在多个时间点评估
    eval_times = [0.0, T_end / 4, T_end / 2, T_end]
    l2_errors = []
    max_errors = []

    model.eval()
    with torch.no_grad():
        x_t = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(1).to(device)
        y_t = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1).to(device)
        for t_val in eval_times:
            t_t = torch.full_like(x_t, t_val)
            u_pred = model(x_t, y_t, t_t).cpu().numpy().flatten()

            u_exact = case1_exact(x_flat, y_flat, t_val, alpha)
            l2_err = relative_l2_error(u_pred, u_exact)
            max_err = max_absolute_error(u_pred, u_exact)
            l2_errors.append(l2_err)
            max_errors.append(max_err)
            print(f"  t={t_val:.3f}: L2={l2_err:.6e}, Max={max_err:.6e}")

    # 5. 保存结果
    summary = {
        "case": case,
        "method": "PINN",
        "mode": "forward",
        "alpha": alpha,
        "layers": layers,
        "total_params": total_params,
        "epochs": pinn_cfg["epochs"],
        "lr": pinn_cfg["lr"],
        "num_collocation": n_col,
        "device": str(device),
        "eval_times": eval_times,
        "l2_errors": [float(e) for e in l2_errors],
        "max_errors": [float(e) for e in max_errors],
        "final_loss": float(loss_history[-1]),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    shutil.copy2(args.config, os.path.join(run_dir, "config_snapshot.yaml"))

    # 6. 可视化
    # 最终时刻温度场
    with torch.no_grad():
        x_t = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(1).to(device)
        y_t = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1).to(device)
        t_t = torch.full_like(x_t, T_end)
        u_final = model(x_t, y_t, t_t).cpu().numpy().flatten()

    u_exact_final = case1_exact(x_flat, y_flat, T_end, alpha)

    plot_temperature_field(nodes, u_final, f"PINN 解 (t={T_end})", os.path.join(fig_dir, "pinn_temperature.png"))
    plot_temperature_field(nodes, u_exact_final, f"解析解 (t={T_end})", os.path.join(fig_dir, "exact_temperature.png"))
    plot_error_field(nodes, u_final, u_exact_final, f"误差分布 (t={T_end})", os.path.join(fig_dir, "error_field.png"))

    # 损失曲线
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.semilogy(loss_history, label="总损失")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("PINN 训练损失曲线")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(fig_dir, "loss_curve.png"), dpi=150)
    plt.close(fig)

    print(f"\n结果摘要: {run_dir}/summary.json")
    print(f"可视化图像: {fig_dir}/")


if __name__ == "__main__":
    main()
