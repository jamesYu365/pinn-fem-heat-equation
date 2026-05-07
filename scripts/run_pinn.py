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
from src.utils.visualization import plot_comparison_2x3, plot_loss_with_components
from src.utils.seed import set_seed

# 监测点坐标
MONITOR_LOCS = [(0.25, 0.25), (0.25, 0.50), (0.50, 0.50)]


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

    set_seed(config.get("seed", 42))

    # 时间泛化：训练只在 [0, T_train]，外推 (T_train, T_end]
    val_time_ratio = pinn_cfg.get("val_time_ratio", 1.0)
    T_train = val_time_ratio * T_end
    time_generalization = val_time_ratio < 1.0

    # 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PINN 求解器 (Case {case}, 正问题) ===")
    print(f"设备: {device}")
    if time_generalization:
        print(f"时间泛化: 训练 t∈[0, {T_train:.3f}], 外推 t∈({T_train:.3f}, {T_end:.3f}]")

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

    # ---- 1. 训练集采样（t ∈ [0, T_train]） ----
    n_col = pinn_cfg["num_collocation"]
    n_ic = pinn_cfg["num_ic"]
    n_bc = pinn_cfg["num_bc"]

    train_x_r, train_y_r, train_t_r = sample_collocation(n_col, T_train, device)
    train_x_ic, train_y_ic = sample_initial(n_ic, device)
    train_x_bc, train_y_bc, train_t_bc = sample_boundary(n_bc, T_train, device)

    print(f"训练配点: 域内 {n_col}, IC {n_ic}, BC {n_bc}, T_train={T_train:.3f}")

    # ---- 2. 验证集采样（随机配点，与训练同域 t ∈ [0, T_train]） ----
    n_val_col = pinn_cfg.get("num_val_collocation", 0)
    n_val_ic = pinn_cfg.get("num_val_ic", 0)
    n_val_bc = pinn_cfg.get("num_val_bc", 0)
    val_data = None
    if n_val_col > 0 or n_val_ic > 0 or n_val_bc > 0:
        val_data = {
            "x_r": torch.rand(n_val_col, 1, device=device),
            "y_r": torch.rand(n_val_col, 1, device=device),
            "t_r": torch.rand(n_val_col, 1, device=device) * T_train,
            "x_ic": torch.rand(n_val_ic, 1, device=device),
            "y_ic": torch.rand(n_val_ic, 1, device=device),
        }
        vx_bc, vy_bc, vt_bc = sample_boundary(n_val_bc, T_train, device)
        val_data["x_bc"] = vx_bc
        val_data["y_bc"] = vy_bc
        val_data["t_bc"] = vt_bc
        print(f"验证配点: 域内 {n_val_col}, IC {n_val_ic}, BC {n_val_bc}")

    # ---- 3. 测试网格 ----
    nx_eval = 50
    x_lin = np.linspace(0, 1, nx_eval)
    y_lin = np.linspace(0, 1, nx_eval)
    xx, yy = np.meshgrid(x_lin, y_lin)
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    nodes = np.column_stack([x_flat, y_flat])

    # 测试评估时间点：区分训练域和外推域
    if time_generalization:
        eval_times = [0.0, T_train / 2, T_train, (T_train + T_end) / 2, T_end]
    else:
        eval_times = [0.0, T_end / 4, T_end / 2, T_end]

    # 网格评估 tensor（复用）
    x_t_grid = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(1).to(device)
    y_t_grid = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1).to(device)

    # 周期性网格验证回调：同时评估训练域和外推域
    grid_l2_log = []

    def on_eval(model, epoch):
        model.eval()
        with torch.no_grad():
            # 训练域末端
            t_t = torch.full_like(x_t_grid, T_train)
            u_pred = model(x_t_grid, y_t_grid, t_t).cpu().numpy().flatten()
        u_exact = case1_exact(x_flat, y_flat, T_train, alpha)
        l2_train = relative_l2_error(u_pred, u_exact)

        if time_generalization:
            with torch.no_grad():
                t_t = torch.full_like(x_t_grid, T_end)
                u_pred = model(x_t_grid, y_t_grid, t_t).cpu().numpy().flatten()
            u_exact = case1_exact(x_flat, y_flat, T_end, alpha)
            l2_extrap = relative_l2_error(u_pred, u_exact)
            grid_l2_log.append({"epoch": epoch, "l2_train": l2_train, "l2_extrap": l2_extrap})
            print(f"  >>> 网格验证 t={T_train:.3f}[训练]: L2={l2_train:.6e} | "
                  f"t={T_end:.3f}[外推]: L2={l2_extrap:.6e}")
        else:
            grid_l2_log.append({"epoch": epoch, "l2_train": l2_train})
            print(f"  >>> 网格验证 t={T_end:.3f}: L2={l2_train:.6e}")
        model.train()

    # ---- 4. 构建模型 ----
    model = PINN(layers=layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"网络参数量: {total_params}")
    print(f"网络结构: {layers}")

    # ---- 5. 训练 ----
    loss_weights = {
        "lambda_r": pinn_cfg["lambda_r"],
        "lambda_ic": pinn_cfg["lambda_ic"],
        "lambda_bc": pinn_cfg["lambda_bc"],
    }
    alpha_t = torch.tensor(alpha, device=device)

    loss_history, components_history, val_history, best_state, best_epoch = train(
        model,
        train_x_r, train_y_r, train_t_r,
        train_x_ic, train_y_ic,
        train_x_bc, train_y_bc, train_t_bc,
        alpha_t, pinn_cfg["epochs"], pinn_cfg.get("lr", pinn_cfg["peak_lr"]),
        loss_weights, log_every=pinn_cfg["log_every"], device=device,
        clip_grad_norm=pinn_cfg.get("clip_grad_norm", None),
        val_data=val_data,
        eval_callback=on_eval,
        early_stop_patience=pinn_cfg.get("early_stop_patience", None),
        adaptive_loss=pinn_cfg.get("adaptive_loss", None),
        warmup_epochs=pinn_cfg.get("warmup_epochs", 0),
        peak_lr=pinn_cfg.get("peak_lr"),
        end_lr=pinn_cfg.get("end_lr"),
    )

    # 加载最优模型
    model.load_state_dict(best_state)
    best_val = min(val_history) if val_history else None
    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))
    print(f"最优模型: epoch={best_epoch}, val loss={best_val:.6e}" if best_val else "最优模型已保存")

    # ---- 6. 测试集评估 ----
    l2_errors = []
    max_errors = []

    model.eval()
    with torch.no_grad():
        for t_val in eval_times:
            t_t = torch.full_like(x_t_grid, t_val)
            u_pred = model(x_t_grid, y_t_grid, t_t).cpu().numpy().flatten()

            u_exact = case1_exact(x_flat, y_flat, t_val, alpha)
            l2_err = relative_l2_error(u_pred, u_exact)
            max_err = max_absolute_error(u_pred, u_exact)
            l2_errors.append(l2_err)
            max_errors.append(max_err)
            tag = "[外推]" if time_generalization and t_val > T_train else "[训练]"
            print(f"  测试 t={t_val:.3f} {tag}: L2={l2_err:.6e}, Max={max_err:.6e}")

    # ---- 7. 计算 2×3 图所需数据 ----
    with torch.no_grad():
        # Row 2 - 时间曲线：3 个监测点在多个时刻的值
        ts_times = np.linspace(0, T_end, 100)
        ts_u_pred = []
        ts_u_exact = []
        for (x0, y0) in MONITOR_LOCS:
            x_pt = torch.full((len(ts_times), 1), x0, dtype=torch.float32, device=device)
            y_pt = torch.full((len(ts_times), 1), y0, dtype=torch.float32, device=device)
            t_pt = torch.tensor(ts_times, dtype=torch.float32, device=device).unsqueeze(1)
            u_p = model(x_pt, y_pt, t_pt).cpu().numpy().flatten()
            u_e = case1_exact(np.full_like(ts_times, x0),
                              np.full_like(ts_times, y0),
                              ts_times, alpha)
            ts_u_pred.append(u_p)
            ts_u_exact.append(u_e)

        # Row 2 - x=0.5 切面
        n_cs = 100
        cs_y = np.linspace(0, 1, n_cs)
        cs_x = torch.full((n_cs, 1), 0.5, dtype=torch.float32, device=device)
        cs_y_t = torch.tensor(cs_y, dtype=torch.float32, device=device).unsqueeze(1)
        cs_t = torch.full((n_cs, 1), T_end, device=device)
        cs_u_pred = model(cs_x, cs_y_t, cs_t).cpu().numpy().flatten()
        cs_u_exact = case1_exact(np.full(n_cs, 0.5), cs_y, T_end, alpha)

    # ---- 8. 保存结果 ----
    summary = {
        "case": case,
        "method": "PINN",
        "mode": "forward",
        "alpha": alpha,
        "layers": layers,
        "total_params": total_params,
        "epochs": pinn_cfg["epochs"],
        "peak_lr": pinn_cfg["peak_lr"],
        "end_lr": pinn_cfg.get("end_lr", 0.0),
        "warmup_epochs": pinn_cfg.get("warmup_epochs", 0),
        "clip_grad_norm": pinn_cfg.get("clip_grad_norm", None),
        "adaptive_loss": pinn_cfg.get("adaptive_loss", None),
        "num_collocation": n_col,
        "num_val_collocation": n_val_col,
        "T_train": T_train,
        "val_time_ratio": val_time_ratio,
        "device": str(device),
        "eval_times": eval_times,
        "l2_errors": [float(e) for e in l2_errors],
        "max_errors": [float(e) for e in max_errors],
        "final_loss": float(loss_history[-1]),
        "final_val_loss": float(val_history[-1]) if val_history else None,
        "best_epoch": best_epoch,
        "best_val_loss": float(min(val_history)) if val_history else None,
        "grid_l2_log": grid_l2_log,
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    shutil.copy2(args.config, os.path.join(run_dir, "config_snapshot.yaml"))

    # ---- 9. 可视化 ----
    # 2×3 对比图（评估时刻取 T_end）
    with torch.no_grad():
        t_t = torch.full_like(x_t_grid, T_end)
        u_final = model(x_t_grid, y_t_grid, t_t).cpu().numpy().flatten()
    u_exact_final = case1_exact(x_flat, y_flat, T_end, alpha)

    plot_comparison_2x3(
        nodes, u_final, u_exact_final, T_end,
        os.path.join(fig_dir, "comparison.png"),
        method="PINN",
        ts_data={"times": ts_times, "locations": MONITOR_LOCS,
                 "u_pred": ts_u_pred, "u_exact": ts_u_exact},
        cs_data={"y": cs_y, "u_pred": cs_u_pred, "u_exact": cs_u_exact},
        T_train=T_train if time_generalization else None,
    )

    # 损失曲线
    plot_loss_with_components(loss_history, components_history,
                             os.path.join(fig_dir, "loss_curve.png"),
                             val_history=val_history if val_history else None)

    print(f"\n结果摘要: {run_dir}/summary.json")
    print(f"可视化图像: {fig_dir}/")


if __name__ == "__main__":
    main()
