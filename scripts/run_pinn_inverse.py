"""PINN 反问题入口脚本（Case 1：参数发现，学习热扩散系数 α）。"""

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
from src.pinn.train import train_inverse
from src.utils.exact_solution import case1_exact
from src.utils.metrics import relative_l2_error, max_absolute_error
from src.utils.visualization import (plot_comparison_2x3, plot_inverse_training,
                                     plot_sampling_points)
from src.utils.seed import set_seed

MONITOR_LOCS = [(0.25, 0.25), (0.25, 0.50), (0.50, 0.50)]


def require_case1(case):
    """当前 PINN 反问题只实现 Case 1，避免配置切换后生成误导性结果。"""
    if case != 1:
        raise NotImplementedError(
            f"当前 PINN 反问题入口只实现 Case 1，但配置为 case={case}。"
            "请先实现对应 PDE 源项、IC、BC、观测生成和解析解后再运行。"
        )


def validate_validation_counts(n_col, n_ic, n_bc):
    """验证损失会同时计算 PDE/IC/BC，三类验证配点必须同时启用或同时关闭。"""
    counts = [n_col, n_ic, n_bc]
    if any(count > 0 for count in counts) and not all(count > 0 for count in counts):
        raise ValueError(
            "验证配点必须同时设置 num_val_collocation、num_val_ic、num_val_bc，"
            f"当前为 {counts}，否则空张量会导致验证损失为 nan。"
        )


def generate_observation_data(n_points, T_end, true_alpha, noise_level, device):
    """用解析解生成稀疏带噪声的观测数据。"""
    x_obs = torch.rand(n_points, 1, device=device)
    y_obs = torch.rand(n_points, 1, device=device)
    t_obs = torch.rand(n_points, 1, device=device) * T_end

    u_exact = case1_exact(
        x_obs.cpu().numpy().flatten(),
        y_obs.cpu().numpy().flatten(),
        t_obs.cpu().numpy().flatten(),
        true_alpha,
    )
    u_obs = torch.tensor(u_exact, dtype=torch.float32, device=device).unsqueeze(1)
    u_obs = u_obs + torch.randn_like(u_obs) * noise_level

    return x_obs, y_obs, t_obs, u_obs


def main():
    parser = argparse.ArgumentParser(description="PINN 反问题：学习热扩散系数 α")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--name", type=str, default=None)
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 读取配置
    T_end = config["fem"]["T_end"]
    true_alpha = config["physics"]["alpha"]
    case = config["case"]
    pinn_cfg = config["pinn"]
    inv_cfg = pinn_cfg["inverse"]
    require_case1(case)

    set_seed(config.get("seed", 42))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"=== PINN 反问题 (Case {case}, 参数发现) ===")
    print(f"设备: {device}")
    print(f"真实 α={true_alpha}, 初始猜测 α={inv_cfg['initial_alpha']}")

    # 时间泛化参数
    val_time_ratio = pinn_cfg.get("val_time_ratio", 1.0)
    T_train = val_time_ratio * T_end
    time_generalization = val_time_ratio < 1.0
    if time_generalization:
        print(f"时间泛化: 训练 t∈[0, {T_train:.3f}], 外推 t∈({T_train:.3f}, {T_end:.3f}]")

    # 实验目录
    layers = pinn_cfg["layers"]
    arch = f"{len(layers)-2}x{layers[1]}"
    if args.name:
        run_dir = os.path.join("results", f"case{case}", "pinn_inverse", args.name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", f"case{case}", "pinn_inverse",
                               f"{timestamp}_{arch}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    print(f"实验目录: {run_dir}")

    # ---- 1. 观测数据（按时间划分训练/验证） ----
    n_obs = inv_cfg["num_observation"]
    noise_level = inv_cfg["noise_level"]
    # 先在全时间域生成，再按时间拆分
    x_obs_all, y_obs_all, t_obs_all, u_obs_all = generate_observation_data(
        n_obs, T_end, true_alpha, noise_level, device,
    )
    if time_generalization:
        train_mask = t_obs_all.flatten() <= T_train
        val_mask = ~train_mask
        x_obs, y_obs, t_obs, u_obs = (
            x_obs_all[train_mask], y_obs_all[train_mask],
            t_obs_all[train_mask], u_obs_all[train_mask],
        )
        if val_mask.sum() > 0:
            x_obs_val, y_obs_val, t_obs_val, u_obs_val = (
                x_obs_all[val_mask], y_obs_all[val_mask],
                t_obs_all[val_mask], u_obs_all[val_mask],
            )
        else:
            x_obs_val = y_obs_val = t_obs_val = u_obs_val = None
            print("  警告：所有观测数据都在训练域内，验证观测数据为空")
        n_val_obs = int(val_mask.sum()) if x_obs_val is not None else 0
        print(f"观测数据: {n_obs} 点 (训练 {int(train_mask.sum())}, "
              f"验证 {n_val_obs}), 噪声 σ={noise_level}")
    else:
        x_obs, y_obs, t_obs, u_obs = x_obs_all, y_obs_all, t_obs_all, u_obs_all
        x_obs_val = y_obs_val = t_obs_val = u_obs_val = None
        print(f"观测数据: {n_obs} 点, 噪声 σ={noise_level}")

    # ---- 2. PDE/IC/BC 配点（训练 t∈[0, T_train]） ----
    n_col = pinn_cfg["num_collocation"]
    n_ic = pinn_cfg["num_ic"]
    n_bc = pinn_cfg["num_bc"]

    x_r, y_r, t_r = sample_collocation(n_col, T_train, device)
    x_ic, y_ic = sample_initial(n_ic, device)
    x_bc, y_bc, t_bc = sample_boundary(n_bc, T_train, device)

    print(f"配点: 域内 {n_col}, IC {n_ic}, BC {n_bc}, T_train={T_train:.3f}")

    # ---- 3. 验证集（外推域 t∈(T_train, T_end]） ----
    n_val_col = pinn_cfg.get("num_val_collocation", 0)
    n_val_ic = pinn_cfg.get("num_val_ic", 0)
    n_val_bc = pinn_cfg.get("num_val_bc", 0)
    validate_validation_counts(n_val_col, n_val_ic, n_val_bc)
    val_data = None
    if n_val_col > 0 or n_val_ic > 0 or n_val_bc > 0:
        if time_generalization:
            # 验证配点在外推域
            val_t_r = torch.rand(n_val_col, 1, device=device) * (T_end - T_train) + T_train
        else:
            val_t_r = torch.rand(n_val_col, 1, device=device) * T_end
        val_data = {
            "x_r": torch.rand(n_val_col, 1, device=device),
            "y_r": torch.rand(n_val_col, 1, device=device),
            "t_r": val_t_r,
            "x_ic": torch.rand(n_val_ic, 1, device=device),
            "y_ic": torch.rand(n_val_ic, 1, device=device),
        }
        if time_generalization:
            vx_bc, vy_bc, vt_bc = sample_boundary(n_val_bc, T_end - T_train, device)
            vt_bc = vt_bc + T_train  # 偏移到外推域
        else:
            vx_bc, vy_bc, vt_bc = sample_boundary(n_val_bc, T_end, device)
        val_data["x_bc"] = vx_bc
        val_data["y_bc"] = vy_bc
        val_data["t_bc"] = vt_bc
        # 验证观测数据
        if x_obs_val is not None and len(x_obs_val) > 0:
            val_data["obs_data"] = (x_obs_val, y_obs_val, t_obs_val, u_obs_val)
        print(f"验证配点: 域内 {n_val_col}, IC {n_val_ic}, BC {n_val_bc}"
              f"{f' + {int(len(x_obs_val))} 观测点' if x_obs_val is not None and len(x_obs_val) > 0 else ''}")

    # 采样点分布图（训练前保存）
    os.makedirs(fig_dir, exist_ok=True)
    train_scatter = {
        "x_r": x_r.cpu(), "y_r": y_r.cpu(),
        "x_bc": x_bc.cpu(), "y_bc": y_bc.cpu(),
        "x_ic": x_ic.cpu(), "y_ic": y_ic.cpu(),
    }
    val_scatter = {k: v.cpu() for k, v in val_data.items() if k != "obs_data"} if val_data else None
    plot_sampling_points(
        train_data=train_scatter, val_data=val_scatter,
        x_obs=x_obs.cpu(), y_obs=y_obs.cpu(),
        x_obs_val=x_obs_val.cpu() if x_obs_val is not None else None,
        y_obs_val=y_obs_val.cpu() if y_obs_val is not None else None,
        filename=os.path.join(fig_dir, "sampling_points.png"),
        title_prefix="反问题 ",
    )

    # ---- 4. 测试网格 ----
    nx_eval = 50
    x_lin = np.linspace(0, 1, nx_eval)
    y_lin = np.linspace(0, 1, nx_eval)
    xx, yy = np.meshgrid(x_lin, y_lin)
    x_flat = xx.flatten()
    y_flat = yy.flatten()
    nodes = np.column_stack([x_flat, y_flat])
    eval_times = [0.0, T_end / 4, T_end / 2, T_end]

    x_t_grid = torch.tensor(x_flat, dtype=torch.float32).unsqueeze(1).to(device)
    y_t_grid = torch.tensor(y_flat, dtype=torch.float32).unsqueeze(1).to(device)

    # ---- 5. 构建模型 ----
    model = PINN(layers=layers).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"网络参数量: {total_params}, 结构: {layers}")

    # ---- 6. 训练 ----
    loss_weights = {
        "lambda_r": pinn_cfg["lambda_r"],
        "lambda_ic": pinn_cfg["lambda_ic"],
        "lambda_bc": pinn_cfg["lambda_bc"],
        "lambda_data": inv_cfg["lambda_data"],
    }

    (loss_history, components_history, alpha_history, val_history,
     best_state, best_alpha, best_epoch) = train_inverse(
        model,
        x_r, y_r, t_r,
        x_ic, y_ic,
        x_bc, y_bc, t_bc,
        obs_data=(x_obs, y_obs, t_obs, u_obs),
        initial_alpha=inv_cfg["initial_alpha"],
        true_alpha=true_alpha,
        epochs=pinn_cfg["epochs"],
        loss_weights=loss_weights,
        log_every=pinn_cfg.get("log_every", 300),
        device=device,
        clip_grad_norm=pinn_cfg.get("clip_grad_norm", None),
        val_data=val_data,
        early_stop_patience=pinn_cfg.get("early_stop_patience", None),
        adaptive_loss=pinn_cfg.get("adaptive_loss", None),
        warmup_epochs=pinn_cfg.get("warmup_epochs", 0),
        peak_lr=pinn_cfg.get("peak_lr"),
        end_lr=pinn_cfg.get("end_lr"),
        alpha_lr=inv_cfg.get("alpha_lr", 0.001),
    )

    # 加载最优模型
    model.load_state_dict(best_state)
    torch.save(best_state, os.path.join(run_dir, "best_model.pt"))

    alpha_err = abs(best_alpha - true_alpha) / true_alpha * 100
    print(f"\n最终结果: α={best_alpha:.6f} (真实={true_alpha}, 误差={alpha_err:.2f}%)")
    print(f"最优 epoch: {best_epoch}")

    # ---- 7. 测试评估 ----
    l2_errors = []
    max_errors = []
    model.eval()
    with torch.no_grad():
        for t_val in eval_times:
            t_t = torch.full_like(x_t_grid, t_val)
            u_pred = model(x_t_grid, y_t_grid, t_t).cpu().numpy().flatten()
            u_exact = case1_exact(x_flat, y_flat, t_val, true_alpha)
            l2_err = relative_l2_error(u_pred, u_exact)
            max_err = max_absolute_error(u_pred, u_exact)
            l2_errors.append(l2_err)
            max_errors.append(max_err)
            print(f"  t={t_val:.3f}: L2={l2_err:.6e}, Max={max_err:.6e}")

    # ---- 8. 可视化数据 ----
    with torch.no_grad():
        ts_times = np.linspace(0, T_end, 100)
        ts_u_pred = []
        ts_u_exact = []
        for (x0, y0) in MONITOR_LOCS:
            x_pt = torch.full((100, 1), x0, dtype=torch.float32, device=device)
            y_pt = torch.full((100, 1), y0, dtype=torch.float32, device=device)
            t_pt = torch.tensor(ts_times, dtype=torch.float32, device=device).unsqueeze(1)
            u_p = model(x_pt, y_pt, t_pt).cpu().numpy().flatten()
            u_e = case1_exact(np.full(100, x0), np.full(100, y0), ts_times, true_alpha)
            ts_u_pred.append(u_p)
            ts_u_exact.append(u_e)

        n_cs = 100
        cs_y = np.linspace(0, 1, n_cs)
        cs_x = torch.full((n_cs, 1), 0.5, dtype=torch.float32, device=device)
        cs_y_t = torch.tensor(cs_y, dtype=torch.float32, device=device).unsqueeze(1)
        cs_t = torch.full((n_cs, 1), T_end, device=device)
        cs_u_pred = model(cs_x, cs_y_t, cs_t).cpu().numpy().flatten()
        cs_u_exact = case1_exact(np.full(n_cs, 0.5), cs_y, T_end, true_alpha)

    # ---- 9. 保存结果 ----
    best_val = min(val_history) if val_history else None
    summary = {
        "case": case,
        "method": "PINN_inverse",
        "true_alpha": true_alpha,
        "initial_alpha": inv_cfg["initial_alpha"],
        "learned_alpha": best_alpha,
        "alpha_error_pct": alpha_err,
        "layers": layers,
        "total_params": total_params,
        "epochs": pinn_cfg["epochs"],
        "num_observation": n_obs,
        "noise_level": noise_level,
        "best_epoch": best_epoch,
        "final_loss": float(loss_history[-1]),
        "best_val_loss": float(best_val) if best_val else None,
        "eval_times": eval_times,
        "l2_errors": [float(e) for e in l2_errors],
        "max_errors": [float(e) for e in max_errors],
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    shutil.copy2(args.config, os.path.join(run_dir, "config_snapshot.yaml"))

    # ---- 10. 可视化 ----
    with torch.no_grad():
        t_t = torch.full_like(x_t_grid, T_end)
        u_final = model(x_t_grid, y_t_grid, t_t).cpu().numpy().flatten()
        # 观测点处的模型预测
        u_pred_obs_train = model(x_obs, y_obs, t_obs).cpu().numpy().flatten()
        u_pred_obs_val = None
        if x_obs_val is not None:
            u_pred_obs_val = model(x_obs_val, y_obs_val, t_obs_val).cpu().numpy().flatten()
    u_exact_final = case1_exact(x_flat, y_flat, T_end, true_alpha)

    obs_plot_data = {
        "t_train": t_obs.cpu().numpy().flatten(),
        "u_train": u_obs.cpu().numpy().flatten(),
        "u_pred_train": u_pred_obs_train,
        "t_val": t_obs_val.cpu().numpy().flatten() if x_obs_val is not None else None,
        "u_val": u_obs_val.cpu().numpy().flatten() if x_obs_val is not None else None,
        "u_pred_val": u_pred_obs_val,
    }

    plot_comparison_2x3(
        nodes, u_final, u_exact_final, T_end,
        os.path.join(fig_dir, "comparison.png"), method="PINN (inverse)",
        ts_data={"times": ts_times, "locations": MONITOR_LOCS,
                 "u_pred": ts_u_pred, "u_exact": ts_u_exact},
        cs_data={"y": cs_y, "u_pred": cs_u_pred, "u_exact": cs_u_exact},
        T_train=T_train if time_generalization else None,
        obs_data=obs_plot_data,
    )

    plot_inverse_training(
        loss_history, components_history, alpha_history, true_alpha,
        os.path.join(fig_dir, "inverse_training.png"),
    )

    print(f"\n结果: {run_dir}/summary.json")
    print(f"图像: {fig_dir}/")


if __name__ == "__main__":
    main()
