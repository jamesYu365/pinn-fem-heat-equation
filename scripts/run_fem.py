"""FEM 求解器入口脚本（Case 1：无源项齐次 Dirichlet）。"""

import argparse
import yaml
import numpy as np
import json
import shutil
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.fem.mesh import generate_structured_mesh
from src.fem.assemble import assemble_global
from src.fem.solver import solve_implicit_euler
from src.utils.exact_solution import case1_exact
from src.utils.metrics import relative_l2_error, max_absolute_error
from src.utils.visualization import plot_comparison_2x3, plot_error_over_time


def main():
    parser = argparse.ArgumentParser(description="FEM 求解二维热传导方程")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="配置文件路径")
    parser.add_argument("--name", type=str, default=None, help="自定义实验目录名")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 读取配置
    Lx = config["domain"]["Lx"]
    Ly = config["domain"]["Ly"]
    nx = config["fem"]["nx"]
    ny = config["fem"]["ny"]
    dt = config["fem"]["dt"]
    T_end = config["fem"]["T_end"]
    alpha = config["physics"]["alpha"]
    case = config["case"]

    # 实验目录：results/case{N}/fem/{时间戳}_{参数}/
    if args.name:
        run_dir = os.path.join("results", f"case{case}", "fem", args.name)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("results", f"case{case}", "fem", f"{timestamp}_{nx}x{ny}_dt{dt}")
    fig_dir = os.path.join(run_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    print(f"=== FEM 求解器 (Case {case}) ===")
    print(f"网格: {nx}x{ny}, dt={dt}, T={T_end}, alpha={alpha}")
    print(f"实验目录: {run_dir}")

    # 1. 生成网格
    nodes, elements, boundary_nodes = generate_structured_mesh(nx, ny, Lx, Ly)
    print(f"节点数: {len(nodes)}, 单元数: {len(elements)}, 边界节点数: {len(boundary_nodes)}")

    # 2. 组装全局矩阵
    K, M = assemble_global(nodes, elements, alpha)
    num_nodes = len(nodes)

    # 3. 初始条件
    U0 = np.sin(np.pi * nodes[:, 0]) * np.sin(np.pi * nodes[:, 1])
    U0[boundary_nodes] = 0.0

    # 5. 时间推进
    U_history, times = solve_implicit_euler(K, M, U0, dt, T_end, boundary_nodes)

    # 6. 计算误差
    l2_errors = []
    max_errors = []
    for i, (U, t) in enumerate(zip(U_history, times)):
        u_exact = case1_exact(nodes[:, 0], nodes[:, 1], t, alpha)
        l2_err = relative_l2_error(U, u_exact)
        max_err = max_absolute_error(U, u_exact)
        l2_errors.append(l2_err)
        max_errors.append(max_err)

    print(f"\n最终时刻 t={T_end}:")
    print(f"  相对 L2 误差: {l2_errors[-1]:.6e}")
    print(f"  最大绝对误差: {max_errors[-1]:.6e}")

    # 7. 保存 summary.json
    summary = {
        "case": case,
        "method": "FEM",
        "nx": nx, "ny": ny,
        "dt": dt, "T_end": T_end, "alpha": alpha,
        "num_nodes": int(num_nodes),
        "num_elements": int(len(elements)),
        "final_l2_error": float(l2_errors[-1]),
        "final_max_error": float(max_errors[-1]),
    }
    with open(os.path.join(run_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 8. 保存配置副本
    shutil.copy2(args.config, os.path.join(run_dir, "config_snapshot.yaml"))

    print(f"结果摘要: {run_dir}/summary.json")

    # 9. 可视化
    U_final = U_history[-1]
    u_exact_final = case1_exact(nodes[:, 0], nodes[:, 1], T_end, alpha)

    # 时间曲线：3 个监测点
    monitor_locs = [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)]
    ts_u_pred = []
    ts_u_exact = []
    for (x0, y0) in monitor_locs:
        dists = np.sqrt((nodes[:, 0] - x0)**2 + (nodes[:, 1] - y0)**2)
        nearest = np.argmin(dists)
        u_ts = [U[nearest] for U in U_history]
        u_e_ts = [case1_exact(nodes[nearest, 0], nodes[nearest, 1], t, alpha) for t in times]
        ts_u_pred.append(u_ts)
        ts_u_exact.append(u_e_ts)

    # x=0.5 切面
    x_half_idx = nx // 2
    cs_node_ids = [x_half_idx + j * (nx + 1) for j in range(ny + 1)]
    cs_y = nodes[cs_node_ids, 1]
    cs_u_pred = U_final[cs_node_ids]
    cs_u_exact = u_exact_final[cs_node_ids]

    plot_comparison_2x3(
        nodes, U_final, u_exact_final, T_end,
        os.path.join(fig_dir, "comparison.png"), method="FEM",
        ts_data={"times": times, "locations": monitor_locs,
                 "u_pred": ts_u_pred, "u_exact": ts_u_exact},
        cs_data={"y": cs_y, "u_pred": cs_u_pred, "u_exact": cs_u_exact},
    )
    plot_error_over_time(times, l2_errors, max_errors, os.path.join(fig_dir, "error_over_time.png"))

    print(f"可视化图像: {fig_dir}/")


if __name__ == "__main__":
    main()
