# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 在本仓库中工作时提供指引。

## 语言规则

- 所有文档、注释、commit message、对话回复均使用**中文**
- 代码中的变量名、函数名、类名使用英文，但 docstring 和注释用中文
- 数学符号和公式保持标准国际写法

## 项目概述

二维非稳态热传导方程求解器，对比 Galerkin 有限元方法（FEM）与物理信息神经网络（PINN）。FEM 使用 NumPy/SciPy，PINN 使用 PyTorch。项目当前处于开发初期。

## 环境配置

本仓库使用 conda 虚拟环境 `agent`，运行 Python 脚本时需设置 UTF-8 编码避免终端乱码：

```bash
conda activate agent
export PYTHONIOENCODING=utf-8
pip install -r requirements.txt
```

## 架构规划

- `src/fem/` — Galerkin FEM 流水线：网格生成、质量/刚度矩阵组装、边界条件、隐式 Euler 时间推进
- `src/pinn/` — PINN 模型、PDE 残差损失、训练循环、配点采样
- `src/utils/` — 解析解、误差度量、可视化
- `configs/default.yaml` — 实验配置（网格大小、时间步长、网络结构、学习率）
- `scripts/` — 入口脚本：`run_fem.py`、`run_pinn.py`、`compare.py`

## 常用命令

```bash
conda activate agent
python scripts/run_fem.py --config configs/default.yaml
python scripts/run_pinn.py --config configs/default.yaml
python scripts/compare.py --config configs/default.yaml
```

## 问题定义

求解的 PDE：∂u/∂t = α(∂²u/∂x² + ∂²u/∂y²) + f(x,y,t)，定义域 [0,1]²，Dirichlet 边界条件。计划四个实验场景（详见 `docs/experiments.md`）：无源项齐次 Dirichlet、制造解、高斯局部热源、非齐次边界温度驱动。

FEM 半离散系统：(M + Δt·K)Uⁿ⁺¹ = M·Uⁿ + Δt·Fⁿ⁺¹
PINN 总损失：λ_r·L_PDE + λ_ic·L_IC + λ_bc·L_BC，通过 autograd 计算残差。

## 文档

- `docs/derivations.md` — 弱形式与解析解推导
- `docs/experiments.md` — 实验配置与结果记录
- `docs/daily_log.md` — 开发日志
- `docs/references.md` — 参考文献

## 实验目录结构

每次实验运行的结果按场景和方法组织：

```
results/
├── case1/
│   ├── fem/
│   │   └── {日期}_{时间}_{网格}_{步长}/
│   │       ├── figures/            # 可视化图像
│   │       ├── summary.json        # 误差指标 + 运行参数
│   │       └── config_snapshot.yaml # 配置文件副本
│   └── pinn/
│       └── {日期}_{时间}_{网络结构}/
├── case2/
│   ├── fem/
│   └── pinn/
├── case3/
└── case4/
```

命名规则：`{YYYYMMDD}_{HHMMSS}_{关键参数}`
- FEM：网格分辨率 + 时间步长，如 `20260502_143000_20x20_dt0.01`
- PINN：网络结构，如 `20260503_100000_4x64`
- 可通过 `--name` 参数自定义目录名

每次实验必须包含：figures/、summary.json、config_snapshot.yaml
