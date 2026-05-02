# pinn-fem-heat-equation

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green)

无源项齐次 Dirichlet 验证 FEM 精度，解析解可得： $$ u_{\text{exact}}(x, y, t) = e^{-2\alpha \pi^2 t}\sin(\pi x)\sin(\pi y) $$
带源项制造解问题 $$ u_{\text{exact}}(x, y, t) = e^{-t}\sin(\pi x)\sin(\pi y), \quad f(x, y, t) = (-1 + 2\alpha\pi^2) e^{-t}\sin(\pi x)\sin(\pi y) $$
局部热源扩散问题 初始条件高斯分布，时间相关局部热源： $$ u(x, y, 0) = \exp\left(-\frac{(x - x_c)^2 + (y - y_c)^2}{2\sigma^2}\right) $$ $$ f(x, y, t) = A\exp\left(-\frac{(x - x_s)^2 + (y - y_s)^2}{2\sigma_s^2}\right)\exp(-\beta t) $$
非齐次边界温度驱动 左边界高温，其他边界低温，模拟热量扩散。

本项目探索传统数值方法 **Galerkin 有限元方法（FEM）** 与 **物理信息神经网络（PINN）** 在二维非稳态热传导方程求解上的性能差异。对比维度包括求解精度、计算开销、连续场重建能力及物理约束满足程度。

> 当前项目处于开发初期，README 将随着代码实现、实验结果和可视化内容持续更新。

---

## 功能亮点 (Features)

* 实现二维非稳态热传导方程 **Galerkin FEM** 求解器
* 基于 PyTorch 构建 **PINN** 模型，学习连续温度场
* 支持多实验场景：解析解验证、制造解、局部热源扩散、非齐次边界温度驱动
* 对比两类方法在精度、效率、温度场重建上的差异
* 自动生成可视化图像：温度场、误差分布、损失曲线
* 文档记录解析解推导、实验日志和 daily log

---

## 项目结构

```text
pinn-fem-heat-equation/
├── README.md
├── requirements.txt
├── docs/                  # 存放文档：解析解推导、实验记录、daily log
├── configs/
│   └── default.yaml
├── src/
│   ├── fem/               # FEM 相关模块
│   │   ├── mesh.py
│   │   ├── assemble.py
│   │   ├── solver.py
│   │   └── boundary.py
│   ├── pinn/              # PINN 相关模块
│   │   ├── model.py
│   │   ├── loss.py
│   │   ├── train.py
│   │   └── sampling.py
│   ├── utils/             # 工具函数
│   │   ├── exact_solution.py
│   │   ├── metrics.py
│   │   └── visualization.py
│   └── main.py
├── scripts/
│   ├── run_fem.py
│   ├── run_pinn.py
│   └── compare.py
├── results/
│   ├── figures/
│   └── logs/
└── tests/
    ├── test_fem.py
    └── test_pinn.py
```

---

## 快速开始

安装依赖：

```bash
pip install -r requirements.txt
```

运行 FEM 求解器：

```bash
python scripts/run_fem.py --config configs/default.yaml
```

训练 PINN：

```bash
python scripts/run_pinn.py --config configs/default.yaml
```

对比 FEM 与 PINN 结果：

```bash
python scripts/compare.py --config configs/default.yaml
```

---

## 问题定义

二维非稳态热传导方程：

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + f(x, y, t), \quad (x, y) \in [0,1]^2, \ t \in [0, T]
$$

初始条件：

$$
u(x, y, 0) = u_0(x, y)
$$

Dirichlet 边界条件：

$$
u(x, y, t) = g(x, y, t), \quad (x, y) \in \partial \Omega
$$

可扩展到 Neumann 或 Robin 边界条件。

---

## 实验场景示例

### 1. 无源项齐次 Dirichlet

验证 FEM 精度，解析解可得：

$$
u_{\text{exact}}(x, y, t) = e^{-2\alpha \pi^2 t}\sin(\pi x)\sin(\pi y)
$$

### 2. 带源项制造解问题

$$
u_{\text{exact}}(x, y, t) = e^{-t}\sin(\pi x)\sin(\pi y)
$$

$$
f(x, y, t) = (-1 + 2\alpha\pi^2) e^{-t}\sin(\pi x)\sin(\pi y)
$$

### 3. 局部热源扩散问题

初始条件高斯分布，时间相关局部热源：

$$
u(x, y, 0) = \exp\left(-\frac{(x - x_c)^2 + (y - y_c)^2}{2\sigma^2}\right)
$$

$$
f(x, y, t) = A\exp\left(-\frac{(x - x_s)^2 + (y - y_s)^2}{2\sigma_s^2}\right)\exp(-\beta t)
$$

### 4. 非齐次边界温度驱动

左边界高温，其他边界低温，模拟热量扩散。

---

## 方法概述

**Galerkin FEM**

弱形式：

$$
\int_{\Omega} \frac{\partial u}{\partial t} v \, d\Omega + \int_{\Omega} \alpha \nabla u \cdot \nabla v \, d\Omega = \int_{\Omega} f v \, d\Omega
$$

空间离散：$u_h(x, y, t) = \sum_j U_j(t)\phi_j(x, y)$

半离散系统：$M \frac{dU}{dt} + KU = F$

时间推进（隐式 Euler）：

$$
(M + \Delta t K) U^{n+1} = M U^n + \Delta t F^{n+1}
$$

**PINN**

网络近似温度场：$u_\theta(x, y, t) \approx u(x, y, t)$

PDE 残差：

$$
r_\theta(x, y, t) = \frac{\partial u_\theta}{\partial t} - \alpha\left(\frac{\partial^2 u_\theta}{\partial x^2} + \frac{\partial^2 u_\theta}{\partial y^2}\right) - f(x, y, t)
$$

总损失：

$$
\mathcal{L} = \lambda_r \mathcal{L}_{PDE} + \lambda_{ic} \mathcal{L}_{IC} + \lambda_{bc} \mathcal{L}_{BC}
$$

---

## 对比指标

* 精度：相对 $L_2$ 误差、最大绝对误差
* 计算效率：FEM 求解时间、PINN 训练/推理时间
* 可视化：温度场、误差分布、损失下降曲线

---

## 当前进度

* [ ] 确定测试问题与解析解
* [ ] FEM 核心模块实现（网格生成、矩阵组装、边界条件、时间推进）
* [ ] PINN 网络结构与损失函数实现
* [ ] 实验对比与可视化

---

## 技术栈

Python + NumPy + SciPy + PyTorch + Matplotlib
FEM 与 PINN 的联合实现与对比研究

---

## 文档 (docs)

在 `docs/` 文件夹中可添加：

* `derivations.md`：解析解与弱形式推导
* `experiments.md`：实验记录
* `daily_log.md`：日常开发日志
* `references.md`：文献和公式笔记

---

