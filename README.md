# pinn-fem-heat-equation

![Python](https://img.shields.io/badge/python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green)

本项目对比 **Galerkin 有限元方法（FEM）** 与 **物理信息神经网络（PINN）** 在二维非稳态热传导方程上的求解能力，涵盖正问题与反问题（参数发现）。

**正问题**：FEM 在求解精度和计算速度上均远优于 PINN（L2 误差差 1-2 个数量级）。**反问题**：当观测数据稀疏且含噪声时，PINN 可直接从物理约束出发学习未知参数（如热扩散系数 α），这是传统数值方法难以胜任的优势。

> Case 1（无源项齐次 Dirichlet）的 FEM 和 PINN 正问题已完整实现。PINN 反问题（参数发现）已实现：从稀疏带噪声观测数据中学习未知热扩散系数 α。

---

## 功能亮点 (Features)

* 实现二维非稳态热传导方程 **Galerkin FEM** 求解器（隐式 Euler 时间推进）
* 基于 PyTorch 构建 **PINN** 模型，学习连续温度场
* 支持多实验场景：解析解验证、制造解、局部热源扩散、非齐次边界温度驱动
* 完整的训练/验证/测试体系：独立验证配点、Early Stop、最优模型保存
* PINN 反问题（参数发现）：从稀疏带噪声观测数据中学习未知热扩散系数 α
* 自适应损失权重平衡（EMA）、warmup + linear decay 学习率调度
* 时间泛化验证：训练在部分时间域，测试模型在未见时间段上的外推能力
* 2×3 综合对比可视化：温度场 + 时间曲线 + 切面分析
* 损失分量曲线可视化：训练/验证/PDE/IC/BC 五线对比
* 支持 GPU 加速训练（CUDA）
* 文档记录解析解推导、实验日志和 daily log

---

## 项目结构

```text
pinn-fem-heat-equation/
├── README.md
├── requirements.txt
├── configs/
│   └── default.yaml        # 实验配置（FEM/PINN 参数）
├── docs/                    # 文档：推导、实验记录、日志
├── scripts/
│   ├── run_fem.py           # FEM 入口脚本
│   ├── run_pinn.py          # PINN 正问题入口脚本
│   └── run_pinn_inverse.py  # PINN 反问题入口脚本
├── src/
│   ├── fem/                 # FEM 模块：网格、组装、求解器
│   ├── pinn/                # PINN 模块：模型、损失、采样、训练、学习率调度
│   └── utils/               # 工具：解析解、误差度量、可视化、随机种子
└── results/                 # 实验结果（按 case/method 组织）
```

---

## 快速开始

安装依赖：

```bash
conda activate agent
pip install -r requirements.txt
```

运行 FEM 求解器：

```bash
PYTHONIOENCODING=utf-8 python scripts/run_fem.py --config configs/default.yaml
```

训练 PINN：

```bash
PYTHONIOENCODING=utf-8 python scripts/run_pinn.py --config configs/default.yaml
```

PINN 反问题（参数发现）：

```bash
PYTHONIOENCODING=utf-8 python scripts/run_pinn_inverse.py --config configs/default.yaml
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

### Case 1：自然冷却（无源项齐次 Dirichlet）

一块正方形金属板（1m×1m），初始温度呈中心高、边缘零的分布。四周边界始终与环境温度（0°C）接触，没有外部热源。热量从中心向四周扩散，温度单调下降直至均匀为零——这就是一个典型的自然冷却过程。

- **初始条件**：$u(x,y,0) = \sin(\pi x)\sin(\pi y)$，中心最高约 1°C，边缘为 0
- **边界条件**：四边固定为 0（齐次 Dirichlet），模拟恒温环境
- **热源**：无（$f=0$）
- **解析解**：$u = e^{-2\alpha\pi^2 t}\sin(\pi x)\sin(\pi y)$，指数衰减

### Case 2：制造解验证（带源项）

假设温度场按已知规律衰减：$u = e^{-t}\sin(\pi x)\sin(\pi y)$。反过来推导需要加什么样的热源才能维持这个温度变化，用于验证数值方法的正确性。

- **初始条件**：$u(x,y,0) = \sin(\pi x)\sin(\pi y)$
- **边界条件**：四边固定为 0（齐次 Dirichlet）
- **热源**：$f = (-1 + 2\alpha\pi^2)e^{-t}\sin(\pi x)\sin(\pi y)$
- **解析解**：$u = e^{-t}\sin(\pi x)\sin(\pi y)$

### Case 3：局部热源扩散

一块初始温度均匀的金属板，中心有一个高斯形状的热源在持续加热（功率随时间衰减）。热量从热源向外扩散，温度场呈现以热源为中心的扩散模式。

- **初始条件**：$u(x,y,0) = \exp\left(-\frac{(x-x_c)^2+(y-y_c)^2}{2\sigma^2}\right)$，中心高斯分布
- **边界条件**：四边固定为 0（齐次 Dirichlet）
- **热源**：$f = A\exp\left(-\frac{(x-x_s)^2+(y-y_s)^2}{2\sigma_s^2}\right)\exp(-\beta t)$，局部加热且功率衰减

### Case 4：边界温度驱动

一块初始均匀低温的金属板，左侧突然被加热到高温，其他三边保持低温。热量从左边界向右扩散，温度场逐渐从左到右建立梯度，模拟边界温差驱动的传热过程。

- **初始条件**：$u(x,y,0) = T_{\text{cold}}$，全区域低温
- **边界条件**：左边界 $u(0,y,t) = T_{\text{hot}}$，其余三边 $u = T_{\text{cold}}$（非齐次 Dirichlet）
- **热源**：无（$f=0$）

---

## 方法概述

**Galerkin FEM**

弱形式：

$$
\int_{\Omega} \frac{\partial u}{\partial t} v \, d\Omega + \int_{\Omega} \alpha \nabla u \cdot \nabla v \, d\Omega = \int_{\Omega} f v \, d\Omega
$$

空间离散：

$$
u_h(x, y, t) = \sum_j U_j(t)\phi_j(x, y)
$$

半离散系统：

$$
M \frac{dU}{dt} + KU = F
$$

时间推进（隐式 Euler）：

$$
(M + \Delta t K) U^{n+1} = M U^n + \Delta t F^{n+1}
$$

**PINN**

网络近似温度场：

$$
u_\theta(x, y, t) \approx u(x, y, t)
$$

PDE 残差：

$$
r_\theta(x, y, t) = \frac{\partial u_\theta}{\partial t} - \alpha\left(\frac{\partial^2 u_\theta}{\partial x^2} + \frac{\partial^2 u_\theta}{\partial y^2}\right) - f(x, y, t)
$$

总损失：

$$
\mathcal{L} = \lambda_r \mathcal{L}_{PDE} + \lambda_{ic} \mathcal{L}_{IC} + \lambda_{bc} \mathcal{L}_{BC}
$$

**PINN 反问题（参数发现）**

给定稀疏带噪声的观测数据 $\{(x_i, y_i, t_i, u_i^{obs})\}_{i=1}^{N}$，学习未知热扩散系数 $\alpha$：

$$
\alpha = \text{softplus}(\hat{\alpha}), \quad \hat{\alpha} \in \mathbb{R} \text{（可学习参数）}
$$

总损失增加数据拟合项：

$$
\mathcal{L} = \lambda_r \mathcal{L}_{PDE} + \lambda_{ic} \mathcal{L}_{IC} + \lambda_{bc} \mathcal{L}_{BC} + \lambda_{data} \mathcal{L}_{data}
$$

其中 $\mathcal{L}_{data} = \frac{1}{N}\sum_{i}(u_\theta(x_i,y_i,t_i) - u_i^{obs})^2$。

---

## 对比指标

* 精度：相对 $L_2$ 误差、最大绝对误差
* 计算效率：FEM 求解时间、PINN 训练/推理时间
* 可视化：温度场、误差分布、损失下降曲线

---

## 当前进度

* [x] 确定测试问题与解析解
* [x] FEM 核心模块实现（网格生成、矩阵组装、边界条件、隐式 Euler 时间推进）
* [x] PINN 正问题实现（模型、PDE 残差损失、配点采样、训练循环）
* [x] 验证体系（独立验证配点、网格误差回调、Early Stop、最优模型保存）
* [x] 时间泛化验证（训练域/外推域分离评估）
* [x] 综合可视化（2×3 对比图：温度场 + 时间曲线 + 切面分析）
* [x] PINN 反问题（参数发现，从稀疏带噪声观测数据学习 α）
* [ ] Case 2-4 实验场景
* [ ] FEM 与 PINN 对比脚本

---

## 技术栈

Python + NumPy + SciPy + PyTorch + Matplotlib
FEM 与 PINN 的联合实现与对比研究

---

## 文档 (docs)

* `docs/derivations.md`：解析解与弱形式推导
* `docs/experiments.md`：实验配置与结果记录
* `docs/daily_log.md`：开发日志
* `docs/references.md`：参考文献

---

