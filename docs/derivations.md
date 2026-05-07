
# 解析解与弱形式推导

## 1. 问题定义

二维非稳态热传导方程：

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + f(x, y, t)
$$

初始条件：

$$
u(x, y, 0) = u_0(x, y)
$$

边界条件：

$$
u(x, y, t) = g(x, y, t), \quad (x, y) \in \partial \Omega
$$

## 2. 弱形式推导

### 2.1 从强形式出发

强形式（原始 PDE）：

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right) + f(x, y, t), \quad (x, y) \in \Omega
$$

等价地用 Laplacian 记号：

$$
\frac{\partial u}{\partial t} - \alpha \nabla^2 u = f
$$

### 2.2 乘以测试函数并积分

引入测试函数 $v(x, y) \in H_0^1(\Omega)$（在边界上 $v = 0$），两边乘以 $v$ 并在 $\Omega$ 上积分：

$$
\int_\Omega \frac{\partial u}{\partial t} v \, d\Omega - \int_\Omega \alpha \nabla^2 u \cdot v \, d\Omega = \int_\Omega f v \, d\Omega
$$

### 2.3 对二阶导数项使用分部积分（Green 第一公式）

对第二项使用分部积分，将一个导数从 $u$ 转移到 $v$：

$$
\int_\Omega \alpha \nabla^2 u \cdot v \, d\Omega = \int_{\partial \Omega} \alpha \frac{\partial u}{\partial n} v \, ds - \int_\Omega \alpha \nabla u \cdot \nabla v \, d\Omega
$$

因为 $v$ 在边界 $\partial \Omega$ 上为零（$v|_{\partial \Omega} = 0$），边界项消失：

$$
\int_\Omega \alpha \nabla^2 u \cdot v \, d\Omega = - \int_\Omega \alpha \nabla u \cdot \nabla v \, d\Omega
$$

### 2.4 最终弱形式

代回原式得到弱形式：

$$
\int_\Omega \frac{\partial u}{\partial t} v \, d\Omega + \int_\Omega \alpha \nabla u \cdot \nabla v \, d\Omega = \int_\Omega f v \, d\Omega
$$

或简记为：求 $u(\cdot, t) \in H^1(\Omega)$ 满足 $u|_{\partial \Omega} = g$，使得对任意 $v \in H_0^1(\Omega)$：

$$
\left( \frac{\partial u}{\partial t}, v \right) + a(u, v) = (f, v)
$$

其中双线性型 $a(u, v) = \int_\Omega \alpha \nabla u \cdot \nabla v \, d\Omega$。

### 2.5 空间离散（Galerkin 投影）

将 $u$ 用有限元基函数展开：

$$
u_h(x, y, t) = \sum_{j=1}^{N} U_j(t) \phi_j(x, y)
$$

取测试函数 $v = \phi_i$，代入弱形式得到半离散常微分方程组：

$$
M \frac{dU}{dt} + K U = F
$$

其中：
- 质量矩阵 $M_{ij} = \int_\Omega \phi_i \phi_j \, d\Omega$
- 刚度矩阵 $K_{ij} = \int_\Omega \alpha \nabla \phi_i \cdot \nabla \phi_j \, d\Omega$
- 载荷向量 $F_i = \int_\Omega f \phi_i \, d\Omega$

### 2.6 时间离散（隐式 Euler）

对 $\frac{dU}{dt}$ 用后向差分：

$$
M \frac{U^{n+1} - U^n}{\Delta t} + K U^{n+1} = F^{n+1}
$$

整理得线性系统：

$$
(M + \Delta t \cdot K) U^{n+1} = M U^n + \Delta t \cdot F^{n+1}
$$

每步求解一个稀疏线性系统即可得到 $U^{n+1}$。

## 3. Case 1：无源项齐次 Dirichlet 解析解推导

### 3.1 问题设定

PDE（无源项）：

$$
\frac{\partial u}{\partial t} = \alpha \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right), \quad (x, y) \in [0,1]^2, \ t \in [0, T]
$$

边界条件（齐次 Dirichlet）：

$$
u(0, y, t) = u(1, y, t) = u(x, 0, t) = u(x, 1, t) = 0
$$

初始条件：

$$
u(x, y, 0) = \sin(\pi x) \sin(\pi y)
$$

### 3.2 分离变量法

**Step 1**：设 $u(x, y, t) = X(x) Y(y) T(t)$，代入 PDE：

$$
X(x) Y(y) T'(t) = \alpha \left( X''(x) Y(y) T(t) + X(x) Y''(y) T(t) \right)
$$

两边除以 $\alpha X(x) Y(y) T(t)$：

$$
\frac{T'(t)}{\alpha T(t)} = \frac{X''(x)}{X(x)} + \frac{Y''(y)}{Y(y)}
$$

**Step 2**：等式左边仅依赖 $t$，右边仅依赖 $x$ 和 $y$，因此各部分必须为常数。设：

$$
\frac{X''(x)}{X(x)} = -\lambda_x, \quad \frac{Y''(y)}{Y(y)} = -\lambda_y
$$

$$
\frac{T'(t)}{\alpha T(t)} = -(\lambda_x + \lambda_y)
$$

得到三个 ODE：

$$
X'' + \lambda_x X = 0, \quad Y'' + \lambda_y Y = 0, \quad T' + \alpha(\lambda_x + \lambda_y) T = 0
$$

**Step 3**：求解 $X(x)$ 的边值问题。

边界条件 $u(0,y,t) = 0$ 和 $u(1,y,t) = 0$ 要求 $X(0) = 0$, $X(1) = 0$。

特征值问题 $X'' + \lambda_x X = 0$, $X(0) = X(1) = 0$ 的解为：

$$
\lambda_x = (m\pi)^2, \quad X_m(x) = \sin(m\pi x), \quad m = 1, 2, 3, \ldots
$$

同理对 $Y(y)$：

$$
\lambda_y = (n\pi)^2, \quad Y_n(y) = \sin(n\pi y), \quad n = 1, 2, 3, \ldots
$$

**Step 4**：求解 $T(t)$。

$$
T'(t) + \alpha(\lambda_x + \lambda_y) T(t) = 0 \implies T(t) = e^{-\alpha(\lambda_x + \lambda_y) t}
$$

一般解为叠加：

$$
u(x, y, t) = \sum_{m=1}^{\infty} \sum_{n=1}^{\infty} A_{mn} \, e^{-\alpha(m^2 + n^2)\pi^2 t} \sin(m\pi x) \sin(n\pi y)
$$

**Step 5**：由初始条件确定系数。

$$
u(x, y, 0) = \sum_{m=1}^{\infty} \sum_{n=1}^{\infty} A_{mn} \sin(m\pi x) \sin(n\pi y) = \sin(\pi x) \sin(\pi y)
$$

比较得 $A_{11} = 1$，其余 $A_{mn} = 0$。

### 3.3 最终解析解

$$
u_{\text{exact}}(x, y, t) = e^{-2\alpha \pi^2 t} \sin(\pi x) \sin(\pi y)
$$

### 3.4 验证

- $t = 0$ 时：$u = \sin(\pi x)\sin(\pi y)$ ✓ 满足初始条件
- $x = 0$ 或 $x = 1$：$\sin(\pi x) = 0$ ✓ 满足边界条件
- $y = 0$ 或 $y = 1$：$\sin(\pi y) = 0$ ✓ 满足边界条件
- 当 $t \to \infty$：$u \to 0$ ✓ 物理合理（无热源，热量耗散）

## 4. 符号说明

| 符号 | 说明 |
|------|------|
| \(u(x,y,t)\) | 温度场 |
| \(\alpha\) | 热扩散系数 |
| \(f(x,y,t)\) | 热源项 |
| \(\Omega\) | 二维计算区域 |