
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

（在此记录 FEM 弱形式推导步骤与公式）

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