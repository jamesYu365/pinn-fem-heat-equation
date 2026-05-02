
# 解析解与弱形式推导

> 文件创建日期：2026-05-01  
> 作者：算法开发者

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

## 3. 解析解推导示例

（例如：无源项齐次 Dirichlet 情况）

$$
u_{\text{exact}}(x, y, t) = e^{-2\alpha \pi^2 t}\sin(\pi x)\sin(\pi y)
$$

## 4. 符号说明

| 符号 | 说明 |
|------|------|
| \(u(x,y,t)\) | 温度场 |
| \(\alpha\) | 热扩散系数 |
| \(f(x,y,t)\) | 热源项 |
| \(\Omega\) | 二维计算区域 |