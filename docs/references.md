# 参考文献与公式笔记

## 1. FEM 与 PINN 相关文献

1. Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. *Journal of Computational Physics*, 378, 686–707.
2. Zienkiewicz, O. C., Taylor, R. L., & Zhu, J. Z. (2013). The Finite Element Method: Its Basis and Fundamentals. Elsevier.

## 2. 公式笔记

- FEM 弱形式：
$$
\int_{\Omega} \frac{\partial u}{\partial t} v \, d\Omega + \int_{\Omega} \alpha \nabla u \cdot \nabla v \, d\Omega = \int_{\Omega} f v \, d\Omega
$$

- PINN PDE 残差：
$$
r_\theta(x, y, t) = \frac{\partial u_\theta}{\partial t} - \alpha \left(\frac{\partial^2 u_\theta}{\partial x^2} + \frac{\partial^2 u_\theta}{\partial y^2}\right) - f(x, y, t)
$$