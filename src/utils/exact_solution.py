import numpy as np


def case1_exact(x, y, t, alpha):
    """Case 1 解析解：无源项齐次 Dirichlet。

    u(x,y,t) = exp(-2*alpha*pi^2*t) * sin(pi*x) * sin(pi*y)
    """
    return np.exp(-2 * alpha * np.pi**2 * t) * np.sin(np.pi * x) * np.sin(np.pi * y)
