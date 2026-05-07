import numpy as np


def relative_l2_error(u_h, u_exact):
    """相对 L2 误差：||u_h - u_exact||_2 / ||u_exact||_2"""
    diff = u_h - u_exact
    denom = np.linalg.norm(u_exact)
    if denom < 1e-12:
        return np.linalg.norm(diff)
    return np.linalg.norm(diff) / denom


def max_absolute_error(u_h, u_exact):
    """最大绝对误差：max|u_h - u_exact|"""
    return np.max(np.abs(u_h - u_exact))
