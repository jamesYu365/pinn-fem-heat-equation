import numpy as np


def relative_l2_error(u_h, u_exact):
    """相对 L2 误差：||u_h - u_exact||_2 / ||u_exact||_2"""
    diff = u_h - u_exact
    return np.linalg.norm(diff) / np.linalg.norm(u_exact)


def max_absolute_error(u_h, u_exact):
    """最大绝对误差：max|u_h - u_exact|"""
    return np.max(np.abs(u_h - u_exact))
