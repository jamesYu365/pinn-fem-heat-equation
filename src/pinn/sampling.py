import torch
import numpy as np


def sample_collocation(num_points, T_end, device="cpu"):
    """域内配点采样：(x,y,t) ∈ (0,1)² × (0,T)。"""
    x = torch.rand(num_points, 1, device=device)
    y = torch.rand(num_points, 1, device=device)
    t = torch.rand(num_points, 1, device=device) * T_end
    return x, y, t


def sample_initial(num_points, device="cpu"):
    """初始条件采样：(x,y) ∈ (0,1)², t=0。"""
    x = torch.rand(num_points, 1, device=device)
    y = torch.rand(num_points, 1, device=device)
    return x, y


def sample_boundary(num_points, T_end, device="cpu"):
    """边界条件采样：四条边上随机采样 (x,y,t)。"""
    n4 = num_points // 4
    remainder = num_points - 4 * n4
    counts = [n4 + (1 if i < remainder else 0) for i in range(4)]

    # x=0
    x0 = torch.zeros(counts[0], 1, device=device)
    y0 = torch.rand(counts[0], 1, device=device)
    # x=1
    x1 = torch.ones(counts[1], 1, device=device)
    y1 = torch.rand(counts[1], 1, device=device)
    # y=0
    x2 = torch.rand(counts[2], 1, device=device)
    y2 = torch.zeros(counts[2], 1, device=device)
    # y=1
    x3 = torch.rand(counts[3], 1, device=device)
    y3 = torch.ones(counts[3], 1, device=device)

    x_bc = torch.cat([x0, x1, x2, x3], dim=0)
    y_bc = torch.cat([y0, y1, y2, y3], dim=0)
    t_bc = torch.rand(x_bc.shape[0], 1, device=device) * T_end

    return x_bc, y_bc, t_bc
