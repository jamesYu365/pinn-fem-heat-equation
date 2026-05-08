import torch


def compute_pde_residual(model, x, y, t, alpha):
    """计算 PDE 残差：r = ∂u/∂t - α(∂²u/∂x² + ∂²u/∂y²)。

    x, y, t 需要 requires_grad=True。
    """
    x = x.clone().requires_grad_(True)
    y = y.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)

    u = model(x, y, t)

    # 一阶导数
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u), create_graph=True)[0]

    # 二阶导数
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, grad_outputs=torch.ones_like(u_y), create_graph=True)[0]

    residual = u_t - alpha * (u_xx + u_yy)
    return residual


def pde_loss(model, x, y, t, alpha):
    """PDE 残差损失（MSE）。"""
    r = compute_pde_residual(model, x, y, t, alpha)
    return torch.mean(r ** 2)


def ic_loss(model, x, y):
    """初始条件损失：u(x,y,0) = sin(πx)sin(πy)。"""
    t = torch.zeros_like(x)
    u_pred = model(x, y, t)
    u_exact = torch.sin(torch.pi * x) * torch.sin(torch.pi * y)
    return torch.mean((u_pred - u_exact) ** 2)


def bc_loss(model, x, y, t):
    """边界条件损失：u|∂Ω = 0。"""
    u_pred = model(x, y, t)
    return torch.mean(u_pred ** 2)


def data_loss(model, x, y, t, u_obs):
    """观测数据拟合损失：MSE(u_pred, u_obs)。"""
    u_pred = model(x, y, t)
    return torch.mean((u_pred - u_obs) ** 2)


def component_losses(model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc, alpha,
                     obs_data=None):
    """返回可反传的各分量损失张量。

    obs_data: None 或 (x_data, y_data, t_data, u_data) 元组。
    """
    losses = {
        "pde": pde_loss(model, x_r, y_r, t_r, alpha),
        "ic": ic_loss(model, x_ic, y_ic),
        "bc": bc_loss(model, x_bc, y_bc, t_bc),
    }
    if obs_data is not None:
        x_d, y_d, t_d, u_d = obs_data
        losses["data"] = data_loss(model, x_d, y_d, t_d, u_d)
    return losses


def total_loss(model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc, alpha,
               lambda_r=1.0, lambda_ic=100.0, lambda_bc=100.0,
               effective_weights=None):
    """计算总损失。"""
    losses = component_losses(model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc, alpha)
    weights = effective_weights or {
        "pde": lambda_r,
        "ic": lambda_ic,
        "bc": lambda_bc,
    }

    loss = weights["pde"] * losses["pde"] + weights["ic"] * losses["ic"] + weights["bc"] * losses["bc"]
    return loss, {name: value.item() for name, value in losses.items()}
