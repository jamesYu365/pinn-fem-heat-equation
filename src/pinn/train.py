import torch
from .loss import total_loss


def train(model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc,
          alpha, epochs, lr, loss_weights, log_every=1000, device="cpu"):
    """PINN 训练循环。

    返回:
        loss_history: 每步总损失列表
        components_history: 每步各分量损失列表
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    components_history = []

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, components = total_loss(
            model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc, alpha,
            **loss_weights
        )
        loss.backward()
        optimizer.step()

        loss_history.append(loss.item())
        components_history.append(components)

        if epoch % log_every == 0:
            print(f"  Epoch {epoch:6d} | Loss: {loss.item():.6e} "
                  f"| PDE: {components['pde']:.6e} "
                  f"| IC: {components['ic']:.6e} "
                  f"| BC: {components['bc']:.6e}")

    return loss_history, components_history
