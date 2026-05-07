import torch
import copy
from .loss import total_loss


def train(model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc,
          alpha, epochs, lr, loss_weights, log_every=1000, device="cpu",
          clip_grad_norm=None, val_data=None, eval_callback=None,
          early_stop_patience=None):
    """PINN 训练循环。

    参数:
        val_data: 验证集字典，每 epoch 评估随机配点验证损失。
        eval_callback: 回调函数 fn(model, epoch)，每 log_every 步调用。
        early_stop_patience: 验证损失连续不下降的 epoch 数，超过则停止。
                             为 None 时不启用 early stop。

    返回:
        loss_history: 每步训练总损失列表
        components_history: 每步各分量损失列表
        val_history: 每步验证总损失列表
        best_state: 验证损失最低时的模型 state_dict
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []
    components_history = []
    val_history = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        loss, components = total_loss(
            model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc, alpha,
            **loss_weights
        )
        loss.backward()
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        loss_history.append(loss.item())
        components_history.append(components)

        # 随机配点验证（每 epoch）
        if val_data is not None:
            with torch.enable_grad():
                val_loss, _ = total_loss(
                    model,
                    val_data["x_r"], val_data["y_r"], val_data["t_r"],
                    val_data["x_ic"], val_data["y_ic"],
                    val_data["x_bc"], val_data["y_bc"], val_data["t_bc"],
                    alpha, **loss_weights
                )
                val_history.append(val_loss.item())

            # 追踪最优模型
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            # Early stop
            if (early_stop_patience is not None
                    and patience_counter >= early_stop_patience):
                print(f"  Early stop at epoch {epoch}: "
                      f"val loss 连续 {early_stop_patience} 步未改善 "
                      f"(best={best_val_loss:.6e})")
                break

        if epoch % log_every == 0:
            msg = (f"  Epoch {epoch:6d} | Loss: {loss.item():.6e} "
                   f"| PDE: {components['pde']:.6e} "
                   f"| IC: {components['ic']:.6e} "
                   f"| BC: {components['bc']:.6e}")
            if val_data is not None:
                msg += f" | Val: {val_history[-1]:.6e} (best={best_val_loss:.6e})"
            print(msg)

            if eval_callback is not None:
                eval_callback(model, epoch)

    # 没有验证集时用最后一个 epoch 的状态
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    return loss_history, components_history, val_history, best_state, best_epoch
