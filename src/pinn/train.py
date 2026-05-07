import torch
import copy
from .loss import component_losses
from .lr_scheduler import WarmupLinearScheduler


class AdaptiveLossBalancer:
    """用 EMA 平衡 PDE/IC/BC 的加权贡献，抑制单个约束项突然主导更新。"""

    def __init__(self, base_weights, enabled=False, beta=0.98, min_scale=0.2,
                 max_scale=5.0, eps=1e-12):
        self.base_weights = {
            "pde": float(base_weights["lambda_r"]),
            "ic": float(base_weights["lambda_ic"]),
            "bc": float(base_weights["lambda_bc"]),
        }
        self.enabled = bool(enabled)
        self.beta = float(beta)
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.eps = float(eps)
        self.ema_losses = None
        self.current_weights = dict(self.base_weights)

    def weights(self, losses):
        if not self.enabled:
            self.current_weights = dict(self.base_weights)
            return self.current_weights

        detached = {name: losses[name].detach().clamp_min(self.eps) for name in self.base_weights}
        if self.ema_losses is None:
            self.ema_losses = detached
            self.current_weights = dict(self.base_weights)
            return self.current_weights
            self.ema_losses = {
                name: self.beta * self.ema_losses[name] + (1.0 - self.beta) * detached[name]
                for name in self.base_weights
            }

        contributions = {
            name: self.base_weights[name] * self.ema_losses[name]
            for name in self.base_weights
        }
        target = sum(contributions.values()) / len(contributions)

        weights = {}
        for name, base_weight in self.base_weights.items():
            scale = (target / contributions[name]).clamp(self.min_scale, self.max_scale)
            weights[name] = float(base_weight * scale.item())

        self.current_weights = weights
        return weights


def _weighted_total(losses, weights):
    return weights["pde"] * losses["pde"] + weights["ic"] * losses["ic"] + weights["bc"] * losses["bc"]


def _serialize_components(losses, weights):
    return {
        "pde": losses["pde"].item(),
        "ic": losses["ic"].item(),
        "bc": losses["bc"].item(),
        "w_pde": float(weights["pde"]),
        "w_ic": float(weights["ic"]),
        "w_bc": float(weights["bc"]),
    }


def train(model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc,
          alpha, epochs, lr, loss_weights, log_every=1000, device="cpu",
          clip_grad_norm=None, val_data=None, eval_callback=None,
          early_stop_patience=None, adaptive_loss=None, warmup_epochs=0,
          peak_lr=None, end_lr=None):
    """PINN 训练循环。

    参数:
        peak_lr: warmup 到达的最大学习率；为 None 时使用 lr（向后兼容）。
        end_lr: 衰减结束时的学习率；为 None 时使用 lr（不衰减）。
        其余参数同前。
    """
    peak_lr = float(peak_lr if peak_lr is not None else lr)
    end_lr = float(end_lr if end_lr is not None else lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=peak_lr)
    scheduler = WarmupLinearScheduler(optimizer, peak_lr, end_lr, epochs, warmup_epochs)
    adaptive_loss = adaptive_loss or {}
    balancer = AdaptiveLossBalancer(loss_weights, **adaptive_loss)

    loss_history = []
    components_history = []
    val_history = []

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    patience_counter = 0

    model.train()
    for epoch in range(1, epochs + 1):
        scheduler.step(epoch)
        optimizer.zero_grad()
        losses = component_losses(
            model, x_r, y_r, t_r, x_ic, y_ic, x_bc, y_bc, t_bc, alpha,
        )
        effective_weights = balancer.weights(losses)
        loss = _weighted_total(losses, effective_weights)
        loss.backward()
        if clip_grad_norm is not None and clip_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)
        optimizer.step()

        loss_history.append(loss.item())
        components = _serialize_components(losses, effective_weights)
        components_history.append(components)

        # 随机配点验证（每 epoch）
        if val_data is not None:
            with torch.enable_grad():
                val_losses = component_losses(
                    model,
                    val_data["x_r"], val_data["y_r"], val_data["t_r"],
                    val_data["x_ic"], val_data["y_ic"],
                    val_data["x_bc"], val_data["y_bc"], val_data["t_bc"],
                    alpha,
                )
                val_loss = _weighted_total(val_losses, effective_weights)
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
                   f"| BC: {components['bc']:.6e} "
                   f"| w=({components['w_pde']:.2g},"
                   f"{components['w_ic']:.2g},{components['w_bc']:.2g})")
            if val_data is not None:
                msg += f" | Val: {val_history[-1]:.6e} (best={best_val_loss:.6e})"
            print(msg)

            if eval_callback is not None:
                eval_callback(model, epoch)

    # 没有验证集时用最后一个 epoch 的状态
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())

    return loss_history, components_history, val_history, best_state, best_epoch
