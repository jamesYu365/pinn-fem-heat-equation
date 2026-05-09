class WarmupLinearScheduler:
    """warmup + linear decay 学习率调度器。

    阶段：
    - [0, warmup_epochs): 线性从 0 升到各 param group 的初始学习率
    - [warmup_epochs, total_epochs): 按 peak_lr -> end_lr 的比例衰减

    多个 param group 会保留各自初始学习率的相对倍率。例如反问题中
    网络参数使用 peak_lr，alpha 参数使用 alpha_lr，调度时不应覆盖成同一值。
    """

    def __init__(self, optimizer, peak_lr, end_lr, total_epochs, warmup_epochs=0):
        self.optimizer = optimizer
        self.peak_lr = float(peak_lr)
        self.end_lr = float(end_lr)
        self.total_epochs = max(total_epochs, 1)
        self.warmup_epochs = min(warmup_epochs, self.total_epochs - 1)
        self.initial_lrs = [float(group["lr"]) for group in optimizer.param_groups]

    def step(self, epoch):
        """根据当前 epoch 更新 optimizer 学习率。"""
        if epoch < self.warmup_epochs:
            scale = epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scheduled_lr = self.peak_lr + (self.end_lr - self.peak_lr) * progress
            scale = scheduled_lr / self.peak_lr if self.peak_lr > 0 else 1.0
        for group, initial_lr in zip(self.optimizer.param_groups, self.initial_lrs):
            group["lr"] = initial_lr * scale
