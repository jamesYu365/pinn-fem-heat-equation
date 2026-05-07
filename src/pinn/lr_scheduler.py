class WarmupLinearScheduler:
    """warmup + linear decay 学习率调度器。

    阶段：
    - [0, warmup_epochs): 线性从 0 升到 peak_lr
    - [warmup_epochs, total_epochs): 线性从 peak_lr 降到 end_lr
    """

    def __init__(self, optimizer, peak_lr, end_lr, total_epochs, warmup_epochs=0):
        self.optimizer = optimizer
        self.peak_lr = peak_lr
        self.end_lr = end_lr
        self.total_epochs = max(total_epochs, 1)
        self.warmup_epochs = min(warmup_epochs, self.total_epochs - 1)

    def step(self, epoch):
        """根据当前 epoch 更新 optimizer 学习率。"""
        if epoch < self.warmup_epochs:
            lr = self.peak_lr * epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.peak_lr + (self.end_lr - self.peak_lr) * progress
        for group in self.optimizer.param_groups:
            group["lr"] = lr
