import torch
import torch.nn as nn


class PINN(nn.Module):
    """物理信息神经网络（正问题）。

    输入 (x, y, t) → MLP → 输出 u
    """

    def __init__(self, layers=None):
        super().__init__()
        if layers is None:
            layers = [3, 64, 64, 64, 64, 1]

        net = []
        for i in range(len(layers) - 2):
            linear = nn.Linear(layers[i], layers[i + 1])
            nn.init.xavier_normal_(linear.weight)
            nn.init.zeros_(linear.bias)
            net.append(linear)
            net.append(nn.Tanh())
        # 最后一层无激活
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)

    def forward(self, x, y, t):
        inp = torch.cat([x, y, t], dim=1)
        return self.net(inp)
