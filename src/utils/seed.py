import random
import numpy as np
import torch


def set_seed(seed):
    """固定所有随机源，确保实验可重复。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
