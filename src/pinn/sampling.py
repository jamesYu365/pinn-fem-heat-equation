import torch


def _stratified_unit_samples(num_points, device="cpu"):
    """在 [0, 1] 上分层采样，降低小样本随机聚集造成的 IC/BC 抖动。"""
    if num_points <= 0:
        return torch.empty(0, 1, device=device)
    edges = torch.linspace(0.0, 1.0, num_points + 1, device=device).unsqueeze(1)
    low = edges[:-1]
    high = edges[1:]
    return low + (high - low) * torch.rand(num_points, 1, device=device)


def _stratified_square_samples(num_points, device="cpu"):
    """在单位正方形上做 jittered grid 采样，保持返回数量精确为 num_points。"""
    if num_points <= 0:
        empty = torch.empty(0, 1, device=device)
        return empty, empty

    side = int(torch.ceil(torch.sqrt(torch.tensor(float(num_points)))).item())
    grid_i, grid_j = torch.meshgrid(
        torch.arange(side, device=device),
        torch.arange(side, device=device),
        indexing="ij",
    )
    xy = torch.stack([grid_i.reshape(-1), grid_j.reshape(-1)], dim=1).float()
    xy = (xy + torch.rand_like(xy)) / float(side)
    xy = xy[torch.randperm(xy.shape[0], device=device)[:num_points]]
    return xy[:, 0:1], xy[:, 1:2]


def sample_collocation(num_points, T_end, device="cpu"):
    """域内配点采样：(x,y,t) ∈ (0,1)² × (0,T)。"""
    x = torch.rand(num_points, 1, device=device)
    y = torch.rand(num_points, 1, device=device)
    t = torch.rand(num_points, 1, device=device) * T_end
    return x, y, t


def sample_initial(num_points, device="cpu"):
    """初始条件采样：(x,y) ∈ (0,1)², t=0。"""
    x, y = _stratified_square_samples(num_points, device=device)
    return x, y


def sample_boundary(num_points, T_end, device="cpu"):
    """边界条件采样：四条边上随机采样 (x,y,t)。"""
    n4 = num_points // 4
    remainder = num_points - 4 * n4
    counts = [n4 + (1 if i < remainder else 0) for i in range(4)]

    # x=0
    x0 = torch.zeros(counts[0], 1, device=device)
    y0 = _stratified_unit_samples(counts[0], device=device)
    # x=1
    x1 = torch.ones(counts[1], 1, device=device)
    y1 = _stratified_unit_samples(counts[1], device=device)
    # y=0
    x2 = _stratified_unit_samples(counts[2], device=device)
    y2 = torch.zeros(counts[2], 1, device=device)
    # y=1
    x3 = _stratified_unit_samples(counts[3], device=device)
    y3 = torch.ones(counts[3], 1, device=device)

    x_bc = torch.cat([x0, x1, x2, x3], dim=0)
    y_bc = torch.cat([y0, y1, y2, y3], dim=0)
    t_bc = torch.cat(
        [_stratified_unit_samples(count, device=device) for count in counts],
        dim=0,
    ) * T_end

    return x_bc, y_bc, t_bc
