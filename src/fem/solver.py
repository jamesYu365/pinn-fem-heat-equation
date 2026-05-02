import numpy as np
from scipy.sparse.linalg import spsolve


def solve_implicit_euler(K, M, U0, dt, T_end, boundary_nodes, boundary_values=None):
    """隐式 Euler 时间推进求解半离散系统。

    (M + dt*K) U^{n+1} = M * U^n + dt * F^{n+1}

    对 Case 1 齐次 Dirichlet 无源项，F = 0。

    返回:
        U_history: 每个时间步的解向量列表（含初始条件）
        times: 对应的时间点列表
    """
    num_steps = int(round(T_end / dt))
    A = M + dt * K

    U = U0.copy()
    U_history = [U.copy()]
    times = [0.0]

    for step in range(1, num_steps + 1):
        rhs = M @ U
        # 边界条件：保持边界节点为指定值
        if boundary_values is None:
            boundary_values = np.zeros(len(boundary_nodes))
        for idx, bd_idx in enumerate(boundary_nodes):
            rhs[bd_idx] = boundary_values[idx]

        U = spsolve(A, rhs)
        U_history.append(U.copy())
        times.append(step * dt)

    return U_history, times
