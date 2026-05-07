import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve


def solve_implicit_euler(K, M, U0, dt, T_end, boundary_nodes, boundary_values=None):
    """隐式 Euler 时间推进求解半离散系统。

    (M + dt*K) U^{n+1} = M * U^n

    边界条件在组合后的系统矩阵上施加，保证 A[bd,bd] = 1。

    返回:
        U_history: 每个时间步的解向量列表（含初始条件）
        times: 对应的时间点列表
    """
    num_steps = int(round(T_end / dt))
    num_nodes = K.shape[0]
    bd_list = boundary_nodes.tolist() if hasattr(boundary_nodes, 'tolist') else list(boundary_nodes)
    bd_set = set(bd_list)

    if boundary_values is None:
        boundary_values = np.zeros(len(bd_list))
    bd_vals = np.asarray(boundary_values, dtype=float)

    # 组合系统矩阵 A = M + dt*K
    A = lil_matrix(M + dt * K)

    # 保存原始 A 的边界列数据（用于右端修正）
    n_bd = len(bd_list)
    bd_col_data = np.zeros((num_nodes, n_bd))
    for col_idx, bd_idx in enumerate(bd_list):
        for j in range(num_nodes):
            if j != bd_idx:
                bd_col_data[j, col_idx] = A[j, bd_idx]

    # 修改 A：边界行/列归零，对角线置 1
    for bd_idx in bd_list:
        for j in range(num_nodes):
            A[bd_idx, j] = 0
            A[j, bd_idx] = 0
        A[bd_idx, bd_idx] = 1.0

    A = A.tocsc()

    U = U0.copy()
    U_history = [U.copy()]
    times = [0.0]

    for step in range(1, num_steps + 1):
        rhs = M @ U

        # 右端修正：减去已知边界值的列贡献
        rhs -= bd_col_data @ bd_vals

        # 设置边界节点右端
        for idx, bd_idx in enumerate(bd_list):
            rhs[bd_idx] = bd_vals[idx]

        U = spsolve(A, rhs)
        U_history.append(U.copy())
        times.append(step * dt)

    return U_history, times
