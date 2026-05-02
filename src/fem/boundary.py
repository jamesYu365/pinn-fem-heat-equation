import numpy as np
from scipy.sparse import lil_matrix


def apply_dirichlet(K, M, F, boundary_nodes, boundary_values=None):
    """施加 Dirichlet 边界条件。

    修改 K, M, F 使得边界节点上 u = boundary_values。
    齐次情况 boundary_values 全为零。

    返回修改后的 K, M, F。
    """
    num_nodes = K.shape[0]
    K = lil_matrix(K).copy()
    M = lil_matrix(M).copy()
    F = F.copy()

    if boundary_values is None:
        boundary_values = np.zeros(len(boundary_nodes))

    for idx, bd_idx in enumerate(boundary_nodes):
        val = boundary_values[idx]
        # 修正右端项：减去边界列的贡献
        for j in range(num_nodes):
            if j not in boundary_nodes:
                F[j] -= K[j, bd_idx] * val
                F[j] -= M[j, bd_idx] * val

        # 边界行归零，对角线置 1
        for j in range(num_nodes):
            K[bd_idx, j] = 0
            M[bd_idx, j] = 0
        K[bd_idx, bd_idx] = 1.0
        M[bd_idx, bd_idx] = 1.0
        F[bd_idx] = val

    return K.tocsc(), M.tocsc(), F
