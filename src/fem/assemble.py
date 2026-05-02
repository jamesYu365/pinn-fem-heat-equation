import numpy as np
from scipy.sparse import lil_matrix


def _element_stiffness(coords, alpha):
    """计算单个三角形单元的刚度矩阵（3x3）。"""
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

    b = np.array([y2 - y3, y3 - y1, y1 - y2])
    c = np.array([x3 - x2, x1 - x3, x2 - x1])

    # Ke = alpha * (b*b^T + c*c^T) / (4*area)
    Ke = alpha * (np.outer(b, b) + np.outer(c, c)) / (4.0 * area)
    return Ke, area


def _element_mass(coords):
    """计算单个三角形单元的一致质量矩阵（3x3）。"""
    x1, y1 = coords[0]
    x2, y2 = coords[1]
    x3, y3 = coords[2]

    area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    Me = area / 12.0 * np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
    return Me


def assemble_global(nodes, elements, alpha):
    """组装全局刚度矩阵 K 和质量矩阵 M。

    返回:
        K: (num_nodes, num_nodes) 稀疏刚度矩阵
        M: (num_nodes, num_nodes) 稀疏质量矩阵
    """
    num_nodes = len(nodes)
    K = lil_matrix((num_nodes, num_nodes))
    M = lil_matrix((num_nodes, num_nodes))

    for el in elements:
        coords = nodes[el]
        Ke, _ = _element_stiffness(coords, alpha)
        Me = _element_mass(coords)

        for i in range(3):
            for j in range(3):
                K[el[i], el[j]] += Ke[i, j]
                M[el[i], el[j]] += Me[i, j]

    return K.tocsc(), M.tocsc()
