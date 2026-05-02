import numpy as np


def generate_structured_mesh(nx, ny, Lx=1.0, Ly=1.0):
    """生成结构化三角形网格。

    将 [0,Lx]x[0,Ly] 均匀分为 nx*ny 个矩形，每个矩形分为 2 个三角形。

    返回:
        nodes: (num_nodes, 2) 节点坐标
        elements: (num_elements, 3) 三角形单元连接表（节点索引）
        boundary_nodes: 边界节点索引数组
    """
    x = np.linspace(0, Lx, nx + 1)
    y = np.linspace(0, Ly, ny + 1)
    xx, yy = np.meshgrid(x, y)
    nodes = np.column_stack([xx.ravel(), yy.ravel()])

    elements = []
    for j in range(ny):
        for i in range(nx):
            # 当前矩形四个角节点（按行优先编号）
            n0 = j * (nx + 1) + i
            n1 = n0 + 1
            n2 = n0 + (nx + 1)
            n3 = n2 + 1
            elements.append([n0, n1, n3])
            elements.append([n0, n3, n2])
    elements = np.array(elements)

    # 边界节点：x=0, x=Lx, y=0, y=Ly 上的节点
    boundary_nodes = []
    for idx, (px, py) in enumerate(nodes):
        if np.isclose(px, 0) or np.isclose(px, Lx) or np.isclose(py, 0) or np.isclose(py, Ly):
            boundary_nodes.append(idx)
    boundary_nodes = np.array(boundary_nodes)

    return nodes, elements, boundary_nodes
