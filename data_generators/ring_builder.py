"""
环构建器 — 参数化构建碳环。

用法：
    from ring_builder import build_ring
    G = build_ring(6)  # 6 元环
"""

import networkx as nx


def build_ring(n_nodes: int = 6) -> nx.Graph:
    """
    构建 n_nodes 个碳原子组成的环（0−1−2−...−(n_nodes-1)−0）。

    Args:
        n_nodes: 环的大小（默认 6，即苯环骨架）

    Returns:
        G: networkx Graph，节点含 'atom' 属性（均为 0=碳）
    """
    G = nx.Graph()
    for i in range(n_nodes):
        G.add_node(i, atom=0)
    for i in range(n_nodes):
        G.add_edge(i, (i + 1) % n_nodes)
    return G
