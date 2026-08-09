"""
图组装器 — 将环与取代基拼接为完整分子图。

用法：
    from ring_builder import build_ring
    from assembler import assemble
    ring = build_ring(6)
    G = assemble(ring, 'isopropyl', attach_positions=[0, 2, 4])
"""

import numpy as np
import networkx as nx
from typing import List, Tuple

try:
    from .substituents import get_recipe
except ImportError:
    from substituents import get_recipe


def assemble(
    ring: nx.Graph,
    subst_name: str,
    attach_positions: List[int],
) -> nx.Graph:
    """
    将取代基拼接到环的指定位置，返回完整分子图。

    环上节点保留原编号，取代基节点从 ring_size 开始递增。
    所有节点的 'atom' 属性为 0（碳）。

    Args:
        ring: 环图（来自 build_ring）
        subst_name: 取代基名称（'methyl' | 'ethyl' | 'isopropyl'）
        attach_positions: 环上挂载位置列表（如 [0, 2, 4]）

    Returns:
        G: 完整分子图
            - G.nodes[i]['atom']: 原子类型
            - G.graph['orbit_labels']: np.ndarray (N,)，轨道编号
            - G.graph['ring_size']: 环大小
            - G.graph['attach_positions']: 挂载位置
            - G.graph['subst_name']: 取代基名称
    """
    recipe = get_recipe(subst_name)
    G = ring.copy()
    ring_size = ring.number_of_nodes()

    # ── 环的轨道标签 ──
    attach_set = set(attach_positions)
    orbit_labels = np.zeros(ring_size, dtype=int)
    for i in range(ring_size):
        orbit_labels[i] = 0 if i in attach_set else 1

    # ── 记录每步新增的节点（用于后续步骤的 parent 引用）──
    step_nodes: List[List[int]] = []

    # ── 逐步挂甲基 ──
    for step_idx, step in enumerate(recipe):
        orbit_id = 2 + step_idx

        # 确定本步的父节点列表
        parent_ref = step['parent']
        if parent_ref == 'ring':
            parents = list(attach_positions)
        elif parent_ref.startswith('step_'):
            ref_idx = int(parent_ref.split('_')[1])
            parents = step_nodes[ref_idx]
        else:
            raise ValueError(f"Unknown parent reference: {parent_ref}")

        new_nodes = []
        for parent in parents:
            new_id = G.number_of_nodes()
            G.add_node(new_id, atom=0)
            G.add_edge(parent, new_id)
            new_nodes.append(new_id)

        step_nodes.append(new_nodes)
        orbit_labels = np.append(
            orbit_labels,
            np.full(len(new_nodes), orbit_id, dtype=int),
        )

    # ── 存储元数据 ──
    G.graph['orbit_labels'] = orbit_labels
    G.graph['ring_size'] = ring_size
    G.graph['attach_positions'] = list(attach_positions)
    G.graph['subst_name'] = subst_name

    return G
