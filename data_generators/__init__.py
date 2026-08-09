"""
data_generators — 可扩展的分子图数据生成器。

通过环 + 取代基组装的方式生成具有 C_n 旋转对称的分子图结构。

主要组件：
    ring_builder  — 参数化环构建
    substituents  — 取代基注册表（可扩展）
    assembler     — 环 + 取代基拼接
    utils         — 打乱、SMILES、可视化、格式转换
    generate      — CLI 批量生成入口

用法示例：
    >>> from ring_builder import build_ring
    >>> from assembler import assemble
    >>> from utils import shuffle_graph, graph_to_dict
    >>>
    >>> ring = build_ring(6)
    >>> G = assemble(ring, 'isopropyl', attach_positions=[0, 2, 4])
    >>> G = shuffle_graph(G, seed=42)
    >>> data = graph_to_dict(G, n_sym=3)
"""

from .ring_builder import build_ring
from .substituents import SUBSTITUENT_RECIPES, list_substituents, get_recipe
from .assembler import assemble
from .utils import (
    shuffle_graph,
    graph_to_smiles,
    visualize_grid,
    graph_to_dict,
)

__all__ = [
    'build_ring',
    'SUBSTITUENT_RECIPES',
    'list_substituents',
    'get_recipe',
    'assemble',
    'shuffle_graph',
    'graph_to_smiles',
    'visualize_grid',
    'graph_to_dict',
]
