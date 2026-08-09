"""
工具函数 — 打乱、SMILES 转换、可视化、格式转换。

用法：
    from utils import shuffle_graph, graph_to_smiles, visualize_grid, graph_to_dict
"""

import os
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional


# ===========================================================================
# 顶点打乱
# ===========================================================================

def shuffle_graph(G: nx.Graph, seed: int = None) -> nx.Graph:
    """
    随机打乱顶点顺序，同时保持 atom 属性和 orbit_labels 同步更新。

    Args:
        G: 原始图（顶点按构建顺序排列）
        seed: 随机种子

    Returns:
        G_new: 顶点顺序被打乱的新图
    """
    rng = np.random.RandomState(seed)
    N = G.number_of_nodes()

    perm = np.arange(N)
    rng.shuffle(perm)                    # perm[old_idx] = new_idx
    inv_perm = np.argsort(perm)          # inv_perm[new_idx] = old_idx

    G_new = nx.Graph()

    # 复制节点（带属性）
    for new_i in range(N):
        old_i = inv_perm[new_i]
        G_new.add_node(new_i, **G.nodes[old_i])

    # 复制边
    for old_u, old_v in G.edges():
        G_new.add_edge(int(perm[old_u]), int(perm[old_v]))

    # 同步 graph 级元数据
    for key in ('ring_size', 'attach_positions', 'subst_name'):
        if key in G.graph:
            G_new.graph[key] = G.graph[key]

    # 同步 orbit_labels（打乱）
    ol = G.graph.get('orbit_labels', None)
    if ol is not None:
        ol_arr = np.asarray(ol)
        G_new.graph['orbit_labels'] = ol_arr[inv_perm]

    return G_new


# ===========================================================================
# SMILES 转换
# ===========================================================================

def graph_to_smiles(G: nx.Graph) -> Optional[str]:
    """
    将 networkx 图转换为 SMILES 字符串。

    所有原子视为碳（原子序数 6），边视为单键。

    Args:
        G: 分子图

    Returns:
        SMILES 字符串，或 None（转换失败时）
    """
    try:
        from rdkit import Chem
    except ImportError:
        return None

    N = G.number_of_nodes()
    mol = Chem.RWMol()

    for i in range(N):
        atom_type = G.nodes[i].get('atom', 0)
        atomic_num = {0: 6, 1: 7, 2: 8, 3: 9}.get(atom_type, 6)
        mol.AddAtom(Chem.Atom(atomic_num))

    for u, v in G.edges():
        if u < v:
            mol.AddBond(int(u), int(v), Chem.BondType.SINGLE)

    try:
        Chem.SanitizeMol(mol)
        mol = Chem.RemoveHs(mol)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ===========================================================================
# 可视化
# ===========================================================================

def visualize_grid(
    graphs: List[nx.Graph],
    save_path: str,
    n_cols: int = 4,
    with_smiles: bool = True,
) -> bool:
    """
    将图列表可视化为分子结构网格图（通过 RDKit）。

    Args:
        graphs: 图列表
        save_path: 输出图片路径 (.png)
        n_cols: 每行列数
        with_smiles: 是否显示 SMILES 作为图例

    Returns:
        True 如果成功保存
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import Draw
    except ImportError:
        print("[warn] RDKit not installed, skipping visualization")
        return False

    mols = []
    legends = []
    for G in graphs:
        smiles = graph_to_smiles(G)
        if smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                mols.append(mol)
                legends.append(smiles if with_smiles else '')

    if not mols:
        print("[warn] No valid molecules to visualize")
        return False

    n_rows = max(1, (len(mols) + n_cols - 1) // n_cols)
    img = Draw.MolsToGridImage(
        mols,
        molsPerRow=n_cols,
        subImgSize=(300, 300),
        legends=legends[:len(mols)],
    )

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    img.save(save_path)
    print(f"Saved visualization → {save_path}")
    return True


# ===========================================================================
# 格式转换
# ===========================================================================

def graph_to_dict(G: nx.Graph, n_sym: int = 3) -> dict:
    """
    将 networkx 图转换为输出格式。

    Args:
        G: 分子图
        n_sym: C_n 对称阶数（默认 3）

    Returns:
        dict with keys: adj, x, n, k, subst_type
    """
    N = G.number_of_nodes()
    adj = nx.to_numpy_array(G).astype(np.int8)

    # 节点特征 one-hot（维度 4：C, N, O, F）
    d = 4
    x = np.zeros((N, d), dtype=np.float32)
    for i in range(N):
        atom_type = G.nodes[i].get('atom', 0)
        x[i, min(atom_type, d - 1)] = 1.0

    return {
        'adj': adj,
        'x': x,
        'n': n_sym,
        'k': N // n_sym,
        'subst_type': G.graph.get('subst_name', 'unknown'),
    }
