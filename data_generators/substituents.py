"""
取代基注册表 — 可扩展的取代基构建模块。

每个取代基通过分步「挂甲基」的方式构建，每一步在指定父节点上新增一个碳原子。
所有新增原子自动获得新的轨道编号。

新增取代基方式：
    1. 在 SUBSTITUENT_RECIPES 中添加条目
    2. 每个 step 指定 parent（挂载到哪些节点）和 name（描述）

当前支持的取代基：
    - methyl    : 1 步，挂 1 个碳到环上
    - ethyl     : 2 步，挂碳到环 → 挂碳到上一步
    - isopropyl : 3 步，挂碳到环 → 挂碳到上一步 → 挂碳到上一步（分支）
"""

from typing import List, Dict, Any

# ===========================================================================
# 取代基配方定义
# ===========================================================================
#
# 每个取代基是一个 list of steps，每一步描述一次「批量挂甲基」操作。
#
# step 字段：
#   parent   : 'ring' → 挂到环的挂载点上
#              'step_N' → 挂到第 N 步新增的节点上
#   name     : 可读名称（用于调试/统计）
#
# 轨道分配：
#   环上节点 → 轨道 0（挂载点）或轨道 1（非挂载点）
#   第 i 步新增的节点 → 轨道 (2 + i)

SUBSTITUENT_RECIPES: Dict[str, List[Dict[str, Any]]] = {
    'methyl': [
        {'parent': 'ring',  'name': 'methyl_c'},
    ],
    'ethyl': [
        {'parent': 'ring',   'name': 'first_c'},
        {'parent': 'step_0', 'name': 'terminal_c'},
    ],
    'isopropyl': [
        {'parent': 'ring',   'name': 'branch_c'},
        {'parent': 'step_0', 'name': 'tip_c'},
        {'parent': 'step_0', 'name': 'branch_methyl'},
    ],
}


# ===========================================================================
# 辅助
# ===========================================================================

def list_substituents() -> List[str]:
    """列出所有已注册的取代基名称。"""
    return sorted(SUBSTITUENT_RECIPES.keys())


def get_recipe(subst_name: str) -> List[Dict[str, Any]]:
    """获取取代基的构建配方。"""
    if subst_name not in SUBSTITUENT_RECIPES:
        raise ValueError(
            f"Unknown substituent: '{subst_name}'. "
            f"Available: {list_substituents()}"
        )
    return SUBSTITUENT_RECIPES[subst_name]
