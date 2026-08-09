import torch
from torch import Tensor


def sampleAfterSingleEmbed(self, atomFeature: Tensor, uni_idx, max_graph_len) -> Tensor:
    """
    单个晶体特征预提取后,根据原子类型和邻域进行抽样

    Args:
        atomFeature (Tensor): 预提取后的原子特征
        uni_idx (list[list]): 标识独特原子
        max_graph_len (int): 随机挑选后的最大长度

    Returns:
        Tensor: 抽样预提取后的原子特征
    """
    uniqAtomIdxList = torch.LongTensor([random.choice(u) for u in uni_idx])[: max_graph_len]
    return atomFeature[uniqAtomIdxList]