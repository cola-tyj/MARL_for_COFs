# MOFTransformer version 2.0.0
import random
import os
import torch
import torch.nn as nn
import numpy as np


class ConvLayer(nn.Module):
    """
    Convolutional operation on graphs
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(self, atom_fea_len, nbr_fea_len):
        """
        Args:
            atom_fea_len (int): 输入每个原子的维度
            nbr_fea_len (int): 原子间连接的特征向量的维度
        """
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.fc_full = nn.Linear(
            2 * self.atom_fea_len + self.nbr_fea_len, 2 * self.atom_fea_len  # 维度变化:两个原子特征+原子间连接的特征->两个原子特征
        )
        self.sigmoid = nn.Sigmoid()
        self.softplus1 = nn.Softplus()  # ReLU函数的平滑
        self.softplus2 = nn.Softplus()
        self.bn1 = nn.BatchNorm1d(2 * self.atom_fea_len)  # nn.BatchNorm1d 声明时指定要归一化的维度
        self.bn2 = nn.BatchNorm1d(self.atom_fea_len)

    def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
        """
        Forward pass

        N: Total number of atoms in the batch
        M: Max number of neighbors

        Args:
        atom_in_fea: Variable(torch.Tensor) shape (N, atom_fea_len)
          Atom hidden features before convolution: 输入原子的特征
        nbr_fea: Variable(torch.Tensor) shape (N, M, nbr_fea_len)
          Bond features of each atom's M neighbors: 每个原子直接相连的化学键特征
        nbr_fea_idx: torch.LongTensor shape (N, M)
          Indices of M neighbors of each atom: 每个原子的邻接原子index

        Returns:
        atom_out_fea: nn.Variable shape (N, atom_fea_len)
          Atom hidden features after convolution: 输出原子的特征,维度和输入一致

        """

        # N为原子数,M为最大邻接原子数
        N, M = nbr_fea_idx.shape

        # 邻接原子转化为向量[N, M, atom_fea_len]
        atom_nbr_fea = atom_in_fea[nbr_fea_idx, :]

        # 再拼接上自身特征、原子间连接的特征,得到[N, M, atom_fea_len*2+nrb_fea_len]作为输入
        total_nbr_fea = torch.cat(
            [
                atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
                # [N, atom_fea_len] -> [N, M, atom_fea_len] -> v_i
                atom_nbr_fea,  # [N, M, atom_fea_len] -> v_j
                nbr_fea,
            ],  # [N, M, nbr_fea_len] -> u(i,j)_k
            dim=2,
        )

        # 全连接层: [N, M, atom_fea_len*2+nrb_fea_len] -> [N, M, atom_fea_len*2]
        total_gated_fea = self.fc_full(total_nbr_fea)
        # BN: [N, M, atom_fea_len*2] -> [N*M, atom_fea_len*2] -> BN处理维度1 -> [N, M, atom_fea_len*2]
        total_gated_fea = self.bn1(
            total_gated_fea.view(-1, self.atom_fea_len * 2)
        ).view(
            N, M, self.atom_fea_len * 2
        )  

        # 沿着维度2均匀分割成2块,每块[N, M, atom_fea_len]
        nbr_filter, nbr_core = total_gated_fea.chunk(2, dim=2)

        # 两块分别走不同的激活函数
        nbr_filter = self.sigmoid(nbr_filter)
        nbr_core = self.softplus1(nbr_core)

        # 按位置相乘,指定dim=1将所有邻接位置信息相加,再BN处理维度1
        nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)  # [N, atom_fea_len]
        nbr_sumed = self.bn2(nbr_sumed)

        # 增加残差连接后再执行激活函数,得到[N, atom_fea_len]
        out = self.softplus2(atom_in_fea + nbr_sumed)

        return out


class GraphEmbeddings(nn.Module):
    """
    Generate Embedding layers made by only convolution layers of CGCNN (not pooling)
    (https://github.com/txie-93/cgcnn)
    """

    def __init__(
        self, atom_fea_len, nbr_fea_len, max_graph_len, hid_dim, n_conv=3, vis=False
    ):
        """
        构造无Polling的CGCNN

        Args:
            atom_fea_len (int): 输入每个原子的维度
            nbr_fea_len (int): 原子间连接特征的维度
            max_graph_len (int): 每个晶体截断后的最大原子数
            hid_dim (int): 隐层维度
            n_conv (int, optional): 卷积总层数. Defaults to 3.
            vis (bool, optional): 是否可视化展示. Defaults to False.
        """
        super().__init__()
        self.atom_fea_len = atom_fea_len
        self.nbr_fea_len = nbr_fea_len
        self.max_graph_len = max_graph_len
        self.hid_dim = hid_dim
        self.vis = vis
        self.embedding = nn.Embedding(119, atom_fea_len)  # 元素周期表共118种元素,每种映射为指定维度
        self.convs = nn.ModuleList(
            [
                ConvLayer(atom_fea_len=atom_fea_len, nbr_fea_len=nbr_fea_len)
                for _ in range(n_conv)
            ]
        )
        self.fc = nn.Linear(atom_fea_len, hid_dim)

    def forward(
        self, atom_num, nbr_idx, nbr_fea, crystal_atom_idx, uni_idx, uni_count, moc=None, cifIdList=None
    ):
        """
        B: batch_size
        N': 当前batch的总原子数
        M: 一个原子的最大连接数
        atom_fea_len: 一个原子的特征维度
        nbr_fea_len: 一个连接的特征维度

        Args:
            atom_num (tensor): [N'],输入原子序号
            nbr_idx (tensor): [N', M],邻接原子的序号
            nbr_fea (tensor): [N', M, nbr_fea_len],连接原子的化学键特征
            crystal_atom_idx (list): [B],列表,每个晶体对应其中一个位置
            uni_idx (list) : [B]按原子类型、邻域类型划分后的原子下标
            uni_count (list) : [B]原子重复数
        Returns:
            new_atom_fea (tensor): [B, max_graph_len, hid_dim],经过卷积并截断后的原子特征
            mask (tensor): [B, max_graph_len],掩码
        """
        assert self.nbr_fea_len == nbr_fea.shape[-1]

        # 将原子序号embedding [N', atom_fea_len]
        atom_fea = self.embedding(atom_num)

        # 若干层卷积+一次全连接
        for conv in self.convs:
            atom_fea = conv(atom_fea, nbr_fea, nbr_idx)  # [N', atom_fea_len]
        atom_fea = self.fc(atom_fea)  # [N', hid_dim]

        # TODO:直接保存未抽样的嵌入结果,需要改化合物类型
        # saveNoSampleAtomEmbedding(cifIdList=cifIdList, atomFeature=atom_fea, atomIdx=crystal_atom_idx, savePath="/home/zhangyi/dataset/COF-hcb-dataset/graph_embed_full")

        # 将每个晶体的原子收到对应的晶体中,并执行裁剪 [N', hid_dim] -> [B, max_graph_len, hid_dim]
        new_atom_fea, mask, mo_label = self.reconstruct_batch(
            atom_fea, crystal_atom_idx, uni_idx, uni_count, moc
        )
        # [B, max_graph_len, hid_dim], [B, max_graph_len]
        return new_atom_fea, mask, mo_label  # None will be replaced with MOC

    def reconstruct_batch(self, atom_fea, crystal_atom_idx, uni_idx, uni_count, moc):
        """
        重构输出特征,对于超出max_graph_len的原子特征做截断

        Args:
            atom_fea (tensor): [N', hid_dim],卷积层输出的特征
            crystal_atom_idx (_type_): 从晶体索引到原子索引的映射
            uni_idx (list): 按原子类型、邻域类型划分后的原子下标
            moc (Bool): 任务是否包含MOC,对于COFs总为False/None

        Returns:
            重构后的特征[batch, max_graph_len, hid_dim],以及掩码[batch, max_graph_len]
            返回的每个晶体最多对应max_graph_len个原子,每个原子的对应hid_dim维度的特征
        """
        batch_size = len(crystal_atom_idx)

        # 构建返回结果,tensorA.to(tensorB)转移到相同的设备上
        new_atom_fea = torch.full(
            size=[batch_size, self.max_graph_len, self.hid_dim], fill_value=0.0
        ).to(atom_fea)

        mo_label = torch.full(
            size=[batch_size, self.max_graph_len], fill_value=-100.0
        ).to(atom_fea)

        for bi, c_atom_idx in enumerate(crystal_atom_idx):  # 枚举每个晶体,bi是batch编号
            # 重复原子中,每一种只选一个,且选择个数最多为max_graph_len
            idx_ = torch.LongTensor([random.choice(u) for u in uni_idx[bi]])[
                : self.max_graph_len
            ]
            
            # min(当前晶体的总独特原子种类数, max_graph_len)
            curr_atom_num = len(idx_)

            # NOTE:如果用于蒸馏则不允许打乱
            rand_idx = idx_
            # randperm(n):将0~n-1（包括0和n-1）随机打乱后获得的数字序列,rand_idx是idx_随机打乱顺序的结果
            # rand_idx = idx_[torch.randperm(curr_atom_num)]

            # 第bi个晶体特征
            new_atom_fea[bi][: curr_atom_num] = atom_fea[c_atom_idx][rand_idx]

            # 晶体含有金属原子,且任务为MOC时的处理
            if moc:
                mo = torch.zeros(len(c_atom_idx))
                metal_idx = moc[bi]
                mo[metal_idx] = 1
                mo_label[bi][: len(rand_idx)] = mo[rand_idx]

        # 掩码,在最后一个维度(特征维度)上求和,如果为0则当前原子被掩蔽,即晶体不具有那么多的原子
        mask = (new_atom_fea.sum(dim=-1) != 0).float()

        return new_atom_fea, mask, mo_label

def saveNoSampleAtomEmbedding(cifIdList, atomFeature, atomIdx, savePath):
    os.makedirs(savePath, exist_ok=True)
    for i, cif in enumerate(cifIdList):
        startIdx, endIdx = atomIdx[i][0].item(), atomIdx[i][-1].item()
        feature = atomFeature[startIdx: endIdx + 1].cpu().numpy()
        np.save(f"{savePath}/{cif}_graph.npy", feature)