'''
 # @ author: zhangyi
 # @ date: 2023-10-17 10:31:37
 # @ desc: 主体模块,输入预嵌入结果+人工提取特征,输出嵌入结果,送入不同head
 '''

from functools import partial

import torch
import torch.nn as nn

from myutils.ConstructModelUtils import initWeights


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        numheads=8,
        qkvbias=False,
        attnDropRatio=0.0,
        fcDropoutRatio=0.0,
    ):
        """
        自注意力初始化

        Args:
            dim (int): 维度
            numheads (int, optional): head数. Defaults to 8.
            qkvbias (bool, optional): 输入转为QKV时,是否带有偏置. Defaults to False.
            attnDropRatio (float, optional): Defaults to 0.0.
            fcDropoutRatio (float, optional): Defaults to 0.0.
        """
        super().__init__()
        self.numheads = numheads
        headDim = dim // numheads  # 输入维度被均分到每个head
        self.scale = headDim**-0.5  # 1/math.sqrt(d_k),标准化QK^T的结果
        self.qkv = nn.Linear(dim, dim * 3, bias=qkvbias)  # x -> Q, K, V
        self.attnDropout = nn.Dropout(attnDropRatio)
        self.pointWiseFc = nn.Linear(dim, dim)
        self.fcDropout = nn.Dropout(fcDropoutRatio)

    def forward(self, x, mask=None):
        """
        attention

        Args:
            x (tensor): [B(batchSize), N(输入长度), C(每个位置对应的维度)]
            mask (bool矩阵): Defaults to None.

        Returns:
            x(tensor): [B(batchSize), N(输入长度), C(每个位置对应的维度)]
            atten(tensor): [B, numheads, N, N] 其值为dropout(softmax(QK/d))
        """
        B, N, C = x.shape
        assert C % self.numheads == 0

        # 将x映射为QKV
        qkv = (
            self.qkv(x)  # [B, N, 3*C]
                .reshape(B, N, 3, self.numheads, C // self.numheads)  # [B, N, 3, numheads, C//numheads],所有的位置共享参数生成QKV
                .permute(2, 0, 3, 1, 4)  # [3, B, numheads, N, C//numheads]
        )
        q, k, v = (
            qkv[0],  # [B, numheads, N, C//numheads]
            qkv[1],  # [B, numheads, N, C//numheads]
            qkv[2],  # [B, numheads, N, C//numheads]
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, numheads, N, N],@代表矩阵乘法
        # 将被掩蔽的位置,相似度设置为-inf
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)  # [B, numheads, N, N]
        attn = self.attnDropout(attn)  # 训练时对softmax之后的矩阵做dropout

        x = (
            (attn @ v).transpose(1, 2).reshape(B, N, C)
        )  # [B, numheads, N, C//numheads] -> [B, N, C],将多头的结果拼接在一起
        x = self.pointWiseFc(x)
        x = self.fcDropout(x)
        return x, attn

class MLP(nn.Module):
    def __init__(
        self,
        inDim,
        hiddenDim=None,
        outDim=None,
        activationFunc=nn.GELU,
        dropoutRatio=0.0,
    ):
        """
        构造两层MLP

        Args:
            inDim (int): 输入维度
            hiddenDim (int, optional): 隐层维度. Defaults to None.
            outDim (int, optional): 输出维度. Defaults to None.
            activationFunc (optional): 激活函数. Defaults to nn.GELU.
            dropoutRatio (float, optional): dropout概率. Defaults to 0.0.
        """
        super().__init__()
        out = outDim if outDim else inDim
        hidden = hiddenDim if hiddenDim else inDim
        self.fc1 = nn.Linear(inDim, hidden)
        self.act = activationFunc()
        self.fc2 = nn.Linear(hidden, out)
        self.drop = nn.Dropout(dropoutRatio)

    def forward(self, x):
        """
        执行MLP

        Args:
            x (tensor): [B, N, in_features]

        Returns:
            tensor: [B, N, out_features]
        """
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CLSPooler(nn.Module):
    def __init__(self, hidden_size, index=0):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.Tanh()
        self.index = index

    def forward(self, hidden_states):
        """
        对输入的隐层指定位置(默认首个位置)执行线性层+激活,得到的向量用于进一步预测
        
        hidden_states: [B, 3+max_graph_len, dim]
        """
        first_token_tensor = hidden_states[:, self.index]  # [B, 1, dim]
        pooled_output = self.dense(first_token_tensor)  # [B, 1, dim]
        pooled_output = self.activation(pooled_output)  # [B, 1, dim]
        return pooled_output  # [B, 1, dim]

class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        dim,
        numheads,
        mlpRatio=4.0,
        qkvbias=False,
        fcDropoutRatio=0.0,
        attnDropRatio=0.0,
        activationFunc=nn.GELU,
        norm=nn.LayerNorm,
    ):
        """
        构建Transformer Layer

        Args:
            dim (int): 每个原子的维度
            numheads (int): 自注意力头数
            mlpRatio (float, optional): MLP 隐层维度/输入维度. Defaults to 4.0.
            fcDropoutRatio (float, optional): 线性层dropout ratio. Defaults to 0.0.
            attnDropRatio (float, optional): softmax(QK^T/sqrt(d_k))的dropout ratio. Defaults to 0.0.
            activationFunc (optional): MLP激活层. Defaults to nn.GELU.
            norm (optional): Norm函数. Defaults to nn.LayerNorm.
        """
        super().__init__()
        self.norm1 = norm(dim)
        self.attn = Attention(
            dim,
            numheads=numheads,
            qkvbias=qkvbias,
            attnDropRatio=attnDropRatio,
            fcDropoutRatio=fcDropoutRatio,
        )
        self.norm2 = norm(dim)
        mlpHiddenDim = int(dim * mlpRatio)
        self.mlp = MLP(
            inDim=dim,
            hiddenDim=mlpHiddenDim,
            activationFunc=activationFunc,
            dropoutRatio=fcDropoutRatio,
        )

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + _x
        x = x + self.mlp(self.norm2(x))
        return x, attn
    
class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim,
        numheads=12,
        layerNum=12,
        qkvBias=True,
        dropoutRatio=0.0,
        attnDropRatio=0.0,
        mlpRatio=4.0,
        vis=False
    ):
        """
        构建Transformer

        Args:
            dim (int): 维度
            numheads (int, optional): 头数. Defaults to 12.
            layerNum (int, optional): 层数. Defaults to 12.
            qkvBias (bool, optional): QKV生成时是否有bias. Defaults to True.
            dropoutRatio (float, optional): Defaults to 0.0.
            attnDropRatio (float, optional): Defaults to 0.0.
            mlpRatio (float, optional): MPL 隐层维度/输入维度. Defaults to 4.0.
            vis (bool, optional): 是否对自注意力矩阵可视化,默认False
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        
        # 构建TransformerLayer
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(
                dim=dim,
                numheads=numheads,
                mlpRatio=mlpRatio,
                qkvbias=qkvBias,
                fcDropoutRatio=dropoutRatio,
                attnDropRatio=attnDropRatio
            ) for i in range(layerNum)
        ])
        
        # CLS构建编码,以及提取CLS的pooler层
        self.CLSLinear = nn.Linear(1, dim)
        self.CLSPooler = CLSPooler(hidden_size=dim)
        
        # 拓扑类型嵌入,共96种拓扑类型(包含unknown)
        self.topoTypeEmbedding = nn.Embedding(96, dim)
        
        # 晶胞体积嵌入
        self.cellVolumnLinear = nn.Linear(1, dim)
        
        # 模态编码,输入信息共包含3类
        self.modalTypeEmbedding = nn.Embedding(3, dim)
        
        # 是否可视化自注意力矩阵
        self.vis = vis
        
        # 初始化模型所有参数
        self.apply(initWeights)
        
    def forward(self, graphEmbed, graphMask, cellVolumn, topoTypeIdx):
        # 当前batch的大小
        batchSize = len(topoTypeIdx)
        
        # 构建图特征模态,将CLS拼接到图特征
        CLS = torch.zeros(batchSize).to(graphEmbed)  # [B]
        CLSEmbed = self.CLSLinear(CLS[:, None, None])  # [B, 1, dim]
        CLSMask = torch.ones(batchSize, 1).to(graphEmbed)  # [B, 1]
        graphEmbed = torch.cat(
            [CLSEmbed, graphEmbed], dim=1
        )  # [B, max_graph_len+1, dim]
        graphMask = torch.cat(
            [CLSMask, graphMask], dim=1
        )  # [B, max_graph_len+1]
        
        # 构造其他输入模态
        topoTypeEmbed = self.topoTypeEmbedding(topoTypeIdx).unsqueeze(1)  # [B, 1, dim]
        topoTypeMask = torch.ones(batchSize, 1).to(graphEmbed)  # [B, 1]
        cellVolumnEmbed = self.cellVolumnLinear(cellVolumn[:, None, None])  # [B, 1, dim]
        cellVolumnMask = torch.ones(batchSize, 1).to(graphEmbed)  # [B, 1]
        
        # 增加模态类型标识,分别用0,1,2标识图特征,晶胞体积,拓扑类型
        graphEmbed = graphEmbed + self.modalTypeEmbedding(
            torch.full_like(graphMask, fill_value=0).to(graphMask).long()
        )  # [B, max_graph_len+1, dim]
        cellVolumnEmbed = cellVolumnEmbed + self.modalTypeEmbedding(
            torch.full_like(cellVolumnMask, fill_value=1).to(cellVolumnMask).long()
        )  # [B, 1, dim]
        topoTypeEmbed = topoTypeEmbed + self.modalTypeEmbedding(
            torch.full_like(topoTypeMask, fill_value=2).to(topoTypeMask).long()
        )  # [B, 1, dim]
        
        # 拼接得到输入序列
        allEmbed = torch.cat(
            [graphEmbed, cellVolumnEmbed, topoTypeEmbed], dim=1
        )  # [B, 3+max_graph_len, dim]
        allMask = torch.cat(
            [graphMask, cellVolumnMask, topoTypeMask], dim=1
        )  # [B, 3+max_graph_len]
        
        # 执行TransformerEncoderLayers
        x = allEmbed
        for i, layer in enumerate(self.layers):
            x, _attn = layer(x, mask=allMask)
        
        # 获取输出
        x = self.norm(x)  # [B, 3+max_graph_len, dim]
        clsFeature = self.CLSPooler(x)  # [B, dim]
        
        return clsFeature