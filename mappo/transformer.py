from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch
import math
from functools import partial

import torch.nn as nn

import torch
import torch.nn as nn
import numpy as np
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        #d_model是每个词embedding后的维度
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term2 = torch.pow(torch.tensor(10000.0),torch.arange(0, d_model, 2).float()/d_model)
        div_term1 = torch.pow(torch.tensor(10000.0),torch.arange(1, d_model, 2).float()/d_model)
        #高级切片方式，即从0开始，两个步长取一个。即奇数和偶数位置赋值不一样。
        pe[:, 0::2] = torch.sin(position * div_term2)
        pe[:, 1::2] = torch.cos(position * div_term1)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x + self.pe.repeat(x.size(0), 1, 1)
        return x

class Embedding_Layer(nn.Module):

    def __init__(self,dim=128,pad=32):
        super(Embedding_Layer, self).__init__()
        # dim: 嵌入维度 pad: 补齐长度
        self.Position = PositionalEncoding(dim,max_len=pad)
        self.embedding = nn.Embedding(num_embeddings=64, embedding_dim=dim, padding_idx=0)
    
    def forward(self,x):
        '''seq_len = [32 for i in range(self.n)]
        for i in range(len(x[0])):
            if x[0,i].item() == 0:
                seq_len[0] = i
        for i in range(len(x[1])):
            if x[1,i].item() == 0:
                seq_len[1] = i
        #seq_len = torch.tensor([next((i for i, value in enumerate(row) if value == 0), len(row)) for row in x])
        positional_encoding_0 = np.zeros((seq_len[0],self.dim))
        positional_encoding_1 = np.zeros((seq_len[1],self.dim))
        # 计算需要补全的数量
        padding_size = 32 - torch.tensor(seq_len, dtype=torch.long)
        # 创建全零张量
        #padding_tensor = torch.zeros(padding_size, 128)
        padding_tensor_0 = torch.zeros(padding_size[0], 128)
        padding_tensor_1 = torch.zeros(padding_size[1], 128)
    
        for pos in range(positional_encoding_0.shape[0]):
            for i in range(positional_encoding_0.shape[1]):
                positional_encoding_0[pos][i] = math.sin(pos/(10000**(2*i/self.dim))) if i % 2 == 0 else math.cos(pos/(10000**(2*i/self.dim)))
        for pos in range(positional_encoding_1.shape[0]):
            for i in range(positional_encoding_1.shape[1]):
                positional_encoding_1[pos][i] = math.sin(pos/(10000**(2*i/self.dim))) if i % 2 == 0 else math.cos(pos/(10000**(2*i/self.dim)))

        # 补全张量
        positional_tensor_0 = torch.cat([torch.from_numpy(positional_encoding_0), padding_tensor_0], dim=0)
        positional_tensor_1 = torch.cat([torch.from_numpy(positional_encoding_1), padding_tensor_1], dim=0)
        positional_tensor = torch.stack((positional_tensor_0, positional_tensor_1), dim=0)

        # 嵌入
        embedded = self.embedding(x)
        # 相加
        input_tensor = positional_tensor + embedded'''
        
        x = self.embedding(x.long())
        x = self.Position(x)
        return x

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

        """
        如果在使用 torch.nn.Embedding 时指定了 padding_idx 参数，
        那么在模型的训练过程中,PyTorch 将会自动处理填充位置的梯度，
        无需再手动创建填充掩码或者处理这些填充位置的注意力权重。
        """
        # 将被掩蔽的位置,相似度设置为-inf
        '''padding_mask = torch.all(x == 0, dim=-1)
        padding_mask = padding_mask.long()
        mask = padding_mask
        print('mask.shape',mask.shape)'''
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
            dim (int): 维度
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
    
def initWeights(module: nn.Module) -> None:
    """
    初始化模块权重
    linear/embedding/ln
    """
    if isinstance(module, (nn.Linear, nn.Embedding)):
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, nn.LayerNorm):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    if isinstance(module, nn.Linear) and module.bias is not None:
        module.bias.data.zero_()    


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        dim=128,
        numheads=8,
        layerNum=12,
        qkvBias=True,
        dropoutRatio=0.0,
        attnDropRatio=0.0,
        mlpRatio=4.0
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
              
        
        # 初始化模型所有参数
        self.apply(initWeights)
        
    def forward(self, x):
        # 当前batch的大小
        batchSize = len(x)
        
        # 执行TransformerEncoderLayers
        for i, layer in enumerate(self.layers):
            x, _attn = layer(x)
        
        # 获取输出
        x = self.norm(x)  # [B, 3+max_graph_len, dim]
        clsFeature = self.CLSPooler(x)  # [B, dim]
        
        return clsFeature
    
