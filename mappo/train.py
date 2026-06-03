import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from transformer import Embedding_Layer  # 从 transformer.py 文件导入 Embedding_Layer
from transformer import TransformerEncoder
import matplotlib.pyplot as plt

class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        y = self.fc3(x)
        return y

class TrainingWrapper(nn.Module):
    def __init__(self, embedding_layer, transformer_encoder, decoder, seq_length):
        super(TrainingWrapper, self).__init__()
        self.embedding_layer = embedding_layer
        self.encoder = transformer_encoder
        self.decoder = decoder  # 假设编码后的维度是 128

    def forward(self, x):
        embedded = self.embedding_layer(x)
        encoded = self.encoder(embedded)
        decoded = self.decoder(encoded)
        return decoded

# 实例化原始模型和包装模型
vocab_size = 64  # 假设词汇表大小为 64
embedding_dim = 128  # 嵌入层的维度
padding_idx = 32  # 填充索引
seq_length = 32  # 序列长度
transformer_encoder = TransformerEncoder(dim=128)
embedding_layer = Embedding_Layer(dim=128, pad=32)
decoder = Decoder(128, 32)
training_model = TrainingWrapper(embedding_layer, transformer_encoder, decoder, seq_length)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(training_model.parameters(), lr=0.001)

# 生成随机数据的函数
def generate_random_data(batch_size, seq_length):
    inputs = torch.randint(0, vocab_size, (batch_size, seq_length)).float()
    inputs[:, 0] = 1.0
    return inputs, inputs  # 使用相同的数据作为输入和目标

# 训练函数
def train(model, criterion, optimizer, epochs, batch_size, seq_length):
    loss_values = []
    for epoch in range(epochs):
        inputs, targets = generate_random_data(batch_size, seq_length)
        optimizer.zero_grad()
        outputs = model(inputs)
        print(outputs)
        print(targets)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        loss_values.append(loss.item())
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')
    return loss_values

# 调用训练函数并记录损失值
#loss_values = train(training_model, criterion, optimizer, epochs=1000, batch_size=128, seq_length=seq_length)

'''
# 绘制损失曲线
plt.plot(loss_values)
plt.title('Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.savefig('loss_curve.png')
'''

# 保存 Embedding_Layer 参数
def save_embedding_layer(embedding_layer, transformer_encoder, path1, path2):
    torch.save(embedding_layer.state_dict(), path1)
    torch.save(transformer_encoder.state_dict(), path2)

# 加载 Embedding_Layer 参数
def load_embedding_layer(embedding_layer, transformer_encoder, path1, path2):
    embedding_layer.load_state_dict(torch.load(path1))
    transformer_encoder.load_state_dict(torch.load(path2))

#torch.save(decoder.state_dict(), 'decoder.pth')
# 保存 Embedding_Layer 参数
#save_embedding_layer(embedding_layer, transformer_encoder, 'embedding_parameters.pth', 'transformer_encoder.pth')

'''
# 加载 Embedding_Layer 参数（假设你已经保存了参数）
loaded_embedding_layer = Embedding_Layer(dim=128, pad=32)
loaded_transformer_encoder = TransformerEncoder(dim=128)
load_embedding_layer(embedding_layer, transformer_encoder, 'embedding_parameters.pth', 'transformer_encoder.pth')

# 测试加载的 Embedding_Layer
test_inputs, _ = generate_random_data(1, seq_length)
print(test_inputs)
x = loaded_embedding_layer(test_inputs)
x = loaded_transformer_encoder(x)
print(x)
'''