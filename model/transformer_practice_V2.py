import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import numpy as np
import math
import argparse
#%%
# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

class TrajectoryDataset(data.Dataset):
    def __init__(self, seq_length, num_samples):
        self.seq_length = seq_length
        self.num_samples = num_samples
        self.data = self.generate_data()
        self.normalize_data()

    def generate_data(self):
        # 生成假设的车辆轨迹数据
        return np.random.rand(self.num_samples, self.seq_length, 2)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # 获取单个样本
        sample = self.data[idx]
        src = torch.tensor(sample[:-1, :], dtype=torch.float32)
        tgt = torch.tensor(sample[1:, :], dtype=torch.float32)
        return src, tgt

    def normalize_data(self):
        # 归一化数据
        max_val = np.max(self.data)
        min_val = np.min(self.data)
        self.data = (self.data - min_val) / (max_val - min_val)

# 定义参数
def get_args():
    parser = argparse.ArgumentParser(description='Trajectory Prediction')
    parser.add_argument('--seq_length', type=int, default=10, help='Sequence length')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--test_size', type=float, default=0.2, help='Test size')
    return parser.parse_args()

# 使用自定义数据集
args = get_args()
dataset = TrajectoryDataset(args.seq_length, args.num_samples)

# 将数据分成训练集和测试集
split_idx = int((1 - args.test_size) * len(dataset))
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [split_idx, len(dataset) - split_idx])

# 创建DataLoader
train_loader = data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
#%%
# 定义位置编码
'''
在这段代码中，我们定义了一个`PositionalEncoding`类，它是一个PyTorch模块，用于给输入序列添加位置信息。这是Transformer模型中的一个重要组件，因为Transformer模型本身不具备处理序列顺序的信息。位置编码通过向每个输入元素添加一个唯一的位置向量来解决这个问题。
下面是代码中每个参数和变量的详细解释：
# 1. `d_model`（int）：
   - 这是模型的维度，即每个输入向量的维度。在Transformer模型中，这个维度通常与嵌入层的输出维度相同。
2. `max_len`（int）：
   - 这是位置编码的最大长度。它定义了位置编码向量的最大时间步长。在这个实现中，我们使用5000作为默认值，这意味着我们可以处理最多5000个时间步长的序列。实际上，这个值可以根据具体任务进行调整。
3. `self.encoding`（torch.zeros）：
   - 这是一个形状为`(max_len, d_model)`的张量，用于存储位置编码。它被初始化为全零。
4. `position`（torch.arange）：
   - 这是一个从0到`max_len`的连续整数序列，形状为`(max_len,)`。它表示序列中每个位置的索引。
5. `div_term`（torch.exp）：
   - 这是一个用于计算位置编码的除法项的张量。它的形状为`(d_model/2,)`，因为对于每个维度，我们需要计算两个位置编码（一个用于sin，一个用于cos）。`div_term`的计算公式为`torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))`，这个公式确保了随着维度的增加，位置编码的频率会降低，这样可以捕捉到不同的位置信息。
6. `self.encoding[:, 0::2]`和`self.encoding[:, 1::2]`：
   - 这两行代码分别计算位置编码的正弦和余弦部分。对于每个位置`pos`，我们计算两个值：`\(\sin(pos / 10000^{2i/d_{\text{model}}})\)`和`\(\cos(pos / 10000^{2i/d_{\text{model}}})\)`，其中`i`是维度索引。这些值被存储在`self.encoding`张量中。
7. `self.encoding = self.encoding.unsqueeze(0)`：
   - 这行代码在`self.encoding`张量的最前面添加了一个新维度，使其形状变为`(1, max_len, d_model)`，这样就可以通过广播机制轻松地将其添加到任何形状为`(batch_size, seq_len, d_model)`的输入张量上。
8. `forward(x)`：
   - 这是模块的前向传播函数。它接受一个输入张量`x`，其形状为`(batch_size, seq_len, d_model)`。函数将位置编码添加到输入张量上，并返回结果。`self.encoding[:, :x.size(1)]`确保位置编码与输入张量在时间步上对齐。`detach()`方法用于确保位置编码不会在反向传播中被计算梯度，因为它是一个固定的张量。
总的来说，`PositionalEncoding`类为Transformer模型提供了一种将位置信息嵌入到输入序列中的方法，这对于模型理解序列数据中的顺序关系至关重要。
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)

    def forward(self, x):
        return x + self.encoding[:, :x.size(1)].detach()

#%%
# 定义多头自注意力机制
"""
在这个`MultiHeadAttention`类的实现中，我们定义了多头自注意力机制，这是Transformer模型的核心组成部分之一。以下是代码中每个参数和变量的详细解释：

1. `d_model`（int）：
   - 这是模型的维度，即每个输入向量的维度。这个维度应该与模型中其他部分的维度保持一致。
2. `num_heads`（int）：
   - 这是注意力机制中的头数。多头注意力机制将输入分割成多个头，每个头学习输入数据的不同部分。
3. `self.d_k`（int）：
   - 这是每个头的维度。由于我们将`d_model`分割成`num_heads`个部分，每个头的维度就是`d_model`除以`num_heads`。
4. `self.query`（nn.Linear）：
   - 这是一个线性层，用于将输入查询（query）转换为查询向量。
5. `self.key`（nn.Linear）：
   - 这是一个线性层，用于将输入键（key）转换为键向量。
6. `self.value`（nn.Linear）：
   - 这是一个线性层，用于将输入值（value）转换为值向量。
7. `self.fc_out`（nn.Linear）：
   - 这是一个线性层，用于将多头注意力的输出合并回原始的模型维度。
8. `query`（torch.Tensor）：
   - 这是自注意力机制的查询输入，它的形状通常是`(batch_size, seq_len, d_model)`。
9. `key`（torch.Tensor）：
   - 这是自注意力机制的键输入，它的形状通常是`(batch_size, seq_len, d_model)`。
10. `value`（torch.Tensor）：
    - 这是自注意力机制的值输入，它的形状通常是`(batch_size, seq_len, d_model)`。
11. `mask`（torch.Tensor，可选）：
    - 这是一个可选的掩码，用于屏蔽（忽略）某些位置的注意力权重。这通常用于解码器的自注意力层，以防止位置信息泄露。
12. `batch_size`（int）：
    - 这是输入张量的批次大小。
13. `query`, `key`, `value`（torch.Tensor）：
    - 这些是通过应用相应的线性层转换后的查询、键和值向量。
14. `scores`（torch.Tensor）：
    - 这是计算得到的注意力分数，它的形状是`(batch_size, num_heads, seq_len, seq_len)`。这些分数表示每个查询向量与每个键向量的相似度。
15. `attention`（torch.Tensor）：
    - 这是应用softmax函数后的注意力权重，它的形状与`scores`相同。
16. `out`（torch.Tensor）：
    - 这是多头自注意力的输出，它是通过将注意力权重与值向量相乘得到的。
17. `out`（torch.Tensor）：
    - 在经过转置和重塑操作后，这是多头自注意力的最终输出，它的形状是`(batch_size, seq_len, d_model)`。

这个多头自注意力机制的实现允许模型在不同的表示子空间中并行地学习信息，每个头可以捕捉输入数据的不同方面。通过这种方式，模型可以更有效地处理序列数据，并且能够捕捉到序列中长距离的依赖关系。
"""
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.fc_out = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = self.query(query), self.key(key), self.value(value)
        query = query.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            # 扩展掩码以匹配scores的头维度
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
            scores = scores.masked_fill(mask == 0, -1e9)
        attention = torch.softmax(scores, dim=-1)
        out = torch.matmul(attention, value)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.d_k)
        return self.fc_out(out)

#%%
# 定义前馈网络
"""
在这段代码中，我们定义了一个名为 `PositionwiseFeedForwardNet` 的类，它代表了一个位置-wise（逐位置）的前馈网络，这是Transformer模型中的一个组件。以下是代码中每个参数和变量的详细解释：

1. `d_model`（int）：
   - 这是模型的输入和输出维度。在这个前馈网络中，它代表了输入张量的特征维度，也是输出张量的特征维度。
2. `d_ff`（int）：
   - 这是前馈网络中间层的维度。它通常大于或等于 `d_model`，用于增加网络的容量，允许网络在两个线性层之间学习更复杂的函数。
3. `dropout`（float）：
   - 这是 dropout 正则化的比例，默认值为0.1。Dropout 是一种防止过拟合的技术，它在训练过程中随机地将一部分网络输出置为零。
4. `self.fc1`（nn.Linear）：
   - 这是前馈网络的第一个线性层，它将输入从 `d_model` 维度映射到 `d_ff` 维度。
5. `self.fc2`（nn.Linear）：
   - 这是前馈网络的第二个线性层，它将中间层的输出从 `d_ff` 维度映射回原始的 `d_model` 维度。
6. `self.dropout`（nn.Dropout）：
   - 这是 dropout 层，它根据指定的比例随机地将输入的一部分元素置为零。
7. `x`（torch.Tensor）：
   - 这是前馈网络的输入张量，它的形状通常是 `(batch_size, seq_len, d_model)`，其中 `batch_size` 是批次大小，`seq_len` 是序列长度。
8. `forward(self, x)`：
   - 这是模块的前向传播函数。它接受一个输入张量 `x`，并返回通过网络层后的输出张量。
9. `x = torch.relu(self.fc1(x))`：
   - 这是前馈网络的第一个操作，输入张量 `x` 通过第一个线性层 `self.fc1`，然后应用 ReLU 激活函数。
10. `x = self.dropout(x)`：
    - 这是应用 dropout 正则化的操作。如果 dropout 比例不为零，一些输出元素将被随机置为零。
11. `x = self.fc2(x)`：
    - 这是前馈网络的最后一个操作，dropout 后的输出通过第二个线性层 `self.fc2`。
12. `return x`：
    - 这是前向传播函数的返回值，即前馈网络的输出张量。
这个前馈网络是Transformer模型中的一个关键组件，它对每个位置的输入独立地应用相同的操作，从而在模型中引入非线性，增强模型的表达能力。位置-wise前馈网络的设计允许模型并行处理序列中的每个位置，这有助于提高训练效率。

"""
class PositionwiseFeedForwardNet(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 定义编码器层
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.pos_ffn = PositionwiseFeedForwardNet(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_output = self.self_attn(x, x, x, mask)
        x = self.dropout(self.layer_norm1(x + attn_output))
        ff_output = self.pos_ffn(x)
        x = self.dropout(self.layer_norm2(x + ff_output))
        return x

# 定义解码器层
class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(DecoderLayer, self).__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_attn = MultiHeadAttention(d_model, num_heads)
        self.pos_ffn = PositionwiseFeedForwardNet(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.layer_norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_attn_mask=None, enc_attn_mask=None):
        self_attn_output = self.self_attn(x, x, x, self_attn_mask)
        x = self.dropout(self.layer_norm1(x + self_attn_output))
        enc_attn_output = self.enc_attn(x, enc_output, enc_output, enc_attn_mask)
        x = self.dropout(self.layer_norm2(x + enc_attn_output))
        ff_output = self.pos_ffn(x)
        x = self.dropout(self.layer_norm3(x + ff_output))
        return x


# 定义解码器
class Decoder(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.pos_enc = PositionalEncoding(d_model)

    def forward(self, tgt, enc_output, self_attn_mask=None, enc_attn_mask=None):
        batch_size, seq_len, _ = tgt.size()
        # 创建掩码以防止位置信息泄露
        if self_attn_mask is True:
            self_attn_mask = torch.triu(
                torch.ones((seq_len, seq_len), device=tgt.device), diagonal=1).bool()
            self_attn_mask = self_attn_mask.unsqueeze(0).repeat(batch_size, 1, 1)
        tgt = self.pos_enc(tgt)
        for layer in self.layers:
            tgt = layer(tgt, enc_output, self_attn_mask, enc_attn_mask)
        return tgt

# 定义完整的Transformer模型
class TransformerModel(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, num_layers, dropout=0.1, input_size=2):
        super(TransformerModel, self).__init__()
        self.input_embedding = nn.Linear(input_size, d_model)  # 输入嵌入层
        self.encoder = nn.ModuleList([EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)])
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers, dropout)
        self.pos_enc = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.d_model = d_model
        self.prediction_layer = nn.Linear(d_model, 2)  # 预测2个坐标(x, y)

    def forward(self, src, tgt):
        src = self.input_embedding(src)  # 应用输入嵌入层
        src = self.pos_enc(src)
        for layer in self.encoder:
            src = layer(src)
        tgt = self.input_embedding(tgt)  # 应用输入嵌入层
        tgt = self.pos_enc(tgt)
        tgt_output = self.decoder(tgt, src)
        # predictions = self.prediction_layer(tgt_output[:, -1, :])  # 只预测最后一个时间步的输出
        predictions = self.prediction_layer(tgt_output)
        return predictions


# 创建模型实例
model = TransformerModel(d_model=16, num_heads=2, d_ff=16, num_layers=6)

# 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):
    for src, tgt in train_loader:
        model.train()
        optimizer.zero_grad()
        predictions = model(src, tgt)
        # loss = criterion(predictions, tgt[:, -1, :])  # 只计算最后一个时间步的损失
        loss = criterion(predictions, tgt)  #
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# 测试模型
# 测试模型
# for src, tgt in test_loader:
#     model.eval()
#     with torch.no_grad():
#         predictions = model(src, tgt)
#         loss = criterion(predictions, tgt[:, -1, :])
#         print(f'Test Loss: {loss.item()}')
