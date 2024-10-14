import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from data_process.data_process import TrackDataset
from model.model import TransformerWithPE

# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 加载数据
# 假设 TrackDataset 已经定义好了，并且可以从CSV文件中加载数据
data = pd.read_csv('../58_tracks.csv')
train_dataset = TrackDataset(data)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)



# 创建模型实例并移动到GPU
model = TransformerWithPE(in_dim=2, out_dim=2, embed_dim=16, num_heads=2, num_layers=6)
model.to(device)

# 定义优化器和损失函数
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# 训练模型
NUM_EPOCHS = 10
for epoch in range(NUM_EPOCHS):
    for src_seq, tgt_seq, tgt_lab_seq in train_loader:
        # 将数据移动到GPU
        src_seq = src_seq.to(device)
        tgt_seq = tgt_seq.to(device)
        tgt_lab_seq = tgt_lab_seq.to(device)

        model.train()
        optimizer.zero_grad()
        predictions = model(src_seq, tgt_seq)
        loss = criterion(predictions, tgt_lab_seq)
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')