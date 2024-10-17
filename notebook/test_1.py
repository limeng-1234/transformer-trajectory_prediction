import torch
import numpy as np

#%%
# 设置随机种子以确保结果可复现
np.random.seed(42)
torch.manual_seed(42)
from torch.utils.data import Dataset, DataLoader
#%%
# 加载数据
# 示例使用
import pandas as pd
from data_process.data_process import TrackDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# 加载数据
data = pd.read_csv('../58_tracks.csv')

# 创建 Dataset
train_dataset = TrackDataset(data)
# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

import torch
from torch.utils.data import DataLoader
from model.model import TransformerWithPE



BS = 512
FEATURE_DIM = 128
NUM_HEADS = 8
NUM_EPOCHS = 10
NUM_VIS_EXAMPLES = 10
NUM_LAYERS = 2
LR = 0.001
model = TransformerWithPE(in_dim=2, out_dim=2, embed_dim=16, num_heads=2, num_layers=6)


optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = torch.nn.MSELoss()

for epoch in range(2):
    for src_seq, tgt_seq, tgt_lab_seq in train_loader:
        model.train()
        optimizer.zero_grad()
        predictions = model(src_seq, tgt_seq)
        # loss = criterion(predictions, tgt[:, -1, :])  # 只计算最后一个时间步的损失
        loss = criterion(predictions, tgt_lab_seq)  #
        loss.backward()
        optimizer.step()
        print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# # 测试模型
# for src_seq, tgt_seq, tgt_lab_seq in test_loader:
#     model.eval()
#     with torch.no_grad():
#         # Run inference with model
#         pred_infer = model.infer(src_seq, tgt_seq.shape[1])
#         loss_infer = criterion(pred_infer, tgt_seq)
#         predictions = model(src, tgt)
#         loss = criterion(predictions, tgt[:, -1, :])
#         print(f'Test Loss: {loss.item()}')