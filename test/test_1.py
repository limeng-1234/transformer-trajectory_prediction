# 示例使用
import pandas as pd
from data_process.data_process_1 import TrackDataset, process_data
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
# 加载数据
1
data = pd.read_csv('58_tracks.csv')

# 分割数据集
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
# 处理数据
train_src, train_tgt, train_tgt_lab = process_data(train_data, normalize=True)
test_src, test_tgt, test_tgt_lab = process_data(test_data, normalize=True)
# 创建 Dataset
train_dataset = TrackDataset(train_src, train_tgt, train_tgt_lab)
test_dataset = TrackDataset(test_src, test_tgt, test_tgt_lab)
# 创建 DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 迭代训练数据和测试数据
for src_seq, tgt_seq, tgt_lab_seq in train_loader:
    print("Training Data - src_seq shape:", src_seq.shape, "tgt_seq shape:", tgt_seq.shape, "tgt_lab_seq shape:", tgt_lab_seq.shape)
    # 训练模型...
    break

for src_seq, tgt_seq, tgt_lab_seq in test_loader:
    print("Testing Data - src_seq shape:", src_seq.shape, "tgt_seq shape:", tgt_seq.shape, "tgt_lab_seq shape:", tgt_lab_seq.shape)
    # 测试模型...
    break