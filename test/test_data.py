# 示例使用
import pandas as pd
from data_process.data_process import TrackDataset
from torch.utils.data import Dataset, DataLoader

data = pd.read_csv('58_tracks.csv')  # 确保路径正确
dataset = TrackDataset(data, seq_encoder=6, seq_decoder=4, normalize=True)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# 迭代数据集
for src_seq, tgt_seq, tgt_lab_seq in dataloader:
    print(src_seq.shape, tgt_seq.shape, tgt_lab_seq.shape)
    break
