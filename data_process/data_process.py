import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class TrackDataset(Dataset):
    def __init__(self, data: pd.DataFrame, seq_encoder=6, seq_decoder=4, normalize: bool=True):
        self.src_sequences, self.tgt_sequences, self.tgt_lab_sequences = process_data(
            data, seq_encoder, seq_decoder, normalize
        )

    def __len__(self):
        return len(self.src_sequences)

    def __getitem__(self, idx):
        src_seq = torch.tensor(self.src_sequences[idx], dtype=torch.float32)
        tgt_seq = torch.tensor(self.tgt_sequences[idx], dtype=torch.float32)
        tgt_lab_seq = torch.tensor(self.tgt_lab_sequences[idx], dtype=torch.float32)

        return src_seq, tgt_seq, tgt_lab_seq

# 其余代码保持不变

def process_data(data: pd.DataFrame, seq_encoder=6, seq_decoder=4, normalize: bool=True):
    src_sequences = []
    tgt_sequences = []
    tgt_lab_sequences = []

    len_total = seq_encoder + seq_decoder
    # 按 id 分组
    grouped = data.groupby('id')

    for id, group in grouped:
        # 提取 x 和 y 的时序数据
        x_data = group['x'].values
        y_data = group['y'].values

        # 确保数据长度足够
        if len(x_data) >= len_total:
            for i in range(len(x_data) - len_total + 1):
                src_x = x_data[i:i + seq_encoder]
                src_y = y_data[i:i + seq_encoder]

                tgt_x = x_data[i + seq_encoder - 1:i + len_total - 1]
                tgt_y = y_data[i + seq_encoder - 1:i + len_total - 1]

                tgt_x_lab = x_data[i + seq_encoder:i + seq_decoder + seq_encoder]
                tgt_y_lab = y_data[i + seq_encoder:i + seq_decoder + seq_encoder]

                # 归一化处理
                if normalize:
                    last_x = src_x[-1]
                    last_y = src_y[-1]
                    src_x = src_x - last_x
                    src_y = src_y - last_y
                    tgt_x = tgt_x - last_x
                    tgt_y = tgt_y - last_y
                    tgt_x_lab = tgt_x_lab - last_x
                    tgt_y_lab = tgt_y_lab - last_y

                src_sequences.append(np.column_stack((src_x, src_y)))
                tgt_sequences.append(np.column_stack((tgt_x, tgt_y)))
                tgt_lab_sequences.append(np.column_stack((tgt_x_lab, tgt_y_lab)))

    return src_sequences, tgt_sequences, tgt_lab_sequences

