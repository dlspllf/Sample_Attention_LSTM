import pandas as pd 
import numpy as np
import torch
from torch.utils.data import Dataset

class Sample_Attention_LSTM_dataset(Dataset):
    #由于数据集较小，所以直接生成数据存在内存中
    #当数据集大时，可以选择在__getitem__中动态生成样本
    def __init__(self,
                 input_series:torch.tensor,
                 horizon_size:int,#预测长度
                 seq_len:int):
        self.input_series = input_series
        self.horizon_size = horizon_size
        self.seq_len = seq_len
        sample_nums = len(input_series)-horizon_size-seq_len
        sample_input = []
        for i in range(0,sample_nums):
            input_seq = self.input_series[i:i+seq_len]
            sample_input.append(input_seq)
        self.sample_input = torch.stack(sample_input)

        sample_vals = []
        for i in range(seq_len,len(input_series)-horizon_size):
            vals_seq = self.input_series[i:i+horizon_size]
            sample_vals.append(vals_seq)
        self.sample_vals = torch.stack(sample_vals)

    def __len__(self):
        return len(self.sample_vals)
    
    def __getitem__(self,idx):
        return self.sample_input[idx], self.sample_vals[idx]