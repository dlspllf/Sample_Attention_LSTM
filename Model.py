import torch
import torch.nn as nn
import pandas as pd 
import numpy as np

class Sample_Attention_LSTM(nn.Module):
    def __init__(self,
                 horizon_size:int,
                 seq_len:int,
                 drop_out:int,
                 layer_size:int,
                 L1:int,
                 L2:int,
                 hidden_size:int,
                 device):
        super(Sample_Attention_LSTM,self).__init__()
        self.horizon_size = horizon_size
        self.seq_len = seq_len
        self.drop_out = drop_out
        self.layer_size = layer_size
        self.L1 = L1
        self.L2 = L2
        self.hidden_size = hidden_size
        self.device = device
        self.linear1 = nn.Linear(in_features= horizon_size+1,
                                 out_features= L1)
        self.LSTM = nn.LSTM(input_size= L1,
                            hidden_size= hidden_size,
                            num_layers= layer_size,
                            dropout= drop_out,
                            batch_first=True)     
        self.linear2 = nn.Linear(in_features= hidden_size,
                                 out_features= L2)
        self.linear3 = nn.Linear(in_features= L2,
                                 out_features= horizon_size) 
        self.activation = nn.Tanh()
        self.linear1.double()
        self.linear2.double()
        self.linear3.double()
        self.activation.double()
        self.LSTM.double()
        self.apply(custom_weights_init)

    def forward(self,input,attention_key,attention_value):
        #这里本应去掉与当前样本相同的样本，但是，时间问题，暂时先不写
        #这里用去除当前注意力分数最大的值来代替
        distance = torch.mm(input, attention_key.T)
        max_indices = torch.argmax(distance, dim=1)
        # 遍历每一行，将最大值设置为0
        for i, max_index in enumerate(max_indices):
            distance[i, max_index] = 0
        a = nn.functional.softmax(distance, dim=1)#[batch_size,batch_size]
        history_reference = torch.mm(a,attention_value)
        # history_reference [batch_size,value_size]
        # input [batch_size,seq_size]
        input = torch.unsqueeze(input,dim=2)
        reference_features = []
        for batch in history_reference:
            batch_features = []
            for i in range(0,history_reference.shape[1]-self.horizon_size):
                batch_features.append(batch[i:i+self.horizon_size])
            batch_features = torch.stack(batch_features)
            reference_features.append(batch_features)
        reference_features = torch.stack(reference_features)#[batch_size,seq_len,1+horizon_size]
        linear_1_input = torch.concat([input,reference_features],dim=2)
        linear_1_output = self.linear1(linear_1_input)
        LSTM_input = self.activation(linear_1_output)
        LSTM_ouput,_ = self.LSTM(LSTM_input)
        LSTM_ouput = self.activation(LSTM_ouput)
        linear_2_output = self.linear2(LSTM_ouput[:,-1,:])
        linear_2_output = self.activation(linear_2_output)
        output = self.linear3(linear_2_output)
        return output

    def predict(self,input,attention_key,attention_value):
        #这里本应去掉与当前样本相同的样本，但是，时间问题，暂时先不写
        #这里用去除当前注意力分数最大的值来代替
        with torch.no_grad():
            #这里本应去掉与当前样本相同的样本，但是，时间问题，暂时先不写
            #这里用去除当前注意力分数最大的值来代替
            distance = torch.mm(input, attention_key.T)
            max_indices = torch.argmax(distance, dim=1)
            # 遍历每一行，将最大值设置为0
            for i, max_index in enumerate(max_indices):
                distance[i, max_index] = 0
            a = nn.functional.softmax(distance, dim=1)#[batch_size,batch_size]
            history_reference = torch.mm(a,attention_value)
            # history_reference [batch_size,value_size]
            # input [batch_size,seq_size]
            input = torch.unsqueeze(input,dim=2)
            reference_features = []
            for batch in history_reference:
                batch_features = []
                for i in range(0,history_reference.shape[1]-self.horizon_size):
                    batch_features.append(batch[i:i+self.horizon_size])
                batch_features = torch.stack(batch_features)
                reference_features.append(batch_features)
            reference_features = torch.stack(reference_features)#[batch_size,seq_len,1+horizon_size]
            linear_1_input = torch.concat([input,reference_features],dim=2)
            linear_1_output = self.linear1(linear_1_input)
            LSTM_input = self.activation(linear_1_output)
            LSTM_ouput,_ = self.LSTM(LSTM_input)
            LSTM_ouput = self.activation(LSTM_ouput)
            linear_2_output = self.linear2(LSTM_ouput[:,-1,:])
            linear_2_output = self.activation(linear_2_output)
            output = self.linear3(linear_2_output)
        return output


def custom_weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)

