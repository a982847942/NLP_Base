import pickle

import torch
batch_size = 256
ws = pickle.load(open('../model/ws.pkl', 'rb'))
#每个句子的最大长度
max_len = 200
#每一层LSTM单元个数
hidden_size = 120
#网络层数
num_layer = 2
drop_out = 0.4
#双向LSTM
bidirectional = True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
