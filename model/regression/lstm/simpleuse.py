import torch
batch_size = 10 #mini_batch大小
seq_len = 20#向量长度
embedding_dim = 30#词嵌入空间特征维度
word_vocab = 100#词典大小
hidden_size = 18#LSTM单元个数
num_layer = 2#网络层数
#输入数据
input = torch.randint(low=0,high=100,size=[batch_size,seq_len])
#词嵌入
embedding = torch.nn.Embedding(word_vocab,embedding_dim)
# lstm = torch.nn.LSTM(embedding_dim,hidden_size,num_layer,batch_first=True,bidirectional=True)
lstm = torch.nn.LSTM(embedding_dim,hidden_size,num_layer,batch_first=True)
embed = embedding(input)#[10,20,30]
#初始化参数值  torch默认初始值为0
h_0 = torch.rand(num_layer,batch_size,hidden_size)
c_0 = torch.rand(num_layer,batch_size,hidden_size)
# h_0 = torch.rand(2*num_layer,batch_size,hidden_size)
# c_0 = torch.rand(2*num_layer,batch_size,hidden_size)
output,(h_1,c_1) = lstm(embed,(h_0,c_0))
print(output.size())#[batch_size,seq_len,hidden_size * 双向/单向]
print(h_1.size())#[num_layer * 单双,batch_size,hidden_szie]
print(c_1.size())#[num_layer * 单双,batch_size,hidden_szie]
last_output = output[:,-1,:]
last_hidden_state = h_1[-1,::]
print(last_output == last_hidden_state)