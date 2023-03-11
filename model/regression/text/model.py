import torch.nn

from dataset import getDataloader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import load
import os

#普通神经网络 效果极差 基本没有效果 和后面RNN LSTM作对比
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        #100代表的分词后单词的特征空间
        self.embedding = nn.Embedding(len(load.ws),100)
        self.lstm = nn.LSTM(input_size=100,hidden_size=load.hidden_size,
                            num_layers=load.num_layer,batch_first=True,
                            bidirectional=load.bidirectional,dropout=load.drop_out)
        #max_len是为了统一让输入的句子长度的向量表示一致
        # self.fc = nn.Linear(load.max_len*100,2)
        self.fc = nn.Linear(load.hidden_size * 2,2)

    def forward(self,x):
        x = self.embedding(x)#[batch_size,max_len,100]
        # x = x.view(-1,load.max_len * 100)
        # x = self.fc(x)
        x,(h_n,c_n) = self.lstm(x)
        output_fw = h_n[-2,:,:]#正向最后一次输出
        output_bw = h_n[-1,:,:]#反向最后一次输出
        output  = torch.cat([output_fw,output_bw],dim=-1)#[batch_size,hidden_size*2]
        out = self.fc(output)
        return F.log_softmax(out,dim=-1)


mymodel = model().to(load.device)
data_loader = getDataloader()
optimizer = optim.Adam(mymodel.parameters(), lr=0.001)
if os.path.exists('../model/imdbModel.pkl'):
    model.load_state_dict(torch.load('../model/imdbModel.pkl'))
    optimizer.load_state_dict(torch.load('../model/imdbOptimizer.pkl'))
def train(epoch):
    for index,(tokens,label)in enumerate(data_loader):
        # print(tokens)
        # print(label)
        tokens.to(load.device)
        label.to(load.device)
        optimizer.zero_grad()
        target = mymodel(tokens)
        loss = F.nll_loss(target,label)
        loss.backward()
        optimizer.step()
        print(f'epoch:{epoch},index:{index},loss:{loss.item()}')
        if index % 1000 == 0:
            torch.save(mymodel.state_dict(),'../model/imdbModel.pkl')
            torch.save(optimizer.state_dict(),'../model/imdbOptimizer.pkl')

if __name__ == '__main__':
    for i in range(5):
        train(i)