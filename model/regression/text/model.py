import torch.nn

from dataset import getDataloader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from load import ws,max_len

#普通神经网络 效果极差 基本没有效果 和后面RNN LSTM作对比
class model(nn.Module):
    def __init__(self):
        super(model,self).__init__()
        #100代表的分词后单词的特征空间
        self.embedding = nn.Embedding(len(ws),100)
        #max_len是为了统一让输入的句子长度的向量表示一致
        self.fc = nn.Linear(max_len*100,2)
    def forward(self,x):
        x = self.embedding(x)#[2,max_len,100]
        x = x.view(-1,max_len * 100)
        x = self.fc(x)
        return F.log_softmax(x,dim=-1)

mymodel = model()
data_loader = getDataloader()
optimizer = optim.Adam(mymodel.parameters(), lr=0.001)
def train(epoch):
    for index,(tokens,label)in enumerate(data_loader):
        # print(tokens)
        # print(label)
        optimizer.zero_grad()
        target = mymodel(tokens)
        loss = F.nll_loss(target,label)
        loss.backward()
        optimizer.step()
        print(f'epoch:{epoch},index:{index},loss:{loss.item()}')

if __name__ == '__main__':
    for i in range(1):
        train(i)