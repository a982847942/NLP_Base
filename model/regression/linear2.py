import torch
import matplotlib.pyplot as plt
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
x_data = (torch.rand([50,1]) * 9 + 1).to(device)
y_truth = x_data * 3 + 0.8
learning_rate= 0.01

class Linear(torch.nn.Module):
    def __init__(self):
        super(Linear,self).__init__() #调用父类来初始化
        self.linear1 = torch.nn.Linear(1,3) #设置自己的linear属性
        self.linear2 = torch.nn.Linear(3,1)

    def forward(self,x):
        out = self.linear2(self.linear1(x))
        return out

# 实例化模型 损失函数 优化器
model = Linear().to(device)
criterion = torch.nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=1e-3)
epoch = 30000
for i in range(epoch):
    out = model(x_data)#重写了__call__方法，实际调用forward方法
    loss = criterion(y_truth,out)#计算损失函数
    optimizer.zero_grad()#梯度清零
    loss.backward()#反向传播，计算梯度
    optimizer.step()#更新梯度(通过学习率)
    if (i + 1) % 20 == 0:
        print('Epoch[{}/{}],loss:{:.6f}'.format(i + 1,epoch,loss.data))
model.eval()#开启评估模式,不再反向传播
predict = model(x_data)
predict = predict.data.numpy()
plt.scatter(x_data.data.numpy(),y_truth.data.numpy(),c='r')
plt.plot(x_data.data.numpy(),predict)
plt.show()

