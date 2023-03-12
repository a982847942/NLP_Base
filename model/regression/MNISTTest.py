import torchvision
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn

# print(torchvision.__version__)
# mnist = torchvision.datasets.MNIST('./data',train=True,download=False)
# print(mnist[0])
# mnist[0][0].show()
# 1.准备数据
BATCH_SIZE = 128


def getDataloader(train=True, batch_size=BATCH_SIZE):
    data_set = torchvision.datasets.MNIST('./data', train=train, download=True, transform=
    torchvision.transforms.Compose(
        [torchvision.transforms.ToTensor(),
         torchvision.transforms.Normalize(
             mean=(0.1307,), std=(0.3081))]))
    # print(len(data_set))
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=True, drop_last=True)
    # print(len(data_loader))
    return data_loader


# for i,content in enumerate(getDataloader()):
#     print(i,content[0].size())
#     break

class MyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = torch.nn.Linear(28 * 28 * 1, 28)
        self.fc2 = torch.nn.Linear(28, 10)

    def forward(self, x):
        # x.size()
        x = x.view(x.size(0), 28 * 28 * 1)
        x = self.fc1(x)
        x = F.relu(x)
        out = self.fc2(x)  # [batchsize,10]
        return F.log_softmax(out, dim=-1)  # 行


mnist_net = MyNet()
optimizer = optim.Adam(mnist_net.parameters(), lr=0.001)
# criterion = nn.NLLLoss()
# criterion = nn.CrossEntropyLoss()
train_loss_list = []
train_count_list = []


def train(epoch):
    mode = True
    mnist_net.train(mode=mode)
    train_dataloader = getDataloader(train=mode)
    # print(len(train_dataloader.dataset()))
    # print(len(train_dataloader))
    for index, (data, target) in enumerate(train_dataloader):

        optimizer.zero_grad()
        output = mnist_net(data)
        print("target",target.size())
        print('output',output.size())
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        break
        if index % 20 == 0:
            # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss:{.6f}'.format(
            #     epoch,index * len(data),len(train_dataloader.dataset),
            #     100.* index / len(train_dataloader),loss.item()
            # ))
            print(f'index: {index * (epoch + 1)}, loss: {loss.item()}')
            train_loss_list.append(loss.item())
            train_count_list.append(index * 128 + (epoch - 1) * len(train_dataloader))
            torch.save(mnist_net.state_dict(), './model/mnist_net.pkl')
            torch.save(optimizer.state_dict(), './model/mnist_optimizer.pkl')


def test():
    mnist_net.load_state_dict(torch.load('./model/mnist_net.pkl'))
    optimizer.load_state_dict(torch.load('./model/mnist_optimizer.pkl'))
    test_loss = 0
    correct = 0
    mnist_net.eval()
    test_dataloader = getDataloader(train=False, batch_size=BATCH_SIZE)
    with torch.no_grad():
        for data, target in test_dataloader:
            output = mnist_net(data)
            test_loss = F.nll_loss(output, target, reduction='sum').item()
            pred = output.data.max(-1, keepdim=True)[1]  # 最大值的位置
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_dataloader.dataset)
    print('\nTest set: Avg,loss: {:.4f},Accuracy: {}/{}({:.2f}%)\n'.format(
        test_loss, correct, len(test_dataloader.dataset),
        100. * correct / len(test_dataloader.dataset)
    ))


if __name__ == '__main__':
    # test_dataloader = getDataloader(train=False)
    # print(len(test_dataloader))
    # dataloader = getDataloader()
    # print(len(dataloader))
    # test()
    for i in range(5):
        train(i)
    # data_loader = getDataloader()
    # for i, (data, target) in enumerate(data_loader):
    #     print(data.size(0))
    #     break
