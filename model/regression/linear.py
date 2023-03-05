import torch
import numpy as np
import matplotlib.pyplot as plt

# y = 3 * x  + 0.8
x_data = torch.rand([1, 100]) * 9 + 1
y_truth = 3 * x_data + 0.8
w = torch.rand([1, 1], requires_grad=True)
b = torch.tensor(0, requires_grad=True, dtype=torch.float32)
learning_rate = 0.01


def forward(x):
    return torch.matmul(w, x) + b


def loss(x, y):
    y_predict = forward(x)
    return torch.pow((y_predict - y), 2).mean()


Epoch_list = []  # 保存epoch
Loss_list = []  # 保存每个epoch对应的loss
for epoch in range(1000):
    l = loss(x_data, y_truth)
    if w.grad is not None:
        w.grad.data.zero_()
    if b.grad is not None:
        b.grad.data.zero_()
    l.backward()
    w.data = w.data - learning_rate * w.grad.data
    b.data = b.data - learning_rate * b.grad.data
    Epoch_list.append(epoch)
    Loss_list.append(l.item())
    print("Epoch:", epoch, "Loss={:.4f}".format(l.item()))
# plt.scatter(Epoch_list, Loss_list)
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.grid(ls='--')
# plt.show()
# 用训练好的模型计算x = 4时 y的值
print("predict (After Training):", 4, forward(torch.tensor(4,dtype=torch.float32).reshape(1,1)).item())
plt.figure(figsize=[20,8])
plt.scatter(x_data.numpy().reshape(-1),y_truth.numpy().reshape(-1))
y_predict = torch.matmul(w,x_data) + b
plt.plot(x_data.numpy().reshape(-1),y_predict.detach().numpy().reshape(-1),color='red')
plt.show()
