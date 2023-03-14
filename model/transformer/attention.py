import torch
import torch.nn as nn
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, input_vector_dim: int, dim_k=None, dim_v=None):
        """
        初始化SelfAttention，包含如下关键参数：
        input_vector_dim: 输入向量的维度，对应上述公式中的d，例如你将单词编码为了10维的向量，则该值为10
        dim_k: 矩阵W^k和W^q的维度
        dim_v: 输出向量的维度，即b的维度，例如，经过Attention后的输出向量b，如果你想让他的维度为15，则该值为15，若不填，则取input_vector_dim
        """
        super(SelfAttention, self).__init__()

        self.input_vector_dim = input_vector_dim
        # 如果 dim_k 和 dim_v 为 None，则取输入向量的维度
        if dim_k is None:
            dim_k = input_vector_dim
        if dim_v is None:
            dim_v = input_vector_dim

        """
        实际写代码时，常用线性层来表示需要训练的矩阵，方便反向传播和参数更新
        """
        self.W_q = nn.Linear(input_vector_dim, dim_k, bias=False)
        self.W_k = nn.Linear(input_vector_dim, dim_k, bias=False)
        self.W_v = nn.Linear(input_vector_dim, dim_v, bias=False)

        # 这个是根号下d_k
        self._norm_fact = 1 / np.sqrt(dim_k)

    def forward(self, x):
        """
        进行前向传播：
        x: 输入向量，size为(batch_size, input_num, input_vector_dim)
        """
        # 通过W_q, W_k, W_v矩阵计算出，Q,K,V
        # Q,K,V矩阵的size为 (batch_size, input_num, output_vector_dim)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # permute用于变换矩阵的size中对应元素的位置，
        # 即，将K的size由(batch_size, input_num, output_vector_dim)，变为(batch_size, output_vector_dim，input_num)
        # 0,1,2 代表各个元素的下标，即变换前，batch_size所在的位置是0，input_num所在的位置是1
        K_T = K.permute(0, 2, 1)

        # bmm是batch matrix-matrix product，即对一批矩阵进行矩阵相乘
        # bmm详情参见：https://pytorch.org/docs/stable/generated/torch.bmm.html
        atten = nn.Softmax(dim=-1)(torch.bmm(Q, K_T) * self._norm_fact)

        # 最后再乘以 V
        output = torch.bmm(atten, V)

        return output


model = SelfAttention(3, 5, 4)
print(model(torch.Tensor(50, 5, 3)).size())
