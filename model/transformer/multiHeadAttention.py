import torch
import torch.nn as nn
import math
def attention(query, key, value):
    """
    计算Attention的结果。
    这里其实传入的是Q,K,V，而Q,K,V的计算是放在模型中的，请参考后续的MultiHeadedAttention类。

    这里的Q,K,V有两种Shape，如果是Self-Attention，Shape为(batch, 词数, d_model)，
                           例如(1, 7, 128)，即batch_size为1，一句7个单词，每个单词128维

                           但如果是Multi-Head Attention，则Shape为(batch, head数, 词数，d_model/head数)，
                           例如(1, 8, 7, 16)，即Batch_size为1，8个head，一句7个单词，128/8=16。
                           这样其实也能看出来，所谓的MultiHead其实就是将128拆开了。

                           在Transformer中，由于使用的是MultiHead Attention，所以Q,K,V的Shape只会是第二种。

    """

    # 获取d_model的值。之所以这样可以获取，是因为query和输入的shape相同，
    # 若为Self-Attention，则最后一维都是词向量的维度，也就是d_model的值。
    # 若为MultiHead Attention，则最后一维是 d_model / h，h为head数
    d_k = query.size(-1)
    # 执行QK^T / √d_k
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

    # 执行公式中的Softmax
    # 这里的p_attn是一个方阵
    # 若是Self Attention，则shape为(batch, 词数, 次数)，例如(1, 7, 7)
    # 若是MultiHead Attention，则shape为(batch, head数, 词数，词数)
    p_attn = scores.softmax(dim=-1)

    # 最后再乘以 V。
    # 对于Self Attention来说，结果Shape为(batch, 词数, d_model)，这也就是最终的结果了。
    # 但对于MultiHead Attention来说，结果Shape为(batch, head数, 词数，d_model/head数)
    # 而这不是最终结果，后续还要将head合并，变为(batch, 词数, d_model)。不过这是MultiHeadAttention
    # 该做的事情。
    return torch.matmul(p_attn, value)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model):
        """
        h: head的数量
        """
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        # 定义W^q, W^k, W^v和W^o矩阵。
        # 如果你不知道为什么用nn.Linear定义矩阵，可以参考该文章：
        # https://blog.csdn.net/zhaohongfei_358/article/details/122797190
        self.linears = [
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
            nn.Linear(d_model, d_model),
        ]

    def forward(self, x):
        # 获取Batch Size
        nbatches = x.size(0)

        """
        1. 求出Q, K, V，这里是求MultiHead的Q,K,V，所以Shape为(batch, head数, 词数，d_model/head数)
            1.1 首先，通过定义的W^q,W^k,W^v求出SelfAttention的Q,K,V，此时Q,K,V的Shape为(batch, 词数, d_model)
                对应代码为 `linear(x)`
            1.2 分成多头，即将Shape由(batch, 词数, d_model)变为(batch, 词数, head数，d_model/head数)。
                对应代码为 `view(nbatches, -1, self.h, self.d_k)`
            1.3 最终交换“词数”和“head数”这两个维度，将head数放在前面，最终shape变为(batch, head数, 词数，d_model/head数)。
                对应代码为 `transpose(1, 2)`
        """
        query, key, value = [
            linear(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for linear, x in zip(self.linears, (x, x, x))
        ]

        """
        2. 求出Q,K,V后，通过attention函数计算出Attention结果，
           这里x的shape为(batch, head数, 词数，d_model/head数)
           self.attn的shape为(batch, head数, 词数，词数)
        """
        x = attention(
            query, key, value
        )

        """
        3. 将多个head再合并起来，即将x的shape由(batch, head数, 词数，d_model/head数)
           再变为 (batch, 词数，d_model)
           3.1 首先，交换“head数”和“词数”，这两个维度，结果为(batch, 词数, head数, d_model/head数)
               对应代码为：`x.transpose(1, 2).contiguous()`
           3.2 然后将“head数”和“d_model/head数”这两个维度合并，结果为(batch, 词数，d_model)
        """
        x = (
            x.transpose(1, 2)
                .contiguous()
                .view(nbatches, -1, self.h * self.d_k)
        )

        # 最终通过W^o矩阵再执行一次线性变换，得到最终结果。
        return self.linears[-1](x)
# 定义8个head，词向量维度为512
model = MultiHeadedAttention(8, 512)
# 传入一个batch_size为2， 7个单词，每个单词为512维度
x = torch.rand(2, 7, 512)
# 输出Attention后的结果
print(model(x).size())
