import config
import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.num_sequence),
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD)
        self.gru = nn.GRU(input_size=config.embedding_dim, num_layers=config.num_layer,
                          hidden_size=config.hidden_size, batch_first=True,dropout=config.drop_out)
        self.fc = nn.Linear(config.hidden_size, len(config.num_sequence))

    def forward(self, target, encoder_hidden):
        decoder_hidden = encoder_hidden
        batch_size = target.size(0)
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * config.num_sequence.SOS)
        #保存每一步的结果作为预测
        decoder_outputs = torch.zeros([batch_size,config.max_len + 2,len(config.num_sequence)])
        for i in range(config.max_len + 2):
            out_put, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            #每一次结果都加入
            # print("target:",target.size())
            decoder_outputs[:, i, :] = out_put
            if np.random.random() > 0.5:
                decoder_input = target[:,i].view(batch_size,-1)
                # print("decoder_input:",decoder_input.size())
            else:
                value, index = torch.topk(out_put, 1)
                # print("index:",index.size())
                decoder_input = index
            # decoder_input = torch.LongTensor(value)
        return decoder_outputs,decoder_hidden

    def forward_step(self, decoder_input, decoder_hidden):
        decoder_input_embeded = self.embedding(decoder_input)  # [batch_size,1] --> [batch_size,1,100]
        # output[batich_size,1,100]
        # decoder_hidden[1,batch_size,hidden_size]
        out_put, decoder_hidden = self.gru(decoder_input_embeded, decoder_hidden)
        out_put = out_put.squeeze(1)
        # [batch_size,len(config.num_sequence)]
        out_put = F.log_softmax(self.fc(out_put), dim=-1)
        # print("out_put:",out_put.size())
        return out_put, decoder_hidden
    def evaluate(self,encoder_hidden):
        '''评估'''
        decoder_hidden =  encoder_hidden
        batch_size = encoder_hidden.size(1)
        decoder_input = torch.LongTensor(torch.ones([batch_size, 1], dtype=torch.int64) * config.num_sequence.SOS)

        indices = []
        for i in  range(config.max_len + 5):
            decoder_output,decoder_hidden = self.forward_step(decoder_input,decoder_hidden)
            value,index = torch.topk(decoder_output,1)
            decoder_input = index
            # if index.item() == config.num_sequence.EOS:
            #     break
            indices.append(index.squeeze(-1).detach().numpy())
        return indices
