import torch.nn as nn
import config
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=len(config.num_sequence),
                                      embedding_dim=config.embedding_dim,
                                      padding_idx=config.num_sequence.PAD)
        self.gru = nn.GRU(input_size=config.embedding_dim,num_layers=config.num_layer,
                          hidden_size=config.hidden_size,batch_first=True,dropout=config.drop_out)
    def forward(self,input,input_length):
        #input[batch_size,max_len]
        embeded = self.embedding(input)
        # pack_padded_sequence必须要按照句子长度进行 降序操作（即对input按照长度进行降序排序）
        embeded = pack_padded_sequence(input=embeded,lengths=input_length,batch_first=True)
        output,hidden = self.gru(embeded)
        output,output_length = pad_packed_sequence(output,batch_first=True,padding_value=config.num_sequence.PAD)
        # [1*1.batch_size,hidden_size]
        return output,hidden,output_length

if __name__ == '__main__':
    from seq2seqt import dataloader
    encoder = Encoder()
    print(encoder)
    for index, (content, label, content_length, label_length) in enumerate(dataloader):
        out,hidden,output_length = encoder(content,content_length)
        print(out.size())
        print(hidden.size())
        print(output_length)
        break