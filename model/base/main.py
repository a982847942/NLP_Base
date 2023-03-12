import torch

from encoder import Encoder
from decoder import Decoder
from seq2seqt import dataloader
from MySeq2Seq import Seq2seq
from torch.optim import Adam
import torch.nn.functional as F
import config
from tqdm import tqdm
import os

# encoder = Encoder()
# decoder = Decoder()
# print(encoder)
# print(decoder)
seq2seq = Seq2seq()
optimizer = Adam(params=seq2seq.parameters(), lr=config.learning_rate)
if os.path.exists(config.model_save_path):
    seq2seq.load_state_dict(torch.load(config.model_save_path))
    optimizer.load_state_dict(torch.load(config.optimizer_save_path))

def train(epoch):
    bar = tqdm(enumerate(dataloader), total=len(dataloader),ascii=True,desc="train")
    for index, (input, target, input_length, target_length) in bar:
        optimizer.zero_grad()
        decoder_outputs, _ = seq2seq(input, target, input_length, target_length)
        # print("---------",decoder_outputs.size())
        # print("*********",target.size())
        decoder_outputs = decoder_outputs.view(decoder_outputs.size(0) * decoder_outputs.size(1), -1)
        target = target.view(-1)
        loss = F.nll_loss(decoder_outputs, target, ignore_index=config.num_sequence.PAD)
        loss.backward()
        optimizer.step()
        bar.set_description(f'epoch:{epoch},\tindex:{index},\tloss:{loss.item():.3f}')
        if index % 100 == 0:
            torch.save(seq2seq.state_dict(),config.model_save_path)
            torch.save(optimizer.state_dict(),config.optimizer_save_path)


if __name__ == '__main__':
    for i in range(10):
        train(i)

# if __name__ == '__main__':
#     for content, label, content_length, label_length in dataloader:
#         output, hidden, _ = encoder(content, content_length)
#         decoder(output, hidden)
