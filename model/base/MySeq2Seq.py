import torch.nn as nn
from encoder import Encoder
from decoder import Decoder


class Seq2seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, input, target, input_length, target_length):
        encoder_outputs, encoder_hidden, _ = self.encoder(input, input_length)
        # print("encoder：",encoder_outputs.size())
        decoder_outputs, decoder_hidden = self.decoder(target, encoder_hidden)
        return decoder_outputs, decoder_hidden


    def evaluate(self, input, input_length):
        encoder_outputs, encoder_hidden, _ = self.encoder(input, input_length)
        indices = self.decoder.evaluate(encoder_hidden)
        return indices