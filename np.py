import torch
from torch import nn

class NeuralProcess(nn.Module):
    # should be a generic np class allowing for different encoders/decoders and combination methods
    def __init__(self, encoder, decoder, combiner, r_to_dist_encoder):
        super(NeuralProcess, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.combiner = combiner
        self.r_to_dist = r_to_dist_encoder

    def forward(self):
        pass
    __