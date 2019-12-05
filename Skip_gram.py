import torch
import torch.nn as nn
import torch.functional as F
from typing import List, Dict

class NegtiveSample(nn.Module):
    """

    """
    def __init__(self):
        """

        """
        super(NegtiveSample, self).__init__()

    def forward(self, input_vectors:torch.Tensor, output_vectors:torch.Tensor, noise_vecotors:torch.Tensor):
        """

        :param input_vectors:(batch_size, embed_size)
        :param output_vectors:(batch_size, embed_size)
        :param noise_vecotors:(batch_size,noise_size, embed_size)
        :return: the loss of a batch
        """
        input_vectors = input_vectors.unsqueeze(2)          # (batch_size, embed_size, 1)
        output_vectors = output_vectors.unsqueeze(1)        # (batch_size, 1, embed_size)

        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log() # (batch_size, 1)
        out_loss = out_loss.squeeze() # (batch_size)

        # incorrect log

        noise_loss = torch.bmm(noise_vecotors.neg(), input_vectors).sigmoid().log().squeeze()
        noise_loss = noise_loss.sum(1)

        loss = - (noise_loss + out_loss).squeeze().mean()

        return loss







class SkipGram(nn.Module):
    """

    """

    def __init__(self, embed_size:int, n_vocab:int, window_size:int=2):
        """

        :param embed_size:
        :param n_vocab:
        """
        super(SkipGram, self).__init__()

        self.embed_size = embed_size
        self.n_vocab = n_vocab
        self.window_size = window_size

        # define the model
        self.embed_in = nn.Embedding(n_vocab, embed_size)
        self.embed_out = nn.Embedding(n_vocab, embed_size)

        # self.loss_func = NegtiveSample()
        # initialize the embeding tables with uniform distribution

        # self.embed_in.weight.data.uniform_(-1, 1)
        # self.embed_out.weight.data.uniform_(-1, 1)


    def forward(self, input_words:torch.Tensor, output_words:torch.Tensor,
                noise_words:torch.Tensor)->torch.Tensor:
        """

        :param input_words:(batch_size)
        :param output_words:(batch_size)
        :param noise_words: (batch_size, noise_size)
        :return:
        """

        input_vectors = self.embed_in(input_words)
        output_vectors = self.embed_out(output_words)
        noise_vecotors = self.embed_out(noise_words)

        input_vectors = input_vectors.unsqueeze(2)          # (batch_size, embed_size, 1)
        output_vectors = output_vectors.unsqueeze(1)        # (batch_size, 1, embed_size)

        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log() # (batch_size, 1)
        out_loss = out_loss.squeeze() # (batch_size)

        # incorrect log

        noise_loss = torch.bmm(noise_vecotors.neg(), input_vectors).sigmoid().log().squeeze()
        noise_loss = noise_loss.sum(1)

        loss = - (noise_loss + out_loss).squeeze().mean()

        return loss



