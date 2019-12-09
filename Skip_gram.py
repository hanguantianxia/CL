import torch
import torch.nn as nn
import torch.functional as F
from typing import List, Dict


class SkipGram(nn.Module):
    """

    """

    def __init__(self, embed_size:int, vocab_size:int, window_size:int=2):
        """
        initialize the Skip-gram model given the parameters of embed_size,
        vocab_size, window_size.

        Initialize two embedding layers of Skip-gram: one is center word embedding,
        the other is context word embedding.


        :param embed_size: (int) embedding dimension
        :param vocab_size: (int) the size of vocabulary, n words + 1 Unknown word
        :param window_size: (int) the windows size of skip-gram model
        """
        super(SkipGram, self).__init__()

        self.embed_size = embed_size
        self.n_vocab = vocab_size
        self.window_size = window_size

        # define the model
        self.embed_in = nn.Embedding(vocab_size, embed_size)
        self.embed_out = nn.Embedding(vocab_size, embed_size)

        # self.loss_func = NegtiveSample()
        # initialize the embeding tables with uniform distribution

        # self.embed_in.weight.data.uniform_(-1, 1)
        # self.embed_out.weight.data.uniform_(-1, 1)


    def forward(self, input_words:torch.Tensor, output_words:torch.Tensor,
                noise_words:torch.Tensor)->torch.Tensor:
        """
        the forward function of skip-gram model, input the input words indices,
        output words indices, noise word indices(from negative sampling method)

        :param input_words: input words indices ,the type is torch.Tensor(batch_size)
        :param output_words:output words indices ,the type is torch.Tensor(batch_size)
        :param noise_words: noise word indices,the type is torch.Tensor(batch_size, noise_size)
        :return:
        """

        input_vectors = self.embed_in(input_words)
        output_vectors = self.embed_out(output_words)
        noise_vecotors = self.embed_out(noise_words)

        input_vectors = input_vectors.unsqueeze(2)          # (batch_size, embed_size, 1)
        output_vectors = output_vectors.unsqueeze(1)        # (batch_size, 1, embed_size)

        # print(input_vectors.size())
        # print(output_vectors.size())

        out_loss = torch.bmm(output_vectors, input_vectors).sigmoid().log() # (batch_size, 1)
        # print(out_loss.size())
        out_loss = out_loss.squeeze() # (batch_size)
        # print(out_loss.size())


        # incorrect log

        noise_loss = torch.bmm(noise_vecotors.neg(), input_vectors).sigmoid().log().squeeze()
        noise_loss = noise_loss.sum(1)

        # print(noise_loss.size())

        loss = - (noise_loss + out_loss).squeeze().mean()
        # print(loss)
        return loss



