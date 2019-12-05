import os
import numpy
import torch
from typing import Tuple,Dict,List

class Batch_Generator:
    """
    the batch generator
    """

    def __init__(self, corpus:List[List[str]], vocab:Dict, window_size:int=4, neg_samples:int=10,
                 save_path="batch_state.json", load=False):
        """

        :param corpus:
        :param vocab:
        """
        self.corpus = corpus
        self.vocab = vocab
        self.window_size = window_size
        self.neg_samples = neg_samples


        self.save_path = os.path.join(".", save_path)
        if load:
            # read the state
            pass




    def batch_generator(self, batch_size:int=128,
                        state:int=0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        It is a generator to get vectors of center words, context words, negtive sampled words.
        To go through the corpus

        :param corpus:
        :param vocab:
        :param batch_size:
        :param window_size:
        :param neg_samples:
        :param state:
        :return:
        """

        ### TODO:
        #   1.
        current_batch_size = 0

        center_list = [None] * batch_size
        context_list = [None] * batch_size
        negsample_list = [[None] * self.neg_samples] * batch_size # List,List

        for sent in self.corpus:

            for wordID in range(len(sent)):
                # word
                center_word = sent[wordID]
                context = sent[max(0, wordID - self.window_size):wordID]
                if wordID + 1 < len(sent):
                    context += sent[wordID + 1:min(len(sent), wordID + self.window_size + 1)]
                for context_word in context:
                    center_list[current_batch_size] = center_word
                    context_list[current_batch_size] = context_word
                    negtive_sample = self.generate_neg(center_word,context)
                    negsample_list[current_batch_size] = negtive_sample
                    current_batch_size += 1
                    if current_batch_size == batch_size:
                        # yield torch.tensor(context_list), torch.tensor(context_list), torch.tensor(negsample_list)
                        yield  center_list, context_list, negsample_list
                        current_batch_size = 0

    def generate_neg(self, center_word, context):
        """

        :param center_word:
        :param context:
        :return:
        """
        pass

def cheak():
    test_data = "数学 是 利用 符号 语言 研究 数量 结构 变化 以及 空间 等 概念 的 一 门 学科 从某 种 角度 看 属于 形式 科学 的 一 种"
    test_data = [test_data.strip().split()]
    generator = Batch_Generator(test_data, {})
    gen = generator.batch_generator(4)
    print(gen)
    for i in gen:
        print(i)

    print(test_data)


if __name__ == '__main__':
    cheak()