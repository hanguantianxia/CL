import json
from typing import List, Dict
import os


class CorpusPreprocess:

    def __init__(self, vocab=None, filename="vocab.json", unknownword_token="UNK"):
        """

        :param filename:
        """
        if vocab is None:
            self.vocab = self.read_vocab(filename)
        else:
            self.vocab = vocab
        self.unknownword_token = unknownword_token
        self.vocab_size = len(self.vocab)

    def read_vocab(self, filename="vocab.json"):
        """
        read the vocab
        :param filename:
        :return:
        """
        vocab = None
        with open(filename, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        return vocab

    def word2id(self, word: str) -> int:
        """

        :param word:
        :param vocab:
        :return:
        """
        return self.vocab.get(word, self.vocab[self.unknownword_token])

    def read_corpus(self, filename) -> List[List[int]]:
        """

        :param filename:
        :param vocab:
        :return:
        """
        corpus = []
        corpus_str = []
        with open(filename, 'r', encoding='utf-8') as f:
            while True:
                sent = f.readline()
                if not sent:
                    break

                sent_list_str = sent.strip().split(' ')
                sent_list_id = list(map(self.word2id, sent_list_str))

                if sent_list_id.count(self.vocab[self.unknownword_token]) > len(sent_list_id) / 2:
                    # when the number of UNK is larger than 1/2, it's a nonsence sentence , pass it.
                    continue

                corpus.append(sent_list_id)
                corpus_str.append(sent_list_str)

        return corpus, corpus_str


def save_list(filename: str, seg_sents: List[List[str]]):
    """

    :param filename:
    :param seg_sents:
    :return:
    """
    with open(filename, 'w', encoding="utf-8") as f:
        for sent in seg_sents:
            sentence = " ".join(sent)
            # print(sentence)
            f.write(sentence + '\n')


def all2one(dirpath, obj_filename):
    """

    :param dirpath:
    :return:
    """
    file_list = os.listdir(dirpath)
    vocab = CorpusPreprocess()
    corpus = []
    corpus_str = []

    for file in file_list:
        print("Now it's {}".format(file))
        filename = os.path.join(dirpath, file)
        corpus_f, corpus_str_f = vocab.read_corpus(filename)
        corpus.extend(corpus_f)
        corpus_str.extend(corpus_str_f)
        print(len(corpus_str_f))

    save_filename = os.path.join(dirpath, obj_filename)
    save_list(save_filename, corpus_str)

if __name__ == '__main__':
    # test_data = "数学 是 利用 符号 语言 研究 数量 结构 变化 以及 空间 等 概念 的 一 门 学科 从某 种 角度 看 属于 形式 科学 的 一 种"
    # test_data = test_data.strip().split()
    #
    #
    # vocab = CorpusPreprocess()
    #
    # res = list(map(vocab.word2id, test_data))
    #
    # corpus = vocab.read_corpus(os.path.join(os.path.join(".", "train_data") , "wiki_00_seg"))
    all2one(os.path.join(".", "train_data"), "train_data")