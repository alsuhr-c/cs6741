import torch
import numpy as np
from torch import nn
from utils import test_code

smoothing_param = 1


class NaiveBayes(nn.Module):
    def __init__(self, vocabulary):
        super(NaiveBayes, self).__init__()

        self._vocabulary = vocabulary

        self._pos_counts = np.zeros(len(self._vocabulary))
        self._neg_counts = np.zeros(len(self._vocabulary))

        self._log_count_ratio = None

        self._bias = None

    def _featurize(self, sentence):
        fv = np.zeros(len(self._vocabulary))
        for word in sentence:
            if word != self._vocabulary.stoi['<pad>']:
                fv[word] = 1
        return fv

    def train(self, train_iter):
        num_pos = 0
        num_neg = 0
        for batch in train_iter:
            for i in range(len(batch)):
                input_text = batch.text[:, i]
                label = batch.label[i]

                fv = self._featurize(input_text)
                if label == 0:
                    # Negative examples
                    self._neg_counts += fv
                    num_neg += 1
                else:
                    # Positive example
                    self._pos_counts += fv
                    num_pos += 1
        self._pos_counts += smoothing_param
        self._neg_counts += smoothing_param
        self._log_count_ratio = np.log((self._pos_counts / np.linalg.norm(self._pos_counts, 1))
                                       / (self._neg_counts / np.linalg.norm(self._neg_counts, 1)))

        self._bias = np.log(num_pos / num_neg)

    def forward(self, sentences: torch.Tensor):
        sent_length, batch_size = sentences.size()
        scores = list()
        for i in range(batch_size):
            sentence_fv = self._featurize(sentences[:, i])

            score = np.dot(self._log_count_ratio, sentence_fv) + self._bias
            scores.append((-score, score))
        return torch.tensor(scores, names=['_', 'classes'])


def run_nb(train_iter, test, vocab):
    model = NaiveBayes(vocab)

    model.train(train_iter)

    test_code(model, test)
