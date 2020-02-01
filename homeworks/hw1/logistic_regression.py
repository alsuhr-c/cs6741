import torch
from torch import nn

from utils import train_model, test_code


class LogisticRegression(nn.Module):
    def __init__(self, vocabulary):
        super(LogisticRegression, self).__init__()

        self._vocabulary = vocabulary

        # W
        self._weights = nn.Parameter(torch.zeros(len(self._vocabulary), 1))
        nn.init.xavier_normal_(self._weights)

        # b
        self._bias = nn.Parameter(torch.zeros(1))

    def forward(self, sentences: torch.Tensor):
        sent_length, batch_size = sentences.size()
        flattened_sentences = sentences.flatten()

        # W x_i for all i in the batch
        selected_weights = torch.index_select(self._weights, 0, flattened_sentences)

        # Padding: don't allow padding tokens to contribute to the score
        padding = torch.eq(flattened_sentences, self._vocabulary.stoi['<pad>']).float().unsqueeze(1)

        # W x_i + b for all i in batch such that x_i is not a padding token
        all_scores = ((selected_weights + self._bias) * (1 - padding)).view(sent_length, batch_size)

        # Sum for all examples in the batch
        example_scores = torch.sum(all_scores, dim=0).view(1, batch_size)

        # Put through a sigmoid
        prob_neg = torch.sigmoid(example_scores)

        # Two logits: probability of negative (class 0), and probability of positive (class 1)
        class_probs = torch.cat(tuple([prob_neg, 1 - prob_neg]))

        return class_probs.rename('classes', 'items')


def run_logistic_regression(train_iter, val_iter, test, vocab):
    lr_model = LogisticRegression(vocab)
    lr_model.train()
    optimizer = torch.optim.Adam(lr_model.parameters())
    criterion = nn.BCELoss()

    def loss(logits, label):
        return criterion(logits[1], label.float())

    lr_model = train_model(lr_model, optimizer, loss, train_iter, val_iter)
    lr_model.load_state_dict(torch.load('best_model.pt'))

    lr_model.eval()
    test_code(lr_model, test)
