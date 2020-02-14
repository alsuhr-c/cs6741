import torch
import util
from torch import nn


class LSTMLanguageModel(torch.nn.Module):
    def __init__(self, text, hidden_size, num_layers, dropout):
        super(LSTMLanguageModel, self).__init__()

        self._text = text
        self._hidden_size = hidden_size
        self._num_layers = num_layers

        # Glove embeddings
        self._embeddings = self._text.vocab.vectors

        self._update_embeddings = nn.Embedding(*self._embeddings.size())
        self._update_embeddings.weight.data.copy_(self._embeddings)

        # LSTM
        self._rnn = nn.LSTM(self._embeddings.size(1), hidden_size, num_layers, dropout=dropout)

        # Output layer
        self._output_layer = nn.Linear(hidden_size, self._embeddings.size(0))

        # Keeps track of the last RNN hidden states for each item in the batch.
        self._batch_rnn_states = None

    def reset_contexts(self, batch_size):
        self._batch_rnn_states = (torch.zeros((self._num_layers, batch_size, self._hidden_size)).float().to(
            util.DEVICE),
                                  torch.zeros((self._num_layers, batch_size, self._hidden_size)).float().to(
                                      util.DEVICE))

    def forward(self, batch):
        """Computes the unnormalized likelihoods of next-token-prediction for items in a batch."""
        # First, reset the contexts to be their values but not in the computation graph
        self._batch_rnn_states = (self._batch_rnn_states[0].clone().detach().requires_grad_(True),
                                  self._batch_rnn_states[1].clone().detach().requires_grad_(True))

        logits = list()
        for token_index in range(batch.text.size(0)):
            # Given the current RNN state, predict the token
            logits.append(self._output_layer(self._batch_rnn_states[0][1]))

            # Update the RNN state
            # Embed the current inputs
            embedded_tokens = self._update_embeddings(batch.text[token_index])

            _, self._batch_rnn_states = self._rnn(embedded_tokens.unsqueeze(0), self._batch_rnn_states)

        return torch.stack(tuple(logits))


def train_lstm_model(train_iter, val_iter, test_iter, text, hidden_size, num_layers, dropout):
    model = LSTMLanguageModel(text, hidden_size, num_layers, dropout).to(util.DEVICE)
    model.train()

    model.reset_contexts(train_iter.batch_size)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    util.train_model(model, train_iter, val_iter, optimizer, criterion)
    model.load_state_dict(torch.load('best_model.pt'))
    model.eval()

    model.reset_contexts(val_iter.batch_size)
    val_perplexity = util.evaluate_perplexity(model, val_iter, True)
    print('Validation perplexity: %s' % val_perplexity)

    model.reset_contexts(test_iter.batch_size)
    test_perplexity = util.evaluate_perplexity(model, test_iter, True)
    print('Test perplexity: %s' % test_perplexity)

def evaluate_lstm_model(train_iter, val_iter, test_iter, text, hidden_size, num_layers):
    model = LSTMLanguageModel(text, hidden_size, num_layers, 0.).to(util.DEVICE)

    model.load_state_dict(torch.load('best_lstm_model.pt'))
    model.eval()

    model.reset_contexts(val_iter.batch_size)
    val_perplexity = util.evaluate_perplexity(model, val_iter, True)
    print('Validation perplexity: %s' % val_perplexity)

    model.reset_contexts(test_iter.batch_size)
    test_perplexity = util.evaluate_perplexity(model, test_iter, True)
    print('Test perplexity: %s' % test_perplexity)
