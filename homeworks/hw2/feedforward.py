import torch
import util
import numpy as np
from torch import nn


class FeedforwardLanguageModel(torch.nn.Module):
    def __init__(self, text, window_size, hidden_size):
        super(FeedforwardLanguageModel, self).__init__()

        self._text = text
        self._window_size = window_size

        # Glove embeddings
        self._embeddings = self._text.vocab.vectors

        self._update_embeddings = nn.Embedding(*self._embeddings.size())
        self._update_embeddings.weight.data.copy_(self._embeddings)

        # First layer will concatenate the embeddings from the past window_size things
        self._combination_layer = nn.Linear(self._embeddings.size(1) * 2 * window_size, hidden_size)
        self._output_layer = nn.Linear(hidden_size, len(self._text.vocab))

        # Keeps track of the last window_size tokens for each item in the batch.
        # Initially, all 0, indicating that the embedding should be zeroed out.
        self._batch_contexts = None

    def reset_contexts(self, batch_size):
        self._batch_contexts = np.full((batch_size, self._window_size), 0)

    def forward(self, batch):
        """Computes the unnormalized likelihoods of next-token-prediction for items in a batch."""
        batch_size = batch.text.size(1)
        logits = list()
        for token_index in range(batch.text.size(0)):
            inputs = torch.tensor(self._batch_contexts)

            # Embed words and flatten
            update_embeddings = self._update_embeddings(inputs)

            flattened_inputs = inputs.view(-1)
            embeddings = torch.index_select(self._embeddings, 0, flattened_inputs).view(
                batch_size, self._window_size, -1)
            embeddings = torch.cat((embeddings, update_embeddings), dim=2).view(batch_size, -1)

            # Put through middle layer and tanh
            embeddings = torch.tanh(self._combination_layer(embeddings))

            # Put through the output layer
            logits.append(self._output_layer(embeddings))

            # Update the context by shifting contexts left and adding the most recent input
            self._batch_contexts[:, :-1] = self._batch_contexts[:, 1:]
            self._batch_contexts[:, -1] = batch.text[token_index]
        return torch.stack(tuple(logits))


def train_feedforward_language_model(train_iter, val_iter, test_iter, text, window_size, hidden_size):
    model: FeedforwardLanguageModel = FeedforwardLanguageModel(text, window_size, hidden_size)

    model.reset_contexts(train_iter.batch_size)

    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    util.train_model(model, train_iter, val_iter, optimizer, criterion)

    model.reset_contexts(val_iter.batch_size)
    val_perplexity = util.evaluate_perplexity(model, val_iter, True)
    print('Validation perplexity: %s' % val_perplexity)
