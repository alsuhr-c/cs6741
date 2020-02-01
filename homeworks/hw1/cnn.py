import torch
from torch import nn
from utils import train_model, test_code


class CNN(nn.Module):
    def __init__(self, vocab):
        super(CNN, self).__init__()

        self._vocabulary = vocab

        # Glove embeddings
        self._embeddings = vocab.vectors

        self._update_embeddings = nn.Embedding(*self._embeddings.size())
        self._update_embeddings.weight.data.copy_(self._embeddings)

        # CNN
        self._hidden_size = 64

        self._conv = nn.Conv1d(self._embeddings.size(1) * 2, self._hidden_size, 2)

        self._dropout = nn.Dropout(0.5)

        # Output layer
        self._output_layer = nn.Linear(self._hidden_size, 2)

    def forward(self, sentences: torch.Tensor):
        sent_length, batch_size = sentences.size()

        # Look up the embeddings
        flattened_sentences = sentences.flatten()
        embeddings = torch.index_select(self._embeddings, 0, flattened_sentences).reshape(sent_length, batch_size, -1)
        update_embeddings = self._update_embeddings(flattened_sentences).reshape(sent_length, batch_size, -1)

        embeddings = torch.cat((embeddings, update_embeddings), dim=2).permute(1, 2, 0)

        # Pass through conv layer
        conv_output = self._dropout(self._conv(embeddings))

        pooled = torch.zeros(batch_size, self._hidden_size)
        for i in range(batch_size):
            num_padding = sent_length - torch.sum(torch.eq(sentences[:, i], self._vocabulary.stoi['<pad>']))
            relevant_items = conv_output[i, :, :].unsqueeze(0)[:, :, :num_padding]
            pooled[i] = nn.functional.max_pool1d(relevant_items, relevant_items.size(2)).view(1, self._hidden_size)

        # Apply the final layer
        scores = self._output_layer(pooled).view(batch_size, 2)

        return scores.rename('items', 'classes')


def run_cnn(train_iter, val_iter, test, vocab):
    model = CNN(vocab)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, optimizer, criterion, train_iter, val_iter)
    model.load_state_dict(torch.load('best_model.pt'))

    model.eval()
    test_code(model, test)
