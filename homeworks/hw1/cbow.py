import torch
from torch import nn
from utils import train_model, test_code


class CBOW(nn.Module):
    def __init__(self, vocabulary):
        super(CBOW, self).__init__()

        self._vocabulary = vocabulary

        # Embedding size of 64
        self._embeddings = nn.Parameter(torch.zeros(len(vocabulary), 300))
        nn.init.xavier_normal_(self._embeddings)

        # Glove embeddings
        #self._embeddings = vocabulary.vectors

        self._embeddings.data.copy_(self._embeddings)

        # Output layer
        self._output_layer = nn.Linear(300, 2)

    def forward(self, sentences: torch.Tensor):
        sent_length, batch_size = sentences.size()
        flattened_sentences = sentences.flatten()

        # Embed each word in the batch
        selected_embeddings = torch.index_select(
            self._embeddings, 0, flattened_sentences).view(
            sent_length, batch_size, -1).rename('sent_length', 'batch_size', 'embeddings')

        # Padding: don't allow padding tokens to contribute to the score
        padding = torch.eq(flattened_sentences, self._vocabulary.stoi['<pad>']).float().view(sent_length, batch_size)

        padded_embeddings = selected_embeddings * (1 - padding).unsqueeze(2)

        # Sum across the entire sentence
        summed_embs = torch.sum(padded_embeddings, dim='sent_length')

        # Pass through the final layer
        scores = self._output_layer(summed_embs).rename(None).view(batch_size, 2).rename('items', 'classes')

        return scores


def run_cbow(train_iter, val_iter, test, vocab):
    model = CBOW(vocab)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, optimizer, criterion, train_iter, val_iter)
    model.load_state_dict(torch.load('best_model.pt'))

    model.eval()
    test_code(model, test)
