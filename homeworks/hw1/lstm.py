import torch
from torch import nn
from utils import train_model, test_code
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def fast_run_rnn(seq_lens_tensor: torch.Tensor, rnn_input: torch.Tensor, rnn: torch.nn.RNN) -> torch.Tensor:
    rnn_input = rnn_input.rename(None)
    seq_lens_tensor = seq_lens_tensor.rename(None)

    max_length = rnn_input.size(1)

    # Sort the lengths and get the old indices
    sorted_lengths, permuted_indices = seq_lens_tensor.sort(0, descending=True)

    # Resort the input
    sorted_input = rnn_input[permuted_indices]

    # Pack the input
    packed_input = pack_padded_sequence(sorted_input, sorted_lengths.cpu().numpy(), batch_first=True)

    # Run the RNN
    rnn.flatten_parameters()
    packed_output = rnn(packed_input)[0]

    output = pad_packed_sequence(packed_output, batch_first=True, total_length=max_length)[0]

    _, unpermuted_indices = permuted_indices.sort(0)

    # Finally, sort back to original state
    hidden_states = output[unpermuted_indices]

    return hidden_states


class LSTM(nn.Module):
    def __init__(self, vocab):
        super(LSTM, self).__init__()

        self._vocabulary = vocab

        # Glove embeddings
        self._embeddings = vocab.vectors

        self._update_embeddings = nn.Embedding(*self._embeddings.size())
        self._update_embeddings.weight.data.copy_(self._embeddings)

        # Bidirectional LSTM with hidden size of 32
        self._rnn = nn.LSTM(self._embeddings.size(1) * 2, 32, batch_first=True, bidirectional=True)

        # Output layer
        self._output_layer = nn.Linear(64, 2)

    def forward(self, sentences: torch.Tensor):
        sent_length, batch_size = sentences.size()

        flattened_sentences = sentences.flatten()

        # Look up the embeddings
        # Size is now batch_size x sentence_length x embedding_size
        embeddings = torch.index_select(self._embeddings, 0, flattened_sentences).reshape(sent_length, batch_size, -1)
        update_embeddings = self._update_embeddings(flattened_sentences).reshape(sent_length, batch_size, -1)

        embeddings = torch.cat((embeddings, update_embeddings), dim=2).permute(1, 0, 2)

        # Get sentence lengths
        padding = torch.eq(sentences, self._vocabulary.stoi['<pad>']).float().permute(1, 0).rename(
            'batch_items', 'is_pad')

        sent_lengths = torch.sum((1 - padding), dim='is_pad')

        # Run the RNN over the sentences and pad the hidden states
        hidden_states = fast_run_rnn(sent_lengths, embeddings, self._rnn) * (1 - padding.rename(None).unsqueeze(
            2)).rename('batch_items', 'tokens', 'hidden_state')

        # Now do an average over the sentence by dividing by the sentence length
        avg_hidden_state = torch.sum(hidden_states, dim='tokens') / sent_lengths.rename(None).view(batch_size, 1)

        # Apply the final layer
        scores = self._output_layer(avg_hidden_state).rename(None).view(batch_size, 2).rename('items', 'classes')

        return scores


def run_lstm(train_iter, val_iter, test, vocab):
    model = LSTM(vocab)
    model.train()
    optimizer = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    model = train_model(model, optimizer, criterion, train_iter, val_iter)
    model.load_state_dict(torch.load('best_model.pt'))

    model.eval()
    test_code(model, test)
