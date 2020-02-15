import lstm
import feedforward
import torch


def evaluate_models(text, train_iter, val_iter):
    lstm_model = lstm.LSTMLanguageModel(text, 512, 2, 0.)
    lstm_model.load_state_dict(torch.load('best_lstm_model.pt', map_location=torch.device('cpu')))
    lstm_model.reset_contexts(train_iter.batch_size)

    feedforward_model = feedforward.FeedforwardLanguageModel(text, 8, 512)
    feedforward_model.load_state_dict(torch.load('best_feedforward_model.pt', map_location=torch.device('cpu')))
    feedforward_model.reset_contexts(train_iter.batch_size)

    train_iter = iter(train_iter)
    val_iter = iter(val_iter)

    train_batch = next(train_iter)
    val_batch = next(val_iter)

    # Run the models on these
    lstm_train_probs = torch.softmax(lstm_model(train_batch), dim=2)
    lstm_val_probs = torch.softmax(lstm_model(train_batch), dim=2)
    feedforward_train_probs = torch.softmax(lstm_model(train_batch), dim=2)
    feedforward_val_probs = torch.softmax(lstm_model(train_batch), dim=2)

    # Just take the first sentence for train, halfway through
    print('---- Training example ----')
    print(' '.join([text.vocab.itos[train_batch.text[i][2]] for i in range(train_batch.text.size(0))][:10]))
    lstm_train_prob = lstm_train_probs[9][2]
    feedforward_train_prob = feedforward_train_probs[9][2]
    lstm_entropy = -torch.sum(lstm_train_prob * torch.log(lstm_train_prob))
    feedforward_entropy = -torch.sum(feedforward_train_prob * torch.log(feedforward_train_prob))
    print('Entropy of LSTM model: %s' % lstm_entropy)
    print('Entropy of feedforward model: %s' % feedforward_entropy)

    kl_lstm_feedforward = -torch.sum(lstm_train_prob * torch.log(feedforward_train_prob)) - feedforward_entropy
    print('KL(lstm, feedforward): %s' % kl_lstm_feedforward)
    kl_feedforward_lstm = -torch.sum(feedforward_train_prob * torch.log(lstm_train_prob)) - lstm_entropy
    print('KL(feedforward, lstm): %s' % kl_feedforward_lstm)

    print('---- Validation example (1) ----')
    print(' '.join([text.vocab.itos[val_batch.text[i][1]] for i in range(val_batch.text.size(0))][:11]))
    lstm_val_prob = lstm_val_probs[10][1]
    feedforward_val_prob = feedforward_val_probs[10][1]
    lstm_entropy = -torch.sum(lstm_val_prob * torch.log(lstm_val_prob))
    feedforward_entropy = -torch.sum(feedforward_val_prob * torch.log(feedforward_val_prob))
    print('Entropy of LSTM model: %s' % lstm_entropy)
    print('Entropy of feedforward model: %s' % feedforward_entropy)

    kl_lstm_feedforward = -torch.sum(lstm_val_prob * torch.log(feedforward_val_prob)) - feedforward_entropy
    print('KL(lstm, feedforward): %s' % kl_lstm_feedforward)
    kl_feedforward_lstm = -torch.sum(feedforward_val_prob * torch.log(lstm_val_prob)) - lstm_entropy
    print('KL(feedforward, lstm): %s' % kl_feedforward_lstm)

    print('---- Validation example (2) ----')
    print(' '.join([text.vocab.itos[val_batch.text[i][6]] for i in range(val_batch.text.size(0))][:8]))
    lstm_val_prob = lstm_val_probs[7][6]
    feedforward_val_prob = feedforward_val_probs[7][6]
    lstm_entropy = -torch.sum(lstm_val_prob * torch.log(lstm_val_prob))
    feedforward_entropy = -torch.sum(feedforward_val_prob * torch.log(feedforward_val_prob))
    print('Entropy of LSTM model: %s' % lstm_entropy)
    print('Entropy of feedforward model: %s' % feedforward_entropy)

    kl_lstm_feedforward = -torch.sum(lstm_val_prob * torch.log(feedforward_val_prob)) - feedforward_entropy
    print('KL(lstm, feedforward): %s' % kl_lstm_feedforward)
    kl_feedforward_lstm = -torch.sum(feedforward_val_prob * torch.log(lstm_val_prob)) - lstm_entropy
    print('KL(feedforward, lstm): %s' % kl_feedforward_lstm)
