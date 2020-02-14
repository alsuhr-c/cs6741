"""Training and evaluation utilities for language modeling."""
import sys
import torch
from tqdm import tqdm
EOS = '<eos>'
UNK = '<unk>'


def evaluate_perplexity(model, data_iter, logits=False):
    # Perplexity is
    #   exp( (-sum_token log( p( token ) ) ) / n )
    # where n is the number of tokens in the dataset
    log_probabilities_sum = 0.
    num_tokens = 0
    for batch in iter(data_iter):
        # Get the token probabilities.
        probabilities = model(batch)

        if logits:
            # Have to do the softmax and pick the probability of the correct thing
            # Should be timestep x batch_size x vocab_size
            probabilities = torch.softmax(probabilities, dim=2)
            seq_length, batch_size, vocab_size = probabilities.size()

            flattened_probabilities = probabilities.view(seq_length * batch_size, -1)
            flattened_targets = batch.target.view(seq_length * batch_size)

            prob_list = list()
            for i in range(seq_length * batch_size):
                prob_list.append(flattened_probabilities[i][flattened_targets[i]])

            # Picking out the probability of the target
            probabilities = torch.stack(tuple(prob_list))

        log_probabilities_sum += torch.sum(torch.log(probabilities))
        num_tokens += batch.text.size(0) * batch.text.size(1)

    # Final log probability
    return torch.exp(-log_probabilities_sum / num_tokens).item()


def train_model(model, train_iter, val_iter, optimizer, loss_fn):
    patience = 10
    countdown = patience
    min_ppl = sys.maxsize
    epoch_num = 0
    num_batches = len(list(iter(train_iter)))
    while countdown >= 0:
        model.train()
        train_it = iter(train_iter)
        with tqdm(total=num_batches) as pbar:
            for batch in train_it:
                optimizer.zero_grad()
                logits = model(batch)
                labels = batch.target

                timestep, batch_size, vocab_size = logits.size()

                logits = logits.view(batch_size * timestep, -1)
                labels = labels.view(batch_size * timestep)

                loss = loss_fn(logits, labels)

                loss.backward()
                optimizer.step()
                pbar.update(1)

        model.eval()
        best = False
        val_ppl = evaluate_perplexity(model, val_iter, True)
        if val_ppl < min_ppl:
            torch.save(model.state_dict(), 'best_model.pt')
            min_ppl = val_ppl
            patience *= 1.1
            countdown = patience
            best = True
        countdown -= 1
        print('Epoch #%d; countdown = %d; val ppl = ' % (epoch_num, countdown) +
              '{0:.2f}'.format(val_ppl) + ('*' if best else ''))
        epoch_num += 1
        model.train()
