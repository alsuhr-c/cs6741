import torch
import random
import util
import numpy as np


class CountBasedLinearInterpolation(torch.nn.Module):
    def __init__(self, train_iter, text, trigram_coefficient=None, bigram_coefficient=None, val_iter=None):
        super(CountBasedLinearInterpolation, self).__init__()
        unigram_counts = dict()
        bigram_counts = dict()
        trigram_counts = dict()
        self._text = text

        # Ensure the coefficients define a valid simplex
        if trigram_coefficient is not None and bigram_coefficient is not None:
            assert 0 <= trigram_coefficient <= 1
            assert 0 <= bigram_coefficient <= 1
            assert 0 <= trigram_coefficient + bigram_coefficient <= 1

        self._trigram_coefficient = trigram_coefficient
        self._bigram_coefficient = bigram_coefficient

        num_wordtype_instances = 0
        num_bigram_instances = dict()
        num_trigram_instances = dict()

        # Compute the counts
        # Add-1 smoothing: each count starts at 1

        self._batch_prev_prev_tokens = ['<bos>' for _ in range(train_iter.batch_size)]
        self._batch_prev_tokens = ['<bos>' for _ in range(train_iter.batch_size)]

        for batch in iter(train_iter):
            for batch_index in range(batch.text.size(1)):
                for token_index in range(batch.text.size(0)):
                    token = self._text.vocab.itos[batch.text[token_index][batch_index]]

                    unigram = token

                    num_wordtype_instances += 1

                    if unigram not in unigram_counts:
                        unigram_counts[unigram] = 0
                    unigram_counts[unigram] += 1

                    prev_token = self._batch_prev_tokens[batch_index]
                    prev_prev_token = self._batch_prev_prev_tokens[batch_index]

                    if prev_token not in bigram_counts:
                        bigram_counts[prev_token] = dict()
                    if prev_token not in num_bigram_instances:
                        num_bigram_instances[prev_token] = 0
                    if unigram not in bigram_counts[prev_token]:
                        bigram_counts[prev_token][unigram] = 0
                    bigram_counts[prev_token][unigram] += 1
                    num_bigram_instances[prev_token] += 1

                    prev_pair = (prev_prev_token, prev_token)
                    if prev_pair not in trigram_counts:
                        trigram_counts[prev_pair] = dict()
                    if prev_pair not in num_trigram_instances:
                        num_trigram_instances[prev_pair] = 0
                    if unigram not in trigram_counts[prev_pair]:
                        trigram_counts[prev_pair][unigram] = 0
                    trigram_counts[prev_pair][unigram] += 1
                    num_trigram_instances[prev_pair] += 1

                    self._batch_prev_prev_tokens[batch_index] = prev_token
                    self._batch_prev_tokens[batch_index] = token

        # Convert to probabilities for easy lookup later
        # Add 1 for smoothing
        self._unigram_probabilities = dict()
        for word_type, count in unigram_counts.items():
            self._unigram_probabilities[word_type] = count / num_wordtype_instances

        # Add vocab size for smoothing
        self._bigram_probabilities = dict()
        for context, counts in bigram_counts.items():
            self._bigram_probabilities[context] = dict()
            num_occurrences = num_bigram_instances[context]

            for word_type, count in counts.items():
                self._bigram_probabilities[context][word_type] = count / num_occurrences

        # Add number of possible contexts for smoothing
        self._trigram_probabilities = dict()
        for context, counts in trigram_counts.items():
            self._trigram_probabilities[context] = dict()
            num_occurrences = num_trigram_instances[context]

            for word_type, count in counts.items():
                self._trigram_probabilities[context][word_type] = count / num_occurrences

        if trigram_coefficient is None and bigram_coefficient is None:
            self._em(val_iter)

    def _em(self, val_iter):
        """Learns the optimal values of the alpha coefficients using expectation maximization."""
        # [1] Randomly initialize the coefficient values.
        trigram_coefficient = random.random()
        bigram_coefficient = random.random()

        while trigram_coefficient + bigram_coefficient > 1:
            trigram_coefficient = random.random()
            bigram_coefficient = random.random()

        self._trigram_coefficient = trigram_coefficient
        self._bigram_coefficient = bigram_coefficient
        print('trigram = %s; bigram = %s; unigram = %s; val ppl = %s' % (
            trigram_coefficient, bigram_coefficient, (1 - trigram_coefficient - bigram_coefficient),
            util.evaluate_perplexity(self, val_iter)))

        # [2] Until convergence
        prev_values = (self._trigram_coefficient, self._bigram_coefficient)
        converged = False
        while not converged:
            for batch in val_iter:
                self._trigram_coefficient = 1.
                self._bigram_coefficient = 0.
                weighted_trigram_probabilities = trigram_coefficient * self(batch).reshape(-1)

                self._trigram_coefficient = 0.
                self._bigram_coefficient = 1.
                weighted_bigram_probabilities = bigram_coefficient * self(batch).reshape(-1)

                self._trigram_coefficient = 0.
                self._bigram_coefficient = 0.
                weighted_unigram_probabilities = (1 - trigram_coefficient - bigram_coefficient) * self(batch).reshape(-1)

                all_probs = torch.stack((weighted_trigram_probabilities, weighted_bigram_probabilities,
                                         weighted_unigram_probabilities)).permute(1, 0)
                weighted_prob_sums = torch.sum(all_probs, dim=0)
                new_distribution = torch.softmax(weighted_prob_sums, dim=0)
                trigram_coefficient = new_distribution[0].item()
                bigram_coefficient = new_distribution[1].item()

                self._trigram_coefficient = trigram_coefficient
                self._bigram_coefficient = bigram_coefficient

            if (self._trigram_coefficient, self._bigram_coefficient) == prev_values:
                converged = True
            prev_values = (self._trigram_coefficient, self._bigram_coefficient)

            print('trigram = %s; bigram = %s; unigram = %s; val ppl = %s' % (
                trigram_coefficient, bigram_coefficient, (1 - trigram_coefficient - bigram_coefficient),
                util.evaluate_perplexity(self, val_iter)))

    def reset_contexts(self, batch_size):
        self._batch_prev_prev_tokens = ['<bos>' for _ in range(batch_size)]
        self._batch_prev_tokens = ['<bos>' for _ in range(batch_size)]

    def forward(self, batch):
        # Given a batch of sentences, returns a tensor containing the probabilities of each item in the batch
        # according to the model.
        probabilities = torch.zeros(batch.text.size())

        for batch_index in range(batch.text.size(1)):
            # Whole-sentence probability is a product of the conditional probabilities of each word:
            # p(X_{1..T}) = \Pi_{i=1..T} p(x_i | x_{0..i-1})
            # which we approximate by using k=2 (trigram prefix)

            for token_index in range(batch.text.size(0)):
                token = self._text.vocab.itos[batch.text[token_index][batch_index]]

                unigram = token

                # Compute the conditional probability of this token
                # p(x_i | x_{0..i-1}) ~= a_1 p(x_i | x_{i-2}, x_{i-1}) + a_2 p(x_i | x_{i-1}) + (1 - a_1 - a_2) p(x_i)
                if unigram in self._unigram_probabilities:
                    prob_token = self._unigram_probabilities[unigram]
                else:
                    prob_token = self._unigram_probabilities[util.UNK]

                prev_token = self._batch_prev_tokens[batch_index]
                prev_prev_token = self._batch_prev_prev_tokens[batch_index]

                if prev_token in self._bigram_probabilities and unigram in self._bigram_probabilities[prev_token]:
                    prob_bigram = self._bigram_probabilities[prev_token][unigram]
                else:
                    prob_bigram = 1 / len(self._unigram_probabilities)

                prev_pair = (prev_prev_token, prev_token)
                if prev_pair in self._trigram_probabilities and unigram in self._trigram_probabilities[prev_pair]:
                    prob_trigram = self._trigram_probabilities[prev_pair][unigram]
                else:
                    prob_trigram = 1 / (len(self._unigram_probabilities) * len(self._unigram_probabilities))

                token_gen_probability = (self._trigram_coefficient * prob_trigram +
                                         self._bigram_coefficient * prob_bigram +
                                         (1 - self._trigram_coefficient - self._bigram_coefficient) * prob_token)
                probabilities[token_index][batch_index] = token_gen_probability

                self._batch_prev_prev_tokens[batch_index] = prev_token
                self._batch_prev_tokens[batch_index] = token

                # Note: not resetting previous tokens when EOS is present, because this is modeling the entire
                # dataset including relationships between sentences.

        return probabilities


def train_count_based_model(train_iter, val_iter, test_iter, text, trigram_coefficient=None, bigram_coefficient=None):
    model = CountBasedLinearInterpolation(train_iter, text, trigram_coefficient, bigram_coefficient, val_iter)

#    model.reset_prev_tokens(train_iter.batch_size)
#    train_perplexity = util.evaluate_perplexity(model, train_iter)
#    print('Train perplexity: %s' % train_perplexity)

    model.reset_contexts(train_iter.batch_size)
    val_perplexity = util.evaluate_perplexity(model, val_iter)
    print('%s\t%s\t%s' % (trigram_coefficient, bigram_coefficient, val_perplexity))

    model.reset_contexts(train_iter.batch_size)
    test_perplexity = util.evaluate_perplexity(model, test_iter)
    print('Test perplexity: %s' % test_perplexity)
