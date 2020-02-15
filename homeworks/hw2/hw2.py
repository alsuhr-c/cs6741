import torch
import torchtext
import sys
from torchtext.data.iterator import BPTTIterator
from count_based import train_count_based_model
from feedforward import train_feedforward_language_model
from lstm import train_lstm_model, evaluate_lstm_model
from torchtext.vocab import Vectors
import analysis

TEXT = torchtext.data.Field()

train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
#    path='/home/alsuhr/Documents/cs6741/homeworks/hw2/ptb/',
    path='/Users/alsuhr/Documents/Cornell/cs6741/ptb/',
    train='train.txt', validation='valid.txt', test='test.txt',
    text_field=TEXT)

if '--debug' in sys.argv:
    print('Debug is ON')
    TEXT.build_vocab(train, max_size=1000)
else:
    TEXT.build_vocab(train)

url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

print('Loaded %s training examples' % len(train))
print('Loaded %s val examples' % len(val))
print('Loaded %s testing examples' % len(test))
print('Vocab size is %s' % len(TEXT.vocab))

batch_size = 512
bptt_length = 32

if '--em' in sys.argv:
    batch_size = 1
    bptt_length = 1

train_iter, val_iter, test_iter = BPTTIterator.splits((train, val, test), batch_size=batch_size,
#                                                      device=torch.device('cuda'),
                                                      bptt_len=bptt_length, repeat=False)

if '--count_based_model' in sys.argv:
    if '--em' in sys.argv:
        train_count_based_model(train_iter, val_iter, test_iter, TEXT)
    else:
        train_count_based_model(train_iter, val_iter, test_iter, TEXT, 0.3, 0.7)
elif '--feedforward_model' in sys.argv:
    train_feedforward_language_model(train_iter, val_iter, test_iter, TEXT, 8, 512)
elif '--lstm_model' in sys.argv:
    if '--eval_model' in sys.argv:
        print('Evaluating model.')
        evaluate_lstm_model(train_iter, val_iter, test_iter, TEXT, 512, 2)
    else:
        train_lstm_model(train_iter, val_iter, test_iter, TEXT, 512, 2, 0.5)
elif '--analysis' in sys.argv:
    analysis.evaluate_models(TEXT, train_iter, val_iter)
