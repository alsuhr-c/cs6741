import torchtext
from torchtext.vocab import Vectors

from logistic_regression import run_logistic_regression
from cbow import run_cbow
from lstm import run_lstm
from nb import run_nb
from cnn import run_cnn

PAD_TOK = '<pad>'

TEXT = torchtext.data.Field()
LABEL = torchtext.data.Field(sequential=False, unk_token=None)

train, val, test = torchtext.datasets.SST.splits(TEXT, LABEL, filter_pred=lambda ex: ex.label != 'neutral')

TEXT.build_vocab(train)
LABEL.build_vocab(train)

train_iter, val_iter = torchtext.data.BucketIterator.splits((train, val), batch_size=10)

url = 'https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/wiki.simple.vec'
TEXT.vocab.load_vectors(vectors=Vectors('wiki.simple.vec', url=url))

#run_nb(train_iter, test, TEXT.vocab)
#run_logistic_regression(train_iter, val_iter, test, TEXT.vocab)
#run_cbow(train_iter, val_iter, test, TEXT.vocab)
#run_cnn(train_iter, val_iter, test, TEXT.vocab)
run_lstm(train_iter, val_iter, test, TEXT.vocab)

