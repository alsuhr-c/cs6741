import torch
import torchtext
import numpy as np


def train_model(model, optimizer, loss_fn, train_iter, val_iter):
    patience = 10
    countdown = patience
    max_accuracy = 0
    epoch_num = 0

    while countdown >= 0:
        model.train()
        for batch in train_iter:
            optimizer.zero_grad()

            logits = model(batch.text).rename(None)
            labels = batch.label

            loss = loss_fn(logits, labels)

            loss.backward()
            optimizer.step()

        val_acc = val_accuracy(model, val_iter)
        best = False
        if val_acc > max_accuracy:
            torch.save(model.state_dict(), 'best_model.pt')

            max_accuracy = val_acc
            patience *= 1.1
            countdown = patience
            best = True
        countdown -= 1

        print('Epoch #%d; countdown = %d; val accuracy = ' % (epoch_num, countdown) +
              '{0:.2f}'.format(100. * val_acc) + ('*' if best else ''))
        epoch_num += 1
    return model


def val_accuracy(model, val_iter):
    model.eval()
    corrects = list()
    for batch in val_iter:
        probs = model(batch.text)
        _, argmax = probs.max('classes')

        corrects.extend(torch.eq(argmax, batch.label).tolist())
    return np.mean(np.array(corrects))


def test_code(model, test):
    upload = []
    corrects = list()

    test_iter = torchtext.data.BucketIterator(test, train=False, batch_size=10)
    for batch in test_iter:
        # Your prediction data here (don't cheat!)
        probs = model(batch.text)
        # here we assume that the name for dimension classes is `classes`
        _, argmax = probs.max('classes')
        upload += argmax.tolist()

        corrects.extend(torch.eq(argmax, batch.label).tolist())

    print('Test accuracy: ' + '{0:.2f}'.format(float(np.mean(np.array(corrects))) * 100))

    with open("predictions.txt", "w") as f:
        f.write("Id,Category\n")
        for i, u in enumerate(upload):
            f.write(str(i) + "," + str(u) + "\n")
