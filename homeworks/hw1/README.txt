The main file is hw1.py. To run it:

python hw1.py

This script will output validation accuracy and the countdown after each epoch, then the final test performance.

To change which model is trained and evaluated uncomment one of the lines 25 -- 29 in hw1.py.

Files
- hw1.py -- main file
- utils.py -- contains wrapper for training a model, computing validation accuracy, and test_code, which computes test results / writes test results to disk.

Model implementations
- Each model file contains a class (nn.Module) containing model parameters and a forward function.
- Each model file contains a function called run_MODEL_TYPE which will initialize the model, train it, and test it using the test_code function.
- Model files: nb (Naive Bayes), cbow (continuous bag-of-words), lstm (RNN LSTM), cnn (convolutional neural network), logistic_regression.
