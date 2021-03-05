"""
Train and test reblog prediction models.
Runs logistic regression, SVM and simple feedforward network with sklearn.
Runs CNN and more complicated neural nets with PyTorch.

@author Michael Miller Yoder
@date 2021
"""

import os

import numpy as np
from sklearn import linear_model
from sklearn import neural_network
from sklearn import model_selection
from sklearn import svm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torch_model import TorchModel, DatasetMapper, train_epoch, test_model


class Experiment():
    """ Encapsulates training and testing a model in an experiment. """

    def __init__(self, extractor, data, classifier_type, epochs=1):
        """ Args:
                extractor: FeatureExtractor, for the parameters
                data: Dataset
        """
        self.extractor = extractor
        self.data = data
        self.clf_type = classifier_type
        self.epochs = epochs
        self.model = None
        self.score = None
        self.train_pred = None
        self.test_pred = None
        self.clf = None

    def run(self):
        """ Train the model, evaluate on test set """
        if self.clf_type in ['cnn']:
            self.run_pytorch()
        else:
            self.run_sklearn()

    def run_sklearn(self):
        """ Train and evaluate models from sklearn """

        classifier_options = {
            #'lr': linear_model.LogisticRegressionCV(cv=10, n_jobs=10, max_iter=10000, verbose=0),
            'lr': linear_model.LogisticRegression(n_jobs=10, max_iter=10000,
                verbose=0),
            'svm': model_selection.GridSearchCV(svm.LinearSVC(dual=False,
                max_iter=10000, verbose=0),
                {'C': [.01, .1, 1, 10, 100],
                'penalty': ['l2']}, n_jobs=10, cv=10, verbose=2),
            'ffn': neural_network.MLPClassifier(hidden_layer_sizes=(32, 50),
                activation='relu', early_stopping=True, verbose=2)
        }
        self.clf = classifier_options[self.clf_type]
        self.model = self.clf.fit(self.data.X_train, self.data.y_train)
        self.score = self.model.score(self.data.X_test, self.data.y_test)
        self.train_pred = self.model.predict(self.data.X_train)
        self.test_pred = self.model.predict(self.data.X_test)

    def run_pytorch(self):
        """ Train and evaluate models from pytorch """
        use_cuda = True
        self.model = TorchModel(self.clf_type, self.extractor, epochs=self.epochs,
             use_cuda=use_cuda)

        #subset = 100 # for debugging
        #train = DatasetMapper(self.data.X_train[:subset], self.data.y_train[:subset])
        train = DatasetMapper(self.data.X_train, self.data.y_train)
        dev = DatasetMapper(self.data.X_dev, self.data.y_dev)
        test = DatasetMapper(self.data.X_test, self.data.y_test)

        # Initialize loaders
        loader_train = DataLoader(train, batch_size=self.model.clf.batch_size)
        loader_dev = DataLoader(dev, batch_size=self.model.clf.batch_size)
        loader_test = DataLoader(test, batch_size=self.model.clf.batch_size)

        # Define optimizer
        optimizer = optim.RMSprop(self.model.clf.parameters(),
            lr=self.model.clf.learning_rate)

        # Starts training phase
        for epoch in range(self.model.epochs):
            train_epoch(epoch, self.model.clf, loader_train, loader_dev, optimizer,
                self.data.y_train, self.data.y_dev)

        # Test
        self.score = test_model(self.model.clf, loader_test, self.data.y_test)

        # Save model
        outpath = os.path.join('../models/', self.clf_type + self.model.clf.name \
             + '.model')
        torch.save(self.model.clf.state_dict(), outpath)
        print(f"Model saved to {outpath}")

    def save_output(self, output_dirpath):
        """ Save score and predictions """
        if not os.path.exists(output_dirpath):
            os.mkdir(output_dirpath)

        # Save score
        with open(os.path.join(output_dirpath, 'score.txt'), 'w') as f:
            f.write(str(self.score))

        # Save predictions
        np.savetxt(os.path.join(output_dirpath, 'train_preds.txt'), self.train_pred)
        np.savetxt(os.path.join(output_dirpath, 'test_preds.txt'), self.test_pred)
        print(f"\tSaved score and predictions to {output_dirpath}")
