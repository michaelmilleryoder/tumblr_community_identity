"""
Train and test reblog prediction models.
Runs logistic regression, SVM and simple feedforward network with sklearn.
Runs a feedforward network, CNN and more complicated neural nets with PyTorch.

@author Michael Miller Yoder
@date 2021
"""

import os
import pickle
import pdb

import numpy as np
from sklearn import linear_model
from sklearn import neural_network
from sklearn import model_selection
from sklearn import svm

from run_pytorch import RunPyTorch


class Experiment():
    """ Encapsulates training and testing a model in an experiment. """

    def __init__(self, extractor, data, classifier_type, use_cuda=False, epochs=1,
                    debug=False):
        """ Args:
                extractor: FeatureExtractor, for the parameters
                data: Dataset
        """
        self.extractor = extractor
        self.data = data
        self.clf_type = classifier_type
        self.epochs = epochs
        self.debug = debug
        self.use_cuda = use_cuda
        self.model = None
        self.score = None
        self.train_pred = None
        self.test_pred = None
        self.clf = None

    def run(self):
        """ Train the model, evaluate on test set """
        if self.clf_type in ['cnn', 'ffn']:
            exp = RunPyTorch(self.clf_type, self.data, self.extractor, 
                epochs=self.epochs, use_cuda=self.use_cuda, debug=self.debug)
            self.score = exp.run()
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
            'mlp': neural_network.MLPClassifier(hidden_layer_sizes=(32, 50),
                activation='relu', early_stopping=True, verbose=2)
        }
        self.clf = classifier_options[self.clf_type]
        self.model = self.clf.fit(self.data.X_train, self.data.y_train)
        self.score = self.model.score(self.data.X_test, self.data.y_test)
        self.train_pred = self.model.predict(self.data.X_train)
        self.test_pred = self.model.predict(self.data.X_test)

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
