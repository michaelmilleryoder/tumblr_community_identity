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
from sklearn.feature_selection import SequentialFeatureSelector

from run_pytorch import RunPyTorch


class Experiment():
    """ Encapsulates training and testing a model in an experiment. """

    def __init__(self, extractor, data, classifier_type, use_cuda=False, epochs=1, sfs_k=-1,
                    debug=False):
        """ Args:
                extractor: FeatureExtractor, for the parameters
                data: Dataset
        """
        self.extractor = extractor
        self.data = data
        self.clf_type = classifier_type
        self.epochs = epochs
        self.sfs_k = sfs_k
        self.debug = debug
        self.use_cuda = use_cuda
        self.model = None
        self.dev_score = None
        self.test_score = None
        self.train_pred = None
        self.dev_pred = None
        self.test_pred = None
        self.clf = None

    def run(self):
        """ Train the model, evaluate on test set """
        if self.clf_type in ['cnn', 'ffn']:
            exp = RunPyTorch(self.clf_type, self.data, self.extractor, 
                epochs=self.epochs, use_cuda=self.use_cuda, debug=self.debug)
            self.test_score = exp.run()
        else:
            self.run_sklearn()

    def run_sklearn(self):
        """ Train and evaluate models from sklearn """

        classifier_options = {
            #'lr': linear_model.LogisticRegressionCV(cv=10, n_jobs=10, max_iter=10000, verbose=0),
            'lr': linear_model.LogisticRegression(n_jobs=10, max_iter=100000,
                verbose=0),
            'svm': model_selection.GridSearchCV(svm.LinearSVC(dual=False,
                max_iter=10000, verbose=0),
                {'C': [.01, .1, 1, 10, 100],
                'penalty': ['l2']}, n_jobs=10, cv=10, verbose=2),
            'mlp': neural_network.MLPClassifier(hidden_layer_sizes=(32, 50),
                activation='relu', early_stopping=True, verbose=2)
        }
        self.clf = classifier_options[self.clf_type]
        if self.sfs_k > 0:
            # Forward feature selection
            print("Doing forward feature selection...")
            sfs = SequentialFeatureSelector(self.clf, n_features_to_select=self.sfs_k, n_jobs=-1)
            sfs.fit(self.data.X_train, self.data.y_train)
            # Save out selected features
            outpath = os.path.join('/projects/tumblr_community_identity/tmp/', 
                f'sfs{self.sfs_k}_{self.extractor.select_k}.txt')
            np.savetxt(outpath, sfs.get_support())
            sfs_mask = sfs.get_support()
            #sfs_mask = np.loadtxt(
            #    '/projects/tumblr_community_identity/tmp/sfs20_500.txt').astype(bool)
            X_train = self.data.X_train[:,sfs_mask] 
            X_dev = self.data.X_dev[:, sfs_mask]
            X_test = self.data.X_test[:,sfs_mask]
            self.data.X_train, self.data.X_dev, self.data.X_test = X_train, X_dev, X_test

        self.model = self.clf.fit(self.data.X_train, self.data.y_train)
        self.test_score = self.model.score(self.data.X_test, self.data.y_test)
        self.train_pred = self.model.predict(self.data.X_train)
        if self.data.X_dev is not None:
            self.dev_score = self.model.score(self.data.X_dev, self.data.y_dev)
            self.dev_pred = self.model.predict(self.data.X_dev)
        self.test_pred = self.model.predict(self.data.X_test)

    def save_output(self, output_dirpath):
        """ Save score and predictions """
        if not os.path.exists(output_dirpath):
            os.mkdir(output_dirpath)

        # Save score
        with open(os.path.join(output_dirpath, 'scores.txt'), 'w') as f:
            f.write(f'test_score:\t{self.test_score}')
            if self.dev_score:
                f.write(f'dev_score:\t{self.dev_score}')

        # Save predictions
        np.savetxt(os.path.join(output_dirpath, 'train_preds.txt'), self.train_pred)
        if self.dev_pred is not None:
            np.savetxt(os.path.join(output_dirpath, 'dev_preds.txt'), self.dev_pred)
        np.savetxt(os.path.join(output_dirpath, 'test_preds.txt'), self.test_pred)
        print(f"\tSaved score and predictions to {output_dirpath}")

    def save_model(self, output_fpath):
        """ Saves sklearn model """
        with open(output_fpath, 'wb') as f:
            pickle.dump(self.clf, f) 
        print(f"Saved model to {output_fpath}")
