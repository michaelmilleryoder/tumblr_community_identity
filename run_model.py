"""
Train and test reblog prediction models.

@author Michael Miller Yoder
@date 2021
"""
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import neural_network
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm


class Experimenter():
    """ Encapsulates training and testing a model in an experiment. """

    def __init__(self, data, classifier_type):
        self.classifier_options = classifier_settings
        self.clf = self.classifier_options[classifier_type]
        self.X_train = data.X_train
        self.X_test = data.X_test
        self.y_train = data.y_train
        self.y_test = data.y_test

    def classifier_settings(self):
        # Classifier definitions
        classifiers = {
            'lr': linear_model.LogisticRegressionCV(cv=10, n_jobs=10, max_iter=10000, verbose=0),
            'svm': model_selection.GridSearchCV(svm.LinearSVC(dual=False, max_iter=10000, verbose=0), {'C': [.01, .1, 1, 10, 100], 'penalty': ['l2']}, n_jobs=10, cv=10, verbose=2),
            'ffn': neural_network.MLPClassifier(hidden_layer_sizes=(32, 50), activation='relu', early_stopping=True, verbose=2)
        }
        return classifiers

    def run(self):
        self.model = self.clf.fit(self.X_train, self.y_train)
        self.score = model.score(self.X_test, self.y_test)
        self.train_pred = model.predict(self.X_train)
        self.model_pred = model.predict(self.X_test)
