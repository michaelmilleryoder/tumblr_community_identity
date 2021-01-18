"""
Extract features for reblog prediction models.

@author Michael Miller Yoder
@date 2021
"""
import pdb

from sklearn.model_selection import train_test_split
import numpy as np


class FeatureExtractor():
    """ Extract features """

    def __init__(self, feature_type, user_embs=None, word_embs=None):
        self.feature_type = feature_type
        self.user_embs = user_embs
        self.word_embs = word_embs

    def extract(self, data):
        """ Takes pandas DataFrame and extracts features.
            Saves extracted features to self.X_train, self.y_train,
                    self.X_test, self.y_test
            Args:
                data: Dataset()
        """
        # Get embeddings
        for user_type in ['follower', 'followee']:
            data[f'{user_type}_embedding'] = data[f'blog_description_{user_type}'].map(self.word_embeddings)
        data['embedding'] = [np.concat([follower, followee]) for follower, followee in zip(data['follower_embedding', 'followee_embedding'])]

        # Split into train and test sets
        data.set_folds(
            train_test_split(data.loc[:, ['embedding', 'reblog']])
        )
        pdb.set_trace()
        return data

    def word_embeddings(self, text):
        """ Returns an embedding for a given text. """
        tokens = text.split()
        embeddings = []
        for word in tokens:
            if word in self.word_embs.w2v:
                embeddings.append(self.word_embs.w2v[word])
        return np.mean(embeddings, axis=0)
