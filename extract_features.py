"""
Extract features for reblog prediction models.

@author Michael Miller Yoder
@date 2021

Feature extraction
    * baseline features from post content: hashtags, post like count, post media type
    * identity features: configurations of matches and mismatches from
        self-presented identity labels between users who may or may not
        reblog each others' posts
"""
from collections import defaultdict
import pdb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm

import utils


class FeatureExtractor():
    """ Extract features """

    def __init__(self, feature_str, user_embs=None, word_embs=None):
        """ Args:
                features: comma-separated list of feature sets to be included
        """
        features = feature_str.split(',')
        if 'post' in features:
            self.post_features = True
        else:
            self.post_features = False
        if 'text' in features:
            self.user_features = True
        else:
            self.user_features = False
        self.user_embs = user_embs
        self.word_embs = word_embs

    def extract(self, dataset):
        """ Takes a Dataset and extracts features.
            Returns a Dataset with extracted features in
                    dataset.X_train, dataset.y_train,
                    dataset.X_test, dataset.y_test
            Args:
                data: Dataset
        """

        data = dataset.data

        # Baseline features
        # TODO: Make this work not just for binary classification (data.organization)
        # TODO: make separate functions
        feature_parts = {}
        if self.post_features:
            # Post hashtag features
            feature_parts['post_tags_emb'] = np.array([
                self.word_embeddings(utils.string_list2str(tags)) for tags in tqdm(data['post_tags'], ncols=70)])

            # Post notes
            feature_parts['post_note_count'] = data['post_note_count'].fillna(0).values

            # Post type
            # Convert types to ints
            type2id = defaultdict(lambda: len(type2id))
            feature_parts['post_type'] = np.array([type2id[val] for val in data['post_type']])

            post_features = np.hstack([
                feature_parts['post_tags_emb'],
                feature_parts['post_note_count'].reshape(-1,1),
                feature_parts['post_type'].reshape(-1,1)
            ])

        # Build user embeddings
        if self.user_features:
            for user_type in ['follower', 'followee']:
                tqdm.write(f'\t{user_type}')
                feature_parts[f'{user_type}_embedding'] = [self.word_embeddings(desc) for desc in tqdm(data[f'processed_blog_description_{user_type}'], ncols=70)]
            user_embeddings = np.hstack([feature_parts['follower_embedding'], feature_parts['followee_embedding']])

        features = np.empty((len(data), 1))
        if self.post_features:
            features = post_features
        if self.user_features:
            features = np.hstack([features, user_embeddings])

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(features, data['reblog'].values, test_size=.1, random_state=9)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        dataset.set_folds(X_train, X_test, y_train, y_test)
        print(f"\tTotal dataset shape (#instances, #features): {X_train.shape[0] + X_test.shape[0], X_train.shape[1]}")
        print(f"\tTraining set shape: {X_train.shape}")
        print(f"\tTest set shape: {X_test.shape}")
        return dataset

    def word_embeddings(self, text):
        """ Returns an embedding for a given text, which has
            space-separated tokens. """
        return_arr = np.zeros(self.word_embs.vector_size)
        if not isinstance(text, float) and text != '' :
            tokens = text.split()
            embeddings = []
            for word in tokens:
                if word in self.word_embs.wv:
                    embeddings.append(self.word_embs.wv[word])
            if len(embeddings) > 0:
                return_arr = np.mean(embeddings, axis=0)
        return return_arr
