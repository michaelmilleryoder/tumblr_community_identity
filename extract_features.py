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
import pandas as pd
from tqdm import tqdm

import utils


def rank_feature_transform(reblog_feats_list, nonreblog_feats_list, labels):
    """ Transform features to come up with comparison features for
        learning-to-rank formulation """
    comparison_feats = []
    for reblog_feats, nonreblog_feats, label in zip(
        reblog_feats_list, nonreblog_feats_list, labels):
        if label == 0:
            comparison_feats.append(reblog_feats - nonreblog_feats)
        else:
            comparison_feats.append(nonreblog_feats - reblog_feats)
    return np.array(comparison_feats)


class FeatureExtractor():
    """ Extract features """

    def __init__(self, feature_str, word_embs=None, graph_embs=None, sent_embs=None):
        """ Args:
                features: comma-separated list of feature sets to be included
                word_embs: loaded word vectors
                graph_embs: loaded graph embeddings for users
                sent_embs: loaded sentence embeddings for user blog descriptions.
                    If this is not None, then will load blog description text
                    embeddings from this instead of word_embs
        """
        features = feature_str.split(',')
        self.post_features, self.user_features = False, False
        self.text_features, self.graph_features = False, False
        if 'post' in features:
            self.post_features = True
        if 'text' in features:
            self.user_features = True
            self.text_features = True
        if 'graph' in features:
            self.user_features = True
            self.graph_features = True
        self.word_embs = word_embs
        self.graph_embs = graph_embs
        self.sent_embs = sent_embs

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
        features = None
        if self.post_features:
            # Post hashtag features
            post_features = self.extract_post_features(data, dataset.organization)
            features = post_features

        # Build user embeddings
        if self.user_features:
            user_features = self.extract_user_features(data, dataset.organization)
            if features is None:
                features = user_features
            else:
                features = np.hstack([features, user_features])

        # Labels to predict
        if dataset.organization == 'learning-to-rank':
            y = data['label'].values
        elif dataset.organization == 'binary_classification':
            y = data['reblog'].values

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=.1, random_state=9)
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
        dataset.set_folds(X_train, X_test, y_train, y_test)
        print("\tTotal dataset shape (#instances, #features):"
            f"{X_train.shape[0] + X_test.shape[0], X_train.shape[1]}")
        print(f"\tTraining set shape: {X_train.shape}")
        print(f"\tTest set shape: {X_test.shape}")
        return dataset

    def extract_post_features(self, data, organization):
        """ Extract post (baseline) features
            Args:
                data: pandas DataFrame with columns to extract data
                organization: for which task the data is organized
                    {learning-to-rank, binary_classification}
        """
        feature_parts = {}
        if organization == 'learning-to-rank':
            # Post tags
            feature_opts = {} # reblog and nonreblog
            for reblog_type in ['reblog', 'nonreblog']:
                feature_opts[reblog_type] = np.array([self.word_embeddings(
                        utils.string_list2str(tags)) for tags in tqdm(
                        data[f'post_tags_{reblog_type}'], ncols=70)])
            feature_parts['post_tags_emb'] = rank_feature_transform(
                feature_opts['reblog'], feature_opts['nonreblog'], data.label)

            # Post notes
            feature_opts = {} # reblog and nonreblog
            for reblog_type in ['reblog', 'nonreblog']:
                feature_opts[reblog_type] = data[
                    f'post_note_count_{reblog_type}'].fillna(0).values
            feature_parts['post_note_count'] = rank_feature_transform(
                feature_opts['reblog'], feature_opts['nonreblog'], data.label)

            # Post type
            # Convert types to ints
            feature_opts = {} # reblog and nonreblog
            for reblog_type in ['reblog', 'nonreblog']:
                feature_opts[reblog_type] = pd.get_dummies(
                    data[f'post_type_{reblog_type}']).values
            feature_parts['post_type'] = rank_feature_transform(
                feature_opts['reblog'], feature_opts['nonreblog'], data.label)

            post_features = np.hstack([
                feature_parts['post_tags_emb'],
                feature_parts['post_note_count'].reshape(-1,1),
                feature_parts['post_type']
            ])

        elif organization == 'binary_classification':
            # Post tags
            feature_parts['post_tags_emb'] = np.array([
                self.word_embeddings(utils.string_list2str(tags)) for tags in tqdm(
                    data['post_tags'], ncols=70)])

            # Post notes
            feature_parts['post_note_count'] = data['post_note_count'].fillna(0).values

            # Post type
            # Convert types to ints
            type2id = defaultdict(lambda: len(type2id))
                # this should be categorical, not numeric
            feature_parts['post_type'] = np.array(
                [type2id[val] for val in data['post_type']])
            post_features = np.hstack([
                feature_parts['post_tags_emb'],
                feature_parts['post_note_count'].reshape(-1,1),
                feature_parts['post_type'].reshape(-1,1)
            ])
        return post_features

    def text_embeddings_ltr(self, data):
        """ Extract embedding from a text blog description,
            expecting a learning-to-rank framework
        """
        parts = {} # to assemble in the end
        for user_type in ['follower', 'followee_reblog', 'followee_nonreblog']:
            tqdm.write(f'\t{user_type} text embeddings')
            if self.sent_embs is not None:
                parts[user_type] = np.array([self.sent_embeddings(uid) for uid in \
                    tqdm(data[f'tumblog_id_{user_type}'], ncols=70)])
            else:
                parts[user_type] = np.array([self.word_embeddings(desc) for desc in \
                    tqdm(data[f'processed_blog_description_{user_type}'], ncols=70)])
        for reblog_type in ['reblog', 'nonreblog']:
            parts[reblog_type] = np.hstack([
                parts['follower'],
                parts[f'followee_{reblog_type}']])
        text_embeddings = rank_feature_transform(
            parts['reblog'], parts['nonreblog'], data.label)
        return text_embeddings

    def graph_embeddings_ltr(self, data):
        """ Extract embedding from users' graph embeddings,
            expecting a learning-to-rank framework
        """
        parts = {}
        for user_type in ['follower', 'followee_reblog', 'followee_nonreblog']:
            parts[user_type] = []
            for tumblog_id in data[f'tumblog_id_{user_type}']:
                if str(tumblog_id) in self.graph_embs:
                    parts[user_type].append(
                        self.graph_embs[str(tumblog_id)])
                else:
                    parts[user_type].append(
                        np.zeros(self.graph_embs.vector_size))
            parts[user_type] = np.array(parts[user_type])
        for reblog_type in ['reblog', 'nonreblog']:
            parts[reblog_type] = np.hstack([
                parts['follower'],
                parts[f'followee_{reblog_type}']])
        graph_embeddings = rank_feature_transform(
            parts['reblog'], parts['nonreblog'], data.label)
        return graph_embeddings

    def text_embeddings_bin(self, data):
        """ Extract embedding from a text blog description,
            expecting a binary classificaion framework
        """
        feature_parts = {}
        for user_type in ['follower', 'followee']:
            tqdm.write(f'\t{user_type} text embeddings')
            feature_parts[f'{user_type}_text_embedding'] = np.array([
                self.word_embeddings(desc) for desc in tqdm(
                    data[f'processed_blog_description_{user_type}'], ncols=70)])
        text_embeddings = np.hstack([
            feature_parts['follower_text_embedding'],
            feature_parts['followee_text_embedding']])
        return text_embeddings

    def graph_embeddings_bin(self, data):
        """ Extract embedding from users' graph embeddings,
            expecting a learning-to-rank framework
        """
        feature_parts = {}
        for user_type in ['follower', 'followee']:
            feature_parts[f'{user_type}_graph_emb'] = []
            for tumblog_id in data[f'tumblog_id_{user_type}']:
                if str(tumblog_id) in self.graph_embs:
                    feature_parts[f'{user_type}_graph_emb'].append(
                        self.graph_embs[str(tumblog_id)])
                else:
                    feature_parts[f'{user_type}_graph_emb'].append(
                        np.zeros(self.graph_embs.vector_size))
            feature_parts[f'{user_type}_graph_emb'] = np.array(
                feature_parts[f'{user_type}_graph_emb'])
        graph_embeddings = np.hstack([
            feature_parts['follower_graph_emb'],
            feature_parts['followee_graph_emb']])
        return graph_embeddings

    def text_embeddings(self, data, organization):
        """ Extract embedding from a text blog description,
        """
        text_embeddings = None
        if organization == 'learning-to-rank':
            text_embeddings = self.text_embeddings_ltr(data)
        elif organization == 'binary_classification':
            text_embeddings = self.text_embeddings_bin(data)
        return text_embeddings

    def graph_embeddings(self, data, organization):
        """ Extract embedding from user graph
        """
        graph_embeddings = None
        if organization == 'learning-to-rank':
            graph_embeddings = self.graph_embeddings_ltr(data)
        elif organization == 'binary_classification':
            graph_embeddings = self.text_embeddings_bin(data)
        return graph_embeddings

    def extract_user_features(self, data, organization):
        """ Extract user (identity) features
            Args:
                data: pandas DataFrame with columns to extract data
        """
        user_embeddings = None
        if self.text_features:
            text_embeddings = self.text_embeddings(data, organization)
            user_embeddings = text_embeddings

        if self.graph_features:
            graph_embeddings = self.graph_embeddings(data, organization)
            if user_embeddings is None:
                user_embeddings = graph_embeddings
            else:
                user_embeddings = np.hstack([user_embeddings, graph_embeddings])
        return user_embeddings

    def sent_embeddings(self, tumblog_id):
        """ Looks up loaded blog description embedding for a given tumblog_id
        """
        if tumblog_id in self.sent_embs:
            return_arr = self.sent_embs[tumblog_id]
        else:
            ndims = len(self.sent_embs[list(self.sent_embs.keys())[0]])
            return_arr = np.zeros(ndims)
        return return_arr

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
