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


def rank_feature_transform(reblog_feats_list, nonreblog_feats_list, labels, 
        combo='subtract'):
    """ Transform features to come up with comparison features for
        learning-to-rank formulation.
        Args:
            combo: How to combine features. Options {subract, concat}
    """
    comparison_feats = []
    for reblog_feats, nonreblog_feats, label in zip(
        reblog_feats_list, nonreblog_feats_list, labels):
        if label == 0:
            if combo == 'subtract':
                comparison_feats.append(reblog_feats - nonreblog_feats)
            elif combo == 'concat':
                comparison_feats.append(np.hstack([
                    reblog_feats, nonreblog_feats]))
        else:
            if combo == 'subtract':
                comparison_feats.append(nonreblog_feats - reblog_feats)
            elif combo == 'concat':
                comparison_feats.append(np.hstack([
                    nonreblog_feats, reblog_feats]))
    return np.array(comparison_feats)


class FeatureExtractor():
    """ Extract features """

    def __init__(self, feature_str, word_embs=None, graph_embs=None, sent_embs=None,
            word_inds=False, padding_size=-1):
        """ Args:
                features: comma-separated list of feature sets to be included
                word_embs: loaded word vectors
                graph_embs: loaded graph embeddings for users
                sent_embs: loaded sentence embeddings for user blog descriptions.
                    If this is not None, then will load blog description text
                    embeddings from this instead of word_embs
                word_inds: True if output for text features should be returned as
                    indices (for PyTorch) in the word_embs vocab + 1 to allow 0
                    index for padding.
                    If False, then will be converted to embeddings.
                padding_size: Size of padding for variable-length text features,
                    to be used when word_inds is True.
                    If -1 (default), don't do padding.
        """
        features = feature_str.split(',')
        self.post_features, self.user_features = False, False
        self.post_nontext_only = False
        self.text_features, self.graph_features = False, False
        if 'post' in features or 'post_nontext' in features:
            self.post_features = True
            if 'post_nontext' in features: # no hashtags
                self.post_nontext_only = True
        if 'text' in features:
            self.user_features = True
            self.text_features = True
        if 'graph' in features:
            self.user_features = True
            self.graph_features = True
        self.word_embs = word_embs
        self.graph_embs = graph_embs
        self.sent_embs = sent_embs
        self.word_inds = word_inds
        self.padding_size = padding_size
        self.vocab = None
        self.nontext_inds = None # nontext feature vector indices, for PyTorch
        self.reblog_inds = None # reblog feature vector indices, for PyTorch
        self.text_inds = {} # text blog desc feature vector indices, for PyTorch
        self.graph_inds = {} # graph blog desc feature vector indices, for PyTorch

    def extract(self, dataset, dev=False):
        """ Takes a Dataset and extracts features.
            Returns a Dataset with extracted features in
                    dataset.X_train, dataset.y_train,
                    dataset.X_test, dataset.y_test
            Args:
                dataset: Dataset
                dev: Whether to split to include a dev set.
                    If False, will just split into training
                    and test.
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
                # Adjust indices
                if self.text_features:
                    self.text_inds = {key: [el+features.shape[1] for el in val] \
                        for key,val in self.text_inds.items()}
                if self.graph_features:
                    self.graph_inds = {key: [el+features.shape[1] for el in val] \
                        for key,val in self.graph_inds.items()}
                features = np.hstack([features, user_features])

        # Labels to predict
        if dataset.organization == 'learning-to-rank':
            y = data['label'].values
        elif dataset.organization == 'binary_classification':
            y = data['reblog'].values

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            features, y, test_size=.1, random_state=9)
        if dev: # PyTorch
            X_train, X_dev, y_train, y_dev = train_test_split(
                X_train, y_train, test_size=len(y_test), random_state=9)
            dataset.set_folds(X_train, X_test, y_train, y_test,
                X_dev=X_dev, y_dev=y_dev)
            dataset.scale_nontext_features(self.nontext_inds)

        else: # sklearn
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
            dataset.set_folds(X_train, X_test, y_train, y_test)
        print("\tTotal dataset shape (#instances, #features): "
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

            if self.word_inds: # PyTorch
                combo = 'concat'
            else: # sklearn
                combo = 'subtract'

            # TODO: Make these separate functions since they are long
            # Post tags
            if not self.post_nontext_only:
                feature_opts = {} # reblog and nonreblog
                for reblog_type in ['reblog', 'nonreblog']:
                    if self.word_inds: # for PyTorch
                        fn = self.get_word_inds
                        self.build_vocab()
                    else:
                        fn = self.word_embeddings
                    feature_opts[reblog_type] = np.array([fn(
                            utils.string_list2str(tags)) for tags in tqdm(
                            data[f'post_tags_{reblog_type}'], ncols=70)])
                feature_parts['post_tags_emb'] = rank_feature_transform(
                    feature_opts['reblog'], feature_opts['nonreblog'], data.label,
                    combo=combo)

            # Post notes
            feature_opts = {} # reblog and nonreblog
            for reblog_type in ['reblog', 'nonreblog']:
                feature_opts[reblog_type] = data[
                    f'post_note_count_{reblog_type}'].fillna(0).values
            feature_parts['post_note_count'] = rank_feature_transform(
                feature_opts['reblog'], feature_opts['nonreblog'], data.label, 
                combo=combo)

            # Post type
            # Convert types to ints
            feature_opts = {} # reblog and nonreblog
            for reblog_type in ['reblog', 'nonreblog']:
                feature_opts[reblog_type] = pd.get_dummies(
                    data[f'post_type_{reblog_type}']).values
            feature_parts['post_type'] = rank_feature_transform(
                feature_opts['reblog'], feature_opts['nonreblog'], data.label,
                combo=combo)

            if not self.word_inds: # sklearn
                feature_parts['post_note_count'] = feature_parts[
                    'post_note_count'].reshape(-1,1)

            post_features = np.hstack(list(feature_parts.values()))

            # Pass on which indices of feature vectors aren't text
            # and which are reblog (for PyTorch)
            if self.word_inds:
                if self.post_nontext_only: # assuming don't have text and graph
                    self.nontext_inds = range(post_features.shape[1])
                else:
                    self.nontext_inds = range(feature_parts[
                        'post_tags_emb'].shape[1], post_features.shape[1])
                offset = 0
                self.reblog_inds = []
                for feats in feature_parts.values():
                    # Add in first half of segments (reblog/nonreblog concat)
                    self.reblog_inds += range(offset, offset + int(feats.shape[1]/2))
                    offset += feats.shape[1]
                post_features = post_features.astype(int)

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
                if self.word_inds:
                    fn = self.get_word_inds
                    if self.vocab is None:
                        self.build_vocab()
                else:
                    fn = self.word_embeddings
                parts[user_type] = np.array([fn(desc) for desc in \
                    tqdm(data[f'processed_blog_description_{user_type}'], ncols=70)])
        for reblog_type in ['reblog', 'nonreblog']:
            parts[reblog_type] = np.hstack([
                parts['follower'],
                parts[f'followee_{reblog_type}']])
            #parts[reblog_type] = parts['follower'] - parts[f'followee_{reblog_type}']
            # ^ for sklearn
        if self.word_inds: # PyTorch
            combo = 'concat'
        else: # sklearn
            combo = 'subtract'
        text_embeddings = rank_feature_transform(
            parts['reblog'], parts['nonreblog'], data.label, combo=combo)

        # Pass on which indices of feature vectors correspond to which follower
        # (for PyTorch)
        if self.word_inds:
            midpt = int(text_embeddings.shape[1]/2)
            self.text_inds['reblog'] = range(0, midpt)
            self.text_inds['nonreblog'] = range(midpt, text_embeddings.shape[1])
            text_embeddings = text_embeddings.astype(int)

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
            #parts[reblog_type] = parts['follower'] - parts[f'followee_{reblog_type}']
        if self.word_inds: # PyTorch
            combo = 'concat'
        else: # sklearn
            combo = 'subtract'
        graph_embeddings = rank_feature_transform(
            parts['reblog'], parts['nonreblog'], data.label, combo=combo)

        # Pass on which indices of feature vectors correspond to which follower
        # (for PyTorch)
        if self.word_inds:
            midpt = int(graph_embeddings.shape[1]/2)
            self.graph_inds['reblog'] = range(0, midpt)
            self.graph_inds['nonreblog'] = range(midpt, graph_embeddings.shape[1])
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
            graph_embeddings = self.graph_embeddings_bin(data)
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
                self.graph_inds = {key: [el+user_embeddings.shape[1] for el in val] \
                    for key,val in self.graph_inds.items()}
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

    def get_word_inds(self, text):
        """ Returns a list of word indices in the word_embs vocab + 1 (for padding).
            Padded with 0s to self.padding_size.
        """
        if text is None or len(text) == 0:
            return self.pad([])
        return self.pad([(self.vocab[w]) for w in text.split() if w in self.vocab])

    def pad(self, inds):
        """ Pad input with 0s """
        while len(inds) < self.padding_size:
            inds.insert(len(inds), 0)
        return inds[:self.padding_size]

    def build_vocab(self):
        """ Build vocab, save to self.vocab """
        self.vocab = dict()
        for w, vec in self.word_embs.wv.vocab.items():
            self.vocab[w] = vec.index + 1 # add one for padding
