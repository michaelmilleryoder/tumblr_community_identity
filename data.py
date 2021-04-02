"""
Load and filter data for reblog prediction experiments.

@author Michael Miller Yoder
@date 2021
"""
import os
import pdb

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class Dataset():
    """ Encapsulates data and train/test splits,
        as well as load, filter, and scaling methods
    """

    def __init__(self):
        self.data = None
        self.data_location = None
        self.X_train = None
        self.X_dev = None
        self.X_test = None
        self.y_train = None
        self.y_dev = None
        self.y_test = None
        self.organization = None
            # how the data is organized: {learning-to-rank, binary_classification}
        self.filter_settings = None

    def set_folds(self, X_train, X_test, y_train, y_test, X_dev=None, y_dev=None):
        """ Set training and testing folds """
        self.X_train = X_train
        self.X_dev = X_dev
        self.X_test = X_test
        self.y_train = y_train
        self.y_dev = y_dev
        self.y_test = y_test

    def load(self, data_location, organization):
        """ Load data.
            Saves a pandas DataFrame with all relevant columns,
            organized by the appropriate rows, to self.data
            Args:
                data_location: path to the CSV to load
                organization: how to organize each instance
                    {learning-to-rank, binary_classification}
        """
        self.data_location = data_location
        data = pd.read_csv(self.data_location)
        self.organization = organization
        if self.organization == 'learning-to-rank':
            selected_cols = [
                'post_tags_reblog_str',
                'post_tags_nonreblog_str',
                'post_note_count_reblog',
                'post_note_count_nonreblog',
                'post_type_reblog',
                'post_type_nonreblog',
                'tumblog_id_follower_reblog',
                'tumblog_id_followee_reblog',
                'tumblog_id_followee_nonreblog',
                'processed_tumblr_blog_description_follower_reblog',
                'processed_tumblr_blog_description_followee_reblog',
                'processed_tumblr_blog_description_followee_nonreblog',
            ]
            rename_cols = {
                'post_tags_reblog_str': 'post_tags_reblog',
                'post_tags_nonreblog_str': 'post_tags_nonreblog',
                'tumblog_id_follower_reblog': 'tumblog_id_follower',
                'processed_tumblr_blog_description_follower_reblog':
                    'processed_blog_description_follower',
                'processed_tumblr_blog_description_followee_reblog':
                    'processed_blog_description_followee_reblog',
                'processed_tumblr_blog_description_followee_nonreblog':
                    'processed_blog_description_followee_nonreblog'
            }
            self.data = data.loc[:, selected_cols]
            self.data.rename(columns=rename_cols, inplace=True)
            self.add_random_labels()
        elif self.organization == 'binary_classification':
            data_section = {}
            selected_cols = [
                'post_tags_{}_str',
                'post_note_count_{}',
                'post_type_{}',
                'tumblog_id_follower_reblog',
                'tumblog_id_followee_{}',
                'processed_tumblr_blog_description_follower_reblog',
                'processed_tumblr_blog_description_followee_{}'
            ]
            rename_cols = {
                'post_tags_{}_str': 'post_tags',
                'post_note_count_{}': 'post_note_count',
                'post_type_{}': 'post_type',
                'tumblog_id_follower_reblog': 'tumblog_id_follower',
                'processed_tumblr_blog_description_{}_reblog':
                    'processed_blog_description_follower',
                'tumblog_id_followee_{}': 'tumblog_id_followee',
                'processed_tumblr_blog_description_follower_reblog':
                    'processed_blog_description_follower',
                'processed_tumblr_blog_description_followee_{}':
                    'processed_blog_description_followee',
            }
            reblog_labels = {'reblog': True, 'nonreblog': False}
            for reblog_type in ['reblog', 'nonreblog']:
                data_section[reblog_type] = data.loc[:, [
                    col.format(reblog_type) for col in selected_cols]]
                data_section[reblog_type].rename(columns={
                    key.format(reblog_type): value for key,value in rename_cols.items()
                }, inplace=True)
                data_section[reblog_type]['reblog'] = [
                    reblog_labels[reblog_type]] * len(data_section[reblog_type])
            self.data = pd.concat([data_section[r] for r in ['reblog', 'nonreblog']])

    def add_random_labels(self):
        """ Add random 0 and 1 labels for ordering reblog/nonreblogs
            for learning-to-rank organization """
        half_len = int(len(self.data)/2)
        np.random.seed(9)
        labels = [0]*half_len + [1]*half_len
        np.random.shuffle(labels)
        self.data['label'] = labels

    def filter(self, user_ids=None, word_filter=None, word_filter_min=1,
                preprocessed_descs=None):
        """ Filter self.data.
            Args:
                user_ids: list of user tumblog IDs to filter data
                word_filter: filter out rows with at least one blog description
                    that has no words in this word set
                word_filter_min: minimum number of words needed in the word filter
                    list for a user's blog description
                preprocessed_descs: dictionary of tumblog_id: token_list of
                    preprocessed tokens
        """
        data = self.data
        user_cols = ['tumblog_id_follower', 'tumblog_id_followee']
        desc_cols = ['processed_blog_description_follower', 'processed_blog_description_followee']
        if user_ids:
            if self.organization == 'binary_classification':
                for col in user_cols:
                    data = data[data[col].isin(user_ids)]

        if preprocessed_descs:
            # Replace text with preprocessed text
            for user_col, desc_col in zip(user_cols, desc_cols):
                data[desc_col] = data[user_col].map(lambda x: ' '.join(preprocessed_descs[x]))

        if word_filter:
            if self.organization == 'binary_classification':
                for col in desc_cols:
                    data = data[data[col].map(lambda x: sum(
                        tok in word_filter for tok in x.split()) >= word_filter_min)]

        # Balance dataset between reblogs and nonreblogs
        if self.organization == 'binary_classification':
            n_reblogs = data['reblog'].sum()
            n_nonreblogs = len(data) - n_reblogs
            smallest = min([n_reblogs, n_nonreblogs])
            reblogs = data[data['reblog']].sample(n=smallest)
            nonreblogs = data[~data['reblog']].sample(n=smallest)
            data = pd.concat([reblogs, nonreblogs])

        self.data = data
        self.filter_settings = {'user_ids': user_ids is None,
                'word_filter': word_filter is None}

    def save_settings(self, output_dirpath):
        """ Save dataset settings """
        if not os.path.exists(output_dirpath):
            os.mkdir(output_dirpath)
        settings = {}
        settings['original_dataset_path'] = self.data_location
        settings['dataset_organization'] = self.organization
        settings['filter_settings'] = self.filter_settings
        output = pd.DataFrame(settings.values(), index=settings.keys())
        output.to_csv(os.path.join(output_dirpath, 'dataset_settings.csv'), header=False)
        print(f"\tSaved dataset settings to {output_dirpath}")

    def scale_nontext_features(self, nontext_inds):
        """ Scale nontext features, for PyTorch prep """

        X_train_add = self.X_train[:,np.array(nontext_inds)]
        before_inds = range(nontext_inds[0])
        after_inds = range(nontext_inds[-1]+1, self.X_train.shape[1])
        scaler = StandardScaler()
        X_train_add_scaled = scaler.fit_transform(X_train_add)
        self.X_train = np.hstack([
            self.X_train[:, before_inds],
            X_train_add_scaled,
            self.X_train[:, after_inds]])

        X_dev_add = self.X_dev[:,np.array(nontext_inds)]
        X_dev_add_scaled = scaler.transform(X_dev_add)
        self.X_dev = np.hstack([
            self.X_dev[:, before_inds],
            X_dev_add_scaled,
            self.X_dev[:, after_inds]])

        X_test_add = self.X_test[:,np.array(nontext_inds)]
        X_test_add_scaled = scaler.transform(X_test_add)
        X_test = X_test_add_scaled
        self.X_test = np.hstack([
            self.X_test[:, before_inds],
            X_test_add_scaled,
            self.X_test[:, after_inds]])
