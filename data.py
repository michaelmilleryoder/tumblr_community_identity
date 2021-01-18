"""
Load and filter data for reblog prediction experiments.

@author Michael Miller Yoder
@date 2021
"""

import pandas as pd


class Dataset():
    """ Encapsulates data and train/test splits """

    def __init__(self):
        self.data = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def set_folds(self, X_train, X_test, y_train, y_test):
        """ Set training and testing folds """
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test


class DataHandler():
    """ Load, filter data """

    def __init__(self, task='learning-to-rank'):
        self.task = task
        self.dataset = None

    def load(self, data_location):
        """ Load data.
            Saves a pandas dataframe with all relevant columns,
            organized by the appropriate rows, to self.data
        """
        data = pd.read_csv(data_location)
        if self.task == 'learning-to-rank':
            self.dataset.data = data
        elif self.task == 'binary_classification':
            reblog_data = data.loc[:, [
                'tumblog_id_follower_reblog',
                'tumblog_id_followee_reblog',
                'processed_tumblr_blog_description_follower_reblog',
                'processed_tumblr_blog_description_followee_reblog'
            ]]
            reblog_data.rename(columns={
                'tumblog_id_follower_reblog': 'tumblog_id_follower',
                'processed_tumblr_blog_description_follower_reblog': 'processed_blog_description_follower',
                'tumblog_id_followee_reblog': 'tumblog_id_followee',
                'processed_tumblr_blog_description_followee_reblog': 'processed_blog_description_followee',
            }, inplace=True)
            reblog_data['reblog'] = [True] * len(reblog_data)
            nonreblog_data = data.loc[:, [
                'tumblog_id_follower_reblog',
                'tumblog_id_followee_nonreblog',
                'processed_tumblr_blog_description_follower_reblog',
                'processed_tumblr_blog_description_followee_nonreblog'
            ]]
            nonreblog_data.rename(columns={
                'tumblog_id_follower_reblog': 'tumblog_id_follower',
                'processed_tumblr_blog_description_follower_reblog': 'processed_blog_description_follower',
                'tumblog_id_followee_nonreblog': 'tumblog_id_followee',
                'processed_tumblr_blog_description_followee_nonreblog': 'processed_blog_description_followee',
            }, inplace=True)
            nonreblog_data['reblog'] = [False] * len(nonreblog_data)
            self.dataset.data = pd.concat([reblog_data, nonreblog_data])

    def filter(self, user_ids=None, word_filter=None):
        """ Filter self.data.
            Args:
                user_ids: list of user tumblog IDs to filter data
                word_filter: filter out rows with at least one blog description
                    that has no words in this word set
        """
        data = self.dataset.data
        if user_ids:
            if self.task == 'binary_classification':
                user_cols = ['tumblog_id_follower', 'tumblog_id_followee']
            for col in user_cols:
                data = data[data[col].isin(user_ids)]
        if word_filter:
            if self.task == 'binary_classification':
                desc_cols = ['processed_blog_description_follower', 'processed_blog_description_followee']
            for col in desc_cols:
                data = data[data[col].map(lambda x: any(tok in word_filter for tok in x.split()))]

        # Balance dataset between reblogs and nonreblogs
        if self.task == 'binary_classification':
            n_reblogs = data['reblog'].sum()
            n_nonreblogs = len(data) - n_reblogs
            smallest = min([n_reblogs, n_nonreblogs])
            reblogs = data[data['reblog']==True].sample(n=smallest)
            nonreblogs = data[data['reblog']==False].sample(n=smallest)
            data = pd.concat([reblogs, nonreblogs])

        self.dataset.data = data
