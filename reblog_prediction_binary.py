import csv
import os
import codecs
from collections import defaultdict
from sklearn import feature_extraction
from sklearn import linear_model
from sklearn import neural_network
from sklearn import model_selection
from sklearn import preprocessing
from sklearn import svm

import nltk
import gensim
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar
from tqdm import tqdm
import pickle
import pdb
import argparse
import copy
from operator import itemgetter
from scipy.sparse import vstack

from utils import *
from gensim.models import word2vec

"""

This script contains code for experiments predicting Tumblr reblog behavior (content propagation) from post content 
and identity features of users.
It treats this as simply binary classification of a reblog between (follower, followee).

This includes:
* Feature extraction
    * baseline features from post content: hashtags, post like count, post media type
    * identity features: configurations of matches and mismatches from self-presented identity labels between users 
        who may or may not reblog each others' posts
* Experiments 
    * learning-to-rank machine learning formulation with pairs of users who did share a post and pairs who did not 
        (the predicted outcome measure)
    * machine learning models from scikit-learn: logistic regression, SVM, feedforward neural network

Entrance point: main function.

Example run:
python3 reblog_prediction_binary.py --classifier lr --name baseline --feature user --output-dirpath output

"""


def run_mcnemar(baseline_pred, experiment_pred, y_test):
    """ McNemar's Test (Significance) 
    http://www.atyun.com/25532.html
    It is a statistical evaluation of paired nominal data or classifiers.
    There are totally 2 tests constucting 2x2 contingency tables.
        /    | Test2 Pos    Test2 Neg | 
    Test1 Pos|                        |
    Test1 Neg|                        |

    Eg.                       Before grad
                        w. girl         w/o girl
    After    w. girl|     5(A)            18(B)
    Grad    w/o girl|     5(C)            22(D)
    H0 => The number of grads with girl is the same as that after grad.
    Ha => The number of grads with girl is the diff from that after grad.
    Chi-Square Distribution

    It wants to know whether two distributions are different only because of the random noise (Null hypothesis).
    """

    a = 0
    b = 0 # Baseline correct, experiment incorrect
    c = 0 # Baseline incorrect, experiment correct
    d = 0
    for b_pred, ex_pred, true in zip(baseline_pred, experiment_pred, y_test):
        if b_pred == true and ex_pred == true:
            a += 1
        elif b_pred == true and ex_pred != true:
            b += 1
        elif b_pred != true and ex_pred == true:
            c += 1
        else:
            d += 1
            
    table = [[a, b],
             [c, d]]

    # Example of calculating the mcnemar test
    # calculate mcnemar test
    result = mcnemar(table, exact=False, correction=False)
    # summarize the finding
    #print('statistic=%.3f, p-value=%.6f' % (result.statistic, result.pvalue))
    # interpret the p-value
    alpha = 0.05
    if result.pvalue > alpha:
            print('Same proportions of errors (fail to reject H0)')
    else:
            print('Different proportions of errors (reject H0)')
    
    return result


def _str2list(in_str):
    """ Utility function """
    return [el[1:-1] for el in in_str[1:-1].split(', ')]


def extract_features(feature_type='user'):
    # Load users
    if feature_type == 'user':
        user_model_name  = '/projects/websci_exp_20210107/model/user_deepwalk_sg_w2v.model'
        model = word2vec.Word2Vec.load(user_model_name)
        users = list(model.wv.vocab.keys())
        users = set([int(user) for user in users])
    else:
        # id2token
        id2token = defaultdict(list) # tumblog_id: [blog_description_tokens]
        #id2token_file = '/projects/websci_exp_20210107/output/new_id2token_114k.pkl'
        #with open(id2token_file, 'rb') as fp:
        #    id2token = pickle.load(fp)
        #users = set(list(id2token.keys()))
        #tqdm.write(f"\t{len(users)} available users")

    # Load word embedding models
    if feature_type == 'user':
        user_model_name  = '/projects/websci_exp_20210107/model/user_deepwalk_sg_w2v.model'
        model = word2vec.Word2Vec.load(user_model_name)
    elif feature_type == 'pretrained':
        post_model_name  = '/projects/websci_exp_20210107/model/post_sg_w2v.model'
        model = word2vec.Word2Vec.load(post_model_name)
    elif feature_type == 'blog_desc':
        blog_model_name  = '/projects/websci_exp_20210107/model/post+all_blog_sg_w2v.model'
        model = word2vec.Word2Vec.load(blog_model_name)
    elif feature_type == 'deepwalk':
        deep_walk_model_name  =  '/projects/websci_exp_20210107/model/post+deepwalk_sg_w2v.model'
        model = word2vec.Word2Vec.load(deep_walk_model_name)

    # Retrieve (non)reblog pairs
    reblog_pairs, non_reblog_pairs = [], []
    reblog_csv = '/data/tumblr_community_identity/dataset114k/matched_reblogs_nonreblogs_dataset114k.csv'
    #reblog_csv = '/data/tumblr_community_identity/dataset114k/matched_reblogs_nonreblogs_dataset114k_full.csv'
    with open(reblog_csv, 'r') as fpath:
        blog_info = pd.read_csv(fpath)
        #blog_info = pd.read_csv(fpath, sep='\t')
        for follower, reblogger, nonreblogger, desc_follower, desc_reblogger, desc_nonreblogger in zip(
                blog_info['tumblog_id_follower_reblog'], \
                blog_info['tumblog_id_followee_reblog'], \
                blog_info['tumblog_id_followee_nonreblog'], \
                blog_info['processed_tumblr_blog_description_follower_reblog'], \
                blog_info['processed_tumblr_blog_description_followee_reblog'], \
                blog_info['processed_tumblr_blog_description_followee_nonreblog'], \
        ):
            if feature_type == 'user':
                if follower in users:
                    if reblogger in users:
                        reblog_pairs.append((follower, reblogger))
                    if nonreblogger in users:
                        non_reblog_pairs.append((follower, nonreblogger))
            else:
                desc_follower_toks = desc_follower.split()
                desc_reblogger_toks = desc_reblogger.split()
                desc_nonreblogger_toks = desc_nonreblogger.split() 
                if any(tok in model for tok in desc_follower_toks) and \
                    any(tok in model for tok in desc_reblogger_toks) and \
                    any(tok in model for tok in desc_nonreblogger_toks):
                    id2token[follower] = desc_follower_toks
                    id2token[reblogger] = desc_reblogger_toks
                    id2token[nonreblogger] = desc_nonreblogger_toks
                    reblog_pairs.append((follower, reblogger))
                    non_reblog_pairs.append((follower, nonreblogger))

    # Features
    x_train = reblog_pairs[:int(len(reblog_pairs)*0.9)] + non_reblog_pairs[:int(len(non_reblog_pairs)*0.9)]
    y_train = [1]*int(len(reblog_pairs)*0.9) + [0]*int(len(non_reblog_pairs)*0.9)
    
    x_test  = reblog_pairs[int(len(reblog_pairs)*0.9):] + non_reblog_pairs[int(len(non_reblog_pairs)*0.9):]
    y_test  = [1]*(len(reblog_pairs) - int(len(reblog_pairs)*0.9)) + [0]*(len(non_reblog_pairs) - int(len(non_reblog_pairs)*0.9))

    if feature_type == 'user':
        dim = 256
    else:
        dim = 128

    X_train, X_test = [], []
    Y_train, Y_test = [], []
    for pair, label in tqdm(list(zip(x_train, y_train)), ncols=70):
        user1, user2 = pair
        if feature_type == 'user':
            embed_u1 = model.wv[str(user1)].reshape(-1)
            embed_u2 = model.wv[str(user2)].reshape(-1)
            feature = np.concatenate((embed_u1,embed_u2))
        else:
            embed_1 = words2embedding(id2token[user1], model.wv).reshape(-1)
            embed_2 = words2embedding(id2token[user2], model.wv).reshape(-1)
            if np.isnan(np.min(embed_1)) or np.isnan(np.min(embed_1)):
                pdb.set_trace()
            feature = np.concatenate((embed_1,embed_2))
        if feature.shape[0] == dim:
            X_train.append(feature)
            Y_train.append(label)
    
    for i, pair in enumerate(tqdm(x_test, ncols=70)):
        user1, user2 = pair
        if feature_type == 'user':
            embed_u1 = model.wv[str(user1)].reshape(-1)
            embed_u2 = model.wv[str(user2)].reshape(-1)
            feature = np.concatenate((embed_u1,embed_u2))
        else:
            embed_1 = words2embedding(id2token[user1], model.wv).reshape(-1)
            embed_2 = words2embedding(id2token[user2], model.wv).reshape(-1)
            if np.isnan(np.min(embed_1)) or np.isnan(np.min(embed_2)):
                pdb.set_trace()
            feature = np.concatenate((embed_1,embed_2))
        if feature.shape[0] == dim:
            X_test.append(feature)
            Y_test.append(label)

    X_train, X_test = np.stack(X_train), np.stack(X_test)
    tqdm.write(f'\tTrain set shape: {X_train.shape}')
    tqdm.write(f'\tTest set shape: {X_test.shape}')
    return X_train, Y_train, X_test, Y_test


def run_model(model_name, clf, X_train, y_train, X_test, y_test, feat, output_dirpath):
    """ 
    Train model, make predictions 
    model_name = baseline_lr
    X_train = features -> scipy sparse matrix (641403, 14309)
    y_train = [1,0,1,0,1,1,1,1,0,0,0,0, ....] -> list of 1s and 0s (641403)
    X_test  = features -> scipy sparse matrix (71267, 14309) -> random 10% split
    y_test  = [1,0,1,0,1,1,1,1,0,0,0,0, ....] -> list of 1s and 0s (71267)
    """
    model = clf.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    train_pred = model.predict(X_train)
    model_pred = model.predict(X_test)

    # Save predictions
    dirpath = os.path.join(output_dirpath, '{}_{}'.format(model_name, feat), 'predictions')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    np.savetxt(os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}_test_preds.txt'), model_pred)
    np.savetxt(os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}_train_preds.txt'), train_pred)

    # Save classifier (with weights)
    dirpath = os.path.join(output_dirpath, '{}_{}'.format(model_name, feat), 'models')
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    with open(os.path.join(dirpath, f'{model_name.replace("/", "_").replace(" ", "_")}.pkl'), 'wb') as f:
        pickle.dump(model, f)

    return model, score, model_pred


def main():
    parser = argparse.ArgumentParser(description='Extract features and run models')
    parser.add_argument('--classifier', dest='classifier_type', help='lr svm ffn', default='')
    parser.add_argument('--name', dest='model_name', help='model name base, automatically appends experiment features and classifier, None just puts classifier and features', default=None)
    parser.add_argument('--feature', dest='feature', help='feature type', default='user')
    parser.add_argument('--output-dirpath', dest='output_dirpath', help='output dirpath; default /projects/websci2020_tumblr_identity', default='/projects/websci2020_tumblr_identity')
    args = parser.parse_args()

    feature_type   = args.feature
    output_dirpath = args.output_dirpath

    # Classifier definitions
    classifiers = {
        'lr': linear_model.LogisticRegressionCV(cv=10, n_jobs=10, max_iter=10000, verbose=0),
        'svm': model_selection.GridSearchCV(svm.LinearSVC(dual=False, max_iter=10000, verbose=0), {'C': [.01, .1, 1, 10, 100], 'penalty': ['l2']}, n_jobs=10, cv=10, verbose=2),
        'ffn': neural_network.MLPClassifier(hidden_layer_sizes=(32, 50), activation='relu', early_stopping=True, verbose=2)
    }
    
    # ### Post baseline
    print("Extracting features...")
    X_train, y_train, X_test, y_test = extract_features(feature_type)
    clf = classifiers[args.classifier_type]

    print("Running model...")
    if args.model_name is None:
        model_name = f'baseline_{args.classifier_type}'
    else:
        model_name = f'{args.model_name}_{args.classifier_type}'
    
    model, score, baseline_preds = run_model(model_name, clf, X_train, y_train, X_test, y_test, feature_type, output_dirpath)
    print(f'\tScore: {score: .4f}')

if __name__ == '__main__':
    main()
