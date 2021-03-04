"""
This script contains code for experiments predicting Tumblr reblog behavior
    (content propagation) from post content and identity features of users.
It can treat this as simply binary classification of a reblog between 
    (follower, followee) or learning-to-rank, where a follower chooses to 
    reblog a post from one of its followers and not from another.

Entrance point: main function.
Environment: conda_env, included in this directory

Example run:
python reblog_prediction_binary.py --classifier lr --name baseline --feature user --output-dirpath output

"""

import os
import pdb
import argparse

from extract_features import FeatureExtractor
from data import Dataset
from run_model import Experiment
from embeddings import EmbeddingLoader
from utils import load_pickle


def get_args():
    """ Get CLI arguments """
    parser = argparse.ArgumentParser(description='Extract features and run models')
    parser.add_argument('--classifier', dest='classifier_type', 
        help='lr svm ffn cnn',
        default='')
    parser.add_argument('--name', dest='model_name', help='model name base, automatically appends experiment features and classifier', default='')
    parser.add_argument('--features', dest='features', help='Which set of features to include, separated by commas. Options: {post,text,graph}. Default: post,text', default='post,text')
    parser.add_argument('--post-emb-type', dest='post_emb_type', 
        help='Which pretrained word embedding model to use for posts'
                'out of {posts, blog_desc, deepwalk}',
        default='posts')
    parser.add_argument('--text-emb-type', dest='text_emb_type',
        help='Which pretrained word embedding model to use for blog descriptions'
                'out of {posts, blog_desc, deepwalk, fasttext, bert}',
        default='blog_desc')
    parser.add_argument('--task', dest='task', 
        help='Which task to run out of {binary_classification, learning-to-rank}', 
        default='learning-to-rank')
    parser.add_argument('--dataset-location', dest='data_location', help='Path to the CSV of the dataset; default /data/tumblr_community_identity/dataset114k/matched_reblogs_nonreblogs_dataset114k.csv', default='/data/tumblr_community_identity/dataset114k/matched_reblogs_nonreblogs_dataset114k.csv')
    parser.add_argument('--output-dirpath', dest='output_dirpath', help='output dirpath; default /projects/websci2020_tumblr_identity', default='/projects/websci2020_tumblr_identity')
    parser.add_argument('--word-filter', dest='word_filter_min', type=int,
        help="word_filter_min: minimum number of words needed in the word"
            "filter list for a user's blog description; default 1", default=1)
    parser.add_argument('--preprocessed', dest='load_preprocessed', help='path to external preprocessed blog descriptions; default None', default=None)
    args = parser.parse_args()
    return args


def main():
    """ Load data, train and evaluate a model """
    args = get_args()
    if args.classifier_type in ['cnn']: # PyTorch
        run_pkg = 'pytorch'
    else:
        run_pkg = 'sklearn'
    exp_name = '_'.join([
            args.model_name,
            args.features.replace(',', '+'),
            args.text_emb_type,
            args.classifier_type,
        ]).strip('_')
    exp_output_dirpath = os.path.join(args.output_dirpath, exp_name)

    # Load trained embedding models
    print("Loading embeddings...")
    emb_loader = EmbeddingLoader(args.post_emb_type, args.text_emb_type)
    load_word_embs = True
    load_graph_embs, graph_embs = False, None
    load_sent_embs, sent_embs = False, None
    if 'graph' in args.features:
        load_graph_embs = True
    if 'text' in args.features and args.text_emb_type in ['fasttext', 'bert']:
        load_sent_embs = True
    emb_loader.load(word_embs=load_word_embs, graph_embs=load_graph_embs,
        sent_embs=load_sent_embs)
    if load_graph_embs:
        graph_embs = emb_loader.graph_embs
    if load_sent_embs:
        sent_embs = emb_loader.sent_embs
    word_embs = emb_loader.word_embs

    # Load and filter dataset
    print("Loading and filtering data...")
    dataset = Dataset()
    dataset.load(args.data_location, args.task)
    if args.load_preprocessed:
        id2token = load_pickle(args.load_preprocessed)
        user_filter = set(list(id2token.keys()))
        dataset.filter(user_ids=user_filter, word_filter=word_embs.wv, 
            word_filter_min=args.word_filter_min, preprocessed_descs=id2token)
    else:
        dataset.filter(word_filter=word_embs.wv, word_filter_min=args.word_filter_min)

    # Extract features
    print("Extracting features...")
    if run_pkg == 'pytorch':
        extractor = FeatureExtractor(args.features, word_embs=word_embs,
            graph_embs=graph_embs, sent_embs=sent_embs, 
            word_inds=True, padding_size=30)
        dataset = extractor.extract(dataset, dev=True)
    else:
        extractor = FeatureExtractor(args.features, word_embs=word_embs,
            graph_embs=graph_embs, sent_embs=sent_embs)
        dataset = extractor.extract(dataset)

    # Run model
    print("Running model...")
    experiment = Experiment(extractor, dataset, args.classifier_type)
    experiment.run()

    # Print output
    print(f'\tScore: {experiment.score: .4f}')

    # Save settings, output
    if run_pkg == 'sklearn':
        dataset.save_settings(exp_output_dirpath)
        experiment.save_output(exp_output_dirpath)


if __name__ == '__main__':
    main()
