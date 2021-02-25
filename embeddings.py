"""
Load trained embeddings for reblog prediction
@author Michael Miller Yoder
@date 2021
"""
import pdb
import os
import pickle
from gensim.models import word2vec, KeyedVectors

class EmbeddingLoader():
    """ Load, hold pretrained embeddings """

    def __init__(self, word_emb_type, sent_emb_type):
        self.word_emb_type = word_emb_type
        self.sent_emb_type = sent_emb_type
        self.word_embs = None
        self.graph_embs = None
        self.sent_embs = None
        base_dirpath = '/projects/tumblr_community_identity/'
        self.model_paths = {
            'posts': os.path.join(
                base_dirpath, 'websci_exp_20210107/model/post_sg_w2v.model'),
            'blog_desc': os.path.join(base_dirpath,
                'websci_exp_20210107/model/post+all_blog_sg_w2v.model'),
            'deepwalk': os.path.join(base_dirpath,
                'websci_exp_20210107/model/post+deepwalk_sg_w2v.model'),
            'sent_embs': os.path.join(base_dirpath,
                'new_sentence_embed/blog_embed.pkl')
        }

    def load(self, word_embs=False, graph_embs=False, sent_embs=False):
        """ Load embedding models to self
            Args:
                word_embs: whether to load word embeddings
                graph_embs: whether to load user graph embeddings
                sent_embs: whether to load pretrained full blog description embeddings
        """
        if graph_embs:
            # From Feng-Guang
            #graph_model_path  = \
            #    '/projects/websci_exp_20210107/model/user_deepwalk_sg_w2v.model'
            #self.graph_embs = word2vec.Word2Vec.load(graph_model_path)

            # node2vec, from Joseph
            graph_model_path  = \
                '/data/tumblr_community_identity/embeddings/114_d100_e30.emb'
            self.graph_embs = KeyedVectors.load_word2vec_format(graph_model_path)

        if word_embs:
            self.word_embs = word2vec.Word2Vec.load(
                self.model_paths[self.word_emb_type])

        if sent_embs:
            with open(self.model_paths['sent_embs'], 'rb') as f:
                sent_embs = pickle.load(f)
            transform = {'fasttext': 'fastText',
                        'bert': 'BERT'}
            self.sent_embs = {uid: sent_embs[uid][transform[self.sent_emb_type]] \
                for uid in sent_embs}
