"""
Load trained embeddings for reblog prediction
@author Michael Miller Yoder
@date 2021
"""
from gensim.models import word2vec

class EmbeddingLoader():
    """ Load, hold pretrained embeddings """

    def __init__(self, word_feature_type):
        self.word_feature_type = word_feature_type
        self.word_embs = None
        self.graph_embs = None

    def load(self, word_embs=False, graph_embs=False):
        """ Load embedding models to self.graph_embs or self.word_embs 
            Args:
                word_embs: whether to load word embeddings
                graph_embs: whether to load user graph embeddings
        """
        if graph_embs:
            graph_model_name  = '/projects/websci_exp_20210107/model/user_deepwalk_sg_w2v.model'
            self.graph_embs = word2vec.Word2Vec.load(graph_model_name)
        if word_embs:
            if self.word_feature_type == 'posts':
                post_model_name  = '/projects/websci_exp_20210107/model/post_sg_w2v.model'
                self.word_embs = word2vec.Word2Vec.load(post_model_name)
            elif self.word_feature_type == 'blog_desc':
                blog_model_name  = '/projects/websci_exp_20210107/model/post+all_blog_sg_w2v.model'
                self.word_embs = word2vec.Word2Vec.load(blog_model_name)
            elif self.word_feature_type == 'deepwalk':
                deep_walk_model_name  =  '/projects/websci_exp_20210107/model/post+deepwalk_sg_w2v.model'
                self.word_embs = word2vec.Word2Vec.load(deep_walk_model_name)
