"""
Load trained embeddings for reblog prediction
@author Michael Miller Yoder
@date 2021
"""
from gensim.models import word2vec

class EmbeddingLoader()

    def __init__(self, feature_type):
        self.feature_type = feature_type

    def load(self):
        """ Load embedding models to self.user_embs or self.word_embs """
        if feature_type == 'graph':
            user_model_name  = '/projects/websci_exp_20210107/model/user_deepwalk_sg_w2v.model'
            self.user_embs = word2vec.Word2Vec.load(user_model_name)
        elif feature_type == 'posts':
            post_model_name  = '/projects/websci_exp_20210107/model/post_sg_w2v.model'
            self.word_embs = word2vec.Word2Vec.load(post_model_name)
        elif feature_type == 'blog_desc':
            blog_model_name  = '/projects/websci_exp_20210107/model/post+all_blog_sg_w2v.model'
            self.word_embs = word2vec.Word2Vec.load(blog_model_name)
        elif feature_type == 'deepwalk':
            deep_walk_model_name  =  '/projects/websci_exp_20210107/model/post+deepwalk_sg_w2v.model'
            self.word_embs = word2vec.Word2Vec.load(deep_walk_model_name)
