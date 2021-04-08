"""
CNN in PyTorch

@author Michael Miller Yoder
@date 2021
"""

import datetime
import math
import pdb

import numpy as np
import torch
import torch.nn as nn


class CNNTextGraphClassifier(nn.Module):
    """ 
        CNN for post+text+graph features.
        From https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch/
        blob/master/src/model/model.py 
    """

    def __init__(self, extractor, device):
        """ Args:
                extractor: FeatureExtractor, for the parameters
        """

        super().__init__()
        self.extractor = extractor
        self.device = device
        self.clf_type = 'cnn'
        self.name = 'model' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Parameters regarding text preprocessing
        self.word_embs = self.extractor.word_embs
        self.seq_len = self.extractor.padding_size
        self.num_words = len(self.extractor.word_embs.wv.vocab)
        self.embedding_size = self.extractor.word_embs.vector_size

        # CNN parameters definition
        # Kernel sizes
        self.kernel_sizes = range(2,6)
        # Output size for each convolution
        self.out_size = 32
        # Number of strides for each convolution
        self.stride = 2
        # Dropout
        #self.dropout = 0.25
        self.dropout = 0

        ## Embedding layer definition
        weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
        zeros = torch.zeros(1, self.embedding_size).to(device)
        weights_with_padding = torch.cat((zeros, weights), 0).to(device)
        self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
            padding_idx=0).to(device)

        # Convolution layers definition (post)
        self.conv_block_post = {}
        for kernel in self.kernel_sizes:
            self.conv_block_post[kernel] = self.conv_block(self.seq_len, kernel).to(self.device)

        # Convolution layers definition (text)
        self.conv_block_text = {}
        for kernel in self.kernel_sizes:
            self.conv_block_text[kernel] = self.conv_block(self.seq_len*2, kernel).to(self.device)

        # Graph FFN layers
        self.fc_graph = nn.Sequential(
            nn.Linear(400, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
        ).to(self.device)

        #self.fc_block = fc_block(7842, self.dropout)
        self.fc_block = fc_block(7474, self.dropout)

    def forward(self, x):
        """ Called indirectly through model(input).
            TODO: probably separate input x as arguments
        """

        # Post features
        tag_feats = {}
        tag_feats['reblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i in self.extractor.reblog_inds])].long()
        tag_feats['nonreblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i not in self.extractor.graph_inds['reblog'] and \
            i not in self.extractor.graph_inds['nonreblog'] and \
            i not in self.extractor.text_inds['reblog'] and \
            i not in self.extractor.text_inds['nonreblog'] and \
            i not in self.extractor.reblog_inds])].long()
        tag_feats['reblog'] = self.embedding(tag_feats['reblog'])
        tag_feats['nonreblog'] = self.embedding(tag_feats['nonreblog'])
        x_add = x[:,np.array(self.extractor.nontext_inds)].float()

        post_feats = {'reblog': {}, 'nonreblog': {}}
        for reblog_type in ['reblog', 'nonreblog']:
            for kernel in self.kernel_sizes:
                post_feats[reblog_type][kernel] = self.conv_block_post[kernel](
                    tag_feats[reblog_type])

        post_embs_list = sum([[post_feats['reblog'][kernel], 
            post_feats['nonreblog'][kernel]] for kernel in self.kernel_sizes], [])
        post_embs = torch.cat(post_embs_list, 2)
        post_embs = post_embs.reshape(post_embs.size(0), -1)

        # Text blog description features
        text_x = {}
        text_x['reblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.text_inds['reblog']])].long()
        text_x['nonreblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.text_inds['nonreblog']])].long()
        text_x['reblog'] = self.embedding(text_x['reblog'])
        text_x['nonreblog'] = self.embedding(text_x['nonreblog'])

        text_feats = {'reblog': {}, 'nonreblog': {}}
        for reblog_type in ['reblog', 'nonreblog']:
            for kernel in self.kernel_sizes:
                text_feats[reblog_type][kernel] = self.conv_block_text[kernel](
                    text_x[reblog_type])

        text_embs_list = sum([[text_feats['reblog'][kernel], 
            text_feats['nonreblog'][kernel]] for kernel in self.kernel_sizes], [])
        text_embs = torch.cat(text_embs_list, 2)
        text_embs = text_embs.reshape(text_embs.size(0), -1)

        # Graph features
        graph_x = {}
        graph_x['reblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.graph_inds['reblog']])]
        graph_x['nonreblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.graph_inds['nonreblog']])]
        graph_embs = torch.cat((graph_x['reblog'], graph_x['nonreblog']), 1)
        graph_embs = self.fc_graph(graph_embs.float().to(self.device))

        # Final classification layer
        flattened = torch.cat((post_embs, x_add, text_embs, graph_embs), 1)
        out = self.fc_block(flattened.float())
        return out.squeeze()

    def conv_block(self, seq_len, kernel):
        return nn.Sequential(
            nn.Conv1d(seq_len, self.out_size, kernel, self.stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel, self.stride)
        )


class CNNGraphClassifier(nn.Module):
    """ 
        CNN for post+graph features.
        From https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch/
        blob/master/src/model/model.py 
    """

    def __init__(self, extractor, device):
        """ Args:
                extractor: FeatureExtractor, for the parameters
        """

        super().__init__()
        self.extractor = extractor
        self.device = device
        self.clf_type = 'cnn'
        self.name = 'model' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Parameters regarding text preprocessing
        self.word_embs = self.extractor.word_embs
        self.seq_len = self.extractor.padding_size
        self.num_words = len(self.extractor.word_embs.wv.vocab)
        self.embedding_size = self.extractor.word_embs.vector_size

        # CNN parameters definition
        # Kernel sizes
        self.kernel_sizes = range(2,6)
        # Output size for each convolution
        self.out_size = 32
        # Number of strides for each convolution
        self.stride = 2
        # Dropout
        #self.dropout = 0.25
        self.dropout = 0

        ## Embedding layer definition
        weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
        zeros = torch.zeros(1, self.embedding_size).to(device)
        weights_with_padding = torch.cat((zeros, weights), 0).to(device)
        self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
            padding_idx=0).to(device)

        # Convolution layers definition (post)
        self.conv_block_post = {}
        for kernel in self.kernel_sizes:
            self.conv_block_post[kernel] = self.conv_block(self.seq_len, kernel).to(self.device)

        # Graph FFN layers
        self.fc_graph = nn.Sequential(
            nn.Linear(400, 128).to(self.device),
            nn.ReLU().to(self.device),
            nn.Linear(128, 32).to(self.device),
            nn.ReLU().to(self.device),
        ).to(self.device)

        #self.fc_block = fc_block(4130, self.dropout) # for 1-layer FFN graph
        self.fc_block = fc_block(3762, self.dropout) # for graph FFN block

    def forward(self, x):
        """ Called indirectly through model(input).
            TODO: probably separate input x as arguments
        """

        # Post features
        tag_feats = {}
        tag_feats['reblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i in self.extractor.reblog_inds])].long()
        tag_feats['nonreblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i not in self.extractor.graph_inds['reblog'] and \
            i not in self.extractor.graph_inds['nonreblog'] and \
            i not in self.extractor.reblog_inds])].long()
        tag_feats['reblog'] = self.embedding(tag_feats['reblog'])
        tag_feats['nonreblog'] = self.embedding(tag_feats['nonreblog'])
        x_add = x[:,np.array(self.extractor.nontext_inds)].float()

        post_feats = {'reblog': {}, 'nonreblog': {}}
        for reblog_type in ['reblog', 'nonreblog']:
            for kernel in self.kernel_sizes:
                post_feats[reblog_type][kernel] = self.conv_block_post[kernel](
                    tag_feats[reblog_type])

        post_embs_list = sum([[post_feats['reblog'][kernel], 
            post_feats['nonreblog'][kernel]] for kernel in self.kernel_sizes], [])
        post_embs = torch.cat(post_embs_list, 2)
        post_embs = post_embs.reshape(post_embs.size(0), -1)

        # Graph features
        graph_x = {}
        graph_x['reblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.graph_inds['reblog']])]
        graph_x['nonreblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.graph_inds['nonreblog']])]

        graph_embs = torch.cat((graph_x['reblog'], graph_x['nonreblog']), 1)
        graph_embs = self.fc_graph(graph_embs.float().to(self.device)) # cuda error for some reason

        # Final classification layer
        flattened = torch.cat((post_embs, x_add, graph_embs), 1)
        out = self.fc_block(flattened.float())
        return out.squeeze()

    def conv_block(self, seq_len, kernel):
        return nn.Sequential(
            nn.Conv1d(seq_len, self.out_size, kernel, self.stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel, self.stride)
        )

class CNNTextClassifier(nn.Module):
    """ 
        CNN for post+text features.
        From https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch/
        blob/master/src/model/model.py 

        TODO: think about combining or subclassing with post baseline CNN
    """

    def __init__(self, extractor, device):
        """ Args:
                extractor: FeatureExtractor, for the parameters
        """

        super().__init__()
        self.extractor = extractor
        self.device = device
        self.clf_type = 'cnn'
        self.name = 'model' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Parameters regarding text preprocessing
        self.word_embs = self.extractor.word_embs
        self.seq_len = self.extractor.padding_size
        self.num_words = len(self.extractor.word_embs.wv.vocab)
        self.embedding_size = self.extractor.word_embs.vector_size

        # CNN parameters definition
        # Kernel sizes
        self.kernel_sizes = range(2,6)
        # Output size for each convolution
        self.out_size = 32
        # Number of strides for each convolution
        self.stride = 2
        # Dropout
        #self.dropout = 0.25
        self.dropout = 0

        ## Embedding layer definition
        weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
        zeros = torch.zeros(1, self.embedding_size).to(device)
        weights_with_padding = torch.cat((zeros, weights), 0).to(device)
        self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
            padding_idx=0).to(device)

        # Convolution layers definition (post)
        self.conv_block_post = {}
        for kernel in self.kernel_sizes:
            self.conv_block_post[kernel] = self.conv_block(self.seq_len, kernel).to(self.device)

        # Convolution layers definition (text)
        self.conv_block_text = {}
        for kernel in self.kernel_sizes:
            self.conv_block_text[kernel] = self.conv_block(self.seq_len*2, kernel).to(self.device)

        self.fc_block = fc_block(7442, self.dropout)

    def forward(self, x):
        """ Called indirectly through model(input).
            TODO: probably separate input x as arguments
        """

        # Post features
        tag_feats = {}
        tag_feats['reblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i in self.extractor.reblog_inds])].long()
        tag_feats['nonreblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i not in self.extractor.text_inds['reblog'] and \
            i not in self.extractor.text_inds['nonreblog'] and \
            i not in self.extractor.reblog_inds])].long()
        tag_feats['reblog'] = self.embedding(tag_feats['reblog'])
        tag_feats['nonreblog'] = self.embedding(tag_feats['nonreblog'])
        x_add = x[:,np.array(self.extractor.nontext_inds)].float()

        post_feats = {'reblog': {}, 'nonreblog': {}}
        for reblog_type in ['reblog', 'nonreblog']:
            for kernel in self.kernel_sizes:
                post_feats[reblog_type][kernel] = self.conv_block_post[kernel](
                    tag_feats[reblog_type])

        post_embs_list = sum([[post_feats['reblog'][kernel], 
            post_feats['nonreblog'][kernel]] for kernel in self.kernel_sizes], [])
        post_embs = torch.cat(post_embs_list, 2)
        post_embs = post_embs.reshape(post_embs.size(0), -1)

        # Blog description features
        text_x = {}
        text_x['reblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.text_inds['reblog']])].long()
        text_x['nonreblog'] = x[:,np.array([i for i in range(x.shape[1]) if \
            i in self.extractor.text_inds['nonreblog']])].long()
        text_x['reblog'] = self.embedding(text_x['reblog'])
        text_x['nonreblog'] = self.embedding(text_x['nonreblog'])

        text_feats = {'reblog': {}, 'nonreblog': {}}
        for reblog_type in ['reblog', 'nonreblog']:
            for kernel in self.kernel_sizes:
                text_feats[reblog_type][kernel] = self.conv_block_text[kernel](
                    text_x[reblog_type])

        text_embs_list = sum([[text_feats['reblog'][kernel], 
            text_feats['nonreblog'][kernel]] for kernel in self.kernel_sizes], [])
        text_embs = torch.cat(text_embs_list, 2)
        text_embs = text_embs.reshape(text_embs.size(0), -1)

        # Final classification layer
        flattened = torch.cat((post_embs, x_add, text_embs), 1)
        out = self.fc_block(flattened)
        return out.squeeze()

    def conv_block(self, seq_len, kernel):
        return nn.Sequential(
            nn.Conv1d(seq_len, self.out_size, kernel, self.stride),
            nn.ReLU(),
            nn.MaxPool1d(kernel, self.stride)
        )


def fc_block(flattened_size, dropout):
    return nn.Sequential(
        nn.Linear(flattened_size, 1),
        nn.Dropout(dropout),
        nn.Sigmoid()
    )


class CNNPostClassifier(nn.Module):
    """ 
        CNN for post features.
        From https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch/
        blob/master/src/model/model.py 
    """

    def __init__(self, extractor, device):
        """ Args:
                extractor: FeatureExtractor, for the parameters
        """

        super().__init__()
        self.extractor = extractor
        self.device = device
        self.clf_type = 'cnn'
        self.name = 'model' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Parameters regarding text preprocessing
        self.word_embs = self.extractor.word_embs
        #self.seq_len = self.extractor.padding_size * 6
        self.seq_len = self.extractor.padding_size
        self.num_words = len(self.extractor.word_embs.wv.vocab)
        self.embedding_size = self.extractor.word_embs.vector_size

        # CNN parameters definition
        # Kernel sizes
        self.kernel_1 = 2
        self.kernel_2 = 3
        self.kernel_3 = 4
        self.kernel_4 = 5

        # Output size for each convolution
        self.out_size = 32
        # Number of strides for each convolution
        self.stride = 2

        ## Embedding layer definition
        weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
        zeros = torch.zeros(1, self.embedding_size).to(device)
        weights_with_padding = torch.cat((zeros, weights), 0).to(device)
        self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
            padding_idx=0).to(device)

        # Convolution layers definition
        self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
        self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
        self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
        self.conv_4 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_4, self.stride)
        
        # Max pooling layers definition
        self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
        self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
        self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
        self.pool_4 = nn.MaxPool1d(self.kernel_4, self.stride)
        
        # Fully connected layer definition
        #self.fc = nn.Linear(self.in_features_fc() + 36, 1)
        self.fc = nn.Linear(3730, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """ Called indirectly through model(input) """

        x_reblog_text = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i in self.extractor.reblog_inds])].long()
        x_nonreblog_text = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i not in self.extractor.reblog_inds])].long()
        x_add = x[:,np.array(self.extractor.nontext_inds)].float()

        x_reblog_text = self.embedding(x_reblog_text)
        x_nonreblog_text = self.embedding(x_nonreblog_text)

        ## Convolution layer 1 is applied
        x1_reblog = self.conv_1(x_reblog_text)
        x1_reblog = self.relu(x1_reblog)
        x1_reblog = self.pool_1(x1_reblog)
        x1_nonreblog = self.conv_1(x_nonreblog_text)
        x1_nonreblog = self.relu(x1_nonreblog)
        x1_nonreblog = self.pool_1(x1_nonreblog)

        ## Convolution layer 2 is applied
        x2_reblog = self.conv_2(x_reblog_text)
        x2_reblog = self.relu(x2_reblog)
        x2_reblog = self.pool_2(x2_reblog)
        x2_nonreblog = self.conv_2(x_nonreblog_text)
        x2_nonreblog = self.relu(x2_nonreblog)
        x2_nonreblog = self.pool_2(x2_nonreblog)

        ## Convolution layer 3 is applied
        x3_reblog = self.conv_3(x_reblog_text)
        x3_reblog = self.relu(x3_reblog)
        x3_reblog = self.pool_3(x3_reblog)
        x3_nonreblog = self.conv_3(x_nonreblog_text)
        x3_nonreblog = self.relu(x3_nonreblog)
        x3_nonreblog = self.pool_3(x3_nonreblog)

        ## Convolution layer 4 is applied
        x4_reblog = self.conv_4(x_reblog_text)
        x4_reblog = self.relu(x4_reblog)
        x4_reblog = self.pool_4(x4_reblog)
        x4_nonreblog = self.conv_4(x_nonreblog_text)
        x4_nonreblog = self.relu(x4_nonreblog)
        x4_nonreblog = self.pool_4(x4_nonreblog)

        union1 = torch.cat((x1_reblog, x1_nonreblog, x2_reblog, x2_nonreblog,
            x3_reblog, x3_nonreblog, x4_reblog, x4_nonreblog), 2)
        union1 = union1.reshape(union1.size(0), -1)
        flattened = torch.cat((union1, x_add), 1)
    
        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(flattened)
        out = self.sigmoid(out)
        
        return out.squeeze()

    def in_features_fc(self):
        '''Calculates the number of output features after Convolution + Max pooling

        Convolved_Features = ((embedding_size + (2 * padding) - dilation * 
            (kernel - 1) - 1) / stride) + 1
        Pooled_Features = ((embedding_size + (2 * padding) - dilation * 
            (kernel - 1) - 1) / stride) + 1

        source: https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
        '''
        # Calculate size of convolved/pooled features for convolution_1/max_pooling_1 features
        out_conv_1 = ((self.embedding_size - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_conv_1 = math.floor(out_conv_1)
        out_pool_1 = ((out_conv_1 - 1 * (self.kernel_1 - 1) - 1) / self.stride) + 1
        out_pool_1 = math.floor(out_pool_1)

        # Calculate size of convolved/pooled features for convolution_2/max_pooling_2 features
        out_conv_2 = ((self.embedding_size - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_conv_2 = math.floor(out_conv_2)
        out_pool_2 = ((out_conv_2 - 1 * (self.kernel_2 - 1) - 1) / self.stride) + 1
        out_pool_2 = math.floor(out_pool_2)

        # Calculate size of convolved/pooled features for convolution_3/max_pooling_3 features
        out_conv_3 = ((self.embedding_size - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_conv_3 = math.floor(out_conv_3)
        out_pool_3 = ((out_conv_3 - 1 * (self.kernel_3 - 1) - 1) / self.stride) + 1
        out_pool_3 = math.floor(out_pool_3)

        # Calculate size of convolved/pooled features for convolution_4/max_pooling_4 features
        out_conv_4 = ((self.embedding_size - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_conv_4 = math.floor(out_conv_4)
        out_pool_4 = ((out_conv_4 - 1 * (self.kernel_4 - 1) - 1) / self.stride) + 1
        out_pool_4 = math.floor(out_pool_4)

        # Returns "flattened" vector (input for fully connected layer)
        # + space for additional non-text post features
        output_size = (out_pool_1 + out_pool_2 + out_pool_3 + out_pool_4) * \
                self.out_size + len(self.extractor.nontext_inds)
        return output_size * 2
