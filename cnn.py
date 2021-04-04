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


class CNNTextClassifier(nn.Module):
    """ From https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch/
        blob/master/src/model/model.py """

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

        # Dropout definition
        #self.dropout = nn.Dropout(0.25)

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
