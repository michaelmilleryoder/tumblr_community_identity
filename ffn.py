"""
Feedforward network in PyTorch

@author Michael Miller Yoder
@date 2021
"""

import datetime
import pdb

import numpy as np
import torch
import torch.nn as nn


class FFNTextClassifier(nn.Module):
    """ Feedforward NN classifier for mean embeddings
        of post hashtags and nontext features """

    def __init__(self, extractor, device):
        """ Args:
                extractor: FeatureExtractor, for the parameters
        """

        super().__init__()
        self.extractor = extractor
        self.device = device
        self.clf_type = 'ffn'
        self.name = 'model' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Parameters regarding text preprocessing
        self.word_embs = self.extractor.word_embs
        self.seq_len = self.extractor.padding_size
        self.num_words = len(self.extractor.word_embs.wv.vocab)
        self.embedding_size = self.extractor.word_embs.vector_size

        ## Embedding layer definition
        weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
        # embedding for 0 pad
        zeros = torch.zeros(1, self.embedding_size).to(device) 
        weights_with_padding = torch.cat((zeros, weights), 0).to(device)
        self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
            padding_idx=0).to(device)

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(146, 64).to(device)
        self.fc2 = nn.Linear(64, 32).to(device)
        self.fc3 = nn.Linear(32, 1).to(device)
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
        mean_reblog_text = torch.mean(x_reblog_text, 1, False)
        x_nonreblog_text = self.embedding(x_nonreblog_text)
        mean_nonreblog_text = torch.mean(x_nonreblog_text, 1, False)

        union1 = torch.cat((mean_reblog_text, mean_nonreblog_text), 1)
        union1 = union1.reshape(union1.size(0), -1)
        flattened = torch.cat((union1, x_add), 1)

        # The "flattened" vector is passed through fully connected layers
        out = self.fc1(flattened)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        
        return out.squeeze()

class FFNSimpleClassifier(nn.Module):
    """ Feedforward NN classifier for 
        post nontext features """

    def __init__(self, extractor, device):
        """ Args:
                extractor: FeatureExtractor, for the parameters
        """

        super().__init__()
        self.extractor = extractor
        self.device = device
        self.clf_type = 'ffn'
        self.name = 'model' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Parameters regarding text preprocessing
        self.post_nontext_only = self.extractor.post_nontext_only # only nontext feats
        self.word_embs = self.extractor.word_embs
        self.seq_len = self.extractor.padding_size
        self.num_words = len(self.extractor.word_embs.wv.vocab)
        self.embedding_size = self.extractor.word_embs.vector_size

        if self.post_nontext_only:
            self.fc1 = nn.Linear(18, 6).to(device)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(6, 1).to(device)
            self.sigmoid = nn.Sigmoid()        

        else:
            ## Embedding layer definition
            weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
            # embedding for 0 pad
            zeros = torch.zeros(1, self.embedding_size).to(device) 
            weights_with_padding = torch.cat((zeros, weights), 0).to(device)
            self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
                padding_idx=0).to(device)

            self.fc1 = nn.Linear(1042, 512).to(device)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(512, 1).to(device)
            self.sigmoid = nn.Sigmoid()        

            #self.fc3 = nn.Linear(9, 1).to(device)
            #self.fc1 = nn.Linear(146, 146).to(device)
            #self.fc2 = nn.Linear(146, 64).to(device)
            #self.fc3 = nn.Linear(64, 32).to(device)
            #self.fc4 = nn.Linear(32, 1).to(device)

    def forward(self, x):
        """ Called indirectly through model(input) """

        #x_reblog_text = x[:,np.array([i for i in range(x.shape[1]) if \
        #    i not in self.extractor.nontext_inds and \
        #    i in self.extractor.reblog_inds])]
        #x_nonreblog_text = x[:,np.array([i for i in range(x.shape[1]) if \
        #    i not in self.extractor.nontext_inds and \
        #    i not in self.extractor.reblog_inds])]
        #x_add = x[:,np.array(self.extractor.nontext_inds)].float()

        #x_reblog_text = self.embedding(x_reblog_text)
        #mean_reblog_text = torch.mean(x_reblog_text, 1, False)
        #x_nonreblog_text = self.embedding(x_nonreblog_text)
        #mean_nonreblog_text = torch.mean(x_nonreblog_text, 1, False)

        #union1 = torch.cat((mean_reblog_text, mean_nonreblog_text), 1)
        #union1 = union1.reshape(union1.size(0), -1)
        #flattened = torch.cat((union1, x_add), 1)
        #flattened = torch.FloatTensor(x_add.float())

        hidden = self.fc1(torch.FloatTensor(x.float()))
        relu = self.relu(hidden)
        output = self.fc2(relu)
        output = self.sigmoid(output)
        return output.squeeze()
    
        # The "flattened" vector is passed through a fully connected layer
        #out = self.fc1(flattened)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        #out = torch.relu(out)

        # The "flattened" vector is passed through a fully connected layer
        #out = self.fc2(out)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        #out = torch.relu(out)
        
        # The "flattened" vector is passed through a fully connected layer
        #out = self.fc3(out)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        #out = torch.relu(out)
        
        # The "flattened" vector is passed through a fully connected layer
        #out = self.fc4(out)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        #out = torch.relu(out)
        
        #return out.squeeze()
