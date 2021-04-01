"""
Feedforward network in PyTorch

@author Michael Miller Yoder
@date 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFNTextClassifier(nn.Module):
    """ Feedforward NN classifier based on mean embeddings for 
        post hashtags or blog descriptions """

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

        # Dropout definition
        #self.dropout = nn.Dropout(0.25)
        #self.dropout = nn.Dropout(0)

        ## Embedding layer definition
        #weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
        #zeros = torch.zeros(1, self.embedding_size).to(device) # embedding for 0 pad
        #weights_with_padding = torch.cat((zeros, weights), 0).to(device)
        #self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
        #    padding_idx=0).to(device)

        #self.fc = nn.Linear(1042, 1).to(device)
        self.fc1 = nn.Linear(18, 18).to(device) # for non-text baseline
        self.fc2 = nn.Linear(18, 9).to(device)
        self.fc3 = nn.Linear(9, 1).to(device)
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
        x_add = x[:,np.array(self.extractor.nontext_inds)].float()

        #x_reblog_text = self.embedding(x_reblog_text)
        #mean_reblog_text = torch.mean(x_reblog_text, 1, False)
        #x_nonreblog_text = self.embedding(x_nonreblog_text)
        #mean_nonreblog_text = torch.mean(x_nonreblog_text, 1, False)

        #union1 = torch.cat((mean_reblog_text, mean_nonreblog_text), 1)
        #union1 = union1.reshape(union1.size(0), -1)
        #flattened = torch.cat((union1, x_add), 1)
        flattened = torch.FloatTensor(x_add.float())
    
        # The "flattened" vector is passed through a fully connected layer
        out = self.fc1(flattened)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        out = torch.relu(out)

        # The "flattened" vector is passed through a fully connected layer
        out = self.fc2(out)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        out = torch.relu(out)
        
        # The "flattened" vector is passed through a fully connected layer
        out = self.fc3(out)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        out = torch.relu(out)
        
        # The "flattened" vector is passed through a fully connected layer
        #out = self.fc4(out)
        # Dropout is applied        
        #out = self.dropout(out)
        # Activation function is applied
        #out = torch.relu(out)
        
        return out.squeeze()

