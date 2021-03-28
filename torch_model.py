"""
Definitions for PyTorch models.

@author Michael Miller Yoder
@date 2021
"""

import math
import datetime
import pdb

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset


class TorchModel():
    """ Class to invoke one of possible torch models """

    def __init__(self, clf_type: str, extractor, epochs, use_cuda: bool = False):
        """ Args:
               clf_type: classifier type
               extractor: FeatureExtractor, for the parameters
        """
        self.clf_type = clf_type
        self.extractor = extractor
        self.epochs = epochs
        self.clfs = {
            'cnn': CNNTextClassifier
        }
        self.use_cuda = use_cuda
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.clf = self.clfs[self.clf_type](extractor, device, self.epochs).to(device)
        if self.use_cuda:
            self.clf = MyDataParallel(self.clf)


class MyDataParallel(nn.DataParallel):
    """ Class to still be able to access attributes """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class CNNTextClassifier(nn.Module):
    """ From https://github.com/FernandoLpz/Text-Classification-CNN-PyTorch/
        blob/master/src/model/model.py """

    def __init__(self, extractor, device, epochs):
        """ Args:
                extractor: FeatureExtractor, for the parameters
        """

        super().__init__()
        self.extractor = extractor
        self.device = device
        self.name = 'model' + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")

        # Parameters regarding text preprocessing
        self.word_embs = self.extractor.word_embs
        self.seq_len = self.extractor.padding_size * 6
        self.num_words = len(self.extractor.word_embs.wv.vocab)
        self.embedding_size = self.extractor.word_embs.vector_size

        # Dropout definition
        self.dropout = nn.Dropout(0.25)

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

        # Training parameters
        self.epochs = epochs
        self.batch_size = 128
        self.learning_rate = 0.001

        """ ORIGINAL
        # Embedding layer definition
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
        self.fc = nn.Linear(self.in_features_fc(), 1)
        """

        # DEBUG
        # Embedding layer definition
        weights = torch.FloatTensor(self.word_embs.wv.vectors).to(device)
        zeros = torch.zeros(1, self.embedding_size).to(device)
        weights_with_padding = torch.cat((zeros, weights), 0).to(device)
        self.embedding = nn.Embedding.from_pretrained(weights_with_padding,
            padding_idx=0).to(device)

        self.fc = nn.Linear(146, 1).to(device)

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
        return output_size
    
    def forward(self, x):
        """ Called indirectly through model(input)? """

        """ ORIGINAL
        # Separate out text features for CNN and additional features
        x_text = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds])]
        x_add = x[:,np.array(self.extractor.nontext_inds)]

        # Sequence of tokens is filtered through an embedding layer
        x = self.embedding(x_text)
        
        # Convolution layer 1 is applied
        x1 = self.conv_1(x)
        x1 = torch.relu(x1)
        x1 = self.pool_1(x1)
        
        # Convolution layer 2 is applied
        x2 = self.conv_2(x)
        x2 = torch.relu((x2))
        x2 = self.pool_2(x2)
        
        # Convolution layer 3 is applied
        x3 = self.conv_3(x)
        x3 = torch.relu(x3)
        x3 = self.pool_3(x3)
        
        # Convolution layer 4 is applied
        x4 = self.conv_4(x)
        x4 = torch.relu(x4)
        x4 = self.pool_4(x4)
        
        # The output of each convolutional layer is concatenated into a unique vector
        union = torch.cat((x1, x2, x3, x4), 2) # add post notes and post type here
        union = union.reshape(union.size(0), -1)
        flattened = torch.cat((union, x_add), 1)
        """

        # DEBUG
        x_reblog_text = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i in self.extractor.reblog_inds])]
        x_nonreblog_text = x[:,np.array([i for i in range(x.shape[1]) if \
            i not in self.extractor.nontext_inds and \
            i not in self.extractor.reblog_inds])]
        #x_text = x[:,np.array([i for i in range(x.shape[1]) if \
        #    i not in self.extractor.nontext_inds])]
        x_add = x[:,np.array(self.extractor.nontext_inds)]

        x_reblog_text = self.embedding(x_reblog_text)
        mean_reblog_text = torch.mean(x_reblog_text, 1, False)
        x_nonreblog_text = self.embedding(x_nonreblog_text)
        mean_nonreblog_text = torch.mean(x_nonreblog_text, 1, False)
        flattened = torch.cat((mean_reblog_text, mean_nonreblog_text, x_add), 1)
    
        # The "flattened" vector is passed through a fully connected layer
        out = self.fc(flattened)
        # Dropout is applied        
        out = self.dropout(out)
        # Activation function is applied
        out = torch.sigmoid(out)
        
        return out.squeeze()


class DatasetMapper(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y
      
    def __len__(self):
        return len(self.x)
       
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def evaluation(model, loader_test):
        
    # Set the model in evaluation mode
    model.eval()
    predictions = []

    # Start evaluation phase
    with torch.no_grad():
        for x_batch, y_batch in loader_test:
            y_pred = model(x_batch.to(model.device))
            if model.device.type.startswith('cuda'):
                predictions += list(y_pred.detach().cpu().numpy())
            else:
                predictions += list(y_pred.detach().numpy())
    return predictions

def calculate_accuracy(grand_truth, predictions):
    # Metrics calculation
    true_positives = 0
    true_negatives = 0
    for true, pred in zip(grand_truth, predictions):
        if (pred >= 0.5) and (true == 1):
            true_positives += 1
        elif (pred < 0.5) and (true == 0):
            true_negatives += 1
        else:
            pass
    # Return accuracy
    return (true_positives+true_negatives) / len(grand_truth)


def train_epoch(epoch, model, loader_train, loader_dev, optimizer, 
        y_train, y_dev):
    """ Run one epoch of training.
    """

    # Set model in training model
    model.train()
    predictions = []
    log_fpath = f'../log/{model.name}.csv'
    if epoch + 1 == 1:
        with open(log_fpath, 'w') as f:
            f.write('datetime,epoch,iteration,loss,train_accuracy,dev_accuracy\n')

    # Starts batch training
    for i, (x_batch, y_batch) in enumerate(loader_train):

        y_batch = y_batch.type(torch.FloatTensor).to(model.device)

        # Feed the model
        y_pred = model(x_batch.to(model.device))

        # Loss calculation
        loss = F.binary_cross_entropy(y_pred, y_batch)

        # Clean gradients
        optimizer.zero_grad()

        # Gradients calculation
        loss.backward()

        # Gradients update
        optimizer.step()

        # Save predictions
        predictions += list(y_pred.detach().cpu().numpy())
        
        # Evaluation phase
        dev_predictions = evaluation(model, loader_dev)

        # Metrics calculation
        if (i+1) % 50 == 0:
            train_accuracy = calculate_accuracy(y_train, predictions)
            dev_accuracy = calculate_accuracy(y_dev, dev_predictions)
            print("[%s] Epoch: %d, iter: %d, loss: %.3f, train acc: %.4f, " 
                "dev acc: %.4f" % (datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M"), epoch+1, i+1, loss.item(), train_accuracy, 
                dev_accuracy))
            with open(log_fpath, 'a') as f:
                f.write(','.join([
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    str(epoch+1), str(i+1), str(loss.item()), str(train_accuracy), 
                    str(dev_accuracy)]) + '\n')


def test_model(model, loader_test, y_test):
    """ Print final accuracy on a test set """

    # Evaluation phase
    test_predictions = evaluation(model, loader_test)

    # Metrics calculation
    test_accuracy = calculate_accuracy(y_test, test_predictions)
    print("Test accuracy: %.5f" % test_accuracy)
    return test_accuracy
