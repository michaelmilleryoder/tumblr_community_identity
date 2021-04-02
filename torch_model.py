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

from ffn import FFNTextClassifier
from cnn import CNNTextClassifier


class TorchModel():
    """ Class to invoke one of possible torch models """

    def __init__(self, clf_type: str, extractor, use_cuda: bool = False):
        """ Args:
               clf_type: classifier type
               extractor: FeatureExtractor, for the parameters
        """
        self.clf_type = clf_type
        self.extractor = extractor
        #self.epochs = epochs
        self.clfs = {
            'cnn': CNNTextClassifier,
            'ffn': FFNTextClassifier
        }
        self.use_cuda = use_cuda
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.clf = self.clfs[self.clf_type](extractor, device).to(device)
        #if self.use_cuda:
        #    self.clf = MyDataParallel(self.clf)


class MyDataParallel(nn.DataParallel):
    """ Class to still be able to access attributes """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


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
        y_train, y_dev, criterion):
    """ Run one epoch of training.
    """

    # Set model in training model
    model.train()
    log_fpath = f'../log/{model.clf_type}{model.name}.csv'
    if epoch + 1 == 1:
        print(f'Logging to {log_fpath}')
        with open(log_fpath, 'w') as f:
            f.write('datetime,epoch,iteration,loss,train_accuracy,dev_accuracy\n')
    epochloss = 0

    # Starts batch training
    for i, (x_batch, y_batch) in enumerate(loader_train):

        # Clean gradients
        optimizer.zero_grad()

        y_batch = y_batch.type(torch.FloatTensor).to(model.device)

        # Feed the model
        y_pred = model(x_batch.to(model.device))

        # Loss calculation
        loss = criterion(y_pred, y_batch)
        epochloss += loss.item() * x_batch.size(0) 

        # Gradients calculation
        loss.backward()

        # Gradients update
        optimizer.step()

        # Metrics calculation
        #if (i+1) % 500 == 0 or i+1 == len(loader_train):
        if i+1 == len(loader_train):
            print("[%s] Epoch: %d, iter: %d, loss: %.3f" 
                % (datetime.datetime.now().strftime(
                "%Y-%m-%d %H:%M"), epoch+1, i+1, epochloss/len(y_train)))
            with open(log_fpath, 'a') as f:
                f.write(','.join([
                    datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
                    str(epoch+1), str(i+1), str(loss.item())]) + '\n')

            # Evaluation phase
            #dev_predictions = evaluation(model, loader_dev)
            #train_predictions = evaluation(model, loader_train)

            #train_accuracy = calculate_accuracy(y_train, train_predictions)
            #dev_accuracy = calculate_accuracy(y_dev, dev_predictions)
            #print("[%s] Epoch: %d, iter: %d, loss: %.3f, train acc: %.4f, " 
            #    "dev acc: %.4f" % (datetime.datetime.now().strftime(
            #    "%Y-%m-%d %H:%M"), epoch+1, i+1, loss.item(), train_accuracy, 
            #    dev_accuracy))
            #with open(log_fpath, 'a') as f:
            #    f.write(','.join([
            #        datetime.datetime.now().strftime('%Y-%m-%d %H:%M'),
            #        str(epoch+1), str(i+1), str(loss.item()), str(train_accuracy), 
            #        str(dev_accuracy)]) + '\n')


def test_model(model, loader_test, y_test):
    """ Print final accuracy on a test set """

    # Evaluation phase
    test_predictions = evaluation(model, loader_test)

    # Metrics calculation
    test_accuracy = calculate_accuracy(y_test, test_predictions)
    print("Test accuracy: %.5f" % test_accuracy)
    return test_accuracy
