"""
Definitions for PyTorch models.

@author Michael Miller Yoder
@date 2021
"""

import os
import datetime
import pickle
import pdb

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from ffn import FFNTextClassifier, FFNSimpleClassifier
from cnn import CNNTextClassifier


class RunPyTorch():
    """ Class to invoke one of possible pytorch models, store parameters """

    def __init__(self, clf_type: str, data, extractor, epochs: int, 
                    use_cuda: bool = False, debug: bool = False):
        """ Args:
               clf_type: classifier type
               extractor: FeatureExtractor, for the parameters
               data: Dataset
        """
        self.extractor = extractor
        self.clf_type = clf_type
        if self.extractor.post_nontext_only:
            self.clf_type = 'ffn_nontext'
        self.data = data
        self.epochs = epochs
        self.debug = debug
        self.subset = 100 # for debugging
        self.clfs = {
            'cnn': CNNTextClassifier,
            'ffn': FFNTextClassifier,
            'ffn_nontext': FFNSimpleClassifier
        }
        self.use_cuda = use_cuda
        if self.use_cuda:
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        self.model = self.clfs[self.clf_type](extractor, device).to(device)
        #if self.use_cuda:
        #    self.model = MyDataParallel(self.model)

        # Parameters (should make a class, or yaml config file?)
        self.run_params = {}
        self.run_params['ffn'] =  {
            'batch_size': 32,
            'learning_rate': 0.001,
            'lossfunc': nn.BCELoss()
        }
        self.run_params['ffn']['optim'] = optim.SGD(self.model.parameters(), 
            lr=self.run_params['ffn']['learning_rate'])
        self.run_params['cnn'] =  {
            'batch_size': 32,
            'learning_rate': 0.001,
            'lossfunc': nn.BCELoss()
        }
        self.run_params['cnn']['optim'] = optim.SGD(self.model.parameters(), 
            lr=self.run_params['cnn']['learning_rate'])
        self.score = None

    def run(self):
        """ Train and evaluate a PyTorch model """
        
        dev = DatasetMapper(self.data.X_dev, self.data.y_dev)
        test = DatasetMapper(self.data.X_test, self.data.y_test)

        if self.debug:
            train = DatasetMapper(self.data.X_train[:self.subset], 
                self.data.y_train[:self.subset])
        else:
            train = DatasetMapper(self.data.X_train, self.data.y_train)
            
            if self.debug:
                # Save out for debugging
                with open('/projects/tumblr_community_identity/tmp/X_train.pkl', 
                    'wb') as f:
                    pickle.dump(self.data.X_train, f)
                with open('/projects/tumblr_community_identity/tmp/y_train.pkl', 
                    'wb') as f:
                    pickle.dump(self.data.y_train, f)

        # Initialize loaders
        pin = self.use_cuda
        params = self.run_params[self.clf_type]
        loader_train = DataLoader(train, batch_size=params['batch_size'],
            pin_memory=pin)
        loader_dev = DataLoader(dev, batch_size=params['batch_size'],
            pin_memory=pin)
        loader_test = DataLoader(test, batch_size=params['batch_size'],
            pin_memory=pin)

        # Start training
        optimizer = self.run_params[self.clf_type]['optim']
        lossfunc = self.run_params[self.clf_type]['lossfunc']
        for epoch in range(self.epochs):
            if self.debug:
                train_epoch(epoch, self.model, loader_train, loader_dev, 
                    optimizer, self.data.y_train[:self.subset], self.data.y_dev,
                    lossfunc)
            else:
                train_epoch(epoch, self.model, loader_train, loader_dev, 
                    optimizer, self.data.y_train, self.data.y_dev,
                    lossfunc)

        # Test
        self.score = test_model(self.model, loader_test, self.data.y_test)

        # Save model
        outpath = os.path.join('../models/', self.clf_type + self.model.name \
             + '.model')
        torch.save(self.model.state_dict(), outpath)
        print(f"Model saved to {outpath}")

        return self.score


class MyDataParallel(nn.DataParallel):
    """ Class to still be able to access attributes """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)


class DatasetMapper(Dataset):
    """ Creates Dataset object for DataLoader """
    def __init__(self, x, y):
        self.x = x
        self.y = y
      
    def __len__(self):
        return len(self.x)
       
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def evaluation(model, loader_test):
    """ Evaluate the model """

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
            f.write('datetime,epoch,iteration,loss,dev_loss\n')
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

    # Dev loss calculation
    devloss = 0
    with torch.no_grad():
        for x_batch, y_batch in loader_dev:
            y_batch = y_batch.type(torch.FloatTensor).to(model.device)
            y_pred = model(x_batch.to(model.device))
            loss = criterion(y_pred, y_batch)
            devloss += loss.item() * x_batch.size(0) 

    # Metrics calculation
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    train_avgloss = epochloss/len(y_train)
    dev_avgloss = devloss/len(y_dev)
    print("[%s] Epoch: %d, iter: %d, loss: %.3f, dev loss: %.3f" 
        % (timestamp, epoch+1, len(loader_train), train_avgloss, dev_avgloss))
    with open(log_fpath, 'a') as f:
        f.write(','.join([timestamp, str(epoch+1), str(len(loader_train)), 
            str(train_avgloss), str(dev_avgloss)]) + '\n')

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
