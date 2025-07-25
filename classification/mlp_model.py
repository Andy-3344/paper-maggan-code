# -*- coding: utf-8 -*-
# pytorch mlp for binary classification
import math

import torch.nn
from numpy import vstack
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from torch import Tensor
from torch.optim import SGD
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn import Linear, ReLU, Sigmoid, Module, BCELoss, CrossEntropyLoss
from torch.nn.init import kaiming_uniform_, xavier_uniform_
import numpy as np
import pickle
from sklearn import preprocessing

embedding_size = 32
df = read_csv("shuffled_all_sample_old.csv", header=None)
print(df)


# dataset definition
class CSVDataset(Dataset):
    # load the dataset
    def __init__(self, path):
        # load the csv file as a dataframe
        df = read_csv(path)
        # store the inputs and outputs
        self.X = df.values[:, :-1]
        min_max_scaler = preprocessing.MinMaxScaler()
        self.X = min_max_scaler.fit_transform(self.X)
        # print(self.X)
        self.y = df.values[:, -1]
        # ensure input data is floats
        self.X = self.X.astype('float32')
        # label encode target and ensure the values are floats
        self.y = LabelEncoder().fit_transform(self.y)
        self.y = self.y.astype('float32')
        self.y = self.y.reshape((len(self.y), 1))

    # number of rows in the dataset
    def __len__(self):
        return len(self.X)

    # get a row at an index
    def __getitem__(self, idx):
        return [self.X[idx], self.y[idx]]

    # get indexes for train and test rows
    def get_splits(self):
        # determine sizes
        t = len(self.X)/32
        # test_size = round(32*(t//10)*3)
        test_size = round(len(self.X)*0.3)
        train_size = len(self.X) - test_size
        # calculate the split
        return random_split(self, [train_size, test_size])


# model definition
class MLP(Module):
    # define model elements
    def __init__(self, n_inputs):
        super(MLP, self).__init__()
        # input to first hidden layer
        # self.embed = torch.nn.Embedding(33, embedding_size)
        # self.conv1 = torch.nn.Conv1d(embedding_size, embedding_size, 5)
        # self.pool1 = torch.nn.MaxPool1d(3)
        # self.conv2 = torch.nn.Conv1d(embedding_size, embedding_size, 5)
        # self.pool2 = torch.nn.MaxPool1d(3)
        self.hidden1 = Linear(n_inputs, 122)
        kaiming_uniform_(self.hidden1.weight, nonlinearity='relu')
        self.act1 = ReLU()
        # second hidden layer
        self.hidden2 = Linear(122, 32)
        kaiming_uniform_(self.hidden2.weight, nonlinearity='relu')
        self.act2 = ReLU()
        # third hidden layer and output
        self.hidden3 = Linear(32, 8)
        kaiming_uniform_(self.hidden3.weight, nonlinearity='relu')
        self.act3 = ReLU()
        # third hidden layer and output
        self.hidden4 = Linear(8, 1)
        xavier_uniform_(self.hidden4.weight)
        self.act4 = Sigmoid()

    # forward propagate input
    def forward(self, X):
        # input to first hidden layer
        # X = self.embed(X)
        # X = self.conv1(X)
        # X = self.pool1(X)
        # X = self.conv2(X)
        # X = self.pool2(X)
        X = self.hidden1(X)
        X = self.act1(X)
        # second hidden layer
        X = self.hidden2(X)
        X = self.act2(X)
        # third hidden layer
        X = self.hidden3(X)
        X = self.act3(X)
        # fourth hidden layer and output
        X = self.hidden4(X)
        X = self.act4(X)
        return X


# prepare the dataset
def prepare_data(path):
    # load the dataset
    dataset = CSVDataset(path)
    # calculate split
    train, test = dataset.get_splits()
    # prepare data loaders
    train_dl = DataLoader(train, batch_size=32, shuffle=True)
    test_dl = DataLoader(test, batch_size=32, shuffle=False)
    return train_dl, test_dl


# train the model
def train_model(train_dl, model):
    # define the optimization
    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=0.01, momentum=0.9)
    # enumerate epochs
    for epoch in range(10):
        # enumerate mini batches
        for i, (inputs, targets) in enumerate(train_dl):
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            print("epoch: {}, batch: {}, loss: {}".format(epoch, i, loss.data))
            # update model weights
            optimizer.step()


temp = 0


# evaluate the model
def evaluate_model(test_dl, model):
    global temp
    test_loss = 0
    correct = 0

    TP, FN, FP, TN = 0, 0, 0, 0
    predictions, actuals = [], []
    for i, (inputs, targets) in enumerate(test_dl):
        # evaluate the model on the test set
        yhat = model(inputs)
        # retrieve numpy array
        yhat = yhat.detach().numpy()
        actual = targets.numpy()
        actual = actual.reshape((len(actual), 1))
        # round to class values
        yhat = yhat.round()
        # store
        predictions.append(yhat)
        actuals.append(actual)
    predictions, actuals = vstack(predictions), vstack(actuals)
    print(predictions)
    print(actuals)
    # calculate accuracy
    item_ground_truth_list = []
    item_predict_list = []
    for item in actuals:
        item_ground_truth_list.append(item.item())
    for item in predictions:
        item_predict_list.append(item.item())

    for i in range(0, len(item_ground_truth_list)):
        item_ground_truth_value = item_ground_truth_list[i]
        item_predict_value = item_predict_list[i]
        if item_ground_truth_value == 1 and item_predict_value == 1:
            TP += 1
            # print('true', item_ground_truth_list)
            # print('pred', item_predict_list)
        if item_ground_truth_value == 1 and item_predict_value == 0:
            FN += 1
        if item_ground_truth_value == 0 and item_predict_value == 1:
            FP += 1
        if item_ground_truth_value == 0 and item_predict_value == 0:
            TN += 1

    correct += predictions.__eq__(actuals.data).sum()

    threshold = 0  # 阈值
    TPR_list = []  # ROC 的纵坐标 真阳率
    FPR_list = []
    while threshold < 1.000:
        FPR = FP / (FP + TN)
        TPR = TP / (TP + FN)
        TPR_list.append(TPR)
        FPR_list.append(FPR)
        threshold += 0.005

    AUC = 0
    for i in range(0, len(TPR_list)):
        AUC += 0.005 * TPR_list[i]

    ACC = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / (TP + FN)
    precision = TP / (TP + FP + 0.000007)
    F1 = (2.0 * precision * recall) / (precision + recall + 0.000007)
    SE = TP / (TP + FN)
    SP = TN / (TN + FP)
    K1 = (TP + FP) * (TP + FN) * (TN + FP) * (TN + FN)
    MCC = ((TP * TN) - (FP * FN)) / (math.sqrt(K1)+0.000007)
    TPR = TP / (TP + FN)
    TNR = TN / (TN + FP)
    PPV = TP / (TP + FP+0.000007)

    temp = temp + ACC
    test_loss /= len(test_dl)
    print('准确率为：' + str(ACC))
    print('召回率为：' + str(recall))
    print('精确率为：' + str(precision))
    print('F1为：' + str(F1))
    print('敏感度为：' + str(SE))
    print('特异性为：' + str(SP))
    print("MCC为：" + str(MCC))
    print("AUC为 : ", str(AUC))
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.6f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset),
    #     100. * correct / len(test_loader.dataset)))
    print('TPR为：' + str(TPR))
    print('TNR为：' + str(TNR))
    print('PPV为：' + str(PPV))
    acc = accuracy_score(actuals, predictions)
    return acc


# prepare the data
path = 'shuffled_all_sample.csv'
train_dl, test_dl = prepare_data(path)
print(len(train_dl.dataset), len(test_dl.dataset))
# define the network
model = MLP(256)
print(model)
# train the model
train_model(train_dl, model)
# save the model
with open('my_model.pkl', 'wb') as f:
    pickle.dump(model, f)
# evaluate the model
acc = evaluate_model(test_dl, model)
print('Accuracy: %.3f' % acc)
