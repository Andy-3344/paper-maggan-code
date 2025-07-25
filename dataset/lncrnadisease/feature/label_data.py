import dask.dataframe as dd
import pandas as pd
import numpy as np
import random

# 读取特征数据
data_LDA = pd.read_csv('LD_adjmat.csv', header=None)
data_L = pd.read_csv('../../lncrnadisease2.0/data result/lncRNAFeature.csv', header=None)
data_D = pd.read_csv('../../lncrnadisease2.0/data result/diseaseFeature.csv', header=None)
print(data_LDA.shape)
print(data_L.shape)
print(data_D.shape)
L = data_L.shape[0]
D = data_D.shape[0]
S = data_D.shape[1] + data_L.shape[1]
data_L = data_L.T
data_D = data_D.T

# labeled feature
data_label_feature = pd.DataFrame(columns=np.arange(S) + 1)
data_label_feature_position = pd.DataFrame(columns=np.arange(2) + 1)
count = 1
for i in range(0, L):
    for j in range(0, D):
        if data_LDA[i][j] != 0:
            data_label_feature.loc[count] = (dd.concat([data_L[i], data_D[j]], axis=0)).to_dask_array(lengths=True)
            data_label_feature_position.loc[count] = [j, i]
            count += 1
            print(count)
print(data_label_feature)

data_label_feature.to_csv('data_label_feature.csv', header=None, index=None)
data_label_feature_position.to_csv('data_label_feature_position.csv', header=None, index=None)

# unlabeled feature
data_unlabel_feature = pd.DataFrame(columns=np.arange(S) + 1)
data_unlabel_feature_position = pd.DataFrame(columns=np.arange(2) + 1)
count = 1
for i in range(0, L):
    p = range(0, D)
    rl = random.sample(p, 5)
    for j in range(5):
        if data_LDA[i][rl[j]] == 0:
            data_unlabel_feature.loc[count] = (dd.concat([data_L[i], data_D[j]], axis=0)).to_dask_array(lengths=True)
            data_unlabel_feature_position.loc[count] = [j, i]
            count += 1
print(count)

data_unlabel_feature.to_csv('data_unlabel_feature.csv', header=None, index=None)
data_unlabel_feature_position.to_csv('data_unlabel_feature_position.csv', header=None, index=None)
