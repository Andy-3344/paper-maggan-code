import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import random
from numpy import *

AllDisease = pd.read_csv('preALLDisease.csv', header=None)
AllRNA = pd.read_csv('preALLRNA.csv', header=None)
DiseaseAndRNABinary = pd.read_csv('preDiseaseAndRNABinary.csv', header=None)
DiseaseAndRNABinary = np.array(DiseaseAndRNABinary)
DiseaseAndRNABinary = DiseaseAndRNABinary.tolist()
# 计算RNA rd
MDiseaseAndRNABinary = np.array(DiseaseAndRNABinary)  # 列表转为矩阵
RNAAndDiseaseBinary = MDiseaseAndRNABinary.T  # 转置DiseaseAndMiRNABinary
counter1 = 0
sum1 = 0
while counter1 < (len(AllRNA)):  # rna数量
    counter2 = 0
    while counter2 < (len(AllDisease)):  # disease数量
        sum1 = sum1 + pow((RNAAndDiseaseBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nm = len(AllRNA)
rdpie = 0.5
rd = rdpie * Nm / Ak
print('RNA rd', rd)

# 生成RNAGaussian
RNAGaussian = []
counter1 = 0
while counter1 < len(AllRNA):  # 计算rna counter1和counter2之间的similarity
    counter2 = 0
    RNAGaussianRow = []
    while counter2 < len(AllRNA):  # 计算Ai*和Bj*
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllDisease):  # rna的每个属性分量
            sum2 = pow((RNAAndDiseaseBinary[counter1][counter3] - RNAAndDiseaseBinary[counter2][counter3]), 2)
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj / rd))
        RNAGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    RNAGaussian.append(RNAGaussianRow)
    counter1 = counter1 + 1
RNAGaussian = pd.DataFrame(RNAGaussian)
RNAGaussian.to_csv("LG.csv", header=None, index=None)