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
# 计算LncRNA rd
print(len(AllDisease))
counter1 = 0
sum1 = 0
while counter1 < (len(AllDisease)):
    counter2 = 0
    while counter2 < (len(AllRNA)):
        sum1 = sum1 + pow((DiseaseAndRNABinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nd = len(AllDisease)
rdpie = 0.5
rd = rdpie * Nd / Ak
print('disease rd', rd)

# 生成DiseaseGaussian
DiseaseGaussian = []
counter1 = 0
while counter1 < len(AllDisease)-1:  # 计算疾病counter1和counter2之间的similarity
    counter2 = 0
    DiseaseGaussianRow = []
    while counter2 < len(AllDisease)-1:  # 计算Ai*和Bj*
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllRNA)-1:  # 疾病的每个属性分量
            sum2 = pow((DiseaseAndRNABinary[counter1][counter3] - DiseaseAndRNABinary[counter2][counter3]), 2)  # 计算平方
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj / rd))
        DiseaseGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    DiseaseGaussian.append(DiseaseGaussianRow)
    counter1 = counter1 + 1
print('len(DiseaseGaussian)', len(DiseaseGaussian))
DiseaseGaussian = pd.DataFrame(DiseaseGaussian)
DiseaseGaussian.to_csv('DGS.csv', header=None, index=None)