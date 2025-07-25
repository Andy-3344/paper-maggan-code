import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
import math
import random
from numpy import *


# 读取文件
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return


# 写入文件
def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


OriginalData = pd.read_csv("data1.csv", header=None, encoding='ansi')
OriginalData = np.array(OriginalData)
OriginalData = OriginalData.tolist()
# ReadMyCsv(OriginalData, "D:\论文\seqGAN_pytorch-master\seqGAN_pytorch-master\data\LncRNADisease\original "
#                        "data\LncRNADisease.csv")
print(len(OriginalData))

# 预处理
LncDisease = OriginalData

# LncDisease去重，数据集有问题
NewLncDisease = []
counter = 0
while counter < len(LncDisease):
    flag = 0
    counter1 = 0
    while counter1 < len(NewLncDisease):
        if (LncDisease[counter][0] == NewLncDisease[counter1][0]) & (
                LncDisease[counter][1] == NewLncDisease[counter1][1]):
            flag = 1
            break
        counter1 = counter1 + 1
    if flag == 0:
        NewLncDisease.append(LncDisease[counter])
    counter = counter + 1
LncDisease = []
LncDisease = NewLncDisease
print('去重LncDisease')
print('LncDisease长度', len(LncDisease))
storFile(LncDisease, 'LncDisease.csv')

# 构建AllDisease
AllDisease = []
counter1 = 0
while counter1 < len(OriginalData):  # 顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllDisease):  # 遍历AllDisease
        if OriginalData[counter1][1] != AllDisease[counter2]:  # 有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllDisease[counter2]:  # 没有新疾病，用两个if第二个if会越界
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllDisease.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllDisease)', len(AllDisease))
AllDisease1 = pd.DataFrame(AllDisease)
AllDisease1.to_csv('ALLDisease.csv')

# 构建AllRNA
AllRNA = []
counter1 = 0
while counter1 < len(OriginalData):  # 顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllRNA):  # 遍历AllDisease
        if OriginalData[counter1][0] != AllRNA[counter2]:  # 有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllRNA[counter2]:  # 没有新疾病，用两个if第二个if会越界
            flag = 1
            break
    if flag == 0:
        AllRNA.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
print('len(AllRNA)', len(AllRNA))
AllRNA1 = pd.DataFrame(AllRNA)
AllRNA1.to_csv('ALLRNA.csv')

# 由rna-disease生成对应关系矩阵，有关系1，没关系0，行为疾病AllDisease，列为rna AllRNA
# 生成全0矩阵
DiseaseAndRNABinary = []
counter = 0
while counter < len(AllDisease):
    row = []
    counter1 = 0
    while counter1 < len(AllRNA):
        row.append(0)
        counter1 = counter1 + 1
    DiseaseAndRNABinary.append(row)
    counter = counter + 1

print('len(LncDisease)', len(LncDisease))
counter = 0
while counter < len(LncDisease):
    DN = LncDisease[counter][1]  # disease name
    RN = LncDisease[counter][0]  # rna name
    counter1 = 0
    while counter1 < len(AllDisease):
        if AllDisease[counter1] == DN:
            counter2 = 0
            while counter2 < len(AllRNA):
                if AllRNA[counter2] == RN:
                    DiseaseAndRNABinary[counter1][counter2] = 1
                    break
                counter2 = counter2 + 1
            break
        counter1 = counter1 + 1
    counter = counter + 1
print('len(DiseaseAndRNABinary)', len(DiseaseAndRNABinary))
storFile(DiseaseAndRNABinary, 'DiseaseAndRNABinary.csv')

# # 计算LncRNA rd
# counter1 = 0
# sum1 = 0
# while counter1 < (len(AllDisease)):
#     counter2 = 0
#     while counter2 < (len(AllRNA)):
#         sum1 = sum1 + pow((DiseaseAndRNABinary[counter1][counter2]), 2)
#         counter2 = counter2 + 1
#     counter1 = counter1 + 1
# print('sum1=', sum1)
# Ak = sum1
# Nd = len(AllDisease)
# rdpie = 0.5
# rd = rdpie * Nd / Ak
# print('disease rd', rd)
#
# # 生成DiseaseGaussian
# DiseaseGaussian = []
# counter1 = 0
# while counter1 < len(AllDisease):  # 计算疾病counter1和counter2之间的similarity
#     counter2 = 0
#     DiseaseGaussianRow = []
#     while counter2 < len(AllDisease):  # 计算Ai*和Bj*
#         AiMinusBj = 0
#         sum2 = 0
#         counter3 = 0
#         AsimilarityB = 0
#         while counter3 < len(AllRNA):  # 疾病的每个属性分量
#             sum2 = pow((DiseaseAndRNABinary[counter1][counter3] - DiseaseAndRNABinary[counter2][counter3]), 2)  # 计算平方
#             AiMinusBj = AiMinusBj + sum2
#             counter3 = counter3 + 1
#         AsimilarityB = math.exp(- (AiMinusBj / rd))
#         DiseaseGaussianRow.append(AsimilarityB)
#         counter2 = counter2 + 1
#     DiseaseGaussian.append(DiseaseGaussianRow)
#     counter1 = counter1 + 1
# print('len(DiseaseGaussian)', len(DiseaseGaussian))
# storFile(DiseaseGaussian, 'DiseaseGaussian.csv')
#
# # 计算RNA rd
# MDiseaseAndRNABinary = np.array(DiseaseAndRNABinary)  # 列表转为矩阵
# RNAAndDiseaseBinary = MDiseaseAndRNABinary.T  # 转置DiseaseAndMiRNABinary
# counter1 = 0
# sum1 = 0
# while counter1 < (len(AllRNA)):  # rna数量
#     counter2 = 0
#     while counter2 < (len(AllDisease)):  # disease数量
#         sum1 = sum1 + pow((RNAAndDiseaseBinary[counter1][counter2]), 2)
#         counter2 = counter2 + 1
#     counter1 = counter1 + 1
# print('sum1=', sum1)
# Ak = sum1
# Nm = len(AllRNA)
# rdpie = 0.5
# rd = rdpie * Nm / Ak
# print('RNA rd', rd)
#
# # 生成RNAGaussian
# RNAGaussian = []
# counter1 = 0
# while counter1 < len(AllRNA):  # 计算rna counter1和counter2之间的similarity
#     counter2 = 0
#     RNAGaussianRow = []
#     while counter2 < len(AllRNA):  # 计算Ai*和Bj*
#         AiMinusBj = 0
#         sum2 = 0
#         counter3 = 0
#         AsimilarityB = 0
#         while counter3 < len(AllDisease):  # rna的每个属性分量
#             sum2 = pow((RNAAndDiseaseBinary[counter1][counter3] - RNAAndDiseaseBinary[counter2][counter3]), 2)
#             AiMinusBj = AiMinusBj + sum2
#             counter3 = counter3 + 1
#         AsimilarityB = math.exp(- (AiMinusBj / rd))
#         RNAGaussianRow.append(AsimilarityB)
#         counter2 = counter2 + 1
#     RNAGaussian.append(RNAGaussianRow)
#     counter1 = counter1 + 1
# storFile(RNAGaussian,
#          'RNAGaussian.csv')
