# -*- coding:utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, precision_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder


def read_file(path):
    # load the csv file as a dataframe
    df = pd.read_csv(path, header=None)
    X = df.values[:, :-1]
    y = df.values[:, -1]
    # ensure input data is floats
    X = X.astype('float32')
    # label encode target and ensure the values are floats
    y = LabelEncoder().fit_transform(y)
    y = y.astype('float32')
    y = y.reshape((len(y), 1))
    return X, y


def main():
    path = "shuffled_all_sample_lncRNADisease2.0.csv"
    dataMat, labelMat = read_file(path)
    X = np.array(dataMat)
    Y = np.array(labelMat)

    # 划分训练集和测试集
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)

    # 创建一个随机森林分类器的实例
    model = RandomForestClassifier(random_state=42, n_estimators=100)

    # 利用训练集样本对分类器模型进行训练
    model.fit(x_train, y_train)
    mytest(model, x_test, y_test)
    myauc(X, Y)
    myaupr(X, Y)


def mytest(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Accuracy:")
    print(accuracy_score(y_test, y_pred))
    print("f1:")
    print(f1_score(y_test, y_pred))
    print("precision:")
    print(precision_score(y_test, y_pred))
    y_pred = model.predict_proba(x_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
    print(auc(fpr, tpr))
    fpr1, tpr1, thresholds1 = precision_recall_curve(y_test, y_pred[:, 1])
    print(auc(tpr1, fpr1))


def myauc(X, Y):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 1000)
    i = 0
    KF = KFold(n_splits=5)
    data = X
    label = Y
    # data为数据集,利用KF.split划分训练集和测试集
    for train_index, test_index in KF.split(data):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = label[train_index], label[test_index]
        # 建立模型(模型已经定义)
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        # 训练模型
        model.fit(X_train, Y_train)
        # 利用model.predict获取测试集的预测值
        y_pred = model.predict_proba(X_test)
        # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
        fpr, tpr, thresholds = roc_curve(Y_test, y_pred[:, 1])
        # interp:插值 把结果添加到tprs列表中
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(AUC=%0.4f)' % (i + 1, roc_auc))
        i += 1

    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC=%0.4f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC')
    plt.legend(loc='lower right')
    plt.show()


def myaupr(X, Y):
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    i = 0
    KF = KFold(n_splits=5)
    data = X
    label = Y
    # data为数据集,利用KF.split划分训练集和测试集
    for train_index, test_index in KF.split(data):
        # 建立模型，并对训练集进行测试，求出预测得分
        # 划分训练集和测试集
        X_train, X_test = data[train_index], data[test_index]
        Y_train, Y_test = label[train_index], label[test_index]
        # 建立模型(模型已经定义)
        model = RandomForestClassifier(random_state=42, n_estimators=100)
        # 训练模型
        model.fit(X_train, Y_train)
        # 利用model.predict获取测试集的预测值
        y_pred = model.predict_proba(X_test)
        # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
        fpr, tpr, thresholds = precision_recall_curve(Y_test, y_pred[:, 1])
        # interp:插值 把结果添加到tprs列表中
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        # 计算auc
        roc_auc = auc(tpr, fpr)
        aucs.append(roc_auc)
        # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label='fold %d(AUPR=%0.4f)' % (i + 1, roc_auc))
        i += 1

    # 画对角线
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean (AUPR=%0.4f)' % mean_auc, lw=2, alpha=.8)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='gray', alpha=.2)
    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision Recall Curve')
    plt.legend(loc='lower right')
    plt.show()


if __name__ == "__main__":
    main()
