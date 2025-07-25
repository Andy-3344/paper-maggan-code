# 导入需要的库
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score
import numpy as np
from sklearn.metrics import roc_curve, auc
from numpy import interp


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


le = LabelEncoder()
label_mapping = {0: 'FAKE', 1: 'TRUE'}
path = "shuffled_all_sample.csv"
dataMat = []  # 分类的数据集
labelMat = []  # 真实的分类结果
dataMat, labelMat = read_file(path)
X = np.array(dataMat)
y = np.array(labelMat)
# 将数据分为训练数据和测试数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, train_size=0.7, random_state=42)
# 训练XGBoost分类器
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
# xgb.plot_tree(model)
# 使用测试数据预测类别
y_pred = model.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
# 输出混淆矩阵
for i, true_label in enumerate(label_mapping.values()):
    row = ''
    for j, pred_label in enumerate(label_mapping.values()):
        row += f'{cm[i, j]} ({pred_label})\t'
    print(f'{row} | {true_label}')

# 输出混淆矩阵
print(classification_report(y_test, y_pred, target_names=['FAKE', 'TRUE']))  # 输出混淆矩阵
print("Accuracy:")
print(accuracy_score(y_test, y_pred))
y_pred = model.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_pred[:, 1])
print(auc(fpr, tpr))




# label_names 是分类变量的取值名称列表
label_names = ['FAKE', 'TRUE']
cm = confusion_matrix(y_test, y_pred)

# # 绘制混淆矩阵图
# fig, ax = plt.subplots()
# im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
# ax.figure.colorbar(im, ax=ax)
# ax.set(xticks=np.arange(cm.shape[1]),
#        yticks=np.arange(cm.shape[0]),
#        xticklabels=label_names, yticklabels=label_names,
#        title='Confusion matrix',
#        ylabel='True label',
#        xlabel='Predicted label')
#
# # 在矩阵图中显示数字标签
# thresh = cm.max() / 2.
# for i in range(cm.shape[0]):
#     for j in range(cm.shape[1]):
#         ax.text(j, i, format(cm[i, j], 'd'),
#                 ha="center", va="center",
#                 color="white" if cm[i, j] > thresh else "black")

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 1000)
i = 0
KF = KFold(n_splits=5)
data = X
label = y
# data为数据集,利用KF.split划分训练集和测试集
for train_index, test_index in KF.split(data):
    # 建立模型，并对训练集进行测试，求出预测得分
    # 划分训练集和测试集
    X_train, X_test = data[train_index], data[test_index]
    Y_train, Y_test = label[train_index], label[test_index]
    # 建立模型(模型已经定义)
    model = xgb.XGBClassifier()
    # 训练模型
    model.fit(X_train, Y_train)
    # 利用model.predict获取测试集的预测值
    y_pred = model.predict(X_test)
    # 计算fpr(假阳性率),tpr(真阳性率),thresholds(阈值)[绘制ROC曲线要用到这几个值]
    fpr, tpr, thresholds = roc_curve(Y_test, y_pred)
    # interp:插值 把结果添加到tprs列表中
    tprs.append(interp(mean_fpr, fpr, tpr))
    tprs[-1][0] = 0.0
    # 计算auc
    roc_auc = auc(fpr, tpr)
    aucs.append(roc_auc)
    # 画图，只需要plt.plot(fpr,tpr),变量roc_auc只是记录auc的值，通过auc()函数计算出来
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d(area=%0.2f)' % (i, roc_auc))
    i += 1

# 画对角线
plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)  # 计算平均AUC值
std_auc = np.std(aucs)
plt.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (area=%0.2f)' % mean_auc, lw=2, alpha=.8)
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

# fig.tight_layout()
# plt.show()
# plt.savefig('XGBoost_Conclusion.png')
# # 上面的代码首先计算混淆矩阵，然后使用 matplotlib 库中的 imshow 函数将混淆矩阵可视化，最后通过 text 函数在混淆矩阵上添加数字，并使用 show/savefig 函数显示图像。
# print(
#     roc_auc_score(y_test, y_pred, average='macro', sample_weight=None, max_fpr=None, multi_class='raise', labels=None))
