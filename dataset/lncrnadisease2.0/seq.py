import pandas as pd

allRNA = pd.read_csv("ALLRNA.csv", header=None)
chart = pd.read_table("D:\论文\seqGAN_pytorch-master\seqGAN_pytorch-master\data\dNONCODE\dNONCODEv5.txt", header=None)
chart = chart.drop(columns=1)
a = allRNA.shape[0]
b = chart.shape[0]
chart.columns = ["NONCODE", "name"]
list = []
for i in range(a):
    list.append(0)
list = pd.DataFrame(list)
print(list)
for i in range(a):
    index = chart[chart.name == allRNA.iloc[i, 1]].index.tolist()
    if index:
        list.iloc[i] = chart.iloc[index[0], 0]
    print(i)
print(list)
list.to_csv("chart.csv")
seq = pd.read_csv("chart.csv", header=None)
print(seq)
count = 0
for i in range(a):
    if seq.iloc[i, 1] == "0":
        count = count + 1
print(count)