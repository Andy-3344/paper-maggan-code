import pandas as pd

allrna = pd.read_csv("ALLRNA.csv", header=None)
chart = pd.read_csv("chart.csv", header=None)
dr = pd.read_csv("LncDisease.csv", header=None, encoding='ansi')
print(allrna)
print(chart)
print(dr)
dr.columns = ['rna', 'disease']
a = []
for i in range(allrna.shape[0]):
    print(i)
    if chart.iloc[i, 1] == '0':
        a.append(allrna.iloc[i, 1])
print(a)
for i in range(len(a)):
    index_name = dr[dr['rna'] == a[i]].index
    dr.drop(index_name, inplace=True)
print(dr)
dr.to_csv("D:\论文\magcn-seqgan\dataset\lncrnadisease2.0\pre\predata.csv", header=None)
