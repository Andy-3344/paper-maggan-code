import pandas as pd

with open('D:\论文\magcn-seqgan\seq-gan\eval.data', 'r') as f:
    lines = f.readlines()
lis = []
for line in lines:
    l = line.strip().split(' ')
    l = [int(s) for s in l]
    lis.append(l)
lis = pd.DataFrame(lis)
print(lis)
lis.to_csv('ungener1.csv', header=None, index=None)
