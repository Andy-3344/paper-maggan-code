import pandas as pd

df = pd.read_csv('D:\论文\seqGAN_pytorch-master\seqGAN_pytorch-master\data\LncRNA-seq1.csv', header=None)
print(df)
df = df.dropna(axis=0, how='any')
print(df)
df.to_csv('seq.csv', header=None)
