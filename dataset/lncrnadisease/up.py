import pandas as pd
seq = pd.read_csv('rna_seq.csv', header=None)
list = []
for i in range(seq.shape[0]):
    list.append(seq.iloc[i, 2])
list = pd.DataFrame(list)
print(list)
list.to_csv('seq.csv', header=None, index=None)