import pandas as pd

df = pd.read_csv('data_unlabel_feature.csv', header=None)
print(df)
df = df.sample(frac=1)
print(df)
df.to_csv('unshuffled.csv', header=None, index=None)
