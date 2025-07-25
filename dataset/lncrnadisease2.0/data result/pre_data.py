import pandas as pd

list = []
for i in range(1, 258):
    list.append("{}".format(i))
# postive_sample2 = pd.read_csv('fword.csv', header=None, names=list, encoding="utf-8")
postive_sample1 = pd.read_csv('data_label_feature.csv', header=None, names=list, encoding="utf-8")
negative_sample = pd.read_csv("data_unlabel_feature.csv", header=None, names=list, encoding="utf-8")
negative_sample1 = pd.read_csv("unfword.csv", header=None, names=list, encoding="utf-8")
# postive_sample1 = pd.concat([postive_sample1, postive_sample2], axis=0)
negative_sample = pd.concat([negative_sample, negative_sample1], axis=0)
postive_sample = postive_sample1.dropna(axis=0, how='all', subset=None, inplace=False)
print(postive_sample)
print(negative_sample)
postive_sample.insert(loc=257, column='label', value=1)
negative_sample.insert(loc=257, column='label', value=0)

print(postive_sample)
print(negative_sample)
all_sample = pd.concat([postive_sample, negative_sample], axis=0)
print(all_sample)
all_sample = all_sample.dropna(axis=1, how='all', subset=None, inplace=False)
shuffled_all_sample = all_sample.sample(frac=1)

print(shuffled_all_sample)
shuffled_all_sample.to_csv('shuffled_all_sample_lncRNADisease2.0.csv', header=None, index=None)